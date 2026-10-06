"""Benchmark runner. Adapters must run a real model; absent adapters never produce success metrics."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from autodub.media import probe_media
from autodub.storage import atomic_json, sha256_file
from autodub.system import gpu_status

STAGES = ("probe", "asr", "align", "tts", "bandit", "ocr", "inpaint", "propainter")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTERS = {
    "asr": "autodub.adapters.qwen:run_asr",
    "align": "autodub.adapters.qwen:run_alignment",
    "tts": "autodub.adapters.vieneu:run",
    "bandit": "autodub.adapters.bandit:run",
    "ocr": "autodub.adapters.ocr:run",
    "inpaint": "autodub.adapters.inpaint:run",
    "propainter": "autodub.adapters.propainter:run",
}


class Sampler:
    def __init__(self):
        self.stop = threading.Event()
        self.peak_ram = 0
        self.peak_gpu: int | None = None
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        processes = [self.process, *self.process.children(recursive=True)]
        total = 0
        for process in processes:
            try:
                total += process.memory_info().rss
            except psutil.Error:
                pass
        self.peak_ram = max(total, self.peak_ram)
        try:
            gpu = gpu_status()
        except Exception:
            return
        if gpu.get("available"):
            used = gpu["total_mb"] - gpu["free_mb"]
            self.peak_gpu = max(used, self.peak_gpu or 0)

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(0.5)


def revision() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=3, cwd=REPO_ROOT)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def worktree_dirty() -> bool | None:
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, timeout=3, cwd=REPO_ROOT)
        return bool(result.stdout.strip()) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _environment() -> dict:
    try:
        gpu = gpu_status()
    except Exception:
        gpu = {"available": False, "measurement_error": True}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "gpu": gpu, "commit": revision(), "worktree_dirty": worktree_dirty()}


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _adapter_metrics(result: dict) -> dict:
    """Copy only documented numeric measurements; never serialize arbitrary adapter/config data."""
    allowed = {
        "model_load_ms", "cold_load_ms", "load_ms", "inference_ms", "processed_media_ms",
        "device_allocator_peak_bytes", "device_allocator_peak_mb", "peak_allocated_bytes",
        "peak_reserved_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
        "gpu_allocator_peak_bytes", "gpu_reserved_peak_bytes",
    }
    values = result.get("metrics", {})
    if not isinstance(values, dict):
        return {}
    return {key: number for key, value in values.items()
            if key in allowed and (number := _number(value)) is not None}


def _artifact_paths(result: dict, output_dir: Path) -> list[str]:
    artifacts = result.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("Adapter artifacts must be a list")
    root = output_dir.resolve()
    paths = []
    for item in artifacts:
        if not isinstance(item, (str, Path)):
            raise ValueError("Adapter artifacts must be paths")
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root) or resolved == root:
            raise ValueError("Adapter artifact path must stay inside output_dir")
        if not resolved.is_file():
            raise ValueError("Adapter artifact file is missing")
        paths.append(resolved.relative_to(root).as_posix())
    return paths


def _base_report(stage: str, adapter: str | None) -> dict:
    return {
        "schema_version": 1,
        "stage": stage,
        "status": "RUNNING",
        "adapter": adapter,
        "source_sha256": None,
        "source_duration_ms": None,
        "corpus_kind": None,
        "representative_quality_pass": False,
        "environment": _environment(),
        "quality_metrics": {},
        "quality_evidence": {},
        "quality_evidence_missing": [],
        "model_revision": None,
        "weights_sha256": None,
        "artifacts": [],
        "validation": "QUALITY_REVIEW_REQUIRED",
    }


def _safe_write(output: Path, report: dict) -> None:
    # atomic_json is the single report writer used by benchmark consumers.
    atomic_json(output, report)


def _write_cli_failure(stage: str, source: Path, output: Path, adapter: str | None,
                       error_type: str) -> None:
    report = _base_report(stage, adapter)
    report.update(status="FAILED", error_type=error_type,
                  metrics={"stage_wall_ms": 0, "adapter_import_ms": None, "adapter_wall_ms": None,
                           "sampler_join_ms": 0, "sampler_stopped": True,
                           "ram_peak_bytes_sampled": 0, "gpu_device_peak_used_mb_sampled": None,
                           "realtime_factor": None, "gpu_measurement_scope":
                           "whole_first_device_including_other_processes", "sampling_interval_ms": 500})
    try:
        if source.is_file():
            report["source_sha256"] = sha256_file(source)
    except OSError:
        pass
    _safe_write(output, report)


def run_benchmark(stage: str, source: Path, output: Path, *, adapter: str | None = None,
                  config: dict | None = None, ffprobe_bin: str = "ffprobe") -> dict:
    source, output = Path(source), Path(output)
    effective_adapter = adapter or DEFAULT_ADAPTERS.get(stage)
    report = _base_report(stage, effective_adapter)
    sampler = Sampler()
    sampler_started = False
    stage_start = time.perf_counter()
    adapter_import_ms = None
    adapter_wall_ms = None
    error: BaseException | None = None
    interrupted = False
    metadata = None
    adapter_measurements = {}

    try:
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            raise ValueError("Adapter config must be an object")
        else:
            config = dict(config)
        corpus_kind = config.get("corpus_kind")
        if corpus_kind is not None and corpus_kind not in {"representative", "synthetic_smoke"}:
            raise ValueError("Unsupported benchmark corpus_kind")
        report["corpus_kind"] = corpus_kind
        # This runner records measurements but never accepts representative quality.
        report["representative_quality_pass"] = False
        if stage not in STAGES:
            raise ValueError("Unsupported benchmark stage")
        if not source.is_file():
            raise ValueError("Input video not found")
        report["source_sha256"] = sha256_file(source)
        metadata = probe_media(source, ffprobe_bin)
        duration_ms = metadata.get("duration_ms")
        if not isinstance(duration_ms, int) or duration_ms <= 0:
            raise ValueError("FFprobe returned an invalid duration")
        report["source_duration_ms"] = duration_ms
        if stage != "probe" and not effective_adapter:
            raise ValueError("A real model adapter module:function is required for this stage")

        sampler.thread.start()
        sampler_started = True
        if stage == "probe":
            result = {"quality_metrics": {}, "model_revision": None, "weights_sha256": None}
            probe_media(source, ffprobe_bin)
        else:
            if ":" not in effective_adapter:
                raise ValueError("Adapter must use module:function syntax")
            module, name = effective_adapter.split(":", 1)
            import_start = time.perf_counter()
            callback = getattr(importlib.import_module(module), name)
            adapter_import_ms = round((time.perf_counter() - import_start) * 1000)
            output_dir_value = config.get("output_dir")
            output_dir = Path(output_dir_value) if output_dir_value else output.parent / "output"
            if not output_dir.is_absolute():
                output_dir = Path.cwd() / output_dir
            output_dir = output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            adapter_config = {**config, "output_dir": str(output_dir)}
            adapter_start = time.perf_counter()
            result = callback(source, adapter_config)
            adapter_wall_ms = round((time.perf_counter() - adapter_start) * 1000)
            if not isinstance(result, dict) or not isinstance(result.get("quality_metrics"), dict):
                raise ValueError("Adapter must return quality_metrics, model_revision and weights_sha256")
            if not result.get("model_revision") or not result.get("weights_sha256"):
                raise ValueError("Model revision and weights checksum required")
            if not isinstance(result["model_revision"], str) or not isinstance(result["weights_sha256"], str):
                raise ValueError("Model revision and weights checksum must be strings")
            # Ensure metric payload can be represented in JSON before writing the final report.
            try:
                json.dumps(result["quality_metrics"], allow_nan=False)
            except (TypeError, ValueError):
                raise ValueError("Adapter quality_metrics must contain finite JSON values") from None
            report["quality_metrics"] = result["quality_metrics"]
            report["model_revision"] = str(result["model_revision"])
            report["weights_sha256"] = str(result["weights_sha256"])
            evidence = result.get("quality_evidence", {})
            if evidence is not None and not isinstance(evidence, dict):
                raise ValueError("Adapter quality_evidence must be an object")
            try:
                json.dumps(evidence or {}, allow_nan=False)
            except (TypeError, ValueError):
                raise ValueError("Adapter quality_evidence must contain finite JSON values") from None
            report["quality_evidence"] = evidence or {}
            evidence_missing = (evidence or {}).get("missing")
            if evidence_missing is not None and (
                    not isinstance(evidence_missing, list)
                    or not all(isinstance(item, str) for item in evidence_missing)):
                raise ValueError("Adapter quality_evidence.missing must be a list of field names")
            missing = result.get("quality_evidence_missing")
            if missing is None:
                missing = evidence_missing if evidence_missing is not None else ["quality_evidence"]
            elif not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
                raise ValueError("Adapter quality_evidence_missing must be a list of field names")
            if evidence_missing is not None:
                missing = [*missing, *(item for item in evidence_missing if item not in missing)]
            report["quality_evidence_missing"] = missing
            adapter_measurements = _adapter_metrics(result)
            if "processed_media_ms" not in adapter_measurements:
                processed_media_ms = _number(result.get("processed_media_ms"))
                if processed_media_ms is not None:
                    adapter_measurements["processed_media_ms"] = processed_media_ms
            if "model_load_ms" not in adapter_measurements:
                load_ms = adapter_measurements.get("cold_load_ms", adapter_measurements.get("load_ms"))
                if load_ms is not None:
                    adapter_measurements["model_load_ms"] = load_ms
            report["artifacts"] = _artifact_paths(result, output_dir)

        report["status"] = "MEASURED"
    except KeyboardInterrupt as caught:
        interrupted = True
        error = caught
        report.update(status="INTERRUPTED", error_type="KeyboardInterrupt")
    except BaseException as caught:
        error = caught
        report.update(status="FAILED", error_type=type(caught).__name__)
    finally:
        elapsed_ms = round((time.perf_counter() - stage_start) * 1000)
        sampler.stop.set()
        sampler_join_ms = 0
        sampler_stopped = True
        if sampler_started:
            join_start = time.perf_counter()
            sampler.thread.join(timeout=4)
            sampler_join_ms = round((time.perf_counter() - join_start) * 1000)
            sampler_stopped = not sampler.thread.is_alive()
        metrics = {
            "stage_wall_ms": elapsed_ms,
            "adapter_import_ms": adapter_import_ms,
            "adapter_wall_ms": adapter_wall_ms,
            "sampler_join_ms": sampler_join_ms,
            "sampler_stopped": sampler_stopped,
            "ram_peak_bytes_sampled": sampler.peak_ram,
            "gpu_device_peak_used_mb_sampled": sampler.peak_gpu,
            "realtime_factor": None,
            "gpu_measurement_scope": "whole_first_device_including_other_processes",
            "sampling_interval_ms": 500,
            **adapter_measurements,
        }
        processed_ms = adapter_measurements.get("processed_media_ms")
        if processed_ms is None and isinstance(metadata, dict):
            processed_ms = metadata.get("duration_ms")
        if isinstance(processed_ms, (int, float)) and processed_ms > 0:
            inference_ms = adapter_measurements.get("inference_ms")
            wall_basis_ms = inference_ms if inference_ms is not None else adapter_wall_ms
            if wall_basis_ms is None:
                wall_basis_ms = elapsed_ms
            basis_name = ("adapter_inference_ms" if inference_ms is not None else
                          "adapter_wall_ms" if adapter_wall_ms is not None else "stage_wall_ms")
            metrics["realtime_factor"] = round(wall_basis_ms / processed_ms, 6)
            metrics["realtime_factor_basis_ms"] = processed_ms
            metrics["realtime_factor_wall_basis"] = basis_name
        report["metrics"] = metrics
        _safe_write(output, report)

    if interrupted:
        raise KeyboardInterrupt
    if error is not None:
        if isinstance(error, ValueError):
            raise ValueError(f"Benchmark failed ({report['error_type']}); report saved") from None
        raise RuntimeError(f"Benchmark failed ({report['error_type']}); report saved") from None
    return report


def _read_config(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Adapter config must be a JSON object")
    if payload.get("output_dir"):
        output_dir = Path(payload["output_dir"])
        if not output_dir.is_absolute():
            payload["output_dir"] = str((path.resolve().parent / output_dir).resolve())
    return payload


def main(stage: str | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    if stage is None:
        parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/benchmark-report.json"))
    parser.add_argument("--adapter", help="Python module:function; executes the measured model")
    parser.add_argument("--config", type=Path, help="Adapter JSON config; never copied to report")
    parser.add_argument("--ffprobe-bin", help="ffprobe executable (overrides config and FFPROBE_BIN)")
    arguments = parser.parse_args()
    selected_stage = stage or arguments.stage
    selected_adapter = arguments.adapter or DEFAULT_ADAPTERS.get(selected_stage)
    try:
        config = _read_config(arguments.config)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        try:
            _write_cli_failure(selected_stage, arguments.input, arguments.output, selected_adapter,
                               type(error).__name__)
        except OSError:
            pass
        print(f"Benchmark not completed: {type(error).__name__}", file=sys.stderr)
        return 1
    ffprobe_bin = (arguments.ffprobe_bin or (config or {}).get("ffprobe_bin")
                   or os.getenv("FFPROBE_BIN") or "ffprobe")
    try:
        run_benchmark(selected_stage, arguments.input, arguments.output, adapter=selected_adapter,
                      ffprobe_bin=ffprobe_bin, config=config)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Benchmark not completed: {type(error).__name__}", file=sys.stderr)
        return 1
    print(f"{selected_stage}: MEASURED; report: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
