"""Benchmark runner. Adapters must run a real model; absent adapters never produce success metrics."""
import argparse
import importlib
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from autodub.media import probe_media
from autodub.storage import atomic_json, sha256_file
from autodub.system import gpu_status


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
        gpu = gpu_status()
        if gpu["available"]:
            used = gpu["total_mb"] - gpu["free_mb"]
            self.peak_gpu = max(used, self.peak_gpu or 0)

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(0.5)


def revision() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def run_benchmark(stage: str, source: Path, output: Path, *, adapter: str | None = None,
                  config: dict | None = None, ffprobe_bin: str = "ffprobe") -> dict:
    if not source.is_file():
        raise ValueError("Input video not found")
    config = config or {}
    if stage != "probe" and not adapter:
        raise ValueError("A real model adapter module:function is required for this stage")
    metadata = probe_media(source, ffprobe_bin)
    sampler = Sampler()
    report = {"schema_version": 1, "stage": stage, "status": "RUNNING", "adapter": adapter,
              "source_sha256": sha256_file(source), "source_duration_ms": metadata["duration_ms"],
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "gpu": gpu_status(), "commit": revision()},
              "validation": "QUALITY_REVIEW_REQUIRED"}
    sampler.thread.start()
    start = time.perf_counter()
    exit_error = None
    try:
        if stage == "probe":
            result = {"quality_metrics": {}, "model_revision": None, "weights_sha256": None}
            probe_media(source, ffprobe_bin)
        else:
            module, name = adapter.split(":", 1)
            callback = getattr(importlib.import_module(module), name)
            # Adapter owns model invocation and must include human/automated quality measurements.
            result = callback(source, config)
            if not isinstance(result, dict) or not isinstance(result.get("quality_metrics"), dict):
                raise ValueError("Adapter must return quality_metrics, model_revision and weights_sha256")
            if not result.get("model_revision") or not result.get("weights_sha256"):
                raise ValueError("Model revision and weights checksum required")
        report.update(status="MEASURED", quality_metrics=result["quality_metrics"],
                      model_revision=result.get("model_revision"), weights_sha256=result.get("weights_sha256"))
    except Exception as error:
        # Do not serialize exception messages/config, which may contain provider credentials.
        report.update(status="FAILED", error_type=type(error).__name__)
        exit_error = error
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        sampler.stop.set()
        sampler.thread.join(timeout=4)
        report["metrics"] = {"stage_wall_ms": elapsed_ms, "ram_peak_bytes_sampled": sampler.peak_ram,
                             "gpu_device_peak_used_mb_sampled": sampler.peak_gpu,
                             "realtime_factor": round(elapsed_ms / metadata["duration_ms"], 6),
                             "gpu_measurement_scope": "whole_first_device_including_other_processes",
                             "sampling_interval_ms": 500}
        atomic_json(output, report)
    if exit_error:
        raise RuntimeError(f"Benchmark failed ({report['error_type']}); report saved") from None
    return report


def main(stage: str | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    if stage is None:
        parser.add_argument("--stage", required=True, choices=["probe", "asr", "bandit", "tts", "inpaint"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/benchmark-report.json"))
    parser.add_argument("--adapter", help="Python module:function; executes the measured model")
    parser.add_argument("--config", type=Path, help="Adapter JSON config; never copied to report")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    arguments = parser.parse_args()
    try:
        report = run_benchmark(stage or arguments.stage, arguments.input, arguments.output,
                               adapter=arguments.adapter, ffprobe_bin=arguments.ffprobe_bin,
                               config=json.loads(arguments.config.read_text()) if arguments.config else None)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Benchmark not completed: {error}", file=sys.stderr)
        return 1
    print(f"{report['stage']}: {report['status']}; report: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
