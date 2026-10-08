"""Immutable experiment snapshots and honest, best-effort stage reports."""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from autodub.benchmark import Sampler, revision, worktree_dirty
from autodub.checkpoint import fingerprint
from autodub.storage import atomic_json, sha256_file

STAGES = ("PREPARING", "SEPARATION", "DIARIZATION", "ASR", "PUNCTUATION", "OCR",
          "CONTEXT", "TRANSLATION", "TTS", "AUDIO_MIX", "PREVIEW", "SUBTITLE_DETECTION",
          "INPAINT", "SUBTITLE_RENDER", "FINAL_ENCODE", "SPEECH_DETECTION", "MULTIMODAL_GATE",
          "TEMPORAL_RESTORATION", "TIMELINE_MANIFEST")
STATUSES = {"SUCCESS", "DEGRADED", "FAILED_FATAL", "SKIPPED"}


class ExperimentalRun:
    """One frozen implementation/configuration, with no corrective edits or model fallback."""

    def __init__(self, root: Path, source: Path, config: dict, *, repo: Path,
                 source_duration_ms: int, cloud_rate: float = 6000):
        if source_duration_ms <= 0 or cloud_rate < 0:
            raise ValueError("Invalid experiment duration or cloud rate")
        self.id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.root = root / self.id
        self.root.mkdir(parents=True)
        self.started = time.perf_counter()
        self._report_lock = threading.RLock()
        self.prior_wall = 0.0
        self.config = json.loads(json.dumps(config))
        self.cloud_rate = cloud_rate
        self.duration_ms = source_duration_ms
        snapshot = self.root / "snapshot"
        shutil.copytree(repo / "src", snapshot / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for relative in ("config/cloud-runtime.json", "benchmarks/models.lock.json"):
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / relative, target)
        self.package_root = snapshot / "src"
        code_hashes = {path.relative_to(snapshot).as_posix(): sha256_file(path)
                       for path in sorted(snapshot.rglob("*")) if path.is_file()}
        self.report = {
            "schema_version": 1, "run_id": self.id, "pipeline_version": config.get("pipeline_version", "CN2VI-V1"),
            "git_commit": revision(), "git_dirty_at_start": worktree_dirty(),
            "code_snapshot_sha256": fingerprint(code_hashes),
            "model_lock_sha256": sha256_file(snapshot / "benchmarks/models.lock.json"),
            "config_sha256": fingerprint(self.config), "input_sha256": sha256_file(source),
            "source_duration_ms": source_duration_ms, "run_kind": config.get("run_kind", "COLD"),
            "model_process_policy": config.get("model_process_policy", "short-lived isolated process per stage; weights reload per stage"),
            "host_cache_state": "UNVERIFIED",
            "stages": [], "issues": [], "artifacts": [], "status": "RUNNING",
            "automatic_corrections": [], "model_fallbacks": [],
        }
        atomic_json(snapshot / "run-config.json", self.config)
        self._save()

    @classmethod
    def resume(cls, root: Path, source: Path, config: dict, *, repo: Path):
        report = json.loads((root / "run-report.json").read_text(encoding="utf-8"))
        saved = json.loads((root / "snapshot/run-config.json").read_text(encoding="utf-8"))
        if fingerprint(config) != report["config_sha256"] or fingerprint(saved) != fingerprint(config):
            raise ValueError("Resume configuration differs from the frozen experiment")
        if sha256_file(source) != report["input_sha256"]:
            raise ValueError("Resume source differs from the frozen experiment")
        snapshot = root / "snapshot"
        observed = {path.relative_to(snapshot).as_posix(): sha256_file(path)
                    for path in sorted(snapshot.rglob("*")) if path.is_file() and path.name != "run-config.json"
                    and "__pycache__" not in path.parts and path.suffix != ".pyc"}
        if fingerprint(observed) != report["code_snapshot_sha256"]:
            raise ValueError("Frozen code or model lock changed")
        if any(not (repo / relative).is_file() or sha256_file(repo / relative) != digest
               for relative, digest in observed.items()):
            raise ValueError("Restore the original code version to resume this experiment")
        value = cls.__new__(cls)
        value.id, value.root, value.config, value.report = report["run_id"], root, saved, report
        value.duration_ms, value.cloud_rate = report["source_duration_ms"], config.get("cloud_rate", 6000)
        value.package_root = snapshot / "src"
        value.prior_wall = report.get("active_wall_seconds", 0)
        value.started = time.perf_counter()
        value._report_lock = threading.RLock()
        value.report["resume_count"] = report.get("resume_count", 0) + 1
        value.report["status"] = "RUNNING"
        value._save()
        return value

    def issue(self, stage: str, code: str, *, segment_id: str | None = None,
              evidence: dict | None = None, downstream_effect: str = "") -> None:
        self.report["issues"].append({"stage": stage, "code": code, "segment_id": segment_id,
                                     "evidence": evidence or {}, "downstream_effect": downstream_effect,
                                     "automatic_correction": "NONE"})
        self._save()

    def execute(self, stage: str, operation, *, attempts: int = 1) -> dict | None:
        if stage not in STAGES or not 1 <= attempts <= 3:
            raise ValueError("Unsupported stage or retry count")
        record = {"stage": stage, "status": "FAILED_FATAL", "attempts": [],
                  "model_load_seconds": None, "preprocess_seconds": None,
                  "inference_seconds": None, "postprocess_seconds": None}
        stage_started = time.perf_counter()
        sampler = Sampler()
        sampler.thread.start()
        result = None
        try:
            for attempt in range(attempts):
                started = time.perf_counter()
                try:
                    result = operation()
                    if not isinstance(result, dict):
                        raise TypeError("Stage must return an artifact result")
                    status = result.get("stage_status", "SUCCESS")
                    if status not in STATUSES:
                        raise ValueError("Unknown stage status")
                    if status == "FAILED_FATAL":
                        raise RuntimeError("Stage returned no usable artifact")
                    record["status"] = status
                    metrics = result.get("metrics", {})
                    for name in ("model_load", "preprocess", "inference", "postprocess"):
                        value = metrics.get(f"{name}_ms")
                        record[f"{name}_seconds"] = value / 1000 if value is not None else None
                    record["metrics"] = metrics
                    record["runtime_packages"] = result.get("runtime_packages", {})
                    record["model_revision"] = result.get("model_revision")
                    record["weights_sha256"] = result.get("weights_sha256")
                    record["attempts"].append({"attempt": attempt + 1, "status": status,
                                               "wall_seconds": time.perf_counter() - started})
                    for raw_path in result.get("artifacts", []):
                        path = Path(raw_path).resolve(strict=True)
                        self.report["artifacts"].append({"stage": stage, "path": str(path),
                                                        "sha256": sha256_file(path)})
                    if status == "DEGRADED":
                        self.issue(stage, result.get("failure_code", "QUALITY_REVIEW_REQUIRED"),
                                   evidence=result.get("quality_evidence", {}),
                                   downstream_effect="Existing degraded artifact is used unchanged downstream")
                    break
                except Exception as error:
                    result = None
                    record["status"] = "FAILED_FATAL"
                    error_code = getattr(error, "error_code", type(error).__name__)
                    record["attempts"].append({"attempt": attempt + 1, "error_code": error_code,
                        "error_type": type(error).__name__, "wall_seconds": time.perf_counter() - started})
                    if attempt + 1 == attempts:
                        self.issue(stage, "NO_USABLE_ARTIFACT", evidence={"error_type": type(error).__name__,
                                                                         "error_code": error_code},
                                   downstream_effect="Unavailable artifact is not used; independent artifacts may continue")
        finally:
            sampler.stop.set()
            sampler.thread.join()
            record["wall_seconds"] = round(time.perf_counter() - stage_started, 6)
            record["peak_ram_bytes"] = sampler.peak_ram
            record["observed_device_used_mb"] = sampler.peak_gpu
            record["gpu_measurement_scope"] = "whole device; includes other processes"
            self.report["stages"].append(record)
            self._save()
        return result

    def skip(self, stage: str, reason: str) -> None:
        self.report["stages"].append({"stage": stage, "status": "SKIPPED", "reason": reason,
                                     "wall_seconds": 0})
        self._save()

    def finish(self, video: Path | None, *, api_cost_vnd: float | None = None) -> dict:
        wall = self.prior_wall + time.perf_counter() - self.started
        self.report.update(total_wall_seconds=round(wall, 6),
                           rtf=wall / (self.duration_ms / 1000),
                           cloud_compute_estimate_vnd=wall * self.cloud_rate / 3600,
                           api_cost_vnd=api_cost_vnd,
                           total_estimate_vnd=(wall * self.cloud_rate / 3600 + api_cost_vnd
                                               if api_cost_vnd is not None else None))
        if video is not None and video.is_file():
            self.report["output_video"] = str(video)
            self.report["output_sha256"] = sha256_file(video)
            self.report["status"] = "PARTIAL" if self.report["issues"] else "SUCCESS"
        else:
            self.report["status"] = "FAILED_FATAL"
            self.report["output_video"] = None
        measured = [s for s in self.report["stages"] if s.get("wall_seconds", 0)]
        self.report["bottleneck"] = max(measured, key=lambda s: s["wall_seconds"])["stage"] if measured else None
        for stage in self.report["stages"]:
            stage["fraction_of_wall_time"] = stage.get("wall_seconds", 0) / wall if wall else 0
        self._save()
        lines = [f"# Run {self.id}", "", f"Status: {self.report['status']}",
                 f"Source: {self.duration_ms / 1000:.3f} s", f"Total wall: {wall:.3f} s",
                 f"RTF: {self.report['rtf']:.3f}", "", "| Stage | Status | Seconds |",
                 "| --- | --- | ---: |"]
        lines += [f"| {s['stage']} | {s['status']} | {s.get('wall_seconds', 0):.3f} |"
                  for s in self.report["stages"]]
        lines += ["", "## Issues (no automatic corrections)", ""]
        lines += [f"- {issue['stage']}: {issue['code']}. {issue['downstream_effect']}"
                  for issue in self.report["issues"]]
        (self.root / "run-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.report

    def _save(self) -> None:
        with self._report_lock:
            self.report["active_wall_seconds"] = self.prior_wall + time.perf_counter() - self.started
            atomic_json(self.root / "run-report.json", self.report)
