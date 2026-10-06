import json
import subprocess
import sys
from pathlib import Path

import pytest

from autodub import bench_suite, benchmark


def _plan(tmp_path, jobs):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"test")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"output_dir": "adapter-output"}), encoding="utf-8")
    plan = tmp_path / "suite.json"
    plan.write_text(json.dumps({"schema_version": 1, "hourly_rate_vnd": 72000,
                                "jobs": jobs(media, config)}), encoding="utf-8")
    return plan, media, config


def test_dry_run_validates_relative_files_and_python_without_model_run(tmp_path):
    plan, media, config = _plan(tmp_path, lambda source, cfg: [
        {"id": "asr-1", "stage": "asr", "input": source.name, "config": cfg.name,
         "python": sys.executable}])
    summary_path = tmp_path / "dry-run.json"
    summary = bench_suite.dry_run(plan, summary_path)
    assert summary["status"] == "DRY_RUN"
    assert summary["jobs"][0]["status"] == "READY"
    assert summary["jobs"][0]["input"] == str(media.resolve())
    assert summary["jobs"][0]["config"] == str(config.resolve())
    assert json.loads(summary_path.read_text())["validation"] == "QUALITY_REVIEW_REQUIRED"


def test_suite_continues_after_timeout_and_measures_actual_elapsed_cost(tmp_path, monkeypatch):
    plan, _media, config = _plan(tmp_path, lambda source, cfg: [
        {"id": "hang", "stage": "asr", "input": source.name, "config": cfg.name, "timeout_seconds": 1},
        {"id": "next", "stage": "asr", "input": source.name, "config": cfg.name}])
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    plan_payload.update(corpus_kind="synthetic_smoke", representative_quality_pass=True)
    plan.write_text(json.dumps(plan_payload), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    calls = []

    class FakeProcess:
        next_pid = 987654321

        def __init__(self, command, **_kwargs):
            self.command = command
            self.pid = self.next_pid
            self.next_pid += 1
            self.calls = len(calls)
            calls.append(command)

        def wait(self, timeout=None):
            if self.calls == 0:
                raise subprocess.TimeoutExpired(self.command, timeout)
            output = Path(self.command[self.command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "status": "MEASURED", "quality_evidence_missing": ["ground_truth"],
                "quality_evidence": {"status": "REVIEW_REQUIRED", "missing": ["ground_truth"]},
                "corpus_kind": "synthetic_smoke", "representative_quality_pass": True,
            }), encoding="utf-8")
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(bench_suite.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(bench_suite, "_terminate_tree", lambda _process: None)
    ticks = iter([0.0, 1.2, 2.0, 3.0])
    monkeypatch.setattr(bench_suite.time, "perf_counter", lambda: next(ticks))
    summary = bench_suite.run_suite(plan, summary_path)
    assert summary["status"] == "INCOMPLETE"
    assert [job["status"] for job in summary["jobs"]] == ["TIMED_OUT", "MEASURED"]
    assert summary["jobs"][1]["quality_evidence_missing"] == ["ground_truth"]
    assert summary["jobs"][1]["quality_evidence"] == {
        "status": "REVIEW_REQUIRED", "missing": ["ground_truth"]}
    assert summary["jobs"][1]["representative_quality_pass"] is False
    assert summary["corpus_kind"] == "synthetic_smoke"
    assert summary["representative_quality_pass"] is False
    assert summary["elapsed_ms"] == 2200
    assert summary["cost_vnd"] == 44
    assert len(calls) == 2
    assert all(command[1:4] == ["-m", "autodub.benchmark", "--stage"] for command in calls)


def test_handwritten_representative_plan_never_claims_quality_pass(tmp_path, monkeypatch):
    plan, _media, config = _plan(tmp_path, lambda source, cfg: [
        {"id": "manual", "stage": "asr", "input": source.name, "config": cfg.name}])
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload.update(corpus_kind="representative", representative_quality_pass=True)
    plan.write_text(json.dumps(payload), encoding="utf-8")

    class MeasuredChild:
        pid = 987654321

        def __init__(self, command, **_kwargs):
            self.command = command

        def wait(self, timeout=None):
            output = Path(self.command[self.command.index("--output") + 1])
            output.write_text(json.dumps({
                "status": "MEASURED", "corpus_kind": "representative",
                "representative_quality_pass": True,
            }), encoding="utf-8")
            return 0

    monkeypatch.setattr(bench_suite.subprocess, "Popen", MeasuredChild)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(bench_suite.time, "perf_counter", lambda: next(ticks))
    summary = bench_suite.run_suite(plan, tmp_path / "summary.json")
    assert summary["status"] == "MEASURED"
    assert summary["corpus_kind"] == "representative"
    assert summary["representative_quality_pass"] is False
    assert summary["jobs"][0]["representative_quality_pass"] is False


def test_propainter_suite_command_forwards_configured_ffprobe_to_child(tmp_path, monkeypatch):
    plan, _media, config = _plan(tmp_path, lambda source, cfg: [
        {"id": "manual-propainter", "stage": "propainter", "input": source.name, "config": cfg.name}])
    ffprobe = r"C:\tools\ffmpeg\bin\ffprobe.exe"
    config.write_text(json.dumps({"output_dir": "adapter-output", "ffprobe_bin": ffprobe}), encoding="utf-8")
    _plan_payload, jobs = bench_suite.load_plan(plan)
    command = bench_suite._command(jobs[0])
    assert command[command.index("--adapter") + 1] == "autodub.adapters.propainter:run"
    assert Path(command[command.index("--config") + 1]) == config.resolve()

    captured = {}
    monkeypatch.setattr("autodub.benchmark.run_benchmark",
                        lambda *args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr("sys.argv", ["autodub.benchmark", *command[3:]])
    assert benchmark.main() == 0
    assert captured["ffprobe_bin"] == ffprobe


def test_suite_continues_after_child_failure(tmp_path, monkeypatch):
    plan, _media, config = _plan(tmp_path, lambda source, cfg: [
        {"id": "bad", "stage": "asr", "input": source.name, "config": cfg.name},
        {"id": "next", "stage": "asr", "input": source.name, "config": cfg.name}])
    calls = []

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            self.command = command
            self.pid = 987654321
            self.index = len(calls)
            calls.append(command)

        def wait(self, timeout=None):
            output = Path(self.command[self.command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            state = "FAILED" if self.index == 0 else "MEASURED"
            error = "RuntimeError" if self.index == 0 else None
            output.write_text(json.dumps({"status": state, "error_type": error}), encoding="utf-8")
            return 1 if self.index == 0 else 0

        def poll(self):
            return 0

    monkeypatch.setattr(bench_suite.subprocess, "Popen", FakeProcess)
    ticks = iter([0.0, 0.1, 0.2, 0.4])
    monkeypatch.setattr(bench_suite.time, "perf_counter", lambda: next(ticks))
    summary = bench_suite.run_suite(plan, tmp_path / "summary.json")
    assert summary["status"] == "FAILED"
    assert [job["status"] for job in summary["jobs"]] == ["FAILED", "MEASURED"]
    assert summary["jobs"][0]["error_type"] == "RuntimeError"
    assert len(calls) == 2


def test_plan_rejects_unconfigured_model_stage(tmp_path):
    plan, media, _config = _plan(tmp_path, lambda source, _cfg: [
        {"id": "asr", "stage": "asr", "input": source.name}])
    with pytest.raises(bench_suite.PlanError, match="requires a config"):
        bench_suite.load_plan(plan)
