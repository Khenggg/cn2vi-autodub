import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodub.benchmark import run_benchmark


def test_benchmark_requires_adapter_and_writes_honest_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    monkeypatch.setattr("autodub.benchmark.probe_media", lambda *_: {"duration_ms": 1000})
    monkeypatch.setattr("autodub.benchmark.gpu_status", lambda: {"available": False})
    monkeypatch.setattr("autodub.benchmark.Sampler.sample", lambda *_: None)
    with pytest.raises(ValueError, match="real model"):
        run_benchmark("asr", source, tmp_path / "missing.json")
    assert not (tmp_path / "missing.json").exists()
    def fail(source: Path, config: dict):
        raise RuntimeError("API_KEY_DO_NOT_REPORT")
    monkeypatch.setattr("autodub.benchmark.importlib.import_module", lambda _: SimpleNamespace(run=fail))
    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError):
        run_benchmark("asr", source, output, adapter="fixture:run")
    report = json.loads(output.read_text())
    assert report["status"] == "FAILED" and report["validation"] == "QUALITY_REVIEW_REQUIRED"
    assert "API_KEY_DO_NOT_REPORT" not in output.read_text()


def test_measured_result_still_requires_quality_review(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    monkeypatch.setattr("autodub.benchmark.probe_media", lambda *_: {"duration_ms": 1000})
    monkeypatch.setattr("autodub.benchmark.gpu_status", lambda: {"available": False})
    monkeypatch.setattr("autodub.benchmark.Sampler.sample", lambda *_: None)
    monkeypatch.setattr("autodub.benchmark.importlib.import_module", lambda _: SimpleNamespace(
        run=lambda *_: {"model_revision": "fixture-version", "weights_sha256": "fixture-checksum", "quality_metrics": {"cer": None}}))
    report = run_benchmark("asr", source, tmp_path / "report.json", adapter="fixture:run")
    assert report["status"] == "MEASURED" and report["quality_metrics"]["cer"] is None
    assert report["validation"] == "QUALITY_REVIEW_REQUIRED"
