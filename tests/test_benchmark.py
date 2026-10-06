import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodub.benchmark import DEFAULT_ADAPTERS, STAGES, _read_config, main, run_benchmark


def _fixture(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    monkeypatch.setattr("autodub.media.probe_media", lambda *_: {"duration_ms": 1000})
    monkeypatch.setattr("autodub.benchmark.probe_media", lambda *_: {"duration_ms": 1000})
    monkeypatch.setattr("autodub.benchmark.gpu_status", lambda: {"available": False})
    monkeypatch.setattr("autodub.benchmark.Sampler.sample", lambda *_: None)
    return source


def test_unconfigured_default_adapter_writes_honest_failure_report(tmp_path, monkeypatch):
    source = _fixture(monkeypatch, tmp_path)
    output = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="report saved"):
        run_benchmark("asr", source, output)
    report = json.loads(output.read_text())
    assert report["status"] == "FAILED"
    assert report["error_type"] == "ModuleNotFoundError"
    assert report["validation"] == "QUALITY_REVIEW_REQUIRED"
    assert report["adapter"] == "autodub.adapters.qwen:run_asr"


def test_propainter_is_available_as_a_manual_stage_only():
    assert "propainter" in STAGES
    assert DEFAULT_ADAPTERS["propainter"] == "autodub.adapters.propainter:run"


@pytest.mark.parametrize(
    ("config_ffprobe", "environment_ffprobe", "cli_ffprobe", "expected"),
    [
        ("config-probe", "environment-probe", "cli-probe", "cli-probe"),
        ("config-probe", "environment-probe", None, "config-probe"),
        (None, "environment-probe", None, "environment-probe"),
        (None, None, None, "ffprobe"),
    ],
)
def test_cli_ffprobe_precedence(tmp_path, monkeypatch, config_ffprobe, environment_ffprobe,
                               cli_ffprobe, expected):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"ffprobe_bin": config_ffprobe}), encoding="utf-8")
    output = tmp_path / "report.json"
    argv = ["autodub-benchmark", "--stage", "propainter", "--input", str(source),
            "--config", str(config), "--output", str(output)]
    if cli_ffprobe:
        argv.extend(["--ffprobe-bin", cli_ffprobe])
    monkeypatch.setattr("sys.argv", argv)
    if environment_ffprobe:
        monkeypatch.setenv("FFPROBE_BIN", environment_ffprobe)
    else:
        monkeypatch.delenv("FFPROBE_BIN", raising=False)
    captured = {}
    monkeypatch.setattr("autodub.benchmark.run_benchmark",
                        lambda *args, **kwargs: captured.update(kwargs))
    assert main() == 0
    assert captured["ffprobe_bin"] == expected
    assert captured["adapter"] == DEFAULT_ADAPTERS["propainter"]


def test_environment_records_only_bounded_worktree_dirty_state(tmp_path, monkeypatch):
    _fixture(monkeypatch, tmp_path)
    calls = []

    def git(command, **kwargs):
        calls.append((command, kwargs))
        stdout = "abc123\n" if command[1] == "rev-parse" else " M private-file.txt\n"
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr("autodub.benchmark.subprocess.run", git)
    from autodub import benchmark
    environment = benchmark._environment()
    assert environment["commit"] == "abc123"
    assert environment["worktree_dirty"] is True
    assert len(calls) == 2
    assert all(kwargs["timeout"] == 3 for _command, kwargs in calls)
    assert "private-file.txt" not in json.dumps(environment)


def test_probe_preflight_failure_still_writes_failure_report(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    output = tmp_path / "report.json"
    monkeypatch.setattr("autodub.benchmark.probe_media", lambda *_: (_ for _ in ()).throw(
        RuntimeError("could contain a credential")))
    with pytest.raises(RuntimeError, match="report saved"):
        run_benchmark("probe", source, output)
    report = json.loads(output.read_text())
    assert report["status"] == "FAILED"
    assert report["error_type"] == "RuntimeError"
    assert "credential" not in output.read_text()


def test_cli_config_preflight_failure_writes_report_without_config_contents(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    config = tmp_path / "secret.json"
    config.write_text('{"token":"CONFIG_SECRET"', encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["autodub-benchmark", "--stage", "asr", "--input", str(source),
                                     "--config", str(config), "--output", str(output)])
    assert main() == 1
    report_text = output.read_text()
    assert json.loads(report_text)["status"] == "FAILED"
    assert json.loads(report_text)["error_type"] == "JSONDecodeError"
    assert "CONFIG_SECRET" not in report_text


def test_relative_config_output_dir_resolves_from_config_file(tmp_path):
    config = tmp_path / "configs" / "asr.json"
    config.parent.mkdir()
    config.write_text('{"output_dir":"../artifacts"}', encoding="utf-8")
    assert _read_config(config)["output_dir"] == str((tmp_path / "artifacts").resolve())


def test_adapter_failure_does_not_leak_exception_or_config(tmp_path, monkeypatch):
    source = _fixture(monkeypatch, tmp_path)

    def fail(_source: Path, _config: dict):
        raise RuntimeError("API_KEY_DO_NOT_REPORT")

    monkeypatch.setattr("autodub.benchmark.importlib.import_module", lambda _: SimpleNamespace(run_asr=fail))
    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="report saved"):
        run_benchmark("asr", source, output, config={"api_key": "CONFIG_SECRET"})
    report_text = output.read_text()
    report = json.loads(report_text)
    assert report["status"] == "FAILED"
    assert "API_KEY_DO_NOT_REPORT" not in report_text
    assert "CONFIG_SECRET" not in report_text


def test_measured_report_keeps_review_gate_and_runner_timings(tmp_path, monkeypatch):
    source = _fixture(monkeypatch, tmp_path)
    artifacts = tmp_path / "outputs"
    artifacts.mkdir()
    (artifacts / "result.wav").write_bytes(b"audio")

    def run(_source: Path, config: dict):
        assert Path(config["output_dir"]).is_absolute()
        return {
            "model_revision": "fixture-version",
            "weights_sha256": "fixture-checksum",
            "quality_metrics": {"cer": None},
            "quality_evidence": {},
            "processed_media_ms": 2000,
            "metrics": {"model_load_ms": 12, "inference_ms": 120,
                        "credential": "must not be copied"},
            "artifacts": ["result.wav"],
        }

    monkeypatch.setattr("autodub.benchmark.importlib.import_module", lambda _: SimpleNamespace(run_asr=run))
    output = tmp_path / "report.json"
    report = run_benchmark("asr", source, output, adapter="fixture:run_asr", config={"output_dir": str(artifacts)})
    assert report["status"] == "MEASURED"
    assert report["validation"] == "QUALITY_REVIEW_REQUIRED"
    assert report["quality_evidence_missing"] == ["quality_evidence"]
    assert report["metrics"]["model_load_ms"] == 12
    assert report["metrics"]["inference_ms"] == 120
    assert report["metrics"]["sampler_stopped"] is True
    assert report["metrics"]["realtime_factor_basis_ms"] == 2000
    assert report["artifacts"] == ["result.wav"]
    assert "credential" not in output.read_text()


def test_report_preserves_review_evidence_and_allocator_peaks(tmp_path, monkeypatch):
    source = _fixture(monkeypatch, tmp_path)
    evidence = {"status": "REVIEW_REQUIRED", "missing": ["human_review"]}
    monkeypatch.setattr("autodub.benchmark.importlib.import_module", lambda _: SimpleNamespace(run=lambda *_: {
        "model_revision": "fixture-v1", "weights_sha256": "fixture-hash",
        "quality_metrics": {"cer": None}, "quality_evidence": evidence,
        "metrics": {"gpu_allocator_peak_bytes": 1234, "gpu_reserved_peak_bytes": 2345},
    }))
    output = tmp_path / "review.json"
    report = run_benchmark("asr", source, output, adapter="fixture:run")
    assert report["quality_evidence"] == evidence
    assert report["quality_evidence_missing"] == ["human_review"]
    assert report["metrics"]["gpu_allocator_peak_bytes"] == 1234
    assert report["metrics"]["gpu_reserved_peak_bytes"] == 2345


def test_adapter_artifact_must_remain_under_output_dir(tmp_path, monkeypatch):
    source = _fixture(monkeypatch, tmp_path)
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    monkeypatch.setattr("autodub.benchmark.importlib.import_module", lambda _: SimpleNamespace(run_asr=lambda *_: {
        "model_revision": "v1", "weights_sha256": "abc", "quality_metrics": {}, "artifacts": [outside]}))
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="report saved"):
        run_benchmark("asr", source, output, adapter="fixture:run_asr")
    assert json.loads(output.read_text())["status"] == "FAILED"
