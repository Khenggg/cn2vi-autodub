import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodub.adapters.v1_audio import sentence_segments, speech_windows
from autodub.adapters.v1_vision import infer_layout
from autodub.character_context import build_context, update_memory
from autodub.experimental_run import ExperimentalRun
from autodub.storage import atomic_json
from autodub.v1_pipeline import RunDrained, V1Pipeline


@pytest.fixture
def frozen_repo(tmp_path, monkeypatch):
    monkeypatch.setattr("autodub.experimental_run.Sampler", lambda: SimpleNamespace(
        thread=SimpleNamespace(start=lambda: None, join=lambda: None),
        stop=SimpleNamespace(set=lambda: None), peak_ram=0, peak_gpu=None))
    repo = tmp_path / "repo"
    for name, content in {"src/model.py": "value = 1\n", "config/cloud-runtime.json": '{"assets": ["new"]}',
                          "benchmarks/models.lock.json": '{"schema_version": 1, "models": []}'}.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    return repo, source


def test_freeze_resume_rejects_changed_code_or_input(frozen_repo, tmp_path):
    repo, source = frozen_repo
    config = {"duration_ms": 1000}
    run = ExperimentalRun(tmp_path / "runs", source, config, repo=repo, source_duration_ms=1000)
    resumed = ExperimentalRun.resume(run.root, source, config, repo=repo)
    assert resumed.report["resume_count"] == 1
    (repo / "src/model.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="original code version"):
        ExperimentalRun.resume(run.root, source, config, repo=repo)
    assert (run.package_root / "model.py").read_text() == "value = 1\n"


def test_quality_continues_and_fatal_has_no_fake_success(frozen_repo, tmp_path):
    repo, source = frozen_repo
    run = ExperimentalRun(tmp_path / "runs", source, {}, repo=repo, source_duration_ms=1000)
    result = run.execute("ASR", lambda: {"stage_status": "DEGRADED", "artifacts": [str(source)]})
    assert result is not None
    assert run.execute("TTS", lambda: {"stage_status": "FAILED_FATAL"}) is None
    report = run.finish(source)
    assert report["status"] == "PARTIAL"
    assert report["api_cost_vnd"] is None and report["total_estimate_vnd"] is None
    assert report["automatic_corrections"] == report["model_fallbacks"] == []
    assert (run.root / "run-report.md").is_file()


def fake_model_runner(monkeypatch, *, fail=None):
    calls = []

    def execute(self, stage, source, config):
        calls.append(stage)
        if stage == fail:
            raise RuntimeError("do not serialize credential content")
        root = Path(config["output_dir"])
        root.mkdir(parents=True)
        payload = {"schema_version": 1}
        result = {"stage_status": "SUCCESS"}
        if stage == "V1_SEPARATION":
            stems = {}
            for name in ("speech", "music", "effects"):
                target = root / f"{name}.wav"
                target.write_bytes(b"wav")
                stems[name] = str(target)
            payload["stems"] = stems
        if stage in {"V1_ASR", "V1_PUNCTUATION"}:
            payload["segments"] = [{"id": "s1", "start_ms": 0, "end_ms": 1000, "zh_text": "你好",
                "speaker_id": "speaker_0", "action": "DUB", "words": [{"t": "你好", "s": 0, "e": 800}]}]
        if stage == "V1_TRANSLATION":
            result["segments"] = [{**s, "subtitle_vi": "Xin chào", "dub_vi": "Xin chào"}
                                  for s in config["segments"]]
        if stage == "V1_TTS":
            result["clips"] = {}
        path = root / "result.json"
        atomic_json(path, payload)
        result["artifacts"] = [str(path)]
        return result

    monkeypatch.setattr("autodub.model_runner.ModelRunner.run", execute)
    def encode(self, video, audio, target, srt=None):
        target.write_bytes(b"rendered")
        return {"video": str(target), "artifacts": [str(target)]}
    monkeypatch.setattr(V1Pipeline, "encode", encode)
    return calls


def test_audio_dependency_failure_still_exports_video(frozen_repo, tmp_path, monkeypatch):
    repo, source = frozen_repo
    calls = fake_model_runner(monkeypatch, fail="V1_ASR")
    engine = V1Pipeline(source, tmp_path / "runs", {"duration_ms": 1000}, repo)
    report = engine.execute()
    assert Path(report["output_video"]).read_bytes() == b"rendered"
    assert report["status"] == "PARTIAL"
    assert calls == ["V1_SEPARATION", "V1_DIARIZATION", "V1_ASR", "V1_OCR"]
    assert {s["stage"] for s in report["stages"] if s["status"] == "SKIPPED"} >= {"PUNCTUATION", "TTS"}


def test_drain_resume_reuses_verified_model_artifacts(frozen_repo, tmp_path, monkeypatch):
    repo, source = frozen_repo
    calls = fake_model_runner(monkeypatch)
    config = {"duration_ms": 1000}
    root = tmp_path / "runs"
    first = V1Pipeline(source, root, config, repo, should_stop=lambda: len(calls) >= 3)
    with pytest.raises(RunDrained):
        first.execute()
    first_id = first.run.id
    second = V1Pipeline(source, root, config, repo)
    second.execute()
    assert second.run.id == first_id
    assert calls.count("V1_ASR") == calls.count("V1_SEPARATION") == 1
    state = json.loads(second.checkpoint.read_text())
    assert state["state"] == "FINISHED"
    assert state["completed"]["ASR"]["artifact_hashes"]


def test_checkpoint_tamper_is_not_silently_rerun(frozen_repo, tmp_path, monkeypatch):
    repo, source = frozen_repo
    calls = fake_model_runner(monkeypatch)
    first = V1Pipeline(source, tmp_path / "runs", {"duration_ms": 1000}, repo,
                       should_stop=lambda: len(calls) >= 1)
    with pytest.raises(RunDrained):
        first.execute()
    artifact = Path(first.completed["SEPARATION"]["result"]["artifacts"][0])
    artifact.write_text("{}")
    resumed = V1Pipeline(source, tmp_path / "runs", {"duration_ms": 1000}, repo)
    with pytest.raises(ValueError, match="checkpoint changed"):
        resumed.execute()
    assert calls.count("V1_SEPARATION") == 1


def test_native_sentence_boundaries_and_overlaps():
    segment = {"id": "a", "zh_text": "你好。再见！", "start_ms": 0, "end_ms": 3000,
               "words": [{"t": "你好", "s": 100, "e": 900}, {"t": "再见", "s": 1200, "e": 2200}]}
    split = sentence_segments(segment)
    assert [(s["zh_text"], s["start_ms"], s["end_ms"]) for s in split] == [
        ("你好。", 100, 900), ("再见！", 1200, 2200)]
    turns = [{"start_ms": 0, "end_ms": 1000, "speaker_id": "a"},
             {"start_ms": 500, "end_ms": 1500, "speaker_id": "b"}]
    assert speech_windows(turns, 2000) == [{"start_ms": 0, "end_ms": 1500, "speaker_ids": ["a", "b"]}]


def test_context_does_not_invent_character_identity():
    segments = [{"id": "s", "speaker_id": "speaker_0", "zh_text": "你好"}]
    context = build_context(segments, {}, {"characters": {"c1": {"name": "王"}}}, {})
    assert context["segments"]["s"]["character_id"] is None
    assert context["segments"]["s"]["uncertain"] is True
    memory = update_memory({"speaker_character_map": {"speaker_0": "c1"}}, segments, "run")
    assert "speaker_character_map" not in memory


def test_layout_requires_temporal_support_and_ignores_upper_text():
    line = {"box": [[300, 600], [700, 600], [700, 640], [300, 640]]}
    assert infer_layout([{"lines": [line]}], 1000, 720) is None
    layout = infer_layout([{"lines": [line]} for _ in range(4)], 1000, 720)
    assert layout and layout["source"] == "TEMPORAL_OCR_CANDIDATE"
