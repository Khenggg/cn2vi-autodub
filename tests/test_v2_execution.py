import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodub.adapters.ocr import build_engine
from autodub.storage import atomic_json, sha256_file
from autodub.v1_pipeline import RunDrained
from autodub.v2_pipeline import V2Pipeline


def test_cuda_ocr_requires_actual_sessions_and_does_not_silently_use_cpu(tmp_path, monkeypatch):
    calls = []

    class Session:
        def __init__(self, path, sess_options, providers):
            calls.append((Path(path).name, providers))

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def disable_fallback(self):
            calls.append("disable_fallback")

    class Engine:
        def __init__(self, params):
            self.params = params
            assert params["EngineConfig.onnxruntime.use_cuda"] is True
            options = params["EngineConfig.onnxruntime.cuda_ep_cfg"]
            for role, filename in (("text_det", "det.onnx"), ("text_rec", "rec.onnx"), ("text_cls", "cls.onnx")):
                session = Session(str(tmp_path / filename), None, [("CUDAExecutionProvider", options), "CPUExecutionProvider"])
                setattr(self, role, SimpleNamespace(session=SimpleNamespace(session=session)))

    ort = SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
                          SessionOptions=SimpleNamespace, InferenceSession=Session)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(RapidOCR=Engine))
    def resolve(config, identifier):
        return tmp_path, {}
    result, _ = build_engine({"ocr_require_cuda": True}, resolve_asset=resolve)
    assert len(result.autodub_cuda_sessions) == 3
    assert calls.count("disable_fallback") == 3
    assert sum(c[1][0][1]["gpu_mem_limit"] for c in calls if isinstance(c, tuple)) <= 1073741824
    assert result.params["Det.limit_type"] == "max"
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    with pytest.raises(RuntimeError, match="CUDA"):
        build_engine({"ocr_require_cuda": True}, resolve_asset=resolve)
    ort.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setattr(Session, "get_providers", lambda self: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="initialize CUDA"):
        build_engine({"ocr_require_cuda": True}, resolve_asset=resolve)


@pytest.fixture
def v2_repo(tmp_path, monkeypatch):
    monkeypatch.setattr("autodub.experimental_run.Sampler", lambda: SimpleNamespace(
        thread=SimpleNamespace(start=lambda: None, join=lambda: None), stop=SimpleNamespace(set=lambda: None),
        peak_ram=0, peak_gpu=0))
    root = tmp_path / "repo"
    for name, content in {"src/model.py": "value = 1\n", "config/cloud-runtime.json": '{"assets":["firered-asr2-aed"]}',
                          "benchmarks/models.lock.json": '{"schema_version":1,"models":[]}'}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video fixture")
    return root, source


def fake_v2(monkeypatch, *, missing_voice=False):
    calls = []

    def run(self, stage, source, config):
        calls.append(stage)
        folder = Path(config["output_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1}
        result = {"artifacts": [], "stage_status": "SUCCESS"}
        if stage == "V2_DETECT":
            audio = folder / "audio.wav"
            audio.write_bytes(b"original soundtrack")
            payload.update(audio=str(audio), candidate_duration_ms=2000)
            result["audio"] = str(audio)
        elif stage in {"V2_ASR", "V2_PUNCTUATION"}:
            payload["segments"] = [{"id": "line", "start_ms": 1000, "end_ms": 3000,
                "zh_text": "你先走", "action": "KEEP", "words": [], "timing_source": "VAD_WINDOW"}]
        elif stage == "V2_OCR":
            payload.update(events=[], frames=[], subtitle_profile={"y": 0.7, "h": 0.2})
        elif stage == "V2_TRANSLATION":
            result["segments"] = [{**s, "dub_vi": "Anh đi trước", "subtitle_vi": "Anh đi trước"} for s in config["segments"]]
        elif stage == "V2_TTS":
            clip = folder / "voice.wav"
            clip.write_bytes(b"Vietnamese voice")
            result["clips"] = {} if missing_voice else {"line": str(clip)}
        elif stage == "V2_MIX":
            audio = folder / "mixed.wav"
            audio.write_bytes(b"voice over")
            result.update(audio=str(audio), voice_added_count=1)
        elif stage == "V2_RESTORE":
            result["plan"] = str(folder / "result.json")
        elif stage == "V2_ENCODE":
            final = folder / "final.mp4"
            final.write_bytes(b"one pass final")
            result["video"] = str(final)
        path = folder / "result.json"
        atomic_json(path, payload)
        result["artifacts"] = [str(path)]
        for key in ("audio", "video"):
            if key in result:
                result["artifacts"].append(result[key])
        result["artifacts"].extend(result.get("clips", {}).values())
        return result

    monkeypatch.setattr("autodub.model_runner.ModelRunner.run", run)
    return calls


def test_v2_runs_without_legacy_models_and_dubs_candidate_timed_dialogue(v2_repo, tmp_path, monkeypatch):
    repo, source = v2_repo
    calls = fake_v2(monkeypatch)
    engine = V2Pipeline(source, tmp_path / "runs", {"duration_ms": 4000, "source_sha256": sha256_file(source),
                         "pipeline_version": "CN2VI-V2", "subtitle_mode": "replace"}, repo)
    report = engine.execute()
    assert report["pipeline_version"] == "CN2VI-V2"
    assert report["separation_used"] is False and report["generative_video_model"] is None
    assert report["voiceover_coverage"] == {"dialogue_count": 1, "generated_clips": 1, "voiced_count": 1}
    assert report["quality_and_sla_verified"] is False
    assert all(stage.startswith("V2_") for stage in calls)
    assert "SEPARATION" not in {s["stage"] for s in report["stages"]}
    assert len([s for s in report["stages"] if s["stage"] == "FINAL_ENCODE"]) == 1
    assert "PREVIEW" not in {s["stage"] for s in report["stages"]}
    assert Path(report["output_video"]).read_bytes() == b"one pass final"


def test_v2_missing_tts_cannot_be_reported_as_success(v2_repo, tmp_path, monkeypatch):
    repo, source = v2_repo
    fake_v2(monkeypatch, missing_voice=True)
    engine = V2Pipeline(source, tmp_path / "runs", {"duration_ms": 4000, "pipeline_version": "CN2VI-V2"}, repo)
    report = engine.execute()
    assert report["status"] == "PARTIAL"
    assert any(i["code"] == "VOICEOVER_COVERAGE_INCOMPLETE" for i in report["issues"])
    assert Path(report["output_video"]).is_file()


def test_v2_drain_preserves_completed_stage_and_resumes_frozen_code(v2_repo, tmp_path, monkeypatch):
    repo, source = v2_repo
    calls = fake_v2(monkeypatch)
    config = {"duration_ms": 4000, "pipeline_version": "CN2VI-V2"}
    engine = V2Pipeline(source, tmp_path / "runs", config, repo, should_stop=lambda: len(calls) >= 3)
    with pytest.raises(RunDrained):
        engine.execute()
    completed = set(engine.completed)
    before = len(calls)
    resumed = V2Pipeline(source, tmp_path / "runs", config, repo)
    report = resumed.execute()
    assert resumed.run.id == engine.run.id and report["resume_count"] == 1
    assert all(f"V2_{stage}" not in calls[before:] for stage in completed if stage in {"ASR", "OCR"})
    assert Path(report["output_video"]).is_file()
