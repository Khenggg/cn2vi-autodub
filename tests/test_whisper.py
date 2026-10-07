import json
import sys
import types
from types import SimpleNamespace

from autodub.adapters import whisper
from autodub.storage import sha256_file

ASR_MANIFEST = {"id": "whisper-asr", "model_revision": "pinned-rev", "weights_sha256": "asr-hash"}
ALIGN_MANIFEST = {"id": "alignment-en", "model_revision": "align-rev", "weights_sha256": "align-hash"}


def test_asr_uses_local_pinned_large_turbo_and_keeps_detected_language(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "asr"
    output.mkdir()
    captured = {}

    class FakeModel:
        def __init__(self, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs

        def transcribe(self, path, **kwargs):
            captured["transcribe"] = kwargs
            return iter([SimpleNamespace(start=1.25, end=2.5, text="Hello there.")]), SimpleNamespace(
                language="en", language_probability=0.98)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    monkeypatch.setattr(whisper, "configure_offline", lambda _config: None)
    monkeypatch.setattr(whisper, "asset", lambda _config, _asset_id: (tmp_path / "whisper-asr", ASR_MANIFEST))
    monkeypatch.setattr(whisper.media, "probe_media", lambda *_args: {"duration_ms": 5000})

    result = whisper.run_asr(source, {"output_dir": str(output)})
    transcript = json.loads((output / "transcript.zh.json").read_text(encoding="utf-8"))

    assert captured["path"] == str(tmp_path / "whisper-asr")
    assert captured["kwargs"]["local_files_only"] is True
    assert captured["kwargs"]["compute_type"] == "int8_float16"
    assert captured["transcribe"]["language"] is None
    assert transcript["language"] == "en"
    assert transcript["source_sha256"] == sha256_file(source)
    assert transcript["segments"][0]["start_ms"] == 1250
    assert transcript["segments"][0]["words"] == []
    assert transcript["segments"][0]["needs_review"] is True
    assert result["language"] == "en"


def test_missing_language_aligner_preserves_coarse_bounds_and_requests_review(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"source")
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps({
        "schema_version": 1, "source_sha256": sha256_file(source), "language": "ja",
        "segments": [{"id": "seg1", "start_ms": 1000, "end_ms": 2000,
                      "zh_text": "こんにちは", "words": []}],
    }), encoding="utf-8")
    output = tmp_path / "align"
    output.mkdir()

    def asset(_config, asset_id):
        if asset_id == "whisper-asr":
            return tmp_path / asset_id, ASR_MANIFEST
        raise FileNotFoundError(asset_id)

    monkeypatch.setattr(whisper, "asset", asset)
    result = whisper.run_alignment(
        source, {"output_dir": str(output), "transcript_path": str(transcript_path)})

    report = json.loads((output / "alignment.json").read_text(encoding="utf-8"))
    segment = report["segments"][0]
    assert report["alignment_status"] == "UNSUPPORTED_LANGUAGE_NO_WORD_TIMES"
    assert report["required_asset"] == "alignment-ja"
    assert segment["start_ms"] == 1000 and segment["end_ms"] == 2000
    assert segment["words"] == []
    assert segment["action"] == "NEEDS_REVIEW" and segment["needs_review"] is True
    assert "pinned_alignment_asset:alignment-ja" in result["quality_evidence"]["missing"]


def test_alignment_uses_local_language_asset_and_emits_word_times(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"source")
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps({
        "schema_version": 1, "source_sha256": sha256_file(source), "language": "en",
        "segments": [{"id": "seg1", "start_ms": 1000, "end_ms": 3000,
                      "zh_text": "hello world.", "words": []}],
    }), encoding="utf-8")
    output = tmp_path / "align"
    output.mkdir()
    align_folder = tmp_path / "alignment-en"
    align_folder.mkdir()
    calls = {}

    def asset(_config, asset_id):
        return ((tmp_path / asset_id, ASR_MANIFEST) if asset_id == "whisper-asr"
                else (align_folder, ALIGN_MANIFEST))

    fake_whisperx = types.ModuleType("whisperx")

    def load_align_model(**kwargs):
        calls["load"] = kwargs
        return object(), {"language": "en"}

    def align(segments, _model, _metadata, _audio, _device, **kwargs):
        calls["align"] = kwargs
        return {"segments": [{"words": [
            {"word": "hello", "start": 1.1, "end": 1.7},
            {"word": "world", "start": 2.0, "end": 2.7},
        ]}]}

    fake_whisperx.load_align_model = load_align_model
    fake_whisperx.load_audio = lambda _path: object()
    fake_whisperx.align = align
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(whisper, "configure_offline", lambda _config: None)
    monkeypatch.setattr(whisper, "asset", asset)

    result = whisper.run_alignment(
        source, {"output_dir": str(output), "transcript_path": str(transcript_path),
                 "device": "cuda:0", "cache_root": str(tmp_path / "cache")})

    report = json.loads((output / "alignment.json").read_text(encoding="utf-8"))
    assert calls["load"]["language_code"] == "en"
    assert calls["load"]["model_name"] == str(align_folder)
    assert calls["load"]["model_dir"] == str(tmp_path / "cache" / "alignment")
    assert calls["align"]["return_char_alignments"] is False
    assert report["word_timing_available"] is True
    assert [word["t"] for word in report["segments"][0]["words"]] == ["hello", "world"]
    assert result["language"] == "en"
