import json
import sys
import types
from types import SimpleNamespace

import pytest

from autodub.adapters import whisper
from autodub.contracts import Segment
from autodub.storage import sha256_file

ASR_MANIFEST = {"id": "whisper-asr", "model_revision": "pinned-rev", "weights_sha256": "asr-hash"}
ALIGN_MANIFEST = {"id": "alignment-en", "model_revision": "align-rev", "weights_sha256": "align-hash"}


def test_asr_uses_local_pinned_large_turbo_and_keeps_detected_language(tmp_path, monkeypatch):
    source = tmp_path / "vocals.wav"
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
    monkeypatch.setattr(whisper.media, "probe_audio", lambda *_args: {"duration_ms": 5000})

    result = whisper.run_asr(source, {"output_dir": str(output),
                                     "original_source_sha256": "original-video-hash",
                                     "audio_input_kind": "separated_vocals"})
    transcript = json.loads((output / "transcript.zh.json").read_text(encoding="utf-8"))

    assert captured["path"] == str(tmp_path / "whisper-asr")
    assert captured["kwargs"]["local_files_only"] is True
    assert captured["kwargs"]["compute_type"] == "int8_float16"
    assert captured["transcribe"]["language"] is None
    assert transcript["language"] == "en"
    assert transcript["source_sha256"] == sha256_file(source)
    assert transcript["original_source_sha256"] == "original-video-hash"
    assert transcript["audio_input_kind"] == "separated_vocals"
    assert transcript["audio_timeline_offset_ms"] == 0
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


@pytest.mark.parametrize("failed_match", [False, True])
def test_alignment_uses_local_language_asset_and_emits_word_times(tmp_path, monkeypatch, failed_match):
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
            {"word": "hello", "start": 1.1, "end": 1.7, "score": 0.0 if failed_match else 0.9},
            {"word": "world", "start": 2.0, "end": 2.7, "score": 0.9},
        ]}]}

    fake_whisperx.load_align_model = load_align_model
    fake_whisperx.load_audio = lambda _path: object()
    fake_whisperx.align = align
    monkeypatch.setitem(sys.modules, "nltk", SimpleNamespace(data=SimpleNamespace(path=[]), download=None))
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
    assert calls["load"]["model_cache_only"] is True
    assert calls["align"]["return_char_alignments"] is False
    if failed_match:
        assert report["alignment_status"] == "PARTIAL_REVIEW_REQUIRED"
        assert report["word_timing_available"] is False
        assert report["segments"][0]["words"] == []
        assert report["segments"][0]["zh_text"] == "hello world."
        assert report["alignment_diagnostics"][0]["reason"] == "LOW_CTC_MATCH_SCORE"
    else:
        assert report["word_timing_available"] is True
        assert [word["t"] for word in report["segments"][0]["words"]] == ["hello", "world"]
    assert report["segments"][0]["start_ms"] == 1000
    assert report["segments"][0]["end_ms"] == 3000
    assert result["language"] == "en"


@pytest.mark.parametrize("words,reason", [
    ([{"word": "三", "start": 27.26, "end": 27.281, "score": 0.0}], "LOW_CTC_MATCH_SCORE"),
    ([{"word": "三", "start": 27.26, "end": 27.281, "score": 0.9}], "COLLAPSED_UTTERANCE_ALIGNMENT"),
    ([{"word": "三", "start": 27.26, "end": 27.86}], "MISSING_OR_INVALID_WORD_ALIGNMENT"),
    ([{"word": "三", "start": 27.0, "end": 27.86, "score": 0.9}], "WORD_OUTSIDE_ASR_WINDOW"),
])
def test_unusable_ctc_output_never_becomes_a_speech_slot(words, reason):
    source = Segment(id="countdown", start_ms=27260, end_ms=27860, zh_text="三")
    accepted, failure = whisper._validated_word_group(words, source)
    assert accepted == [] and failure == reason
    preserved = whisper._review_segments([source])[0]
    assert (preserved["start_ms"], preserved["end_ms"]) == (27260, 27860)
    assert preserved["words"] == [] and preserved["needs_review"] is True


def test_missing_or_overlapping_words_cannot_pass_alignment_validation():
    source = Segment(id="line", start_ms=1000, end_ms=3000, zh_text="hello world")
    _, reason = whisper._validated_word_group(
        [{"word": "hello", "start": 1.1, "end": 1.7, "score": 0.9}], source)
    assert reason == "INCOMPLETE_TEXT_ALIGNMENT"
    _, reason = whisper._validated_word_group([
        {"word": "hello", "start": 1.1, "end": 2.0, "score": 0.9},
        {"word": "world", "start": 1.8, "end": 2.7, "score": 0.9},
    ], source)
    assert reason == "NON_MONOTONIC_WORD_ALIGNMENT"


def test_whisperx_sentence_splits_stay_attached_to_their_source():
    source = Segment(id="line", start_ms=1000, end_ms=3000, zh_text="Hello. Goodbye.")
    calls = []

    def align(inputs, *_args, **_kwargs):
        calls.append(inputs)
        return {"segments": [
            {"words": [{"word": "Hello.", "start": 1.1, "end": 1.6, "score": 0.9}]},
            {"words": [{"word": "Goodbye.", "start": 2.0, "end": 2.7, "score": 0.9}]},
        ]}

    raw = whisper._aligned_source_words(SimpleNamespace(align=align), None, {}, None, "cuda", source)
    words, reason = whisper._validated_word_group(raw, source)
    assert reason is None
    assert [word["t"] for word in words] == ["Hello.", "Goodbye."]
    assert calls == [[{"start": 1.0, "end": 3.0, "text": "Hello. Goodbye."}]]
