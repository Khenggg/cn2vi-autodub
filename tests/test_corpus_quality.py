import json
import os
import shutil
import wave
from array import array

import pytest

from autodub import benchmedia
from autodub.corpus import create_synthetic_smoke, validate_manifest
from autodub.quality import audio_qc, cer, word_boundary_error


def _manifest(tmp_path, monkeypatch, *, kind="representative", duration_ms=600_000,
              categories=None, cases=None):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake media bytes")
    if cases is None:
        categories = categories or ["speech_clear", "music_loud", "nonverbal", "overlap",
                                    "subtitle_static", "subtitle_motion", "dark"]
        cases = [{"id": category, "source": source.name, "start_ms": 0, "end_ms": duration_ms,
                  "category": category, "source_sha256": ""} for category in categories]
    for case in cases:
        if case.get("source_sha256") == "":
            import hashlib
            case["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"schema_version": 1, "kind": kind, "cases": cases}), encoding="utf-8")
    monkeypatch.setattr("autodub.corpus.probe_media", lambda *_args: {"duration_ms": duration_ms})
    return path, source


def test_representative_corpus_requires_category_coverage_and_ten_minutes(tmp_path, monkeypatch):
    path, _ = _manifest(tmp_path, monkeypatch, duration_ms=90_000)
    report = validate_manifest(path)
    assert report["status"] == "READY_FOR_REPRESENTATIVE_QUALITY_EVALUATION"
    assert report["total_case_duration_ms"] == 630_000
    assert report["representative_quality_pass"] is False

    short_path, _ = _manifest(tmp_path, monkeypatch, duration_ms=60_000,
                               categories=["speech_clear", "music_loud", "nonverbal", "overlap",
                                           "subtitle_static", "subtitle_motion", "dark"])
    short = validate_manifest(short_path)
    assert short["status"] == "INVALID"
    assert any("600000" in message for message in short["errors"])


def test_corpus_rejects_duplicate_ids_bad_hash_and_out_of_bounds_timing(tmp_path, monkeypatch):
    cases = [
        {"id": "same", "source": "clip.mp4", "start_ms": 0, "end_ms": 10,
         "category": "speech_clear", "source_sha256": "0" * 64,
         "words": [{"t": "你", "s": 5, "e": 11}]},
        {"id": "same", "source": "clip.mp4", "start_ms": 0, "end_ms": 10,
         "category": "speech_clear"},
    ]
    path, _ = _manifest(tmp_path, monkeypatch, kind="synthetic_smoke", duration_ms=10, cases=cases)
    report = validate_manifest(path)
    assert report["status"] == "INVALID"
    assert any("duplicate id" in message for message in report["errors"])
    assert any("does not match" in message for message in report["errors"])
    assert any("word times" in message for message in report["errors"])


def test_corpus_rejects_manifest_path_escape(tmp_path, monkeypatch):
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"x")
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"schema_version": 1, "kind": "synthetic_smoke", "cases": [
        {"id": "escape", "source": "../outside.mp4", "start_ms": 0, "end_ms": 1,
         "category": "nonverbal"}]}), encoding="utf-8")
    report = validate_manifest(path)
    assert report["status"] == "INVALID"
    assert any("inside the manifest directory" in message for message in report["errors"])


def test_cer_and_word_boundary_error_report_no_confidence():
    assert cer("你好，世界！", "你好世界") == 0
    assert cer("你好", "你号") == pytest.approx(0.5)
    assert cer("", "x") is None
    stats = word_boundary_error(
        [{"t": "你好", "s": 100, "e": 300}, {"t": "世界", "s": 310, "e": 500}],
        [{"t": "你好", "s": 110, "e": 320}, {"t": "世界", "s": 320, "e": 520}],
    )
    assert stats == {"matched_words": 2, "compared_boundaries": 4,
                     "median_abs_error_ms": 15.0, "p95_abs_error_ms": 20.0}
    assert "confidence" not in stats


def test_audio_qc_reads_pcm16_duration_clipping_and_silence(tmp_path):
    path = tmp_path / "qc.wav"
    values = array("h", [0] * 1600 + [32767, -32768] * 800)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(values.tobytes())
    metrics = audio_qc(path)
    assert metrics["duration_ms"] == 200
    assert metrics["clipped_sample_ratio"] > 0.49
    assert metrics["silence_window_ratio"] > 0


def test_chunk_windows_have_configured_overlap_and_cover_tail():
    assert benchmedia.chunk_windows(500_000) == [(0, 240_000), (239_000, 479_000), (478_000, 500_000)]
    with pytest.raises(ValueError):
        benchmedia.chunk_windows(20, chunk_ms=10, overlap_ms=10)


def test_ffmpeg_helpers_create_audio_chunk_production_audio_and_roi_when_available(tmp_path):
    ffmpeg = os.getenv("FFMPEG_BIN", "ffmpeg")
    ffprobe = os.getenv("FFPROBE_BIN", "ffprobe")
    if not shutil.which(ffmpeg) or not shutil.which(ffprobe):
        pytest.skip("FFmpeg/FFprobe not installed")
    manifest = create_synthetic_smoke(tmp_path, ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe, duration_s=2)
    report = validate_manifest(manifest, ffprobe)
    assert report["status"] == "SYNTHETIC_SMOKE_ONLY"
    assert report["representative_quality_pass"] is False
    source = tmp_path / "synthetic-smoke.mp4"
    chunk = benchmedia.extract_audio(source, tmp_path / "audio" / "chunk.wav", 100, 1100,
                                     ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe)
    production = benchmedia.extract_audio(source, tmp_path / "audio" / "production.wav",
                                          sample_rate=48_000, channels=2, ffmpeg_bin=ffmpeg,
                                          ffprobe_bin=ffprobe)
    frame = benchmedia.extract_roi_frame(source, tmp_path / "frames" / "roi.png",
                                         {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}, 300,
                                         ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe)
    with wave.open(str(chunk), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16_000, 1, 2)
    with wave.open(str(production), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (48_000, 2, 2)
    assert frame.is_file() and frame.stat().st_size > 0
    # YUV420 crop defaults round to chroma coordinates; masks need exact odd pixel geometry.
    from PIL import Image

    metadata = benchmedia.probe_media(source, ffprobe)
    width, height = metadata["video"]["width"], metadata["video"]["height"]
    odd_roi = {"x": 3 / width, "y": 5 / height, "w": 101 / width, "h": 77 / height}
    odd_frame = benchmedia.extract_roi_frame(source, tmp_path / "frames" / "odd.png", odd_roi, 300,
                                             ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe)
    with Image.open(odd_frame) as picture:
        assert picture.size == (101, 77)
    with pytest.raises(ValueError, match="normalized"):
        benchmedia.extract_roi_frame(source, tmp_path / "bad.png", {"x": 0.8, "y": 0, "w": 0.3, "h": 1},
                                     100, ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe)
