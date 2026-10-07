import hashlib
import json
import shutil
import subprocess
import wave

import pytest

from autodub.subtitle_render import render_subtitles, subtitle_srt


def test_srt_orders_cues_clamps_bounds_and_escapes_markup():
    segments = [
        {"start_ms": 700, "end_ms": 1400, "subtitle_vi": "B & <i>unsafe</i>"},
        {"start_ms": 100, "end_ms": 600, "subtitle_vi": "Xin chào\nViệt Nam"},
        {"start_ms": 1500, "end_ms": 1700, "subtitle_vi": "beyond"},
    ]

    result = subtitle_srt(segments, 1600)

    assert result.startswith("1\n00:00:00,100 --> 00:00:00,600\nXin chào\nViệt Nam")
    assert "B &amp; &lt;i&gt;unsafe&lt;/i&gt;" in result
    assert "00:00:01,500 --> 00:00:01,600\nbeyond" in result
    assert result.index("Xin chào") < result.index("B &amp;") < result.index("beyond")


def test_srt_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="invalid bounds"):
        subtitle_srt([{"start_ms": -1, "end_ms": 10, "subtitle_vi": "bad"}], 100)


def _make_source(path, duration=1, size="96x64", fps=2):
    command = [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", f"color=c=blue:s={size}:r={fps}:d={duration}",
               "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration}",
               "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-y", str(path)]
    subprocess.run(command, check=True, capture_output=True)


def _make_audio(path, duration=1):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(b"\0\0\0\0" * 48000 * duration)


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


@needs_ffmpeg
def test_burn_mode_renders_caption_and_streams_a_bounded_video(tmp_path):
    source, audio, output = tmp_path / "source.mp4", tmp_path / "dub.wav", tmp_path / "final.mp4"
    _make_source(source)
    _make_audio(audio)
    segment = {"id": "s1", "start_ms": 100, "end_ms": 900,
               "subtitle_vi": "Xin chào Việt Nam", "dub_vi": "Xin chào Việt Nam"}

    result = render_subtitles(source, audio, output, [segment], {
        "output_dir": str(tmp_path), "subtitle_mode": "burn", "ffmpeg_bin": "ffmpeg",
        "ffprobe_bin": "ffprobe", "timeout_seconds": 60,
    })

    assert output.is_file() and result["artifacts"][0] == str(output)
    assert (tmp_path / "subs.vi.srt").read_text(encoding="utf-8").find("Xin chào") >= 0
    report = json.loads((tmp_path / "subtitle_render.qc.json").read_text(encoding="utf-8"))
    assert report["technical_render_verified"] is True
    assert report["semantic_quality"] == "UNVERIFIED"
    assert report["weights_sha256"] is None
    assert report["duration_delta_ms"] <= 500


@needs_ffmpeg
def test_replace_mode_requires_explicit_validated_roi(tmp_path):
    source, audio = tmp_path / "source.mp4", tmp_path / "dub.wav"
    _make_source(source)
    _make_audio(audio)

    result = render_subtitles(source, audio, tmp_path / "final.mp4", [], {
        "output_dir": str(tmp_path), "subtitle_mode": "replace", "timeout_seconds": 60,
    })

    assert result["review_required"] is True
    assert result["failure_code"] == "SUBTITLE_ROI_REQUIRED"
    assert not (tmp_path / "final.mp4").exists()
    assert (tmp_path / "subtitle_render.qc.json").is_file()


@needs_ffmpeg
def test_replace_mode_uses_mocked_ocr_and_lama_and_decodes_video_result(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import cv2
    import numpy as np

    from autodub.adapters import common, inpaint, ocr

    source, audio, output = tmp_path / "source.mp4", tmp_path / "dub.wav", tmp_path / "final.mp4"
    _make_source(source, duration=1, size="96x64", fps=2)
    _make_audio(audio)
    seen = []

    class FakeOCR:
        def __call__(self, _crop):
            return SimpleNamespace(boxes=np.array([[[5, 5], [18, 5], [18, 14], [5, 14]]]),
                                   txts=["旧字幕"], scores=[0.99])

    class FakeSession:
        pass

    manifest_lama = {"id": "lama-onnx", "model_revision": "fixture-lama", "weights_sha256": "a" * 64}
    manifest_ocr = {"id": "rapidocr-v6", "model_revision": "fixture-ocr", "weights_sha256": "b" * 64}
    monkeypatch.setattr(common, "configure_offline", lambda _config: None)
    monkeypatch.setattr(common, "asset", lambda _config, _identifier: (tmp_path, manifest_lama))
    monkeypatch.setattr(ocr, "build_engine", lambda _config: (FakeOCR(), manifest_ocr))
    monkeypatch.setattr(inpaint, "inpaint_masked", lambda image, mask, _session, **_kwargs:
                        _mock_inpaint(np, seen, image, mask))
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda *_args, **_kwargs: FakeSession()))

    segment = {"id": "s1", "start_ms": 100, "end_ms": 900,
               "subtitle_vi": "Xin chào", "dub_vi": "Xin chào"}
    result = render_subtitles(source, audio, output, [segment], {
        "output_dir": str(tmp_path), "subtitle_mode": "replace",
        "roi": {"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6, "scope": "episode"},
        "timeout_seconds": 60, "ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe",
    })

    assert len(seen) == 2
    assert all(np.array_equal(before[mask == 0], after[mask == 0]) for before, mask, after in seen)
    assert result["artifacts"][0] == str(output) and output.is_file()
    cap = cv2.VideoCapture(str(output))
    decoded = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    assert decoded == 2
    report = json.loads((tmp_path / "subtitle_render.qc.json").read_text(encoding="utf-8"))
    assert report["models"][0]["weights_sha256"] == "a" * 64
    assert report["models"][1]["weights_sha256"] == "b" * 64
    assert report["ocr_residual_text_pass"] is None
    assert report["semantic_quality"] == "UNVERIFIED"
    assert report["frame_result"]["frame_count"] == 2
    assert report["frame_result"]["unmasked_pixels_preserved_before_encode"] is True
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def _mock_inpaint(np, seen, image, mask):
    after = image.copy()
    selected = mask > 0
    after[selected] = (255, 0, 0)
    seen.append((image.copy(), mask.copy(), after.copy()))
    return after
