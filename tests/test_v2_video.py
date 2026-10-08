import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autodub.adapters import v2_vision
from autodub.storage import atomic_json, sha256_file


def test_text_change_ignores_motion_outside_observed_boxes_but_detects_replacement():
    image = np.zeros((64, 256, 3), dtype="uint8")
    cv2.putText(image, "OLD", (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    boxes = [[[75, 15], [130, 15], [130, 45], [75, 45]]]
    before = v2_vision.text_signature(image)
    moving = image.copy()
    moving[:, 170:] = np.random.default_rng(42).integers(0, 256, moving[:, 170:].shape, dtype="uint8")
    assert not v2_vision.signature_changed(before, v2_vision.text_signature(moving), boxes, image.shape)
    moving[15:46, 75:131] = 0
    cv2.putText(moving, "NEW", (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    assert v2_vision.signature_changed(before, v2_vision.text_signature(moving), boxes, image.shape)


def source_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg media checks run on Ubuntu/CI")
    clean = np.full((180, 320, 3), (100, 50, 20), dtype="uint8")
    frames = []
    for index in range(20):
        image = clean.copy()
        if 5 <= index < 15:
            cv2.putText(image, "OLD", (90, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frames.append(image)
    source = tmp_path / "source.mp4"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
                    "-video_size", "320x180", "-framerate", "10", "-i", "pipe:0", "-f", "lavfi", "-i",
                    "sine=frequency=200:sample_rate=48000", "-t", "2", "-c:v", "libx264", "-c:a", "aac",
                    "-y", str(source)], input=b"".join(frame.tobytes() for frame in frames),
                   check=True, capture_output=True, timeout=30)
    return source, clean, ffmpeg


def test_real_temporal_render_restores_known_background_in_one_encode(tmp_path, monkeypatch):
    source, clean, ffmpeg = source_video(tmp_path)
    donor = tmp_path / "donor.png"
    cv2.imwrite(str(donor), clean[90:145])
    plan = tmp_path / "plan.json"
    atomic_json(plan, {"source_sha256": sha256_file(source), "width": 320, "height": 180,
        "crop": [90, 145, 320, 55], "events": [{"id": "caption", "start_ms": 500, "end_ms": 1500,
            "boxes": [[[88, 15], [145, 15], [145, 40], [88, 40]]], "donors": [str(donor)]}]})
    subtitles = tmp_path / "subtitles.srt"
    subtitles.write_text("1\n00:00:00,500 --> 00:00:01,500\nXin chào\n")
    monkeypatch.setenv("AUTODUB_WORKER_OUTPUT", str(tmp_path / "render"))
    monkeypatch.setenv("AUTODUB_WORKER_RUN_ROOT", str(tmp_path))
    result = v2_vision.encode(source, {"duration_ms": 2000, "audio_path": str(source),
        "subtitles_path": str(subtitles), "restoration_plan": str(plan), "subtitle_mode": "replace",
        "ffmpeg_bin": ffmpeg, "video_encoder": "libx264", "production": False})
    report = json.loads(Path(result["artifacts"][0]).read_text())
    assert report["restored_frames"] > 0 and report["unrestored_frames"] == 0
    assert report["encode_passes"] == 1 and report["generative_model"] is None
    capture = cv2.VideoCapture(result["video"])
    capture.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert np.max(frame[105:126, 95:138]) < 170  # OLD white glyphs have been removed.


def test_event_ocr_reuses_static_text_and_records_clean_donors(tmp_path, monkeypatch):
    source, _, _ = source_video(tmp_path)
    calls = []

    class Engine:
        def __call__(self, image):
            calls.append(image.copy())
            if np.max(image) > 180:
                return SimpleNamespace(boxes=np.array([[[88, 4], [145, 4], [145, 30], [88, 30]]]),
                                       txts=["你好"], scores=[0.95])
            return SimpleNamespace(boxes=None)

    monkeypatch.setattr(v2_vision, "build_engine", lambda config, resolve_asset: (
        Engine(), {"id": "ocr", "model_revision": "pinned", "weights_sha256": "hash"}))
    monkeypatch.setenv("AUTODUB_WORKER_OUTPUT", str(tmp_path / "ocr"))
    result = v2_vision.scan(source, {"duration_ms": 2000})
    report = json.loads(Path(result["artifacts"][0]).read_text())
    assert report["decoded_frames"] == 20
    assert len(calls) < 10
    assert len(report["events"]) == 1 and report["events"][0]["kind"] == "DIALOGUE"
    assert report["events"][0]["donors"]
    assert report["provider"] == "CUDAExecutionProvider"
