import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autodub.adapters import subtitle_events, v2_vision
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
    monkeypatch.setattr(subtitle_events, 'recognize_lines',
                        lambda engine, images, batch_size: [('你好', 0.95) for image in images])
    monkeypatch.setenv("AUTODUB_WORKER_OUTPUT", str(tmp_path / "ocr"))
    result = v2_vision.scan(source, {"duration_ms": 2000, 'subtitle_line_roi': {
        'x': 0.25, 'y': 0.58, 'w': 0.5, 'h': 0.18}})
    report = json.loads(Path(result["artifacts"][0]).read_text())
    assert report["decoded_frames"] == 20
    assert calls == []  # Localization must not invoke the full OCR detector.
    assert len(report["events"]) == 1 and report["events"][0]["kind"] == "DIALOGUE"
    assert report["events"][0]["donors"]
    assert report["provider"] == "CUDAExecutionProvider"
    assert report['recognition_batch_calls'] > 0
    assert report['materialized_frames'] == 20
    assert report['calibration_calls'] == 0
    assert report['recognized_line_images'] == 1
    assert report['events'][0]['start_ms'] == 500
    assert report['events'][0]['end_ms'] == 1500


def test_locate_dialogue_line_excludes_yellow_title_and_logo():
    image = np.zeros((100, 320, 3), dtype='uint8')
    cv2.putText(image, 'TITLE', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, 'WORDS', (110, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    lines = [
        {'text': '装饰', 'box': [[0, 8], [80, 8], [80, 35], [0, 35]]},
        {'text': '对白', 'box': [[105, 50], [190, 50], [190, 80], [105, 80]]},
        {'text': 'logo', 'box': [[0, 0], [40, 0], [40, 8], [0, 8]]},
    ]
    assert subtitle_events.select_line(image, lines)['text'] == '对白'


def test_line_signature_ignores_colored_decoration():
    image = np.zeros((48, 320, 3), dtype='uint8')
    cv2.putText(image, 'WORDS', (110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    before = subtitle_events.line_signature(image)
    cv2.putText(image, 'TITLE', (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    assert not subtitle_events.changed_line(before, subtitle_events.line_signature(image))
    image[15:40, 110:200] = 0
    assert subtitle_events.changed_line(before, subtitle_events.line_signature(image))


def test_recognition_submits_one_batch_and_keeps_original_image_order(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, 'rapidocr.ch_ppocr_rec.main',
                        SimpleNamespace(TextRecInput=SimpleNamespace))
    calls = []

    class Recognizer:
        rec_batch_num = 1

        def __call__(self, args):
            calls.append(args.img)
            assert self.rec_batch_num == 8
            return SimpleNamespace(txts=['短', '长句'], scores=[0.9, 0.95])

    images = [np.zeros((32, 40, 3), dtype='uint8'), np.zeros((32, 180, 3), dtype='uint8')]
    result = subtitle_events.recognize_lines(SimpleNamespace(text_rec=Recognizer()), images, 8)
    assert len(calls) == 1
    assert all(np.array_equal(left, right) for left, right in zip(calls[0], images, strict=True))
    assert result == [('短', 0.9), ('长句', 0.95)]
    with pytest.raises(ValueError):
        subtitle_events.recognize_lines(SimpleNamespace(), images * 10, 8)


def test_width_buckets_pad_thin_rows_and_restore_id_order(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, 'rapidocr.ch_ppocr_rec.main',
                        SimpleNamespace(TextRecInput=SimpleNamespace))
    calls = []

    class Recognizer:
        rec_image_shape = (3, 48, 320)

        def __call__(self, args):
            width = max(max(320, round(48 * image.shape[1] / image.shape[0])) for image in args.img)
            assert width * len(args.img) <= 2048
            assert all(image.shape[0] >= 32 for image in args.img)
            calls.append(len(args.img))
            return SimpleNamespace(txts=[str(int(image.max())) for image in args.img],
                                   scores=[0.95] * len(args.img))

    images = [np.full((9, width, 3), i + 1, dtype='uint8')
              for i, width in enumerate([600, 50, 400, 60, 500, 70, 550, 80])]
    result = subtitle_events.recognize_lines(SimpleNamespace(text_rec=Recognizer()), images, 8)
    assert [text for text, score in result] == [str(i + 1) for i in range(8)]
    assert len(calls) > 1


def test_ocr_failure_diagnosis_retains_only_shapes_and_numeric_memory(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTODUB_WORKER_OUTPUT', str(tmp_path))
    error = RuntimeError('PRIVATE_TOKEN Available memory of 30154240 is smaller than requested bytes of 88610560')
    subtitle_events.record_recognition_failure(error, {'size': 8, 'normalized_width': 1184})
    value = (tmp_path / 'ocr-failure.json').read_text()
    assert 'PRIVATE_TOKEN' not in value
    assert json.loads(value)['requested_bytes'] == 88610560


def test_glyph_mask_rejects_colored_title_and_tall_background_edges():
    image = np.zeros((48, 320, 3), dtype='uint8')
    cv2.putText(image, 'WORDS', (110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    before = subtitle_events.glyph_signature(image)
    assert np.count_nonzero(before) > 0
    cv2.putText(image, 'TITLE', (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    image[:, 280:285] = 255
    assert np.array_equal(before, subtitle_events.glyph_signature(image))
