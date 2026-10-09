"""CI-only native media integration, synthetic frames and fake OCR; no weights."""

import json
from dataclasses import replace
from fractions import Fraction

import pytest

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from vnle.domain import Observation, Request  # noqa: E402
from vnle.media import file_sha256  # noqa: E402
from vnle.ocr import RapidAdapter  # noqa: E402
from vnle.pipeline import DEFAULT_CONFIG, analyze  # noqa: E402


def test_ocr_sees_only_complement_pixels(monkeypatch):
    from types import SimpleNamespace

    # Red marks the user's subtitle ROI. None of those pixels may reach detector
    # or recognizer, including resize/rectification boundaries.
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[80:100, 30:130, 2] = 255
    from vnle.domain import Rect

    rect = Rect(30 / 160, 80 / 120, 100 / 160, 20 / 120)
    adapter = RapidAdapter.__new__(RapidAdapter)
    adapter.np, adapter.cv2 = np, cv2
    from collections import Counter

    adapter.timings, adapter.calls = Counter(), Counter()
    adapter.counter = 0
    adapter.batch_size = 8
    adapter.config = dict(DEFAULT_CONFIG)
    seen = []

    def detect(tile):
        assert not np.any(tile[:, :, 2])
        seen.append(tile.copy())
        h, w = tile.shape[:2]
        return SimpleNamespace(
            boxes=np.array([[[2, 2], [w - 3, 2], [w - 3, h - 3], [2, h - 3]]], dtype=np.float32)
        )

    def recognize(args):
        assert all(not np.any(c[:, :, 2]) for c in args.img)
        return SimpleNamespace(txts=["系统 100" for _ in args.img], scores=[0.95] * len(args.img))

    adapter.detector, adapter.recognizer = detect, recognize
    adapter.crop = lambda tile, box: tile.copy()
    # Use actual RapidOCR input contract without constructing/loading any model.
    if not pytest.importorskip("rapidocr"):
        return
    rows, _ = adapter.read(image, (rect,), 100, {"num": 1, "den": 100}, 1)
    assert rows and len(seen) >= 2
    assert len({tile.shape for tile in seen}) == 1
    assert len(rows) == len(seen)  # Padded batch rows never become source observations.
    assert all(not (30 <= p[0] < 130 and 80 <= p[1] < 100) for o in rows for p in o.polygon)


def test_vfr_pts_and_durable_partial_events(tmp_path, monkeypatch):
    video = tmp_path / "vfr.mkv"
    timestamps = [0, 100, 370, 620, 900]
    with av.open(str(video), "w") as output:
        stream = output.add_stream("ffv1", rate=10)
        stream.width, stream.height = 160, 120
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for t in timestamps:
            frame = av.VideoFrame.from_ndarray(
                np.zeros((120, 160, 3), dtype=np.uint8), format="bgr24"
            )
            frame.pts, frame.time_base = t, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    det, rec = tmp_path / "det", tmp_path / "rec"
    det.write_bytes(b"fake detector")
    rec.write_bytes(b"fake recognizer")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "PP-OCRv6-small",
                "detector": {
                    "path": "det",
                    "sha256": file_sha256(det),
                    "license": "Apache-2.0",
                    "source": "test",
                },
                "recognizer": {
                    "path": "rec",
                    "sha256": file_sha256(rec),
                    "license": "Apache-2.0",
                    "source": "test",
                },
            }
        )
    )
    seen = []

    class FakeOCR:
        def __init__(self, *args):
            pass

        def read(self, image, exclusions, pts, time_base, time_s):
            seen.append((pts, time_base, time_s))
            row = Observation(
                len(seen),
                pts,
                time_base,
                time_s,
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                "HP 100" if time_s < 0.5 else "HP 90",
                0.99,
                "a" * 64,
                (0, 0, 160, 80),
            )
            return [row], {row.id: image[:10, :10].copy()}

        def finish(self):
            return {"fake": True}

    monkeypatch.setattr("vnle.pipeline.RapidAdapter", FakeOCR)
    req = Request.parse(
        {
            "roi_confirmed": True,
            "start_s": 0,
            "end_s": 1,
            "exclusions": [{"x": 0, "y": 0.8, "width": 1, "height": 0.2}],
        },
        1,
    )
    media = {"width": 160, "height": 120, "stream_index": 0, "origin": {"num": 0, "den": 1}}
    report = analyze(
        video,
        media,
        req,
        tmp_path / "run",
        manifest,
        dict(DEFAULT_CONFIG, provider="CPUExecutionProvider"),
    )
    assert report["status"] == "COMPLETED_UNVERIFIED"
    assert [round(row[2], 2) for row in seen] == [0, 0.37, 0.62, 0.9]
    data = json.loads((tmp_path / "run/events.json").read_text())
    assert [e["text"] for e in data["events"]] == ["HP 100", "HP 90"]
    assert (tmp_path / "run/observations.jsonl").exists()
    assert (tmp_path / "run/evidence/e000001.png").exists()
    assert report["quality_verified"] is False
    assert report["input_sha256"] == file_sha256(video)
    assert report["request"]["roi_hash"] == req.roi_hash


def test_failure_keeps_report_and_partial_output(tmp_path):
    req = Request.parse(
        {
            "roi_confirmed": True,
            "start_s": 0,
            "end_s": 1,
            "exclusions": [{"x": 0, "y": 0.8, "width": 1, "height": 0.2}],
        },
        1,
    )
    bad_video = tmp_path / "missing.mp4"
    report = analyze(
        bad_video, {}, replace(req, end_s=1), tmp_path / "run", tmp_path / "missing.json"
    )
    assert report["status"] == "FAILED"
    assert (tmp_path / "run/run-report.json").exists()
    assert (tmp_path / "run/events.json").exists()
