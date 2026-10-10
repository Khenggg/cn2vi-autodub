import json
from pathlib import Path

import pytest

from vnle.domain import Observation, Rect, Request, allowed_tiles
from vnle.events import EventBuilder
from vnle.media import parse_probe
from vnle.ocr import load_manifest
from vnle.pipeline import DEFAULT_CONFIG, validate_config
from vnle.server import byte_range


def obs(i, time, text="HP 100", x=20, score=0.9):
    return Observation(
        i,
        int(time * 90000),
        {"num": 1, "den": 90000},
        time,
        [[x, 10], [x + 100, 10], [x + 100, 30], [x, 30]],
        text,
        score,
        "a" * 64,
        (0, 0, 640, 480),
    )


def request(rects=None, **changes):
    data = {
        "roi_confirmed": True,
        "start_s": 0,
        "end_s": 30,
        "exclusions": rects or [{"x": 0.1, "y": 0.8, "width": 0.8, "height": 0.1}],
    }
    data.update(changes)
    return Request.parse(data, 30)


@pytest.mark.parametrize(
    "field,value",
    [("x", -0.1), ("width", 0), ("height", float("nan")), ("y", float("inf")), ("x", True)],
)
def test_invalid_roi(field, value):
    row = {"x": 0.1, "y": 0.8, "width": 0.8, "height": 0.1}
    row[field] = value
    with pytest.raises(ValueError):
        request([row])


def test_confirmation_is_required_and_geometry_changes_hash():
    with pytest.raises(ValueError):
        request(roi_confirmed=False)
    a = request()
    b = request([{"x": 0.2, "y": 0.8, "width": 0.7, "height": 0.1}])
    assert a.roi_hash != b.roi_hash


def test_roi_intervals_are_half_open():
    value = request([{"x": 0.1, "y": 0.8, "width": 0.8, "height": 0.1, "start_s": 2, "end_s": 3}])
    assert not value.at(1.99)
    assert value.at(2)
    assert not value.at(3)


def test_exact_complement_for_overlapping_rectangles():
    width, height = 47, 31
    rois = (Rect(0.1, 0.6, 0.8, 0.2), Rect(0.3, 0.5, 0.2, 0.4))
    blocked = [r.pixels(width, height) for r in rois]
    tiles = allowed_tiles(width, height, rois)
    counts = {}
    for x0, y0, x1, y1 in tiles:
        for y in range(y0, y1):
            for x in range(x0, x1):
                counts[x, y] = counts.get((x, y), 0) + 1
    for y in range(height):
        for x in range(width):
            inside = any(a <= x < c and b <= y < d for a, b, c, d in blocked)
            assert counts.get((x, y), 0) == (0 if inside else 1)


def test_whole_frame_exclusion_is_valid_and_empty_complement():
    assert allowed_tiles(100, 50, (Rect(0, 0, 1, 1),)) == []


def test_revisions_keep_changed_numbers_and_reappearances():
    builder = EventBuilder(0)
    builder.update(0, [obs(1, 0)])
    builder.update(0.25, [obs(2, 0.25)])
    builder.update(0.5, [obs(3, 0.5, "HP 90")])
    builder.update(0.75, [])
    builder.update(1, [obs(4, 1, "HP 90")])
    builder.boundary(1.25, "END")
    events = builder.events
    assert [e["text"] for e in events] == ["HP 100", "HP 90", "HP 90"]
    assert events[1]["parent_event_id"] == events[0]["id"]
    assert events[1]["revision"] == 1
    assert events[1]["appearance_id"] != events[2]["appearance_id"]
    assert events[0]["end_s"] == events[1]["start_s"] == 0.375


def test_same_text_different_locations_remains_separate():
    builder = EventBuilder(0)
    builder.update(0, [obs(1, 0, x=20), obs(2, 0, x=300)])
    builder.update(0.25, [obs(3, 0.25, x=300), obs(4, 0.25, x=20)])
    builder.boundary(0.5, "END")
    assert len(builder.events) == 2
    assert all(len(e["observations"]) == 2 for e in builder.events)


def test_one_sample_event_and_uncertainty_are_kept():
    builder = EventBuilder(0)
    builder.update(0.25, [obs(1, 0.25, score=0.4)])
    builder.update(0.5, [])
    event = builder.events[0]
    assert event["observations"] == [1]
    assert event["start_bracket_s"] == [0, 0.25]
    assert event["end_bracket_s"] == [0.25, 0.5]
    assert event["state"] == "NEEDS_REVIEW"
    assert event["decision"] == "PENDING_IMPORTANCE_REVIEW"


def test_shot_cut_closes_instead_of_merging():
    builder = EventBuilder(0)
    builder.update(0, [obs(1, 0)])
    builder.update(0.25, [obs(2, 0.25)], cut=True)
    builder.boundary(0.5, "END")
    assert len(builder.events) == 2
    assert builder.events[0]["shot_id"] != builder.events[1]["shot_id"]


@pytest.mark.parametrize(
    "header,expected",
    [
        (None, (0, 9, False)),
        ("bytes=2-5", (2, 5, True)),
        ("bytes=8-", (8, 9, True)),
        ("bytes=-3", (7, 9, True)),
    ],
)
def test_media_ranges(header, expected):
    assert byte_range(header, 10) == expected


@pytest.mark.parametrize("header", ["bytes=12-", "bytes=6-2", "bytes=-0", "bytes=1-2,4-5"])
def test_invalid_ranges(header):
    with pytest.raises(ValueError):
        byte_range(header, 10)


def probe_data():
    return {
        "streams": [
            {
                "codec_type": "video",
                "index": 0,
                "width": 1280,
                "height": 720,
                "duration": "30",
                "codec_name": "h264",
                "time_base": "1/90000",
                "start_time": "2.5",
                "start_pts": 225000,
                "sample_aspect_ratio": "1:1",
            }
        ]
    }


def test_probe_preserves_pts_origin_and_rejects_unmapped_rotation():
    data = probe_data()
    value = parse_probe(data)
    assert value["origin"] == {"num": 5, "den": 2}
    assert value["time_base"] == {"num": 1, "den": 90000}
    data["streams"][0]["side_data_list"] = [{"rotation": 90}]
    with pytest.raises(ValueError):
        parse_probe(data)


def test_model_manifest_hash_error_never_downloads(tmp_path: Path):
    (tmp_path / "det.onnx").write_bytes(b"bad")
    manifest = {
        "schema_version": 1,
        "family": "PP-OCRv6-small",
        "detector": {
            "path": "det.onnx",
            "sha256": "0" * 64,
            "license": "Apache-2.0",
            "source": "test",
        },
    }
    file = tmp_path / "manifest.json"
    file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(file)


def test_config_cannot_quietly_slow_watchdog_or_choose_unknown_backend():
    config = dict(DEFAULT_CONFIG, watchdog_s=1)
    with pytest.raises(ValueError):
        validate_config(config)
    with pytest.raises(ValueError):
        validate_config(dict(DEFAULT_CONFIG, provider="AUTO"))
    with pytest.raises(ValueError):
        validate_config(dict(DEFAULT_CONFIG, detector_backend="unknown_detector"))
    valid = validate_config(dict(DEFAULT_CONFIG, detector_backend="chinese_detector"))
    assert valid["detector_backend"] == "chinese_detector"


def test_chinese_detector_manifest_family_accepted(tmp_path: Path):
    from vnle.media import file_sha256
    det = tmp_path / "det.onnx"
    rec = tmp_path / "rec.onnx"
    det.write_bytes(b"chinese detector")
    rec.write_bytes(b"chinese recognizer")
    manifest = {
        "schema_version": 1,
        "family": "Chinese-PPOCRv4-small",
        "detector": {
            "path": "det.onnx",
            "sha256": file_sha256(det),
            "license": "Apache-2.0",
            "source": "test",
        },
        "recognizer": {
            "path": "rec.onnx",
            "sha256": file_sha256(rec),
            "license": "Apache-2.0",
            "source": "test",
        },
    }
    file = tmp_path / "manifest.json"
    file.write_text(json.dumps(manifest))
    loaded = load_manifest(file)
    assert loaded["family"] == "Chinese-PPOCRv4-small"

