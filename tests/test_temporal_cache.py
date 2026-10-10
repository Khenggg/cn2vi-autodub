import numpy as np
import cv2
import pytest
from vnle.domain import Observation, Rect
from vnle.cache import TemporalROICache, contains_chinese


def make_text_image(text="测试文本", size=(120, 160), pos=(20, 30), fg=(255, 255, 255), bg=(0, 0, 0)):
    img = np.full((size[0], size[1], 3), bg, dtype=np.uint8)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2, cv2.LINE_AA)
    return img


def test_contains_chinese():
    assert contains_chinese("十八岁的机车梦") is True
    assert contains_chinese("脉动有点慢Lilibili") is True
    assert contains_chinese("No. 1 德源") is True
    assert contains_chinese("hartMoves with Music") is False
    assert contains_chinese("123456") is False
    assert contains_chinese("") is False


def test_static_text_cache_hit():
    cache = TemporalROICache(similarity_threshold=0.80, chinese_only=False)
    img1 = make_text_image("SAMPLE 100")
    counter = 0

    def next_id():
        nonlocal counter
        counter += 1
        return counter

    obs1 = Observation(
        id=next_id(),
        pts=0,
        time_base={"num": 1, "den": 1000},
        time_s=0.0,
        polygon=[[20.0, 10.0], [120.0, 10.0], [120.0, 40.0], [20.0, 40.0]],
        text="SAMPLE 100",
        score=0.98,
        crop_sha256="fake_sha",
        tile=(0, 0, 160, 120),
        touches_tile_edge=False,
    )
    cache.register_observations([obs1], img1)
    assert len(cache.active_rois) == 1

    # Frame 2 is identical (static text)
    img2 = img1.copy()
    hits, evidence, reliable = cache.track_and_emit(img2, (), 100, {"num": 1, "den": 1000}, 0.1, next_id)
    assert len(hits) == 1
    assert hits[0].text == "SAMPLE 100"
    assert hits[0].time_s == 0.1
    assert reliable is True
    assert cache.stats["cache_hits"] == 1


def test_changing_numbers_cache_invalidation():
    cache = TemporalROICache(similarity_threshold=0.85, chinese_only=False)
    img1 = make_text_image("VAL 123")
    counter = 0

    def next_id():
        nonlocal counter
        counter += 1
        return counter

    obs1 = Observation(
        id=next_id(),
        pts=0,
        time_base={"num": 1, "den": 1000},
        time_s=0.0,
        polygon=[[20.0, 10.0], [120.0, 10.0], [120.0, 40.0], [20.0, 40.0]],
        text="VAL 123",
        score=0.98,
        crop_sha256="fake_sha",
        tile=(0, 0, 160, 120),
        touches_tile_edge=False,
    )
    cache.register_observations([obs1], img1)

    # Frame 2 has changing numbers in the scoreboard/table ("VAL 890")
    img2 = make_text_image("VAL 890")
    hits, evidence, reliable = cache.track_and_emit(img2, (), 100, {"num": 1, "den": 1000}, 0.1, next_id)

    # Correlation dropped below threshold -> must NOT be reliable cache hit
    assert reliable is False


def test_motion_compensation():
    cache = TemporalROICache(similarity_threshold=0.80, max_motion_padding=12, chinese_only=False)
    img1 = make_text_image("TRACK ME", pos=(20, 30))
    counter = 0

    def next_id():
        nonlocal counter
        counter += 1
        return counter

    obs1 = Observation(
        id=next_id(),
        pts=0,
        time_base={"num": 1, "den": 1000},
        time_s=0.0,
        polygon=[[20.0, 10.0], [120.0, 10.0], [120.0, 40.0], [20.0, 40.0]],
        text="TRACK ME",
        score=0.95,
        crop_sha256="fake_sha",
        tile=(0, 0, 160, 120),
        touches_tile_edge=False,
    )
    cache.register_observations([obs1], img1)

    # Frame 2 has text shifted by dx=4, dy=3
    img2 = make_text_image("TRACK ME", pos=(24, 33))
    hits, evidence, reliable = cache.track_and_emit(img2, (), 100, {"num": 1, "den": 1000}, 0.1, next_id)

    assert len(hits) == 1
    assert reliable is True
    # The polygon should be compensated
    assert hits[0].polygon[0][0] == pytest.approx(24.0, abs=1.5)
    assert hits[0].polygon[0][1] == pytest.approx(13.0, abs=1.5)


def test_short_lived_text_disappearance():
    cache = TemporalROICache(similarity_threshold=0.80, chinese_only=False)
    img1 = make_text_image("FLASH")
    counter = 0

    def next_id():
        nonlocal counter
        counter += 1
        return counter

    obs1 = Observation(
        id=next_id(),
        pts=0,
        time_base={"num": 1, "den": 1000},
        time_s=0.0,
        polygon=[[20.0, 10.0], [100.0, 10.0], [100.0, 40.0], [20.0, 40.0]],
        text="FLASH",
        score=0.90,
        crop_sha256="fake_sha",
        tile=(0, 0, 160, 120),
        touches_tile_edge=False,
    )
    cache.register_observations([obs1], img1)

    # Frame 2: Text disappeared (blank image)
    blank_img = np.zeros((120, 160, 3), dtype=np.uint8)
    hits1, _, rel1 = cache.track_and_emit(blank_img, (), 100, {"num": 1, "den": 1000}, 0.1, next_id)
    assert len(hits1) == 0
    assert rel1 is False

    # Frame 3: Still blank -> ROI is dropped from cache
    hits2, _, _ = cache.track_and_emit(blank_img, (), 200, {"num": 1, "den": 1000}, 0.2, next_id)
    assert len(hits2) == 0
    assert len(cache.active_rois) == 0  # Dropped after consecutive misses


def test_scene_cut_cache_reset():
    cache = TemporalROICache(similarity_threshold=0.80, chinese_only=False)
    img1 = make_text_image("SCENE 1")
    obs1 = Observation(
        id=1,
        pts=0,
        time_base={"num": 1, "den": 1000},
        time_s=0.0,
        polygon=[[20.0, 10.0], [100.0, 10.0], [100.0, 40.0], [20.0, 40.0]],
        text="SCENE 1",
        score=0.90,
        crop_sha256="fake_sha",
        tile=(0, 0, 160, 120),
        touches_tile_edge=False,
    )
    cache.register_observations([obs1], img1)
    assert len(cache.active_rois) == 1

    cache.reset()
    assert len(cache.active_rois) == 0
