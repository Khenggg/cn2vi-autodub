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


def render_chinese_panel(
    text: str,
    size: tuple[int, int] = (120, 260),
    pos: tuple[int, int] = (22, 16),
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (24, 24, 24),
    panel: bool = True,
) -> np.ndarray:
    """Render Chinese glyphs with system CJK TrueType font if available, or deterministic stroke bitmaps."""
    import os

    img = np.full((size[0], size[1], 3), bg, dtype=np.uint8)
    if panel:
        px0, py0 = pos[0] - 6, pos[1] - 4
        px1, py1 = min(size[1] - 8, pos[0] + 206), min(size[0] - 8, pos[1] + 34)
        cv2.rectangle(img, (px0, py0), (px1, py1), (58, 58, 58), -1)
        cv2.rectangle(img, (px0, py0), (px1, py1), (185, 185, 185), 1)

    cjk_fonts = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    font_path = next((p for p in cjk_fonts if os.path.exists(p)), None)
    if font_path is not None:
        try:
            from PIL import Image, ImageDraw, ImageFont

            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.truetype(font_path, 22)
            draw.text(pos, text, font=font, fill=fg)
            return np.array(pil_img)
        except Exception:
            pass

    # Deterministic multi-stroke CJK glyph fallback for CI environments without system CJK fonts
    x_cursor, y_cursor = pos
    for ch in text:
        if ch == " ":
            x_cursor += 10
            continue
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            rng = np.random.RandomState(code)
            cell = np.zeros((22, 22), dtype=np.uint8)
            for _ in range(6):
                x0, y0 = rng.randint(2, 20), rng.randint(2, 20)
                x1, y1 = rng.randint(2, 20), rng.randint(2, 20)
                cv2.line(cell, (x0, y0), (x1, y1), 255, 2)
            y_end = min(size[0], y_cursor + 22)
            x_end = min(size[1], x_cursor + 22)
            mask = cell[: y_end - y_cursor, : x_end - x_cursor] > 0
            img[y_cursor:y_end, x_cursor:x_end][mask] = fg
            x_cursor += 24
        else:
            cv2.putText(
                img,
                ch,
                (x_cursor, y_cursor + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                fg,
                2,
                cv2.LINE_AA,
            )
            x_cursor += 12
    return img


def test_partial_chinese_task_progress_change():
    """Phase 1B: '任务进度 10/100' -> '任务进度 90/100' must invalidate cache in both track and match."""
    cache = TemporalROICache(similarity_threshold=0.82, chinese_only=True)
    img1 = render_chinese_panel("任务进度 10/100")
    img2 = render_chinese_panel("任务进度 90/100")
    poly = [[16.0, 12.0], [228.0, 12.0], [228.0, 46.0], [16.0, 46.0]]
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
        polygon=poly,
        text="任务进度 10/100",
        score=0.96,
        crop_sha256="sha_10",
        tile=(0, 0, 260, 120),
        touches_tile_edge=False,
    )
    cache.register_observations([obs1], img1)

    # 1. Even if match_box_with_cache is called directly on Frame 2, it must reject stale text
    direct_obs, _ = cache.match_box_with_cache(
        np.array(poly, dtype=np.float32), img2, 100, {"num": 1, "den": 1000}, 0.1, next_id
    )
    assert direct_obs is None

    # 2. track_and_emit on Frame 2 must reject cache hit and mark unreliable
    hits, _, reliable = cache.track_and_emit(img2, (), 100, {"num": 1, "den": 1000}, 0.1, next_id)
    assert len(hits) == 0
    assert reliable is False

    # 3. Subsequent match_box_with_cache after track_and_emit must also return None
    after_obs, _ = cache.match_box_with_cache(
        np.array(poly, dtype=np.float32), img2, 100, {"num": 1, "den": 1000}, 0.1, next_id
    )
    assert after_obs is None


def test_single_chinese_char_change_in_long_string():
    """Phase 1B: One Chinese character changing in a long string must not reuse cached text."""
    cache = TemporalROICache(similarity_threshold=0.82, chinese_only=True)
    img1 = render_chinese_panel("十八岁的机车梦")
    img2 = render_chinese_panel("十八岁的汽车梦")
    poly = [[16.0, 12.0], [228.0, 12.0], [228.0, 46.0], [16.0, 46.0]]
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
        polygon=poly,
        text="十八岁的机车梦",
        score=0.95,
        crop_sha256="sha_moto",
        tile=(0, 0, 260, 120),
        touches_tile_edge=False,
    )
    cache.register_observations([obs1], img1)

    hits, _, reliable = cache.track_and_emit(img2, (), 100, {"num": 1, "den": 1000}, 0.1, next_id)
    assert len(hits) == 0
    assert reliable is False

    matched_obs, _ = cache.match_box_with_cache(
        np.array(poly, dtype=np.float32), img2, 100, {"num": 1, "den": 1000}, 0.1, next_id
    )
    assert matched_obs is None


def test_chinese_text_slight_motion_and_reappearance():
    """Phase 1B: Slight motion of unchanged Chinese text hits cache; disappear + reappear revalidates cleanly."""
    cache = TemporalROICache(similarity_threshold=0.82, max_motion_padding=8, chinese_only=True)
    img1 = render_chinese_panel("任务进度 10/100", pos=(22, 16), panel=False)
    img_shifted = render_chinese_panel("任务进度 10/100", pos=(26, 19), panel=False)
    poly = [[20.0, 12.0], [220.0, 12.0], [220.0, 46.0], [20.0, 46.0]]
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
        polygon=poly,
        text="任务进度 10/100",
        score=0.97,
        crop_sha256="sha_1",
        tile=(0, 0, 260, 120),
        touches_tile_edge=False,
    )
    cache.register_observations([obs1], img1)

    # Shifted by dx=4, dy=3 -> must hit cache
    hits, _, reliable = cache.track_and_emit(
        img_shifted, (), 100, {"num": 1, "den": 1000}, 0.1, next_id
    )
    assert len(hits) == 1
    assert reliable is True
    assert hits[0].text == "任务进度 10/100"

    # Disappear for 2 frames -> evicted from active_rois
    blank = np.full((120, 260, 3), 24, dtype=np.uint8)
    hits_d1, _, rel_d1 = cache.track_and_emit(blank, (), 200, {"num": 1, "den": 1000}, 0.2, next_id)
    assert len(hits_d1) == 0
    assert rel_d1 is False
    # Even after 1st miss, match_box_with_cache must NOT reuse the disappeared ROI
    stale_match, _ = cache.match_box_with_cache(
        np.array(poly, dtype=np.float32), blank, 200, {"num": 1, "den": 1000}, 0.2, next_id
    )
    assert stale_match is None

    hits_d2, _, _ = cache.track_and_emit(blank, (), 300, {"num": 1, "den": 1000}, 0.3, next_id)
    assert len(hits_d2) == 0
    assert len(cache.active_rois) == 0


def test_scene_cut_drains_pending_crops_before_cache_reset():
    """Phase 1A: With batch_size=8 and 7 pending crops in Scene A, Scene B (cut=True) must:
    - Not lose any of Scene A's 7 observations
    - Preserve strict PTS ordering for frames and EventBuilder
    - Not pollute Scene B's cache with Scene A's ROIs
    - Close Scene A events at cut boundary without duplicate events
    """
    from collections import Counter
    from types import SimpleNamespace
    from vnle.events import EventBuilder
    from vnle.ocr import RapidAdapter
    from vnle.pipeline import DEFAULT_CONFIG

    adapter = RapidAdapter.__new__(RapidAdapter)
    adapter.np, adapter.cv2 = np, cv2
    adapter.timings, adapter.calls = Counter(), Counter()
    adapter.counter = 0
    adapter.batch_size = 8
    adapter.config = dict(
        DEFAULT_CONFIG,
        recognition_batch=8,
        enable_temporal_cache=True,
        enable_motion_gated_det=True,
        chinese_only=True,
    )
    adapter.pending_crops = []
    adapter.pending_frames = []
    adapter.cache = TemporalROICache(similarity_threshold=0.82, chinese_only=True)
    adapter.last_detector_time = -999.0
    adapter.scene_epoch = 0

    scene_a_img = render_chinese_panel("场景甲文本一", pos=(20, 16))
    scene_b_img = render_chinese_panel("场景乙全新内容", pos=(20, 60))

    # Scene A has 7 distinct boxes in frame 0 (t=0.0), so 7 crops sit in pending_crops (< batch_size 8)
    boxes_a = np.array(
        [
            [[16, 12 + i * 10], [120, 12 + i * 10], [120, 20 + i * 10], [16, 20 + i * 10]]
            for i in range(7)
        ],
        dtype=np.float32,
    )
    # Scene B has 1 box in frame 1 (t=0.5, cut=True)
    boxes_b = np.array(
        [[[20, 56], [220, 56], [220, 90], [20, 90]]],
        dtype=np.float32,
    )

    call_idx = 0

    def fake_detect(img):
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            return SimpleNamespace(boxes=boxes_a)
        return SimpleNamespace(boxes=boxes_b)

    rec_batches = []

    def fake_recognize(args):
        n = len(args.img)
        rec_batches.append(n)
        # Distinguish Scene A vs Scene B by checking if pending_crops was drained on cut
        if len(rec_batches) == 1:
            txts = [f"场景甲第{i}行" for i in range(n)]
        else:
            txts = [f"场景乙第{i}行" for i in range(n)]
        return SimpleNamespace(txts=txts, scores=[0.95] * n)

    adapter.detector = fake_detect
    adapter.recognizer = fake_recognize
    adapter.crop = lambda img, box: img[
        int(box[:, 1].min()) : int(box[:, 1].max()),
        int(box[:, 0].min()) : int(box[:, 0].max()),
    ].copy()

    # Feed Scene A (7 crops < batch_size 8 -> stays pending)
    out_a = adapter.feed(scene_a_img, (), 0, {"num": 1, "den": 1000}, 0.0, cut=False)
    assert out_a == []
    assert len(adapter.pending_crops) == 7

    # Feed Scene B with cut=True -> must drain Scene A's 7 crops first, reset cache, then queue Scene B's 1 crop
    out_b = adapter.feed(scene_b_img, (), 500, {"num": 1, "den": 1000}, 0.5, cut=True)
    # Scene A frame (t=0.0) must be returned now, completed with all 7 observations!
    assert len(out_b) == 1
    f_time_s, f_obs, _, f_cut = out_b[0]
    assert f_time_s == 0.0
    assert f_cut is False
    assert len(f_obs) == 7
    assert [o.text for o in f_obs] == [f"场景甲第{i}行" for i in range(7)]

    # Crucially, Scene B's active cache must NOT contain any of Scene A's 7 ROIs!
    assert len(adapter.cache.active_rois) == 0

    # Now flush remaining (Scene B's 1 pending crop)
    out_rem = adapter.flush_remaining()
    assert len(out_rem) == 1
    b_time_s, b_obs, _, b_cut = out_rem[0]
    assert b_time_s == 0.5
    assert b_cut is True
    assert len(b_obs) == 1
    assert b_obs[0].text == "场景乙第0行"

    # Now Scene B's cache has only Scene B's 1 ROI
    assert len(adapter.cache.active_rois) == 1
    assert list(adapter.cache.active_rois.values())[0].text == "场景乙第0行"

    # Verify EventBuilder timeline ordering and boundary closing
    tracker = EventBuilder(0.0, 0.68)
    for t_s, obs_list, _, is_cut in out_b + out_rem:
        tracker.update(t_s, obs_list, cut=is_cut)
    tracker.boundary(1.0, "ANALYSIS_END")
    events = tracker.snapshot()
    assert len(events) == 8
    scene_a_events = [e for e in events if e["shot_id"] == 0]
    scene_b_events = [e for e in events if e["shot_id"] == 1]
    assert len(scene_a_events) == 7
    assert len(scene_b_events) == 1
    assert all(e["close_reason"] == "SHOT_OR_EXCLUSION_CHANGE" for e in scene_a_events)
    assert all(e["end_s"] == 0.5 for e in scene_a_events)
    assert scene_b_events[0]["start_s"] == 0.5


def test_resource_sampler_and_ort_profiling_toggle(tmp_path):
    from types import SimpleNamespace
    from collections import Counter
    from vnle.ocr import RapidAdapter
    from vnle.pipeline import DEFAULT_CONFIG
    from vnle.profiler import GranularProfiler, sample_resources

    res = sample_resources()
    assert res["sample_count"] >= 1
    assert "vram_process_peak_mb" in res
    assert "vram_device_peak_mb" in res
    assert "ram_peak_working_set_mb" in res

    adapter = RapidAdapter.__new__(RapidAdapter)
    adapter.config = dict(DEFAULT_CONFIG)
    adapter.runtime_dll_paths = []
    adapter.available_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    adapter.timings = Counter()
    adapter.calls = Counter()
    adapter.cache = SimpleNamespace(stats={"cache_hits": 3})
    adapter.ort_profiling_enabled = False
    adapter.profiler = GranularProfiler(enabled=False)
    adapter.output = tmp_path
    adapter.sessions = {
        "detector": SimpleNamespace(
            get_providers=lambda: ["CUDAExecutionProvider"],
            get_provider_options=lambda: {"CUDAExecutionProvider": {}},
        ),
        "recognizer": SimpleNamespace(
            get_providers=lambda: ["CUDAExecutionProvider"],
            get_provider_options=lambda: {"CUDAExecutionProvider": {}},
        ),
    }

    summary = adapter.finish()
    assert summary["profiles"]["detector"]["ort_profiling_enabled"] is False
    assert summary["profiles"]["detector"]["trace"] is None
    assert summary["resources"] is not None
    assert summary["resources"]["sample_count"] >= 1


