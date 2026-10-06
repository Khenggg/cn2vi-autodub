import pytest

from autodub.domain import plan_separation_windows
from autodub.utterances import build_utterances, locate_words


def words(spec):
    """spec: list of (text, start_ms, end_ms)."""
    return [{"t": t, "s": s, "e": e} for t, s, e in spec]


def test_silence_gap_starts_new_utterance():
    w = words([("你好", 0, 500), ("吗", 500, 900), ("我很好", 1600, 2400)])
    result = build_utterances("你好吗我很好", w)
    assert [(u["start_ms"], u["end_ms"]) for u in result] == [(0, 900), (1600, 2400)]
    assert [u["text"] for u in result] == ["你好吗", "我很好"]


def test_sentence_punctuation_splits_and_is_kept_in_text():
    w = words([("你好", 0, 600), ("再见", 600, 1200), ("谢谢", 1200, 1800), ("你", 1800, 2400)])
    result = build_utterances("你好，再见。谢谢你！", w, min_ms=500)
    assert [u["text"] for u in result] == ["你好，再见。", "谢谢你！"]
    assert [len(u["words"]) for u in result] == [2, 2]


def test_long_run_is_split_at_best_pause_and_respects_max():
    spec = [(f"w{i}", i * 1000, i * 1000 + 1000) for i in range(10)]
    spec[4] = ("w4", 4000, 4800)  # 200 ms pause after w4 is the best internal cut
    spec[5] = ("w5", 5000, 6000)
    result = build_utterances(" ".join(t for t, _, _ in spec), words(spec),
                              gap_ms=500, min_ms=1000, max_ms=6000)
    assert len(result) == 2
    assert all(u["end_ms"] - u["start_ms"] <= 6000 for u in result)
    assert result[0]["words"][-1]["t"] == "w4"


def test_tiny_fragments_merge_with_nearest_neighbour():
    w = words([("嗯", 0, 200), ("我们走吧", 700, 2000)])
    result = build_utterances("嗯 我们走吧", w, gap_ms=400, min_ms=800)
    assert len(result) == 1
    assert (result[0]["start_ms"], result[0]["end_ms"]) == (0, 2000)


def test_window_drops_words_owned_by_the_neighbouring_chunk():
    w = words([("a", 0, 500), ("b", 500, 1000), ("c", 1000, 1500), ("d", 1500, 2000)])
    result = build_utterances("a b c d", w, window=(1000, 2000))
    assert [x["t"] for u in result for x in u["words"]] == ["c", "d"]
    assert build_utterances("a b c d", w, window=(5000, 6000)) == []
    assert build_utterances("", []) == []


def test_unlocatable_words_fall_back_to_joined_text():
    w = words([("甲", 0, 500), ("乙", 500, 1000)])
    result = build_utterances("completely different text", w)
    assert result[0]["text"] == "甲 乙"
    assert locate_words("甲乙", w)[0]["start"] == 0


def test_utterances_never_overlap_and_stay_ordered():
    spec = [(f"w{i}", i * 300, i * 300 + 300) for i in range(80)]
    result = build_utterances(" ".join(t for t, _, _ in spec), words(spec), max_ms=5000)
    assert len(result) > 3
    assert all(a["end_ms"] <= b["start_ms"] for a, b in zip(result, result[1:], strict=False))


def test_separation_windows_merge_pad_cap_and_cover_every_span():
    spans = [(1000, 2000), (2500, 3500), (300_000, 301_000)]
    windows = plan_separation_windows(spans, 600_000)
    assert windows[0]["start_ms"] == 0 and windows[0]["end_ms"] >= 30_000
    assert all(w["end_ms"] - w["start_ms"] >= 30_000 for w in windows)
    for start, end in spans:
        assert any(w["start_ms"] <= start and end <= w["end_ms"] for w in windows)
    long_run = [(i * 5000, i * 5000 + 4500) for i in range(60)]  # 300 s of speech
    windows = plan_separation_windows(long_run, 400_000)
    assert max(w["end_ms"] - w["start_ms"] for w in windows) <= 120_000
    for start, end in long_run:
        covered = start
        for w in windows:
            if w["start_ms"] <= covered < w["end_ms"]:
                covered = max(covered, min(end, w["end_ms"]))
        assert covered >= end
    assert plan_separation_windows([], 10_000)[0] == {"start_ms": 0, "end_ms": 10_000}
    with pytest.raises(ValueError):
        plan_separation_windows([], 0)
