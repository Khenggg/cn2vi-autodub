"""Split forced-aligned words into dubbing-sized utterances.

ASR chunks are fixed-length windows (minutes long); dubbing needs one unit per
spoken line so translation, TTS and timing fit each line to its own slot.
Boundaries come from sentence punctuation in the ASR text, silence between
aligned words, and hard duration limits.  Pure functions, no model imports.
"""
from __future__ import annotations

import unicodedata

SENTENCE_END = frozenset("。！？!?…;；")
CLAUSE_BREAK = frozenset("，,、：:")

GAP_MS = 400        # silence that always starts a new utterance
MIN_MS = 800        # shorter utterances are merged into a neighbour when possible
MAX_MS = 12_000     # longer utterances are split at the best internal pause


def _is_punct(char: str) -> bool:
    return char.isspace() or unicodedata.category(char)[0] in {"P", "S"}


def locate_words(text: str, words: list[dict]) -> list[dict]:
    """Map each aligned word back into the ASR text and read its trailing punctuation.

    Returns one dict per word: ``start``/``end`` character offsets (or None when
    the word cannot be found) and ``trailing`` punctuation that follows it.
    """
    located: list[dict] = []
    cursor = 0
    for word in words:
        token = word["t"]
        index = text.find(token, cursor) if token else -1
        if index < 0:
            located.append({"start": None, "end": None, "trailing": ""})
            continue
        end = index + len(token)
        located.append({"start": index, "end": end, "trailing": ""})
        cursor = end
    for position, item in enumerate(located):
        if item["end"] is None:
            continue
        following = next((later["start"] for later in located[position + 1:]
                          if later["start"] is not None), len(text))
        trailing = []
        for char in text[item["end"]:following]:
            if not _is_punct(char):
                break
            if not char.isspace():
                trailing.append(char)
        item["trailing"] = "".join(trailing)
        # Character offset where this word's punctuation ends, for text slicing.
        run = 0
        for char in text[item["end"]:following]:
            if not _is_punct(char):
                break
            run += 1
        item["text_end"] = item["end"] + run
    return located


def _duration(words: list[dict], group: list[int]) -> int:
    return words[group[-1]]["e"] - words[group[0]]["s"]


def _gap_after(words: list[dict], index: int) -> int:
    return words[index + 1]["s"] - words[index]["e"] if index + 1 < len(words) else 0


def _boundary_score(words: list[dict], located: list[dict], index: int) -> int:
    trailing = located[index]["trailing"]
    score = _gap_after(words, index)
    if any(char in SENTENCE_END for char in trailing):
        score += 1000
    elif any(char in CLAUSE_BREAK for char in trailing):
        score += 250
    return score


def _hard_groups(words: list[dict], located: list[dict], indices: list[int],
                 gap_ms: int, min_ms: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    for position, index in enumerate(indices):
        current.append(index)
        if position + 1 == len(indices):
            break
        following = indices[position + 1]
        gap = words[following]["s"] - words[index]["e"]
        sentence_end = any(char in SENTENCE_END for char in located[index]["trailing"])
        if gap >= gap_ms or (sentence_end and _duration(words, current) >= min_ms):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _split_long(words: list[dict], located: list[dict], group: list[int],
                min_ms: int, max_ms: int) -> list[list[int]]:
    if _duration(words, group) <= max_ms or len(group) < 2:
        return [group]
    best, best_key = None, None
    for cut in range(len(group) - 1):
        left, right = group[:cut + 1], group[cut + 1:]
        if _duration(words, left) < min_ms or _duration(words, right) < min_ms:
            continue
        balance = abs(_duration(words, left) - _duration(words, right))
        key = (_boundary_score(words, located, group[cut]), -balance)
        if best_key is None or key > best_key:
            best, best_key = cut, key
    if best is None:  # no cut keeps both halves above the minimum; cut in the middle
        best = len(group) // 2 - 1 if len(group) > 2 else 0
    return (_split_long(words, located, group[:best + 1], min_ms, max_ms)
            + _split_long(words, located, group[best + 1:], min_ms, max_ms))


def _merge_short(words: list[dict], groups: list[list[int]], gap_ms: int,
                 min_ms: int, max_ms: int) -> list[list[int]]:
    merged = [list(group) for group in groups]
    changed = True
    while changed:
        changed = False
        for position, group in enumerate(merged):
            if _duration(words, group) >= min_ms:
                continue
            options = []
            if position > 0:
                previous = merged[position - 1]
                options.append((words[group[0]]["s"] - words[previous[-1]]["e"], position - 1))
            if position + 1 < len(merged):
                following = merged[position + 1]
                options.append((words[following[0]]["s"] - words[group[-1]]["e"], position))
            options = [(gap, left) for gap, left in options if gap < gap_ms * 2
                       and _duration(words, merged[left] + merged[left + 1]) <= max_ms]
            if not options:
                continue
            _, left = min(options)
            merged[left:left + 2] = [merged[left] + merged[left + 1]]
            changed = True
            break
    return merged


def build_utterances(text: str, words: list[dict], window: tuple[int, int] | None = None, *,
                     gap_ms: int = GAP_MS, min_ms: int = MIN_MS,
                     max_ms: int = MAX_MS) -> list[dict]:
    """Group ``words`` ({t,s,e} dicts, monotonic) into utterances.

    ``window`` keeps only words whose start lies in ``[lo, hi)``; callers use it to
    drop the overlap duplicated between neighbouring ASR chunks.  Each result has
    ``start_ms``, ``end_ms``, ``text`` and ``words``.
    """
    if not words:
        return []
    located = locate_words(text, words)
    lo, hi = window if window else (words[0]["s"], words[-1]["e"] + 1)
    indices = [i for i, word in enumerate(words) if lo <= word["s"] < hi]
    if not indices:
        return []
    groups: list[list[int]] = []
    for group in _hard_groups(words, located, indices, gap_ms, min_ms):
        groups.extend(_split_long(words, located, group, min_ms, max_ms))
    groups = _merge_short(words, groups, gap_ms, min_ms, max_ms)

    utterances = []
    for group in groups:
        first, last = group[0], group[-1]
        if located[first]["start"] is not None and located[last].get("text_end") is not None:
            snippet = text[located[first]["start"]:located[last]["text_end"]].strip()
        else:
            snippet = ""
        utterances.append({
            "start_ms": words[first]["s"], "end_ms": words[last]["e"],
            "text": snippet or " ".join(words[i]["t"] for i in group),
            "words": [dict(words[i]) for i in group],
        })
    return utterances
