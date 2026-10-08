"""Audio supplies coverage; OCR supplies evidence, never a mandatory speech gate."""
from __future__ import annotations

from difflib import SequenceMatcher

from autodub.contracts import Segment

HOMOPHONES = ("他她它", "在再", "的得地", "做作", "是事市", "到道", "已以", "像象")
NONLEXICAL = frozenset("啊呃嗯哎哈呵唉哦喔哼！？!?,，。… ")


def lexical(text: str) -> str:
    return "".join(c for c in text if c.isalnum())


def canonical_text(asr: str, ocr: str) -> tuple[str, list[dict]]:
    """Only verified equal-length homophone substitutions; keep ASR punctuation."""
    left, right = lexical(asr), lexical(ocr)
    if not left or len(left) != len(right):
        return asr, []
    replacements = []
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a != b:
            if not any(a in group and b in group for group in HOMOPHONES):
                return asr, []
            replacements.append({"index": index, "asr": a, "ocr": b})
    if not replacements:
        return asr, []
    chars, cursor = [], 0
    for char in asr:
        if char.isalnum():
            char = right[cursor]
            cursor += 1
        chars.append(char)
    return "".join(chars), replacements


def sequence_evidence(text: str, events: list[dict]) -> dict:
    """Match successive captions to a paragraph without inventing speech timing.

    Captions can contain decorative titles. Count only ordered matching spans of
    the ASR, and never let repeated captions count the same characters twice.
    """
    source = lexical(text)
    cursor, covered, matches = 0, 0, []
    for event in sorted(events, key=lambda e: (e["start_ms"], e["end_ms"])):
        caption = lexical(event["text"])
        blocks = SequenceMatcher(None, source[cursor:], caption, autojunk=False).get_matching_blocks()
        accepted = [block for block in blocks if block.size >= 2]
        if not accepted:
            continue
        size = sum(block.size for block in accepted)
        # Isolated shared characters and largely unrelated title text are weak evidence.
        if size / max(1, len(caption)) < 0.5:
            continue
        start = cursor
        covered += size
        cursor += max(block.a + block.size for block in accepted)
        matches.append({"event_id": event["id"], "asr_start_char": start,
                        "asr_end_char": cursor, "matched_chars": size})
    return {"audio_ocr_sequence_coverage": covered / max(1, len(source)),
            "ordered_caption_matches": matches}


def decide(segment: dict, events: list[dict]) -> dict:
    value = Segment.model_validate(segment).model_dump()
    relevant = [event for event in events
                if event["start_ms"] < value["end_ms"] and value["start_ms"] < event["end_ms"]]
    dialogue = [e for e in relevant if e.get("kind") == "DIALOGUE" and e.get("score", 0) >= 0.7]
    lyrics = [e for e in relevant if e.get("kind") == "LYRIC"]
    evidence = dict(value["dialogue_evidence"])
    evidence["ocr_event_ids"] = [e["id"] for e in relevant]
    text = value["zh_text"].strip()
    sustained = (evidence.get("vocal_run_ms", 0) >= 25000
                 and evidence.get("pitched_fraction", 0) >= 0.85
                 and evidence.get("pitch_span_semitones", 0) >= 5)
    confidence = value["confidence"].get("asr")
    kind, action, review = "AMBIGUOUS", "KEEP", True
    if not text or all(c in NONLEXICAL for c in text):
        kind, review = "NONLEXICAL", False
    elif lyrics and sustained and not dialogue:
        kind, review = "SINGING_OST", False
    elif confidence is not None and confidence < 0.45:
        evidence["ambiguity_reason"] = "LOW_UNCALIBRATED_ASR_SCORE"
    elif sustained:
        # A bottom-center lyric must not become dialogue solely by matching OCR.
        evidence["ambiguity_reason"] = "POSSIBLE_SINGING_REQUIRES_FOREGROUND_DIALOGUE_REVIEW"
    elif dialogue:
        best = max(dialogue, key=lambda e: SequenceMatcher(None, lexical(text), lexical(e["text"])).ratio())
        similarity = SequenceMatcher(None, lexical(text), lexical(best["text"])).ratio()
        evidence["audio_ocr_similarity"] = similarity
        sequence = sequence_evidence(text, dialogue)
        evidence.update(sequence)
        canonical, corrections = canonical_text(text, best["text"])
        if similarity >= 0.7 or corrections or sequence["audio_ocr_sequence_coverage"] >= 0.7:
            value["zh_text"] = canonical
            evidence["homophone_corrections"] = corrections
            kind, action, review = "DIALOGUE", "DUB", False
        else:
            evidence["ambiguity_reason"] = "AUDIO_OCR_CONFLICT"
    else:
        kind, action, review = "UNSUBTITLED_DIALOGUE", "DUB", False
    value.update(dialogue_kind=kind, action=action, needs_review=review, dialogue_evidence=evidence,
                 speech_kind="nonverbal" if kind == "NONLEXICAL" else "lexical" if action == "DUB" else "unknown")
    # OCR text correction preserves the separately recorded timing evidence.
    return value


def merge_dialogue(segments: list[dict], events: list[dict]) -> dict:
    result = [decide(s, events) for s in segments]
    return {"schema_version": 1, "segments": result,
            "issues": [{"segment_id": s["id"], "code": "AMBIGUOUS_DIALOGUE", "evidence": s["dialogue_evidence"]}
                       for s in result if s["needs_review"]],
            "classification_policy": "EXPLICIT_RULES_UNCALIBRATED", "subtitle_required_for_dialogue": False}
