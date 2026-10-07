"""Evidence-bound series memory; diarization labels are never assumed identities."""
from __future__ import annotations

import copy


def build_context(segments: list[dict], ocr: dict, memory: dict, glossary: dict[str, str]) -> dict:
    registry = copy.deepcopy(memory.get("characters", {}))
    evidence = []
    for frame in ocr.get("frames", []):
        for line in frame.get("lines", []):
            text = line.get("text", "")
            for name, translated in glossary.items():
                if name in text:
                    evidence.append({"name": name, "vi": translated, "at_ms": frame["at_ms"],
                                     "source": "OCR_TEXT_MATCH", "character_identity_confirmed": False})
    contexts = {}
    speaker_map = memory.get("speaker_character_map", {})
    for index, segment in enumerate(segments):
        speaker = segment.get("speaker_id")
        character = speaker_map.get(speaker)
        # Labels are episode-local. Only explicit episode speaker maps supplied by
        # the caller may establish a character; saved series labels are not reused.
        segment["character_id"] = character
        contexts[segment["id"]] = {
            "speaker_id": speaker, "character_id": character,
            "addressee_id": segment.get("addressee_id"),
            "addressing": segment.get("addressing", {}),
            "nearby_dialogue": [{"speaker_id": item.get("speaker_id"), "zh_text": item["zh_text"]}
                                for item in segments[max(0, index - 3):index + 4]],
            "uncertain": character is None or segment.get("addressee_id") is None,
        }
    return {"schema_version": 1, "characters": registry, "name_evidence": evidence,
            "relationships": copy.deepcopy(memory.get("relationships", [])),
            "addressing_history": copy.deepcopy(memory.get("addressing_history", [])),
            "segments": contexts, "unconfirmed_facts_are_not_promoted": True}


def update_memory(memory: dict, segments: list[dict], run_id: str) -> dict:
    value = copy.deepcopy(memory)
    value.pop("speaker_character_map", None)
    history = value.setdefault("addressing_history", [])
    for segment in segments:
        if segment.get("character_id") and segment.get("addressee_id") and segment.get("addressing"):
            history.append({"run_id": run_id, "speaker": segment["character_id"],
                            "addressee": segment["addressee_id"], "addressing": segment["addressing"],
                            "provenance": segment.get("context_provenance", {}), "confirmed": False})
    value["addressing_history"] = history[-200:]
    value["last_dialogue"] = [{"zh_text": s["zh_text"], "subtitle_vi": s.get("subtitle_vi", ""),
                               "character_id": s.get("character_id")} for s in segments[-30:]]
    value["last_run_id"] = run_id
    return value
