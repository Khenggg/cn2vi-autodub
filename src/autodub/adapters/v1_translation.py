"""One explicit DeepSeek provider; per-batch failure retains original speech."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

from autodub.adapters.common import output_folder
from autodub.adapters.local_translation import _messages, _parse_completion
from autodub.contracts import Segment
from autodub.storage import atomic_json


def run(source: Path, config: dict) -> dict:
    del source
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is required; no translation model fallback")
    segments = [Segment.model_validate(s) for s in config["segments"]]
    result, issues, calls, proposals = [], [], [], []
    tick = time.perf_counter()
    for start in range(0, len(segments), 10):
        batch = segments[start:start + 10]
        messages = _messages(batch, config.get("glossary", {}), config)
        messages[0]["content"] += (
            " For this V1 request, the outer JSON must contain segments and character_context. "
            "Keep the exact translation item fields already specified. character_context is an array of "
            "objects: id (same segment alias), character_id (existing registry id or null), "
            "addressee_id (existing registry id or null), addressing (self/other string map), "
            "evidence (exact Chinese quotation from this batch). Use null and empty addressing when "
            "identity/relationship is uncertain. Never infer a character merely from diarization labels. "
            "Use addressing history and locked character facts; include tentative name/relationship "
            "findings only in evidence, without promoting them into confirmed series facts."
        )
        request = urllib.request.Request("https://api.deepseek.com/chat/completions", method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
            data=json.dumps({"model": "deepseek-chat", "temperature": 0.2, "max_tokens": 4096,
                             "response_format": {"type": "json_object"},
                             "messages": messages},
                            ensure_ascii=False).encode())
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                value = json.loads(response.read())
            calls.append({"usage": value.get("usage"), "provider_model": value.get("model"),
                          "request_id": value.get("id")})
            decoded = json.loads(value["choices"][0]["message"]["content"])
            translated = _parse_completion(batch, json.dumps({"segments": decoded["segments"]}),
                                           config.get("glossary", {}))
            by_alias = {f"s{index + 1}": s for index, s in enumerate(translated)}
            known = config.get("character_context", {}).get("characters", {})
            for context in decoded.get("character_context", []):
                segment = by_alias.get(context.get("id"))
                evidence = context.get("evidence", "")
                if not segment or not isinstance(evidence, str):
                    continue
                proven = bool(evidence) and any(evidence in s.zh_text for s in batch)
                character, addressee = context.get("character_id"), context.get("addressee_id")
                addressing = context.get("addressing", {})
                if (not proven or (character is not None and character not in known)
                        or (addressee is not None and addressee not in known)
                        or not isinstance(addressing, dict)
                        or any(k not in {"self", "other"} or not isinstance(v, str) for k, v in addressing.items())):
                    issues.append({"segment_id": segment.id, "code": "UNCONFIRMED_CONTEXT_PROPOSAL"})
                    continue
                segment.character_id, segment.addressee_id, segment.addressing = character, addressee, addressing
                proposals.append({**context, "segment_id": segment.id, "run_evidence_verified": proven})
            result.extend(s.model_dump() for s in translated)
        except Exception as error:
            issues.append({"segment_ids": [s.id for s in batch], "code": "TRANSLATION_BATCH_UNAVAILABLE",
                           "error_type": type(error).__name__})
            result.extend(s.model_copy(update={"action": "KEEP", "needs_review": True}).model_dump()
                          for s in batch)
    target = output_folder(config) / "translation.json"
    atomic_json(target, {"schema_version": 1, "segments": result, "issues": issues, "api_calls": calls,
                         "provider": "DeepSeek", "model_fallbacks": [], "api_cost_vnd": None,
                         "context_proposals": proposals})
    return {"stage_status": "DEGRADED" if issues else "SUCCESS", "api_calls": calls,
            "quality_evidence": {"issues": issues}, "segments": result, "context_proposals": proposals,
            "metrics": {"inference_ms": (time.perf_counter() - tick) * 1000}, "artifacts": [str(target)]}
