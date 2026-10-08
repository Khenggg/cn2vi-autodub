"""Bounded DeepSeek Flash batches; singing and nonlexical candidates are not translated."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from autodub.adapters.runtime_paths import output_folder
from autodub.adapters.v1_translation import run as translate_batch
from autodub.storage import atomic_json


def run(source: Path, config: dict) -> dict:
    folder = output_folder(config)
    segments = config["segments"]
    eligible = [s for s in segments if s["action"] == "DUB" and not s.get("context_provenance", {}).get("human_reviewed")]
    if len({s['id'] for s in segments}) != len(segments):
        raise ValueError('Duplicate translation segment identity')
    size = int(config.get("translation_batch_size", 24))
    workers = int(config.get("translation_concurrency", 2))
    if not 1 <= size <= 48 or not 1 <= workers <= 4:
        raise ValueError("Translation limits exceeded")
    groups = [eligible[i:i + size] for i in range(0, len(eligible), size)]

    def operation(item):
        index, group = item
        return translate_batch(source, {**config, "segments": group,
            "translation_output_subdir": f"batch-{index:06d}", "translation_model": "deepseek-flash",
            "translation_batch_size": size, "translation_max_tokens": 8192})

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(operation, enumerate(groups)))
    returned = [s['id'] for result in results for s in result['segments']]
    if len(returned) != len(set(returned)) or set(returned) != {s['id'] for s in eligible}:
        raise ValueError('Translation response identity coverage mismatch')
    translated = {s["id"]: s for result in results for s in result["segments"]}
    complete = [translated.get(s["id"], s) for s in segments]
    calls = [call for result in results for call in result.get("api_calls", [])]
    issues = [issue for result in results for issue in result.get("quality_evidence", {}).get("issues", [])]
    proposals = [p for result in results for p in result.get("context_proposals", [])]
    target = folder / "translation.json"
    atomic_json(target, {"schema_version": 1, "segments": complete, "issues": issues,
                        "api_calls": calls, "provider": "DeepSeek", "requested_model": "deepseek-flash",
                        "api_cost_vnd": None, "context_proposals": proposals, "model_fallbacks": []})
    return {"segments": complete, "api_calls": calls, "context_proposals": proposals,
            "artifacts": [str(target)], "stage_status": "DEGRADED" if issues else "SUCCESS",
            "quality_evidence": {"issues": issues}}
