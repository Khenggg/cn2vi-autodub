"""Durable, tamper-evident stage checkpoints for the pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from autodub.storage import atomic_json, safe_path, sha256_file


class CheckpointError(ValueError):
    """A checkpoint or one of its recorded artifacts is invalid."""

    def __init__(self, message: str, *, stage: str | None = None):
        super().__init__(message)
        self.stage = stage


def fingerprint(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def checkpoint_path(data_dir: Path, episode_id: str) -> Path:
    return data_dir / "checkpoints" / episode_id / "state.json"


def read_checkpoint(data_dir: Path, episode_id: str, source_sha256: str,
                    config_fingerprint: str) -> dict:
    path = checkpoint_path(data_dir, episode_id)
    if not path.is_file():
        return {"schema_version": 1, "episode_id": episode_id,
                "source_sha256": source_sha256, "completed_stages": [],
                "next_stage": "ASR", "artifacts": {}, "results": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError("Checkpoint is unreadable") from error
    if not isinstance(value, dict) or value.get("schema_version") not in (1, 2):
        raise CheckpointError("Checkpoint schema is unsupported")
    if value.get("episode_id") != episode_id or value.get("source_sha256") != source_sha256:
        raise CheckpointError("Checkpoint source identity changed")
    allowed_stages = {"PREPARING", "ASR", "ALIGNING", "TRANSLATING", "SEPARATING",
                      "TTS", "TIMING", "AUDIO_MIX", "QC", "VISION_RENDER", "ENCODING"}
    completed = value.get("completed_stages", [])
    next_stage = value.get("next_stage")
    if (not isinstance(completed, list) or any(stage not in allowed_stages for stage in completed)
            or next_stage is not None and next_stage not in allowed_stages):
        raise CheckpointError("Checkpoint stage cursor is invalid")
    results = value.get("results", {})
    inputs = value.get("stage_inputs", {})
    if not isinstance(results, dict) or not isinstance(inputs, dict):
        raise CheckpointError("Checkpoint stage data is invalid")
    # Schema 1 checkpoints were emitted by preparation only and remain resumable.
    if value.get("schema_version") == 2 and value.get("config_fingerprint") != config_fingerprint:
        raise CheckpointError("Pipeline configuration changed since checkpoint")
    artifacts = value.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise CheckpointError("Checkpoint artifact index is invalid")
    for key, record in artifacts.items():
        stage = key.split(":", 1)[0] if isinstance(key, str) else None
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise CheckpointError("Checkpoint artifact record is invalid", stage=stage)
        try:
            artifact = safe_path(data_dir, record["path"])
        except ValueError as error:
            raise CheckpointError("Checkpoint artifact path is invalid", stage=stage) from error
        if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
            raise CheckpointError("Checkpoint artifact checksum mismatch", stage=stage)
    known_paths = {record["path"] for record in artifacts.values()}
    for stage, result in results.items():
        if not isinstance(result, dict):
            raise CheckpointError("Checkpoint stage result is invalid", stage=stage)
        paths = result.get("artifacts", [])
        clips = result.get("clips", {})
        if not isinstance(paths, list) or not isinstance(clips, dict):
            raise CheckpointError("Checkpoint stage artifact list is invalid", stage=stage)
        for raw_path in [*paths, *clips.values()]:
            if not isinstance(raw_path, str):
                raise CheckpointError("Checkpoint stage artifact path is invalid", stage=stage)
            try:
                resolved = safe_path(data_dir, str(Path(raw_path).resolve().relative_to(data_dir.resolve())))
            except (OSError, ValueError) as error:
                raise CheckpointError("Checkpoint stage artifact path is invalid", stage=stage) from error
            if resolved.relative_to(data_dir.resolve()).as_posix() not in known_paths:
                raise CheckpointError("Checkpoint stage artifact is not indexed", stage=stage)
    return value


def invalidate_from_stage(data_dir: Path, episode_id: str, source_sha256: str,
                          config_fingerprint: str, stage: str) -> Path:
    """Drop a corrupted derived stage and its dependants while preserving its inputs."""
    path = checkpoint_path(data_dir, episode_id)
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("schema_version") not in (1, 2)
            or value.get("episode_id") != episode_id or value.get("source_sha256") != source_sha256):
        raise CheckpointError("Checkpoint source identity changed")
    if value.get("schema_version") != 2 or value.get("config_fingerprint") != config_fingerprint:
        raise CheckpointError("Checkpoint configuration is invalid")
    order = ["PREPARING", "ASR", "ALIGNING", "TRANSLATING", "SEPARATING", "TTS",
             "TIMING", "AUDIO_MIX", "QC", "VISION_RENDER", "ENCODING"]
    if stage not in order:
        raise CheckpointError("Checkpoint artifact stage is unknown")
    invalid = set(order[order.index(stage):])
    value["completed_stages"] = [item for item in value.get("completed_stages", [])
                                  if item not in invalid]
    value["results"] = {key: item for key, item in value.get("results", {}).items()
                        if key not in invalid}
    value["stage_inputs"] = {key: item for key, item in value.get("stage_inputs", {}).items()
                             if key not in invalid}
    value["artifacts"] = {key: item for key, item in value.get("artifacts", {}).items()
                          if key.split(":", 1)[0] not in invalid}
    value["next_stage"] = stage
    atomic_json(path, value)
    return path


def save_checkpoint(data_dir: Path, episode_id: str, source_sha256: str,
                    config_fingerprint: str, *, completed_stages: list[str],
                    next_stage: str | None, artifacts: dict[str, dict],
                    results: dict[str, Any], media: dict | None = None,
                    stage_inputs: dict[str, str] | None = None) -> Path:
    path = checkpoint_path(data_dir, episode_id)
    payload = {"schema_version": 2, "episode_id": episode_id,
               "source_sha256": source_sha256,
               "config_fingerprint": config_fingerprint,
               "completed_stages": list(completed_stages),
               "next_stage": next_stage, "artifacts": artifacts,
               "results": results, "media": media,
               "stage_inputs": stage_inputs or {}}
    atomic_json(path, payload)
    return path


def artifact_record(data_dir: Path, path: Path) -> dict:
    resolved = path.resolve(strict=True)
    relative = resolved.relative_to(data_dir.resolve()).as_posix()
    return {"path": relative, "sha256": sha256_file(resolved)}
