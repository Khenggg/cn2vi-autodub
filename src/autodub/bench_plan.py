"""Generate a corpus-linked, auditable benchmark suite plan without loading models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from autodub.bench_suite import dry_run
from autodub.corpus import CorpusError, sha256_file, validate_manifest
from autodub.storage import atomic_json

PLAN_SCHEMA_VERSION = 1
DEFAULT_STAGES = ("asr", "align", "tts", "bandit", "ocr", "inpaint")
VENV_PROFILES = {"asr": "asr", "align": "asr", "tts": "tts", "bandit": "bandit",
                 "ocr": "vision", "inpaint": "vision"}
_ENV_KEYS = {"env", "env_vars", "environment", "environment_variables"}


class PlanGenerationError(ValueError):
    """Raised when the corpus cannot safely produce a runnable benchmark plan."""


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _contained(root: Path, relative: str) -> Path:
    candidate = root / relative
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise PlanGenerationError("Generated file path must stay inside output_dir")
    return resolved


def _venv_python(venv_root: Path, profile: str) -> Path:
    if os.name == "nt":
        return (venv_root / profile / "Scripts" / "python.exe").resolve()
    return (venv_root / profile / "bin" / "python").resolve()


def _job_id(case_id: str, stage: str, used: set[str] | None = None) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", case_id.strip()).strip("-._").lower() or "case"
    slug = slug[:48].rstrip("-._") or "case"
    identifier = f"{slug}_{stage}"
    if used is not None and identifier in used:
        suffix = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8]
        identifier = f"{slug[:36].rstrip('-._')}_{suffix}_{stage}"
    if used is not None:
        if identifier in used:
            raise PlanGenerationError("Corpus case IDs collide after job ID normalization")
        used.add(identifier)
    return identifier


def _json_copy(value: Any) -> Any:
    """Copy only JSON data and omit environment-like keys at every nesting level."""
    if isinstance(value, dict):
        return {key: _json_copy(item) for key, item in value.items()
                if isinstance(key, str) and key.casefold() not in _ENV_KEYS}
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sanitized_sha256(value: Any) -> str:
    canonical = json.dumps(_json_copy(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanGenerationError(f"Corpus manifest could not be read ({type(error).__name__})") from None
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise PlanGenerationError("Corpus manifest schema is invalid")
    return manifest


def _validate_corpus(path: Path, *, ffprobe_bin: str, allow_synthetic: bool) -> tuple[dict, dict]:
    manifest = _read_manifest(path)
    try:
        report = validate_manifest(path, ffprobe_bin)
    except CorpusError as error:
        # CorpusError messages can embed user paths or source metadata. Keep the CLI/report safe.
        raise PlanGenerationError(f"Corpus validation failed ({type(error).__name__})") from None
    if report.get("status") == "INVALID":
        raise PlanGenerationError("Corpus validation returned INVALID")
    kind = report.get("kind")
    if kind == "representative":
        if report.get("status") != "READY_FOR_REPRESENTATIVE_QUALITY_EVALUATION":
            raise PlanGenerationError("Representative corpus does not meet coverage requirements")
    elif kind == "synthetic_smoke":
        if not allow_synthetic:
            raise PlanGenerationError("Synthetic corpus requires --allow-synthetic")
        if report.get("status") != "SYNTHETIC_SMOKE_ONLY":
            raise PlanGenerationError("Synthetic corpus validation failed")
    else:
        raise PlanGenerationError("Unsupported corpus kind")
    if report.get("representative_quality_pass") is not False:
        raise PlanGenerationError("Corpus validator must never mark quality as passed")
    return manifest, report


def _missing_quality_evidence(stage: str, case: dict, *, synthetic: bool) -> list[str]:
    missing = []
    if stage == "asr" and not case.get("zh_reference"):
        missing.append("zh_reference")
    elif stage == "align" and not case.get("words"):
        missing.append("words")
    elif stage == "tts":
        missing.append("pronunciation_human_rating")
    elif stage == "bandit":
        missing.append("sfx_preservation_reference")
    elif stage == "ocr":
        missing.append("subtitle_text_reference")
    elif stage == "inpaint":
        missing.append("residual_visual_review")
    if synthetic:
        missing.append("representative_ground_truth")
    return missing


def _stage_config(stage: str, case: dict, *, source: Path, output_dir: Path,
                  models_root: Path, cache_root: Path, ffmpeg_bin: str, ffprobe_bin: str,
                  corpus_kind: str, asr_artifact_dir: Path | None = None,
                  ocr_artifact_dir: Path | None = None) -> dict:
    config = {
        "models_root": str(models_root),
        "cache_root": str(cache_root),
        "output_dir": str(output_dir),
        "ffmpeg_bin": ffmpeg_bin,
        "ffprobe_bin": ffprobe_bin,
        "corpus_kind": corpus_kind,
        "representative_quality_pass": False,
    }
    if stage == "asr":
        config.update(start_ms=case["start_ms"], end_ms=case["end_ms"], chunk_ms=240_000, overlap_ms=1_000)
        if case.get("zh_reference"):
            config["reference_text"] = case["zh_reference"]
    elif stage == "align":
        if asr_artifact_dir is None:
            raise PlanGenerationError("Alignment requires an ASR job for the same case")
        config.update(transcript_path=str(asr_artifact_dir / "transcript.zh.json"),
                      reference_words=case.get("words", []))
    elif stage == "tts":
        text = case.get("tts_text")
        if not isinstance(text, str) or not text.strip():
            raise PlanGenerationError("TTS config requires case tts_text")
        config.update(text=text, target_ms=case.get("target_ms", case["end_ms"] - case["start_ms"]),
                      voice_id="Mai Anh", device="cuda:0")
    elif stage == "bandit":
        config["windows"] = [{"start_ms": case["start_ms"], "end_ms": case["end_ms"]}]
    elif stage == "ocr":
        if not isinstance(case.get("roi"), dict):
            raise PlanGenerationError("OCR config requires a case ROI")
        config.update(roi=case["roi"], start_ms=case["start_ms"], end_ms=case["end_ms"], sample_fps=5)
    elif stage == "inpaint":
        if ocr_artifact_dir is None:
            raise PlanGenerationError("Inpainting requires an OCR job for the same case")
        config["ocr_manifest_path"] = str(ocr_artifact_dir / "ocr.json")
    return config


def generate_plan(*, corpus_path: Path, output_dir: Path, models_root: Path = Path("/data/models"),
                  cache_root: Path = Path("/data/cache"), venv_root: Path,
                  allow_synthetic: bool = False, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe",
                  hourly_rate_vnd: float | None = None) -> dict:
    if hourly_rate_vnd is not None and (isinstance(hourly_rate_vnd, bool)
                                        or not isinstance(hourly_rate_vnd, (int, float))
                                        or not math.isfinite(hourly_rate_vnd) or hourly_rate_vnd < 0):
        raise PlanGenerationError("hourly_rate_vnd must be a non-negative finite number")
    corpus_path, output_root = _resolve(corpus_path), _resolve(output_dir)
    models_root, cache_root, venv_root = _resolve(models_root), _resolve(cache_root), _resolve(venv_root)
    manifest, validation = _validate_corpus(corpus_path, ffprobe_bin=ffprobe_bin,
                                            allow_synthetic=allow_synthetic)
    report_cases = {case["id"]: case for case in validation["cases"]}
    checked_cases = []
    for case in manifest["cases"]:
        checked = report_cases.get(case.get("id"))
        if checked is None:
            raise PlanGenerationError("Validated corpus case set did not match the source manifest")
        source = (corpus_path.parent / case["source"]).resolve()
        checked_cases.append({**case, "source_absolute": str(source),
                              "source_sha256": checked["source_sha256"]})

    output_root.mkdir(parents=True, exist_ok=True)
    jobs, snapshot_jobs, skipped = [], [], []
    config_hashes = []
    used_job_ids: set[str] = set()
    synthetic = validation["kind"] == "synthetic_smoke"
    for case in checked_cases:
        case_id = case["id"]
        per_case_paths: dict[str, Path] = {}
        for stage in DEFAULT_STAGES:
            if stage == "tts" and (not isinstance(case.get("tts_text"), str) or not case["tts_text"].strip()):
                skipped.append({"case_id": case_id, "stage": stage, "reason": "missing_tts_text"})
                continue
            if stage in {"ocr", "inpaint"} and not isinstance(case.get("roi"), dict):
                skipped.append({"case_id": case_id, "stage": stage, "reason": "missing_roi"})
                continue

            job_id = _job_id(case_id, stage, used_job_ids)
            report_rel = f"{job_id}.json"
            config_rel = f"configs/{job_id}.json"
            _contained(output_root, report_rel)
            config_path = _contained(output_root, config_rel)
            artifact_dir = _contained(output_root, f"{job_id}.artifacts")
            config = _stage_config(stage, case, source=Path(case["source_absolute"]), output_dir=artifact_dir,
                                   models_root=models_root, cache_root=cache_root,
                                   ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin,
                                    corpus_kind=validation["kind"],
                                   asr_artifact_dir=per_case_paths.get("asr"),
                                   ocr_artifact_dir=per_case_paths.get("ocr"))
            atomic_json(config_path, config)
            config_hash = sha256_file(config_path)
            config_hashes.append({"path": config_rel, "sha256": config_hash})
            profile = VENV_PROFILES[stage]
            python_path = _venv_python(venv_root, profile)
            job = {"id": job_id, "stage": stage, "input": case["source_absolute"],
                   "config": config_rel, "output": report_rel, "python": str(python_path),
                   "timeout_seconds": 3600}
            jobs.append(job)
            missing_evidence = _missing_quality_evidence(stage, case, synthetic=synthetic)
            snapshot_jobs.append({**job, "artifact_dir": str(artifact_dir),
                                  "config_sha256": config_hash,
                                  "quality_evidence_missing": missing_evidence})
            per_case_paths[stage] = artifact_dir

    if not jobs:
        raise PlanGenerationError("No supported benchmark jobs were generated from corpus cases")

    snapshot = {
        "schema_version": 1,
        "corpus_kind": validation["kind"],
        "source_corpus_path": str(corpus_path),
        "source_corpus_sha256": _sanitized_sha256(manifest),
        "representative_quality_pass": False,
        "allow_synthetic": bool(allow_synthetic),
        "covered_categories": validation["covered_categories"],
        "missing_categories": validation["missing_categories"],
        "total_case_duration_ms": validation["total_case_duration_ms"],
        "cases": [{key: _json_copy(value) for key, value in case.items()
                   if key in {"id", "category", "source_absolute", "source_sha256", "start_ms", "end_ms",
                              "zh_reference", "words", "roi", "tts_text", "target_ms"}}
                  for case in checked_cases],
        "jobs": snapshot_jobs,
        "skipped": skipped,
        "configs": config_hashes,
    }
    snapshot = _json_copy(snapshot)
    generation_rel = "generation-manifest.json"
    generation_path = _contained(output_root, generation_rel)
    atomic_json(generation_path, snapshot)
    generation_hash = sha256_file(generation_path)
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "hourly_rate_vnd": hourly_rate_vnd,
        "corpus_kind": validation["kind"],
        "representative_quality_pass": False,
        "validation": "QUALITY_REVIEW_REQUIRED",
        "generation_manifest": generation_rel,
        "generation_manifest_sha256": generation_hash,
        "jobs": jobs,
    }
    suite_rel = "suite.json"
    suite_path = _contained(output_root, suite_rel)
    atomic_json(suite_path, plan)

    # Suite dry-run checks its schema, generated paths and interpreter executables only; no GPU/model is loaded.
    try:
        preflight = dry_run(suite_path)
    except Exception as error:
        preflight = {"status": "PREFLIGHT_BLOCKED", "error_type": type(error).__name__}
    return {"status": preflight["status"], "plan": str(suite_path),
            "generation_manifest": str(generation_path), "generation_manifest_sha256": generation_hash,
            "jobs": len(jobs), "skipped": skipped, "representative_quality_pass": False,
            "preflight": preflight}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, default=Path("/data/models"))
    parser.add_argument("--cache-root", type=Path, default=Path("/data/cache"))
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--ffmpeg-bin", default=os.getenv("FFMPEG_BIN", "ffmpeg"))
    parser.add_argument("--ffprobe-bin", default=os.getenv("FFPROBE_BIN", "ffprobe"))
    parser.add_argument("--hourly-rate-vnd", type=float)
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="allow smoke-only corpus generation; reports remain quality-review-required")
    args = parser.parse_args(argv)
    try:
        result = generate_plan(corpus_path=args.corpus, output_dir=args.output_dir,
                               models_root=args.models_root, cache_root=args.cache_root,
                               venv_root=args.venv_root, allow_synthetic=args.allow_synthetic,
                               ffmpeg_bin=args.ffmpeg_bin, ffprobe_bin=args.ffprobe_bin,
                               hourly_rate_vnd=args.hourly_rate_vnd)
    except Exception as error:
        print(f"Benchmark plan not generated: {type(error).__name__}", file=sys.stderr)
        return 2
    print(f"{result['status']}: {result['jobs']} jobs; plan: {result['plan']}")
    return 0 if result["status"] == "DRY_RUN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
