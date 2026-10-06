"""Validate benchmark corpus manifests and create clearly synthetic smoke inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from autodub.media import MediaError, probe_media

CATEGORIES = {
    "speech_clear", "music_loud", "nonverbal", "overlap", "subtitle_static",
    "subtitle_motion", "dark",
}
REQUIRED_REPRESENTATIVE_MS = 10 * 60 * 1000


class CorpusError(ValueError):
    """Raised when a corpus manifest cannot be read or validated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _case_error(errors: list[str], case_id: str, message: str) -> None:
    errors.append(f"case {case_id}: {message}")


def validate_manifest(manifest_path: Path, ffprobe_bin: str = "ffprobe") -> dict:
    """Validate manifest structure, local source files, hashes and media-relative bounds.

    Returns a JSON-serializable report. A representative corpus is ready only when
    every required category is present and at least ten minutes of unique case time
    is covered. Synthetic corpora are never considered representative-ready.
    """
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusError(f"Cannot read corpus manifest: {error}") from error
    errors: list[str] = []
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise CorpusError("Manifest must be an object with schema_version=1")
    kind = manifest.get("kind")
    if kind not in {"representative", "synthetic_smoke"}:
        errors.append("kind must be representative or synthetic_smoke")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a non-empty array")
        cases = []

    ids: set[str] = set()
    covered: set[str] = set()
    total_duration_ms = 0
    checked_cases = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"case at index {index} must be an object")
            continue
        case_id = case.get("id")
        label = case_id if isinstance(case_id, str) and case_id else str(index)
        if not isinstance(case_id, str) or not case_id.strip():
            _case_error(errors, label, "id must be a non-empty string")
        elif case_id in ids:
            _case_error(errors, label, "duplicate id")
        ids.add(label)
        category = case.get("category")
        if category not in CATEGORIES:
            _case_error(errors, label, f"category must be one of {', '.join(sorted(CATEGORIES))}")
        else:
            covered.add(category)
        start_ms, end_ms = case.get("start_ms"), case.get("end_ms")
        if not _is_int(start_ms) or not _is_int(end_ms) or start_ms < 0 or end_ms <= start_ms:
            _case_error(errors, label, "start_ms/end_ms must be integers with 0 <= start_ms < end_ms")
            continue
        duration_ms = end_ms - start_ms
        source = case.get("source")
        if not isinstance(source, str) or not source.strip():
            _case_error(errors, label, "source must be a relative path")
            continue
        source_path = (manifest_path.parent / source).resolve()
        if Path(source).is_absolute() or not source_path.is_relative_to(manifest_path.parent):
            _case_error(errors, label, "source must stay inside the manifest directory")
            continue
        if not source_path.is_file():
            _case_error(errors, label, f"source file not found: {source}")
            continue
        actual_hash = sha256_file(source_path)
        expected_hash = case.get("source_sha256")
        if expected_hash is not None:
            if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
                _case_error(errors, label, "source_sha256 must be a 64-character hexadecimal SHA-256")
            elif actual_hash.lower() != expected_hash.lower():
                _case_error(errors, label, "source_sha256 does not match source file")
        try:
            metadata = probe_media(source_path, ffprobe_bin)
        except (MediaError, OSError) as error:
            _case_error(errors, label, f"source could not be probed ({type(error).__name__})")
            continue
        if end_ms > metadata["duration_ms"]:
            _case_error(errors, label, f"end_ms exceeds source duration {metadata['duration_ms']} ms")
            continue
        total_duration_ms += duration_ms
        _validate_case_fields(case, label, start_ms, end_ms, errors)
        checked_cases.append({"id": case_id, "source": source, "source_sha256": actual_hash,
                              "source_duration_ms": metadata["duration_ms"],
                              "start_ms": start_ms, "end_ms": end_ms, "category": category})

    missing_categories = sorted(CATEGORIES - covered)
    if kind == "representative":
        if missing_categories:
            errors.append(f"representative corpus missing categories: {', '.join(missing_categories)}")
        if total_duration_ms < REQUIRED_REPRESENTATIVE_MS:
            errors.append("representative corpus must cover at least 600000 ms in total")
    ready = kind == "representative" and not errors
    return {
        "schema_version": 1,
        "kind": kind,
        "status": "READY_FOR_REPRESENTATIVE_QUALITY_EVALUATION" if ready else "INVALID" if errors else "SYNTHETIC_SMOKE_ONLY",
        "representative_quality_pass": False,
        "total_case_duration_ms": total_duration_ms,
        "required_duration_ms": REQUIRED_REPRESENTATIVE_MS if kind == "representative" else None,
        "covered_categories": sorted(covered),
        "missing_categories": missing_categories,
        "case_count": len(checked_cases),
        "cases": checked_cases,
        "errors": errors,
    }


def _validate_case_fields(case: dict, case_id: str, start_ms: int, end_ms: int,
                          errors: list[str]) -> None:
    reference = case.get("zh_reference")
    if reference is not None and not isinstance(reference, str):
        _case_error(errors, case_id, "zh_reference must be a string")
    words = case.get("words")
    if words is not None:
        if not isinstance(words, list):
            _case_error(errors, case_id, "words must be an array")
        else:
            last_start = -1
            for word in words:
                if (not isinstance(word, dict) or not isinstance(word.get("t"), str)
                        or not _is_int(word.get("s")) or not _is_int(word.get("e"))):
                    _case_error(errors, case_id, "each word requires text t and integer s/e milliseconds")
                    continue
                if not start_ms <= word["s"] < word["e"] <= end_ms or word["s"] < last_start:
                    _case_error(errors, case_id, "word times must be ordered and within the case source bounds")
                    break
                last_start = word["s"]
    roi = case.get("roi")
    if roi is not None:
        try:
            x, y, w, h = (roi[key] for key in ("x", "y", "w", "h"))
            if (not isinstance(roi, dict) or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (x, y, w, h))
                    or not all(map(lambda v: float("-inf") < v < float("inf"), (x, y, w, h)))
                    or x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1 or y + h > 1):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            _case_error(errors, case_id, "roi must be normalized x/y/w/h bounded within [0, 1]")
    if "tts_text" in case and not isinstance(case["tts_text"], str):
        _case_error(errors, case_id, "tts_text must be a string")
    if "target_ms" in case and (not _is_int(case["target_ms"]) or case["target_ms"] <= 0):
        _case_error(errors, case_id, "target_ms must be a positive integer")


def create_synthetic_smoke(output_dir: Path, *, ffmpeg_bin: str = "ffmpeg",
                           ffprobe_bin: str = "ffprobe", duration_s: int = 2) -> Path:
    """Write a generated color-bars+sine video and an explicitly synthetic manifest."""
    if not _is_int(duration_s) or not 1 <= duration_s <= 60:
        raise ValueError("duration_s must be an integer from 1 to 60")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "synthetic-smoke.mp4"
    try:
        subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             f"testsrc2=size=320x180:rate=24:duration={duration_s}", "-f", "lavfi", "-i",
             f"sine=frequency=440:sample_rate=48000:duration={duration_s}", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)],
            check=True, timeout=60, capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CorpusError("Could not create synthetic smoke video; check FFMPEG_BIN") from error
    metadata = probe_media(source, ffprobe_bin)
    manifest = {
        "schema_version": 1,
        "kind": "synthetic_smoke",
        "cases": [{"id": "synthetic-smoke-1", "source": source.name,
                   "source_sha256": sha256_file(source), "start_ms": 0,
                   "end_ms": min(duration_s * 1000, metadata["duration_ms"]),
                   "category": "nonverbal"}],
        "notice": "Generated synthetic media for I/O smoke checks only; this is not quality evidence.",
    }
    manifest_path = output_dir / "corpus.synthetic.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    arguments = parser.parse_args()
    try:
        report = validate_manifest(arguments.manifest, arguments.ffprobe_bin)
    except CorpusError as error:
        report = {"schema_version": 1, "status": "INVALID", "errors": [str(error)],
                  "representative_quality_pass": False}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {arguments.output}")
    return 0 if report["status"] in {"READY_FOR_REPRESENTATIVE_QUALITY_EVALUATION", "SYNTHETIC_SMOKE_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
