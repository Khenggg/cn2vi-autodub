"""Prepare or run a clearly synthetic local OCR/LaMa/TTS smoke suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_RENDER_FRAME = r"""
from PIL import Image, ImageDraw, ImageFont
import sys
canvas = Image.new('RGB', (640, 360), '#264a41')
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype(sys.argv[2], 36)
draw.text((30, 30), 'OUTSIDE ROI', font=font, fill='white')
draw.text((160, 280), '中文字幕测试', font=font, fill='white', stroke_width=1)
canvas.save(sys.argv[1])
"""

_IMAGE_CHECK = r"""
import json, sys
import numpy as np
from PIL import Image
results = []
for item in json.loads(sys.argv[1]):
    with Image.open(item['source']) as picture:
        before = np.asarray(picture.convert('RGB'))
    with Image.open(item['mask']) as picture:
        mask = np.asarray(picture.convert('L')) > 0
    with Image.open(item['output']) as picture:
        after = np.asarray(picture.convert('RGB'))
    if before.shape != after.shape or before.shape[:2] != mask.shape:
        results.append({'shape_matches': False, 'mask_pixels': int(mask.sum()),
                        'changed_masked_pixels': 0, 'changed_unmasked_pixels': 0})
        continue
    changed = np.any(before != after, axis=2)
    results.append({'shape_matches': True, 'mask_pixels': int(mask.sum()),
                    'changed_masked_pixels': int(np.count_nonzero(changed & mask)),
                    'changed_unmasked_pixels': int(np.count_nonzero(changed & ~mask))})
print(json.dumps(results))
"""


def _resolve_executable(value: str, label: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or "/" in value or "\\" in value:
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise ValueError(f"{label} was not found")
        return str(resolved)
    found = shutil.which(value)
    if found is None:
        raise ValueError(f"{label} was not found on PATH")
    return str(Path(found).resolve())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _create_source(output_dir: Path, vision_python: str, font: Path,
                   ffmpeg_bin: str) -> tuple[Path, Path]:
    image_path = output_dir / "synthetic-frame.png"
    video_path = output_dir / "synthetic-smoke.mp4"
    try:
        subprocess.run([vision_python, "-c", _RENDER_FRAME, str(image_path), str(font)],
                       check=True, timeout=90, capture_output=True, text=True)
        subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-loop", "1", "-i", str(image_path),
             "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1", "-r", "24",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-y", str(video_path)],
            check=True, timeout=90, capture_output=True, text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Could not create synthetic media ({type(error).__name__})") from None
    if not image_path.is_file() or not video_path.is_file():
        raise RuntimeError("Synthetic frame/video was not created")
    return image_path, video_path


def prepare(output_dir: Path, models_root: Path, vision_python: str, tts_python: str | None,
            font: Path, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> dict:
    output_dir = Path(output_dir).expanduser().resolve()
    models_root = Path(models_root).expanduser().resolve()
    font = Path(font).expanduser().resolve()
    if not font.is_file():
        raise ValueError("Chinese glyph font file was not found")
    output_dir.mkdir(parents=True, exist_ok=True)
    vision_python = _resolve_executable(vision_python, "Vision Python")
    tts_python = _resolve_executable(tts_python, "TTS Python") if tts_python else None
    ffmpeg_bin = _resolve_executable(ffmpeg_bin, "FFmpeg")
    ffprobe_bin = _resolve_executable(ffprobe_bin, "FFprobe")

    image_path, video_path = _create_source(output_dir, vision_python, font, ffmpeg_bin)
    roi = {"x": 0.1, "y": 0.72, "w": 0.8, "h": 0.25}
    corpus_path = output_dir / "corpus.synthetic.json"
    _write_json(corpus_path, {
        "schema_version": 1,
        "kind": "synthetic_smoke",
        "cases": [{"id": "one-second-synthetic-subtitle", "source": video_path.name,
                   "source_sha256": _sha256_file(video_path), "start_ms": 0, "end_ms": 1000,
                   "category": "subtitle_static", "roi": roi,
                   "tts_text": "Chạy mau! Đừng quay đầu lại.", "target_ms": 2000}],
        "notice": "Synthetic frame, overlay text and sine audio are I/O smoke fixtures only.",
    })
    source_path = str(REPO_ROOT / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from autodub.corpus import validate_manifest

    corpus_report = validate_manifest(corpus_path, ffprobe_bin)
    _write_json(output_dir / "corpus-validation.json", corpus_report)
    if corpus_report["status"] != "SYNTHETIC_SMOKE_ONLY" or corpus_report["representative_quality_pass"]:
        raise RuntimeError("Synthetic corpus validation failed")

    base = {"models_root": str(models_root), "cache_root": str((output_dir / "cache").resolve()),
            "ffmpeg_bin": ffmpeg_bin, "ffprobe_bin": ffprobe_bin,
            "corpus_kind": "synthetic_smoke", "representative_quality_pass": False,
            "synthetic_media": True}
    jobs = []

    ocr_dir = output_dir / "ocr"
    ocr_config = {**base, "output_dir": str(ocr_dir), "roi": roi,
                  "start_ms": 0, "end_ms": 1000, "sample_fps": 1, "mask_dilate_px": 4}
    _write_json(output_dir / "ocr.config.json", ocr_config)
    jobs.append({"id": "synthetic-ocr", "stage": "ocr", "input": video_path.name,
                 "config": "ocr.config.json", "python": vision_python,
                 "output": "reports/ocr.json", "timeout_seconds": 600})

    inpaint_dir = output_dir / "inpaint"
    inpaint_config = {**base, "output_dir": str(inpaint_dir),
                      "ocr_manifest_path": str((ocr_dir / "ocr.json").resolve()),
                      "output_range": "0_255", "context_px": 32}
    _write_json(output_dir / "inpaint.config.json", inpaint_config)
    jobs.append({"id": "synthetic-lama", "stage": "inpaint", "input": video_path.name,
                 "config": "inpaint.config.json", "python": vision_python,
                 "output": "reports/inpaint.json", "timeout_seconds": 600})

    if tts_python:
        tts_dir = output_dir / "tts"
        tts_config = {**base, "output_dir": str(tts_dir), "device": "cpu",
                      "text": "Chạy mau! Đừng quay đầu lại.", "target_ms": 2000, "voice_id": "Mai Anh"}
        _write_json(output_dir / "tts.config.json", tts_config)
        jobs.append({"id": "synthetic-tts", "stage": "tts", "input": video_path.name,
                     "config": "tts.config.json", "python": tts_python,
                     "output": "reports/tts.json", "timeout_seconds": 1800})

    plan_path = output_dir / "suite.synthetic.json"
    _write_json(plan_path, {"schema_version": 1, "corpus_kind": "synthetic_smoke",
                            "representative_quality_pass": False, "jobs": jobs,
                            "notice": "All jobs use synthetic media; outputs are not representative quality evidence.",
                            "tts_omitted": tts_python is None})
    return {"output_dir": str(output_dir), "source_video": str(video_path),
            "source_image": str(image_path), "corpus_manifest": str(corpus_path),
            "corpus_validation": str(output_dir / "corpus-validation.json"),
            "suite_plan": str(plan_path), "jobs": jobs,
            "kind": "synthetic_smoke", "representative_quality_pass": False,
            "tts_omitted": tts_python is None}


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _smoke_assertions(output_dir: Path, vision_python: str, tts_selected: bool) -> dict:
    checks = []

    def record(name: str, passed: bool, details: dict) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    ocr_doc = _read_json(output_dir / "ocr" / "ocr.json")
    ocr_report = _read_json(output_dir / "reports" / "ocr.json")
    ocr_frames = ocr_doc.get("frames", []) if ocr_doc else []
    text_items = [str(text) for frame in ocr_frames for text in frame.get("texts", [])]
    record("ocr_recognizes_fixture_text", any("中文字幕测试" in text for text in text_items),
           {"recognized_texts": text_items, "expected_substring": "中文字幕测试"})
    record("ocr_does_not_report_outside_roi_label",
           not any("OUTSIDE" in text.upper() for text in text_items),
           {"recognized_texts": text_items, "roi_only": bool(ocr_doc and ocr_doc.get("ocr_scope") == "ROI_ONLY")})
    record("ocr_processed_at_least_one_frame",
           bool(ocr_frames) and bool(ocr_report and ocr_report.get("status") == "MEASURED"),
           {"frame_count": len(ocr_frames), "report_status": ocr_report.get("status") if ocr_report else None})

    inpaint_doc = _read_json(output_dir / "inpaint" / "inpaint.json")
    inpaint_report = _read_json(output_dir / "reports" / "inpaint.json")
    inpaint_frames = inpaint_doc.get("frames", []) if inpaint_doc else []
    image_pairs = []
    for index, frame in enumerate(ocr_frames):
        if index >= len(inpaint_frames):
            continue
        image_pairs.append({
            "source": str((output_dir / "ocr" / frame["image"]).resolve()),
            "mask": str((output_dir / "ocr" / frame["mask"]).resolve()),
            "output": str((output_dir / "inpaint" / inpaint_frames[index]["image"]).resolve()),
        })
    image_results = []
    image_error = None
    if image_pairs:
        try:
            process = subprocess.run([vision_python, "-c", _IMAGE_CHECK, json.dumps(image_pairs)],
                                     check=True, timeout=60, capture_output=True, text=True)
            image_results = json.loads(process.stdout)
            if not isinstance(image_results, list):
                image_results = []
                image_error = "invalid image check output"
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            image_error = "image comparison failed"
    mask_pixels = sum(item.get("mask_pixels", 0) for item in image_results)
    record("ocr_mask_has_nonzero_pixels", mask_pixels > 0,
           {"masked_pixels": mask_pixels, "frame_count": len(image_results), "error": image_error})
    changed_masked = sum(item.get("changed_masked_pixels", 0) for item in image_results)
    record("inpaint_changes_masked_pixels", changed_masked > 0,
           {"changed_masked_pixels": changed_masked, "frames": image_results, "error": image_error})
    unchanged_unmasked = (bool(image_results)
                          and all(item.get("shape_matches") and item.get("changed_unmasked_pixels") == 0
                                  for item in image_results)
                          and all(frame.get("unmasked_pixels_unchanged") is True for frame in inpaint_frames))
    record("inpaint_preserves_unmasked_pixels", unchanged_unmasked,
           {"frames": image_results,
            "report_flags": [frame.get("unmasked_pixels_unchanged") for frame in inpaint_frames]})
    residual_count = sum(frame.get("ocr_residual_text_count", 0) for frame in inpaint_frames
                         if isinstance(frame.get("ocr_residual_text_count", 0), int))
    inpaint_processed = (bool(inpaint_frames) and bool(inpaint_report)
                         and inpaint_report.get("status") == "MEASURED"
                         and inpaint_report.get("quality_metrics", {}).get("processed_frames", 0) > 0)
    record("inpaint_processed_masked_frame_with_zero_residuals",
           inpaint_processed and mask_pixels > 0 and residual_count == 0,
           {"processed_frames": inpaint_report.get("quality_metrics", {}).get("processed_frames", 0)
            if inpaint_report else 0, "residual_text_count": residual_count,
            "report_status": inpaint_report.get("status") if inpaint_report else None})

    if tts_selected:
        tts_report = _read_json(output_dir / "reports" / "tts.json")
        wav_path = output_dir / "tts" / "tts.wav"
        audio_metrics = None
        audio_error = None
        try:
            source_path = str(REPO_ROOT / "src")
            if source_path not in sys.path:
                sys.path.insert(0, source_path)
            from autodub.quality import audio_qc
            audio_metrics = audio_qc(wav_path)
        except (OSError, ValueError, wave.Error) as error:
            audio_error = type(error).__name__
        tts_ok = bool(tts_report and tts_report.get("status") == "MEASURED")
        record("tts_pcm16_48khz_nonempty_unclipped",
               tts_ok and bool(wav_path.is_file() and wav_path.stat().st_size > 44)
               and bool(audio_metrics and audio_metrics.get("sample_rate_hz") == 48_000
                        and audio_metrics.get("sample_format") == "pcm_s16le"
                        and audio_metrics.get("duration_ms", 0) > 0
                        and audio_metrics.get("clipped_sample_ratio") == 0.0),
               {"report_status": tts_report.get("status") if tts_report else None,
                "wav_bytes": wav_path.stat().st_size if wav_path.is_file() else 0,
                "audio_metrics": audio_metrics, "error": audio_error})
    else:
        record("tts_not_selected", True, {"omitted": True})

    return {"schema_version": 1, "corpus_kind": "synthetic_smoke",
            "representative_quality_pass": False,
            "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
            "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--vision-python", required=True)
    parser.add_argument("--tts-python", help="Optional isolated CPU TTS interpreter")
    parser.add_argument("--font", type=Path, required=True, help="Font file containing Chinese glyphs")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Create media/configs and run Python-only suite preflight; do not load models")
    args = parser.parse_args()
    try:
        prepared = prepare(args.output_dir, args.models_root, args.vision_python, args.tts_python,
                           args.font, args.ffmpeg_bin, args.ffprobe_bin)
        from autodub.bench_suite import dry_run, run_suite

        plan = Path(prepared["suite_plan"])
        summary_path = Path(prepared["output_dir"]) / "suite-summary.json"
        if args.prepare_only:
            result = dry_run(plan, summary_path)
            prepared["suite_status"] = result["status"]
        else:
            result = run_suite(plan, summary_path)
            try:
                smoke_checks = _smoke_assertions(Path(prepared["output_dir"]),
                                                 prepared["jobs"][0]["python"],
                                                 args.tts_python is not None)
            except Exception as error:
                smoke_checks = {"schema_version": 1, "corpus_kind": "synthetic_smoke",
                                "representative_quality_pass": False, "status": "FAIL",
                                "checks": [{"name": "smoke_assertion_execution", "passed": False,
                                            "details": {"error_type": type(error).__name__}}]}
            checks_path = Path(prepared["output_dir"]) / "smoke-checks.json"
            _write_json(checks_path, smoke_checks)
            result["smoke_check_status"] = smoke_checks["status"]
            result["representative_quality_pass"] = False
            if smoke_checks["status"] != "PASS":
                result["status"] = "FAILED"
            _write_json(summary_path, result)
            prepared["smoke_checks"] = str(checks_path)
            prepared["suite_status"] = result["status"]
        prepared["suite_summary"] = str(summary_path)
        print(json.dumps(prepared, ensure_ascii=False, indent=2))
        if args.prepare_only:
            return 0
        return 0 if result["status"] == "MEASURED" and result.get("smoke_check_status") == "PASS" else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"CPU model smoke not completed ({type(error).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
