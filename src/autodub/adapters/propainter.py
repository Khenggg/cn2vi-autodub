"""Bounded ProPainter benchmark on dense, checksum-bound OCR ROI masks.

This adapter stages only the pinned inference modules and checked checkpoints. It
never passes the source video or a full-frame image folder to upstream inference.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Iterator

from autodub.adapters.common import asset, identity, output_folder
from autodub.storage import atomic_json, safe_path, sha256_file

UPSTREAM_COMMIT = "e870e79321c31b733e2031af5aa2fb1fe3ac7eec"
WEIGHT_FILES = ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth")
MAX_CROP_WIDTH = 720
MAX_CROP_HEIGHT = 480
MAX_CROP_AREA = MAX_CROP_WIDTH * MAX_CROP_HEIGHT
MAX_FRAMES = 50
MAX_SUBVIDEO_LENGTH = 50
MIN_FREE_GPU_BYTES = 1_800 * 1024**2
MAX_PROJECTED_VRAM_BUDGET_MIB = 14_200


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalized_roi(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("An explicit normalized ROI is required")
    try:
        coordinates = {key: value[key] for key in ("x", "y", "w", "h")}
    except KeyError:
        raise ValueError("ROI must contain x, y, w, and h") from None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
           for item in coordinates.values()):
        raise ValueError("ROI coordinates must be finite numbers")
    x, y, width, height = (float(coordinates[key]) for key in ("x", "y", "w", "h"))
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise ValueError("ROI must stay within normalized source bounds")
    return {key: float(coordinates[key]) for key in ("x", "y", "w", "h")}


def _config_values(config: dict) -> tuple[dict[str, float], int, int, int, int, int]:
    roi = _normalized_roi(config.get("roi"))
    start_ms = config.get("start_ms")
    end_ms = config.get("end_ms")
    if (isinstance(start_ms, bool) or not isinstance(start_ms, int) or
            isinstance(end_ms, bool) or not isinstance(end_ms, int) or start_ms < 0 or end_ms <= start_ms):
        raise ValueError("Explicit start_ms and end_ms are required")
    context = config.get("crop_context_px", 64)
    if isinstance(context, bool) or not isinstance(context, int) or not 64 <= context <= 128:
        raise ValueError("crop_context_px must be between 64 and 128")
    frame_cap = config.get("max_frames", MAX_FRAMES)
    if isinstance(frame_cap, bool) or not isinstance(frame_cap, int) or not 1 <= frame_cap <= MAX_FRAMES:
        raise ValueError("max_frames must be between 1 and 50")
    subvideo = config.get("subvideo_length", 40)
    if (isinstance(subvideo, bool) or not isinstance(subvideo, int) or
            not 1 <= subvideo <= MAX_SUBVIDEO_LENGTH):
        raise ValueError("subvideo_length must be between 1 and 50")
    if config.get("fp16") is not True:
        raise ValueError("ProPainter benchmark requires fp16=true")
    return roi, start_ms, end_ms, context, frame_cap, subvideo


def crop_geometry(roi: dict, source_width: int, source_height: int, context_px: int) -> dict[str, int]:
    """Calculate expanded ROI bounds and enforce the initial workload cap."""
    normalized = _normalized_roi(roi)
    if source_width <= 0 or source_height <= 0 or not 64 <= context_px <= 128:
        raise ValueError("Invalid source geometry or crop context")
    roi_left = int(normalized["x"] * source_width)
    roi_top = int(normalized["y"] * source_height)
    roi_width = int(normalized["w"] * source_width)
    roi_height = int(normalized["h"] * source_height)
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI is smaller than one source pixel")
    roi_right, roi_bottom = roi_left + roi_width, roi_top + roi_height
    left, top = max(0, roi_left - context_px), max(0, roi_top - context_px)
    right, bottom = min(source_width, roi_right + context_px), min(source_height, roi_bottom + context_px)
    width, height = right - left, bottom - top
    aligned_width, aligned_height = (width + 7) // 8 * 8, (height + 7) // 8 * 8
    extra_x, extra_y = aligned_width - width, aligned_height - height
    shift_left, shift_top = min(left, extra_x), min(top, extra_y)
    left -= shift_left
    top -= shift_top
    right = min(source_width, right + extra_x - shift_left)
    bottom = min(source_height, bottom + extra_y - shift_top)
    width, height = right - left, bottom - top
    if width != aligned_width or height != aligned_height:
        raise ValueError("Source bounds prevent an 8-pixel-aligned ROI crop")
    if width == source_width and height == source_height:
        raise ValueError("Full-frame inference is prohibited; configure a smaller subtitle ROI")
    if width > MAX_CROP_WIDTH or height > MAX_CROP_HEIGHT or width * height > MAX_CROP_AREA:
        raise ValueError("Expanded ROI exceeds 720x480 crop limit; reduce the ROI or context")
    return {"left": left, "top": top, "width": width, "height": height,
            "roi_left": roi_left, "roi_top": roi_top, "roi_width": roi_width,
            "roi_height": roi_height, "mask_left": roi_left - left, "mask_top": roi_top - top}


def _probe_source(source: Path, ffprobe_bin: str = "ffprobe") -> dict:
    command = [ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-count_frames",
               "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,start_time",
               "-of", "json", str(source)]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                encoding="utf-8", timeout=45, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("FFprobe failed while verifying the source timeline") from None
    if result.returncode != 0:
        raise ValueError("FFprobe could not verify the source timeline")
    try:
        stream = json.loads(result.stdout)["streams"][0]
        fps = Fraction(stream["avg_frame_rate"])
        width, height, frame_count = int(stream["width"]), int(stream["height"]), int(stream["nb_read_frames"])
        start_seconds = float(stream.get("start_time") or 0)
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError):
        raise ValueError("Source must provide exact dimensions, frame count, and frame rate") from None
    if width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0 or not math.isfinite(start_seconds):
        raise ValueError("Source timeline metadata is invalid")
    return {"width": width, "height": height, "frame_count": frame_count,
            "fps_num": fps.numerator, "fps_den": fps.denominator,
            "start_ms": round(start_seconds * 1000)}


def _same_roi(expected: dict[str, float], observed: object) -> bool:
    if not isinstance(observed, dict):
        return False
    try:
        actual = _normalized_roi(observed)
    except ValueError:
        return False
    return all(math.isclose(expected[key], actual[key], rel_tol=0, abs_tol=1e-9)
               for key in ("x", "y", "w", "h"))


def _read_dense_manifest(source: Path, config: dict, roi: dict[str, float],
                         start_ms: int, end_ms: int, source_info: dict, frame_cap: int) -> tuple[Path, dict, list[dict]]:
    if not config.get("ocr_manifest_path"):
        raise ValueError("An OCR dense-frame manifest is required")
    manifest_path = Path(config["ocr_manifest_path"]).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("OCR manifest could not be read") from None
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or
            manifest.get("temporal_coverage") != "DENSE_SOURCE_FRAMES" or
            manifest.get("ocr_scope") != "ROI_ONLY"):
        raise ValueError("ProPainter requires a dense ROI-only OCR manifest")
    if manifest.get("source_sha256") != sha256_file(source):
        raise ValueError("OCR manifest source checksum does not match input")
    if not _same_roi(roi, manifest.get("roi")):
        raise ValueError("OCR manifest ROI does not match configured ROI")
    manifest_start, manifest_end = manifest.get("start_ms"), manifest.get("end_ms")
    if (isinstance(manifest_start, bool) or not isinstance(manifest_start, int) or
            isinstance(manifest_end, bool) or not isinstance(manifest_end, int) or
            manifest_start < 0 or manifest_end <= manifest_start or
            start_ms < manifest_start or end_ms > manifest_end):
        raise ValueError("Configured interval must fit inside the dense OCR timeline")
    if manifest.get("crop_context_px") != 0:
        raise ValueError("Dense OCR frames must be ROI-only crops with zero context")
    if manifest.get("mask_origin") != "OCR_POLYGONS":
        raise ValueError("Dense OCR mask provenance must be OCR_POLYGONS")
    frames = manifest.get("frames")
    frame_count = manifest.get("frame_count")
    if not isinstance(frames, list) or not frames or frame_count != len(frames):
        raise ValueError("Dense OCR frame_count must equal its non-empty frames list")
    fps_num, fps_den = manifest.get("fps_num"), manifest.get("fps_den")
    if (isinstance(fps_num, bool) or not isinstance(fps_num, int) or fps_num <= 0 or
            isinstance(fps_den, bool) or not isinstance(fps_den, int) or fps_den <= 0):
        raise ValueError("Dense OCR manifest requires a positive rational fps")
    source_fps = Fraction(source_info["fps_num"], source_info["fps_den"])
    if Fraction(fps_num, fps_den) != source_fps:
        raise ValueError("OCR manifest frame rate does not match source")
    period_ms = 1000 * fps_den / fps_num
    tolerance_ms = max(1.1, 1000 / fps_num)
    timestamps = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("frame_index") != index:
            raise ValueError("Dense OCR frame indices must be contiguous from zero")
        timestamp = frame.get("at_ms")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or not manifest_start <= timestamp < manifest_end:
            raise ValueError("Dense OCR timestamps must lie inside its declared interval")
        timestamps.append(timestamp)
    if len(timestamps) > 1:
        if any(abs((right - left) - period_ms) > tolerance_ms
               for left, right in zip(timestamps[:-1], timestamps[1:], strict=True)):
            raise ValueError("OCR frame timeline is not dense at the source frame rate")
        if (timestamps[0] - manifest_start > period_ms + tolerance_ms or
                manifest_end - timestamps[-1] > period_ms + tolerance_ms):
            raise ValueError("OCR dense timeline does not cover its declared interval")
    elif end_ms - start_ms > period_ms + tolerance_ms:
        raise ValueError("A multi-frame interval cannot use a one-frame OCR manifest")
    frame_offset = max(0, round((timestamps[0] - source_info["start_ms"]) * fps_num / (1000 * fps_den)))
    if frame_offset + len(frames) > source_info["frame_count"]:
        raise ValueError("Dense OCR frame_count exceeds the source timeline")
    for index, timestamp in enumerate(timestamps):
        source_index = frame_offset + index
        expected_ms = source_info["start_ms"] + round(source_index * period_ms)
        if abs(timestamp - expected_ms) > tolerance_ms:
            raise ValueError("OCR timestamps do not match exact source frames")
    selected = [frame for frame in frames if start_ms <= frame["at_ms"] < end_ms]
    if not selected or len(selected) > frame_cap:
        raise ValueError("Configured dense OCR interval must contain between 1 and max_frames frames")
    if (selected[0]["at_ms"] - start_ms > period_ms + tolerance_ms or
            end_ms - selected[-1]["at_ms"] > period_ms + tolerance_ms):
        raise ValueError("Dense OCR frames do not cover the configured ProPainter interval")
    if len(selected) > 1 and any(abs((right["at_ms"] - left["at_ms"]) - period_ms) > tolerance_ms
                                 for left, right in zip(selected[:-1], selected[1:], strict=True)):
        raise ValueError("Configured ProPainter interval is not dense at the source frame rate")
    return manifest_path, manifest, selected


def _validate_ocr_files(manifest_path: Path, manifest: dict, frames: list[dict], geometry: dict) -> list[tuple[Path, Path]]:
    from PIL import Image

    inputs = []
    for frame in frames:
        for key in ("image", "mask"):
            if not isinstance(frame.get(key), str) or not isinstance(frame.get(f"{key}_sha256"), str):
                raise ValueError("Every OCR frame needs image/mask paths and checksums")
        image_path = safe_path(manifest_path.parent, frame["image"])
        mask_path = safe_path(manifest_path.parent, frame["mask"])
        if not image_path.is_file() or not mask_path.is_file():
            raise ValueError("OCR ROI image or mask is missing")
        if sha256_file(image_path) != frame["image_sha256"] or sha256_file(mask_path) != frame["mask_sha256"]:
            raise ValueError("OCR ROI image/mask checksum mismatch")
        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                image_size, mask_size = image.size, mask.size
                ocr_mask = mask.convert("L")
                roi_pixels = image_size[0] == geometry["roi_width"] and image_size[1] == geometry["roi_height"]
                if not roi_pixels or mask_size != image_size:
                    raise ValueError("OCR image/mask must be aligned ROI-only crops")
                if "boxes" in frame:
                    from PIL import ImageChops

                    from autodub.adapters.ocr import polygon_mask
                    expected = polygon_mask(image_size, frame["boxes"],
                                             dilate_px=int(manifest.get("mask_dilate_px", 4)))
                    if ImageChops.difference(expected, ocr_mask).getbbox() is not None:
                        raise ValueError("OCR mask pixels do not match the supplied OCR polygons")
                elif manifest.get("mask_origin") != "OCR_POLYGONS":
                    raise ValueError("OCR mask provenance cannot be verified")
                if ocr_mask.getextrema() == (255, 255):
                    raise ValueError("A full-ROI mask is not an OCR glyph mask")
        except (OSError, ValueError) as error:
            if isinstance(error, ValueError) and str(error).startswith(("OCR", "A full")):
                raise
            raise ValueError("OCR ROI image/mask could not be validated") from None
        inputs.append((image_path, mask_path))
    return inputs


def _extract_crop_frame(source: Path, output: Path, at_ms: int, geometry: dict,
                        ffmpeg_bin: str = "ffmpeg") -> None:
    from decimal import Decimal

    seconds = format(Decimal(at_ms) / Decimal(1000), ".3f")
    vf = (f"crop={geometry['width']}:{geometry['height']}:{geometry['left']}:{geometry['top']}:exact=1")
    command = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-ss", seconds,
               "-i", str(source), "-vf", vf, "-frames:v", "1", "-y", str(output)]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("FFmpeg timed out or could not extract an ROI crop") from None
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError("FFmpeg failed to extract an ROI crop")


def _materialize_inputs(source: Path, frames: list[dict], ocr_inputs: list[tuple[Path, Path]],
                        geometry: dict, input_root: Path, ffmpeg_bin: str) -> tuple[Path, Path, list[dict]]:
    from PIL import Image

    frame_dir, mask_dir = input_root / "frames", input_root / "masks"
    frame_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    artifacts = []
    any_mask_pixels = False
    for index, (frame, (ocr_image_path, ocr_mask_path)) in enumerate(zip(frames, ocr_inputs, strict=True)):
        frame_name = f"{index:06d}.png"
        input_frame, input_mask = frame_dir / frame_name, mask_dir / frame_name
        _extract_crop_frame(source, input_frame, frame["at_ms"], geometry, ffmpeg_bin)
        try:
            with Image.open(input_frame) as crop, Image.open(ocr_image_path) as ocr_image, Image.open(ocr_mask_path) as mask:
                crop_rgb = crop.convert("RGB")
                if crop_rgb.size != (geometry["width"], geometry["height"]):
                    raise ValueError("FFmpeg crop dimensions do not match the bounded ROI")
                if ocr_image.size != (geometry["roi_width"], geometry["roi_height"]):
                    raise ValueError("OCR image dimensions do not match configured ROI")
                mask_canvas = Image.new("L", crop_rgb.size, 0)
                mask_canvas.paste(mask.convert("L"), (geometry["mask_left"], geometry["mask_top"]))
                any_mask_pixels = any_mask_pixels or mask_canvas.getbbox() is not None
                crop_rgb.save(input_frame, format="PNG")
                mask_canvas.save(input_mask, format="PNG")
        except OSError:
            raise ValueError("Could not prepare the bounded ProPainter frame/mask pair") from None
        artifacts.append({"frame_index": index, "at_ms": frame["at_ms"], "input_frame": frame_name,
                          "input_mask": frame_name, "ocr_image_sha256": frame["image_sha256"],
                          "ocr_mask_sha256": frame["mask_sha256"]})
    if not any_mask_pixels:
        raise ValueError("Dense OCR interval contains no recognized glyph masks")
    return frame_dir, mask_dir, artifacts


@contextmanager
def _staged_source(upstream_repo: Path, weights_path: Path, models_root: Path) -> Iterator[Path]:
    """Copy only inference code to a temporary model-root workspace; link verified weights locally."""
    models_root = models_root.resolve()
    try:
        with tempfile.TemporaryDirectory(prefix=".propainter-run-", dir=models_root) as temporary:
            runtime = Path(temporary) / "source"
            runtime.mkdir()
            for filename in ("inference_propainter.py",):
                shutil.copy2(upstream_repo / filename, runtime / filename)
            for directory in ("core", "model", "utils"):
                shutil.copytree(upstream_repo / directory, runtime / directory,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.mp4", "*.mov", "*.avi"))
            runtime_weights = runtime / "weights"
            if runtime_weights.exists() or runtime_weights.is_symlink():
                if runtime_weights.is_dir() and not runtime_weights.is_symlink():
                    shutil.rmtree(runtime_weights)
                else:
                    runtime_weights.unlink()
            resolved_weights = weights_path.resolve(strict=True)
            if not resolved_weights.is_relative_to(models_root):
                raise ValueError("Verified ProPainter weights must stay under models_root")
            runtime_weights.symlink_to(resolved_weights, target_is_directory=True)
            if runtime_weights.resolve(strict=True) != resolved_weights:
                raise ValueError("Staged weight link does not resolve to verified model assets")
            wrapper = runtime / "autodub_inference.py"
            wrapper.write_text(_inference_wrapper_source(), encoding="utf-8")
            yield runtime
    except OSError:
        raise RuntimeError("Could not stage the pinned ProPainter inference code") from None


def _inference_wrapper_source() -> str:
    return (
        "import imageio, json, runpy, sys, torch\n"
        "from pathlib import Path\n"
        "device = 'cuda:0'\n"
        "torch.cuda.synchronize(device)\n"
        "torch.cuda.reset_peak_memory_stats(device)\n"
        "script = Path(__file__).with_name('inference_propainter.py')\n"
        "imageio.mimwrite = lambda *args, **kwargs: None\n"
        "sys.argv[0] = str(script)\n"
        "try:\n"
        "    runpy.run_path(str(script), run_name='__main__')\n"
        "finally:\n"
        "    torch.cuda.synchronize(device)\n"
        "    measurement = {'schema_version': 1, 'device': device,\n"
        "        'gpu_allocator_peak_bytes': int(torch.cuda.max_memory_allocated(device)),\n"
        "        'gpu_reserved_peak_bytes': int(torch.cuda.max_memory_reserved(device))}\n"
        "    Path(__file__).with_name('cuda_peak.json').write_text(\n"
        "        json.dumps(measurement), encoding='utf-8')\n"
    )


def _read_peak_metrics(runtime: Path) -> dict[str, int] | None:
    try:
        payload = json.loads((runtime / "cuda_peak.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("device") != "cuda:0":
        return None
    values = (payload.get("gpu_allocator_peak_bytes"), payload.get("gpu_reserved_peak_bytes"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        return None
    return {"gpu_allocator_peak_bytes": values[0], "gpu_reserved_peak_bytes": values[1]}


def _command(python: str, runtime: Path, frame_dir: Path, mask_dir: Path, output_dir: Path,
             width: int, height: int, subvideo_length: int) -> list[str]:
    if width % 8 or height % 8:
        raise ValueError("ProPainter crop dimensions must be multiples of 8")
    return [python, str(runtime / "autodub_inference.py"), "--video", str(frame_dir),
            "--mask", str(mask_dir), "--output", str(output_dir), "--fp16", "--save_frames",
            "--width", str(width), "--height", str(height),
            "--subvideo_length", str(subvideo_length), "--neighbor_length", "10", "--ref_stride", "10"]


def _invoke_upstream(command: list[str], cwd: Path, timeout_seconds: int) -> int:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=timeout_seconds, check=False,
                                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    except subprocess.TimeoutExpired:
        raise RuntimeError("ProPainter inference timed out") from None
    except OSError:
        raise RuntimeError("ProPainter inference process could not start") from None
    if result.returncode != 0:
        raise RuntimeError(f"ProPainter inference failed (exit {result.returncode})")
    return round((time.perf_counter() - started) * 1000)


def _check_gpu_reserve() -> None:
    try:
        import torch
    except ImportError:
        raise RuntimeError("ProPainter profile requires its pinned PyTorch CUDA environment") from None
    if not torch.cuda.is_available():
        raise RuntimeError("ProPainter requires an available CUDA GPU")
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info("cuda:0")
    except Exception as error:
        raise RuntimeError(f"Could not inspect CUDA free memory ({type(error).__name__})") from None
    if free_bytes < MIN_FREE_GPU_BYTES:
        raise RuntimeError("ProPainter requires at least 1800 MiB free GPU memory")


def _copy_and_validate_outputs(result_frames: Path, output_folder_path: Path, expected_count: int,
                               expected_size: tuple[int, int]) -> list[Path]:
    from PIL import Image

    files = sorted(result_frames.glob("*.png")) if result_frames.is_dir() else []
    if len(files) != expected_count or expected_count <= 0:
        raise RuntimeError("ProPainter output frame count does not match dense input")
    if [path.name for path in files] != [f"{index:04d}.png" for index in range(expected_count)]:
        raise RuntimeError("ProPainter output frame names are incomplete or out of sequence")
    outputs = []
    for index, source_frame in enumerate(files):
        try:
            with Image.open(source_frame) as image:
                image.verify()
            with Image.open(source_frame) as image:
                if image.size != expected_size:
                    raise RuntimeError("ProPainter output frame dimensions do not match ROI crop")
        except OSError:
            raise RuntimeError("ProPainter produced an invalid output frame") from None
        destination = output_folder_path / f"propainter_{index:06d}.png"
        shutil.copy2(source_frame, destination)
        outputs.append(destination)
    return outputs


def run(source: Path, config: dict) -> dict:
    """Run pinned ProPainter on a dense, source-bound sequence of cropped ROI frames."""
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError("Benchmark source video not found")
    roi, start_ms, end_ms, context_px, frame_cap, configured_subvideo = _config_values(config)
    source_info = _probe_source(source, config.get("ffprobe_bin", "ffprobe"))
    if end_ms > max(source_info["start_ms"], source_info["start_ms"] + round(
            source_info["frame_count"] * 1000 * source_info["fps_den"] / source_info["fps_num"])):
        raise ValueError("Configured ProPainter interval exceeds source frame timeline")
    geometry = crop_geometry(roi, source_info["width"], source_info["height"], context_px)
    manifest_path, ocr, frames = _read_dense_manifest(source, config, roi, start_ms, end_ms,
                                                       source_info, frame_cap)
    ocr_inputs = _validate_ocr_files(manifest_path, ocr, frames, geometry)
    folder = output_folder(config)
    _check_gpu_reserve()

    models_root = Path(config.get("models_root", "/data/models")).resolve()
    code_folder, code_manifest = asset(config, "propainter-code")
    weights_folder, weights_manifest = asset(config, "propainter-weights")
    if code_manifest.get("model_revision") != UPSTREAM_COMMIT:
        raise ValueError("ProPainter code asset is not the pinned upstream commit")
    declared = {item.get("path") for item in weights_manifest.get("files", [])}
    if not set(WEIGHT_FILES) <= declared:
        raise ValueError("ProPainter weight manifest is missing a required checkpoint")
    for filename in WEIGHT_FILES:
        path = weights_folder / filename
        if not path.is_file():
            raise ValueError("A checksum-verified ProPainter checkpoint is missing")
    upstream_repo = code_folder / "source"
    if not (upstream_repo / "inference_propainter.py").is_file():
        raise ValueError("Pinned ProPainter inference entry point is missing")
    if not models_root.is_dir():
        raise ValueError("models_root must exist before ProPainter staging")

    timeout = config.get("timeout_seconds", 3600)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 86_400:
        raise ValueError("timeout_seconds must be between 1 and 86400")
    output_records = []
    peak_metrics = None
    with tempfile.TemporaryDirectory(prefix=".propainter-input-", dir=folder) as temp_input:
        input_root = Path(temp_input)
        frame_dir, mask_dir, input_records = _materialize_inputs(
            source, frames, ocr_inputs, geometry, input_root, config.get("ffmpeg_bin", "ffmpeg"))
        with tempfile.TemporaryDirectory(prefix=".propainter-output-", dir=folder) as temp_output:
            results_root = Path(temp_output)
            with _staged_source(upstream_repo, weights_folder, models_root) as runtime:
                video_name = frame_dir.name
                command = _command(sys.executable, runtime, frame_dir, mask_dir, results_root,
                                   geometry["width"], geometry["height"],
                                   min(configured_subvideo, len(frames)))
                _invoke_upstream(command, runtime, timeout)
                peak_metrics = _read_peak_metrics(runtime)
            if list(results_root.rglob("*.mp4")):
                raise RuntimeError("ProPainter video output is disabled; only cropped PNG frames are accepted")
            result_frames = results_root / video_name / "frames"
            outputs = _copy_and_validate_outputs(result_frames, folder, len(frames),
                                                 (geometry["width"], geometry["height"]))
            for index, path in enumerate(outputs):
                output_records.append({"frame_index": index, "at_ms": frames[index]["at_ms"],
                                       "image": path.name, "sha256": sha256_file(path)})

    report_path = folder / "propainter.json"
    measured_peak = peak_metrics is not None
    missing_evidence = ["temporal_warp_quality_review"]
    if not measured_peak:
        missing_evidence.append("measured_per_process_vram_peak")
    atomic_json(report_path, {
        "schema_version": 1,
        "source_sha256": sha256_file(source),
        "roi": roi,
        "interval_ms": {"start": start_ms, "end": end_ms},
        "temporal_coverage": "DENSE_SOURCE_FRAMES",
        "crop_context_px": context_px,
        "crop_geometry": geometry,
        "input_frames": input_records,
        "output_frames": output_records,
        "render_scope": "CROPPED_ROI_FRAME_SEQUENCE",
        "final_video": False,
        "full_frame_composition": False,
        "cuda_peak_measurement": {"scope": "child_process_allocator_peak",
                                  "measured": measured_peak,
                                  **(peak_metrics or {"gpu_allocator_peak_bytes": None,
                                                      "gpu_reserved_peak_bytes": None})},
    })
    identities = identity([code_manifest, weights_manifest])
    return {
        **identities,
        "processed_media_ms": end_ms - start_ms,
        "quality_metrics": {"processed_frames": len(frames), "frame_count_exact": True,
                            "crop_width": geometry["width"], "crop_height": geometry["height"],
                            "crop_area": geometry["width"] * geometry["height"],
                            "mask_provenance": "OCR_POLYGONS"},
        "quality_evidence": {"status": "REVIEW_REQUIRED", "render_scope": "CROPPED_ROI_FRAME_SEQUENCE",
                             "provisional_vram_admission_budget_mib": MAX_PROJECTED_VRAM_BUDGET_MIB,
                             "measured_vram_peak": measured_peak, "missing": missing_evidence},
        "metrics": peak_metrics or {"gpu_allocator_peak_bytes": None, "gpu_reserved_peak_bytes": None},
        "artifacts": [str(report_path), *[str(path) for path in outputs]],
    }
