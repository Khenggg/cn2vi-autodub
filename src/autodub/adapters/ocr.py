"""RapidOCR PP-OCRv6 on explicitly selected ROI frames; no full-frame OCR path."""
import json
import subprocess
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from autodub.adapters.common import asset, identity, milliseconds, output_folder
from autodub.benchmedia import extract_roi_frame
from autodub.contracts import Roi
from autodub.media import probe_media
from autodub.storage import atomic_json, sha256_file


def frame_times(start_ms: int, end_ms: int, fps: int = 5) -> list[int]:
    if (any(not isinstance(v, int) or isinstance(v, bool) for v in (start_ms, end_ms, fps))
            or not 0 <= start_ms < end_ms or not 1 <= fps <= 30):
        raise ValueError("Invalid OCR sampling interval")
    result, index = [], 0
    while True:
        timestamp = start_ms + int(Decimal(index) * 1000 / fps)
        if timestamp >= end_ms:
            return result
        result.append(timestamp)
        index += 1


def dense_frame_times(source: Path, start_ms: int, end_ms: int, ffprobe_bin: str = "ffprobe") -> list[int]:
    """Use decoded frame PTS, not avg_frame_rate, so a dense mask does not silently skip VFR frames."""
    interval = f"{Decimal(start_ms) / 1000}%{Decimal(end_ms) / 1000}"
    process = subprocess.run([ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-read_intervals", interval,
                              "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json",
                              str(source)], capture_output=True, text=True, timeout=180, check=True)
    frames = json.loads(process.stdout)["frames"]
    times = [int(Decimal(frame["best_effort_timestamp_time"]) * 1000) for frame in frames
             if "best_effort_timestamp_time" in frame]
    times = [timestamp for timestamp in times if start_ms <= timestamp < end_ms]
    if not times or len(times) != len(set(times)) or times != sorted(times):
        raise ValueError("Dense frame PTS unavailable or ambiguous at millisecond precision")
    return times


def build_engine(config: dict):
    from rapidocr import RapidOCR
    path, manifest = asset(config, config.get("ocr_asset_id", "rapidocr-v6"))
    engine = RapidOCR(params={"Det.model_path": str(path / "det.onnx"),
                              "Rec.model_path": str(path / "rec.onnx"),
                              "Cls.model_path": str(path / "cls.onnx"),
                              "Global.model_root_dir": str(path), "Global.log_level": "warning",
                              "EngineConfig.onnxruntime.intra_op_num_threads": 4,
                              "Global.text_score": float(config.get("text_score", 0.5))})
    return engine, manifest


def polygon_mask(size: tuple[int, int], boxes, *, dilate_px: int = 4):
    from PIL import Image, ImageDraw, ImageFilter
    if not 0 <= dilate_px <= 12:
        raise ValueError("OCR mask dilation exceeds safe limit")
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        points = [(max(0, min(width - 1, float(x))), max(0, min(height - 1, float(y)))) for x, y in box]
        if len(points) != 4:
            raise ValueError("OCR polygon must have four vertices")
        draw.polygon(points, fill=255)
    if dilate_px:
        mask = mask.filter(ImageFilter.MaxFilter(2 * dilate_px + 1))
    return mask


def run(source: Path, config: dict) -> dict:
    from PIL import Image
    roi = Roi.model_validate(config["roi"]).model_dump()
    media = probe_media(source, config.get("ffprobe_bin", "ffprobe"))
    duration = media["duration_ms"]
    start, end = config.get("start_ms", 0), config.get("end_ms", duration)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (start, end)) or not 0 <= start < end <= duration:
        raise ValueError("OCR interval exceeds source")
    dense = config.get("dense_frames", False)
    dilation = config.get("mask_dilate_px", 4)
    if not isinstance(dense, bool):
        raise ValueError("dense_frames must be boolean")
    if not isinstance(dilation, int) or isinstance(dilation, bool) or not 0 <= dilation <= 12:
        raise ValueError("OCR mask dilation exceeds safe limit")
    rate = Fraction(media["video"]["avg_frame_rate"])
    if dense:
        if end - start > 10000:
            raise ValueError("Dense OCR benchmark window is limited to 10 seconds")
        times = dense_frame_times(source, start, end, config.get("ffprobe_bin", "ffprobe"))
    else:
        times = frame_times(start, end, config.get("sample_fps", 5))
    folder = output_folder(config)
    tick = milliseconds()
    engine, manifest = build_engine(config)
    load_ms = milliseconds() - tick
    frames, inference_ms = [], 0
    for index, at_ms in enumerate(times):
        image = extract_roi_frame(source, folder / f"roi_{index:06d}.png", roi, at_ms,
                                  config.get("ffmpeg_bin", "ffmpeg"), ffprobe_bin=config.get("ffprobe_bin", "ffprobe"))
        tick = milliseconds()
        result = engine(str(image))
        inference_ms += milliseconds() - tick
        boxes = result.boxes.tolist() if result.boxes is not None else []
        text = list(result.txts) if result.txts is not None else []
        scores = [float(score) for score in result.scores] if result.scores is not None else []
        with Image.open(image) as picture:
            mask = polygon_mask(picture.size, boxes, dilate_px=dilation)
            mask_path = folder / f"mask_{index:06d}.png"
            mask.save(mask_path)
        frames.append({"frame_index": index, "at_ms": at_ms, "image": image.name, "mask": mask_path.name,
                       "image_sha256": sha256_file(image), "mask_sha256": sha256_file(mask_path),
                       "boxes": boxes, "texts": text, "scores": scores})
    target = folder / "ocr.json"
    atomic_json(target, {"schema_version": 1, "source_sha256": sha256_file(source), "roi": roi,
                         "frames": frames, "ocr_scope": "ROI_ONLY", "mask_origin": "OCR_POLYGONS",
                         "temporal_coverage": "DENSE_SOURCE_FRAMES" if dense else "SAMPLED_FRAMES",
                         "frame_count": len(frames), "fps_num": rate.numerator, "fps_den": rate.denominator,
                         "crop_context_px": 0, "mask_dilate_px": dilation,
                         "start_ms": start, "end_ms": end, "temporal_hold_applied": False})
    return {**identity([manifest]), "processed_media_ms": end - start,
            "metrics": {"model_load_ms": load_ms, "inference_ms": inference_ms},
            "quality_metrics": {"sampled_frames": len(frames), "detected_text_instances": sum(len(f["texts"]) for f in frames),
                                "subtitle_recall": None, "false_positive_rate": None},
            "quality_evidence": {"status": "REVIEW_REQUIRED", "roi_only": True, "missing": ["labeled_polygons", "fade_review"]},
            "artifacts": [str(target), *[str(folder / f[key]) for f in frames for key in ("image", "mask")]]}
