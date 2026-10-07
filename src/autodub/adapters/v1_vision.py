"""Automatic subtitle layout detection and bounded temporal ProPainter removal."""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from autodub.adapters.common import asset, identity, output_folder
from autodub.adapters.ocr import build_engine
from autodub.storage import atomic_json


def infer_layout(frames: list[dict], width: int, height: int) -> dict | None:
    """A persistent centered lower-screen text line is a candidate, never a fact."""
    candidates = []
    for frame in frames:
        for line in frame["lines"]:
            box = line["box"]
            left, right = min(p[0] for p in box), max(p[0] for p in box)
            top, bottom = min(p[1] for p in box), max(p[1] for p in box)
            if top >= height * 0.55 and abs((left + right) / 2 - width / 2) <= width * 0.22:
                candidates.append((top, bottom))
    if len(candidates) < 3:
        return None
    centers = sorted((a + b) / 2 for a, b in candidates)
    center = centers[len(centers) // 2]
    consistent = [(a, b) for a, b in candidates if abs((a + b) / 2 - center) < height * 0.06]
    if len(consistent) < 3:
        return None
    top = max(0, int(min(a for a, _ in consistent) - 12))
    bottom = min(height, int(max(b for _, b in consistent) + 12))
    return {"x": 0, "y": top / height, "w": 1, "h": (bottom - top) / height,
            "source": "TEMPORAL_OCR_CANDIDATE", "support_count": len(consistent),
            "verified_by_user": False}


def scan(source: Path, config: dict) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Video cannot be decoded for subtitle detection")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tick = time.perf_counter()
    engine, manifest = build_engine({**config, "ocr_asset_id": "rapidocr-v6-medium"})
    load_ms = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    frames = []
    try:
        for at in range(0, config["duration_ms"], config.get("ocr_sample_ms", 1000)):
            capture.set(cv2.CAP_PROP_POS_MSEC, at)
            ok, image = capture.read()
            if not ok:
                continue
            # OCR context is permitted across the frame; inpainting is confined to
            # the automatically detected subtitle band and glyph mask.
            output = engine(image)
            lines = []
            if output.boxes is not None:
                lines = [{"text": str(text), "score": float(score), "box": box.tolist()}
                         for box, text, score in zip(output.boxes, output.txts, output.scores, strict=True)]
            frames.append({"at_ms": at, "lines": lines})
    finally:
        capture.release()
    layout = config.get("subtitle_layout") or infer_layout(frames, width, height)
    target = output_folder(config) / "ocr-context.json"
    atomic_json(target, {"schema_version": 1, "frames": frames, "layout": layout,
                         "width": width, "height": height, "layout_detection_is_verified": False})
    return {**identity([manifest]), "layout": layout,
            "stage_status": "SUCCESS" if layout else "DEGRADED",
            "failure_code": "SUBTITLE_LAYOUT_UNCERTAIN" if not layout else None,
            "metrics": {"model_load_ms": load_ms, "inference_ms": (time.perf_counter() - tick) * 1000},
            "artifacts": [str(target)]}


def glyph_mask(image, boxes):
    """Threshold stroke pixels inside OCR polygons; never erase a filled subtitle band."""
    import cv2
    import numpy as np

    mask = np.zeros(image.shape[:2], dtype="uint8")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for box in boxes:
        polygon = np.round(box).astype("int32")
        x, y, w, h = cv2.boundingRect(polygon)
        x, y = max(0, x), max(0, y)
        w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)
        if w <= 0 or h <= 0:
            continue
        _, strokes = cv2.threshold(gray[y:y + h, x:x + w], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(strokes) > w * h * 0.6:
            strokes = cv2.bitwise_not(strokes)
        region = np.zeros_like(mask)
        cv2.fillPoly(region, [polygon], 255)
        mask[y:y + h, x:x + w] |= strokes & region[y:y + h, x:x + w]
    return cv2.dilate(mask, np.ones((3, 3), dtype="uint8"), iterations=2)


def inpaint(source: Path, config: dict) -> dict:
    import cv2
    import numpy as np

    from autodub.adapters.ocr import dense_frame_times
    from autodub.adapters.propainter import _command, _invoke_upstream, _staged_source

    layout = config.get("subtitle_layout")
    if not layout:
        raise ValueError("No usable automatic subtitle layout; preserve the original video")
    folder = output_folder(config)
    code, code_manifest = asset(config, "propainter-code")
    weights, weight_manifest = asset(config, "propainter-weights")
    engine, ocr_manifest = build_engine({**config, "ocr_asset_id": "rapidocr-v6-medium"})
    capture = cv2.VideoCapture(str(source))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        raise ValueError("Source decode timeline is unavailable")
    timestamps = dense_frame_times(source, 0, config["duration_ms"], config.get("ffprobe_bin", "ffprobe"))
    if any(abs(at - index * 1000 / fps) > 2 for index, at in enumerate(timestamps)):
        capture.release()
        raise ValueError("Variable or offset frame timeline cannot use the frozen CFR inpainting path")
    top = max(0, int(layout["y"] * height) - 64)
    bottom = min(height, int((layout["y"] + layout["h"]) * height) + 64)
    if bottom - top >= height:
        raise ValueError("Full-frame ProPainter is prohibited")
    scaled_w = min(720, (width // 8) * 8)
    scaled_h = max(8, int((bottom - top) * scaled_w / width) // 8 * 8)
    if scaled_h > 480:
        raise ValueError("Detected subtitle crop exceeds the frozen inference size limit")
    output = folder / "clean-video.avi"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Lossless intermediate encoder unavailable")
    count, issues, history = 0, [], []
    tick = time.perf_counter()
    try:
        with _staged_source(code / "source", weights, Path(config["models_root"])) as runtime:
            while True:
                original = list(history)
                for _ in range(40):
                    ok, image = capture.read()
                    if not ok:
                        break
                    original.append(image)
                prefix = len(history)
                if len(original) == prefix:
                    break
                work = folder / f"chunk-{count:08d}"
                frame_dir, mask_dir, results = work / "frames", work / "masks", work / "results"
                frame_dir.mkdir(parents=True)
                mask_dir.mkdir()
                masks = []
                for index, image in enumerate(original):
                    crop = cv2.resize(image[top:bottom], (scaled_w, scaled_h))
                    ocr = engine(crop)
                    boxes = []
                    for box in ocr.boxes if ocr.boxes is not None else []:
                        center_y = float(np.mean(box[:, 1])) * (bottom - top) / scaled_h + top
                        center_x = float(np.mean(box[:, 0])) * width / scaled_w
                        if (layout["y"] * height <= center_y <= (layout["y"] + layout["h"]) * height
                                and abs(center_x - width / 2) <= width * 0.3):
                            boxes.append(box)
                    mask = glyph_mask(crop, boxes)
                    masks.append(mask)
                    cv2.imwrite(str(frame_dir / f"{index:04d}.png"), crop)
                    cv2.imwrite(str(mask_dir / f"{index:04d}.png"), mask)
                try:
                    if any(np.any(mask) for mask in masks):
                        _invoke_upstream(_command(sys.executable, runtime, frame_dir, mask_dir, results,
                                                  scaled_w, scaled_h, min(40, len(original))), runtime,
                                         config.get("timeout_seconds", 3600))
                    completed_frames = []
                    for index in range(prefix, len(original)):
                        image = original[index].copy()
                        if np.any(masks[index]):
                            clean = cv2.imread(str(results / "frames" / "frames" / f"{index:04d}.png"))
                            if clean is None:
                                raise RuntimeError("Missing ProPainter output frame")
                            clean = cv2.resize(clean, (width, bottom - top))
                            mask = cv2.resize(masks[index], (width, bottom - top), interpolation=cv2.INTER_NEAREST)
                            image[top:bottom][mask > 0] = clean[mask > 0]
                        completed_frames.append(image)
                    for image in completed_frames:
                        writer.write(image)
                except Exception as error:
                    issues.append({"start_frame": count, "code": "INPAINT_CHUNK_UNAVAILABLE",
                                   "error_type": type(error).__name__, "action": "KEEP_ORIGINAL_CHUNK"})
                    for image in original[prefix:]:
                        writer.write(image)
                count += len(original) - prefix
                history = original[-8:]
                shutil.rmtree(work)
    finally:
        capture.release()
        writer.release()
    if count != len(timestamps):
        raise RuntimeError("Decoded frame count differs from the source PTS timeline")
    report = folder / "inpaint.json"
    atomic_json(report, {"schema_version": 1, "video": str(output), "frames": count, "issues": issues,
        "mask_origin": "OCR_POLYGONS_AND_STROKE_THRESHOLD", "temporal_context_frames": 8,
        "crop_resize": [scaled_w, scaled_h], "temporal_quality_verified": False})
    return {**identity([code_manifest, weight_manifest, ocr_manifest]), "video": str(output),
        "stage_status": "DEGRADED", "failure_code": "INPAINT_VISUAL_REVIEW_REQUIRED",
        "quality_evidence": {"issues": issues, "mask_and_temporal_quality_verified": False},
        "metrics": {"combined_processing_ms": (time.perf_counter() - tick) * 1000},
        "artifacts": [str(report), str(output)]}
