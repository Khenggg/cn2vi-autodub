"""Pinned LaMa ONNX tight crop benchmark. Does not render or modify a full episode."""
import json
from pathlib import Path

from autodub.adapters.common import asset, identity, milliseconds, output_folder
from autodub.storage import atomic_json, safe_path, sha256_file


def inpaint_masked(image, mask, session, *, context_px: int = 32, output_range: str = "0_1"):
    import numpy as np
    from PIL import Image
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("Inpaint expects RGB image and matching grayscale mask")
    if not 0 <= context_px <= 128 or output_range not in {"0_1", "0_255"}:
        raise ValueError("Invalid LaMa configuration")
    locations = np.argwhere(mask > 0)
    if not len(locations):
        return image.copy()
    top, left = locations.min(axis=0)
    bottom, right = locations.max(axis=0) + 1
    left, top = max(0, left - context_px), max(0, top - context_px)
    right, bottom = min(image.shape[1], right + context_px), min(image.shape[0], bottom + context_px)
    crop, crop_mask = image[top:bottom, left:right], mask[top:bottom, left:right]
    height, width = crop_mask.shape
    scale = min(512 / width, 512 / height, 1)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    patch = np.asarray(Image.fromarray(crop).resize(resized, Image.Resampling.LANCZOS))
    patch_mask = np.asarray(Image.fromarray(crop_mask).resize(resized, Image.Resampling.NEAREST))
    padded = np.pad(patch, ((0, 512 - resized[1]), (0, 512 - resized[0]), (0, 0)), mode="edge")
    padded_mask = np.pad(patch_mask, ((0, 512 - resized[1]), (0, 512 - resized[0])), mode="constant")
    feeds = {"image": padded.transpose(2, 0, 1)[None].astype(np.float32) / 255,
             "mask": (padded_mask[None, None] > 0).astype(np.float32)}
    names = {item.name for item in session.get_inputs()}
    if names != set(feeds):
        raise ValueError("LaMa ONNX input schema mismatch")
    prediction = session.run(None, feeds)[0]
    if prediction.shape != (1, 3, 512, 512) or not np.isfinite(prediction).all():
        raise ValueError("LaMa ONNX output schema mismatch")
    patch_result = prediction[0].transpose(1, 2, 0)[:resized[1], :resized[0]]
    if output_range == "0_1":
        patch_result = patch_result * 255
    patch_result = np.clip(patch_result, 0, 255).astype(np.uint8)
    restored = np.asarray(Image.fromarray(patch_result).resize((width, height), Image.Resampling.LANCZOS))
    result = image.copy()
    # Preserve every unmasked pixel exactly, including all pixels outside ROI.
    region = result[top:bottom, left:right]
    selected = crop_mask > 0
    region[selected] = restored[selected]
    return result


def run(source: Path, config: dict) -> dict:
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    # This pinned Carve export returns RGB values in [0,255], unlike native LaMa tensors.
    if config.get("output_range", "0_255") != "0_255":
        raise ValueError("Pinned LaMa ONNX output_range must be 0_255")
    manifest_path = Path(config["ocr_manifest_path"]).resolve()
    ocr = json.loads(manifest_path.read_text(encoding="utf-8"))
    if ocr.get("schema_version") != 1 or ocr.get("ocr_scope") != "ROI_ONLY" or ocr["source_sha256"] != sha256_file(source):
        raise ValueError("OCR mask manifest does not match source/ROI contract")
    inputs = []
    masked_frames = 0
    for frame in ocr["frames"]:
        image_path, mask_path = [safe_path(manifest_path.parent, frame[key]) for key in ("image", "mask")]
        if sha256_file(image_path) != frame["image_sha256"] or sha256_file(mask_path) != frame["mask_sha256"]:
            raise ValueError("OCR crop/mask checksum mismatch")
        with Image.open(mask_path) as picture:
            masked_frames += picture.convert("L").getbbox() is not None
        inputs.append((image_path, mask_path))
    if not masked_frames:
        raise ValueError("Inpainting benchmark requires at least one non-empty OCR glyph mask")
    model_folder, manifest = asset(config, "lama-onnx")
    folder = output_folder(config)
    tick = milliseconds()
    session = ort.InferenceSession(str(model_folder / "lama_fp32.onnx"), providers=["CPUExecutionProvider"])
    load_ms = milliseconds() - tick
    from autodub.adapters.ocr import build_engine
    residual_engine, residual_manifest = build_engine(config)
    frames, inference_ms, qc_ms, residual_count = [], 0, 0, 0
    for index, frame in enumerate(ocr["frames"]):
        image_path, mask_path = inputs[index]
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            before, binary_mask = np.asarray(image.convert("RGB")), np.asarray(mask.convert("L"))
        tick = milliseconds()
        after = inpaint_masked(before, binary_mask, session, context_px=int(config.get("context_px", 32)),
                               output_range="0_255")
        inference_ms += milliseconds() - tick
        output = folder / f"inpaint_{index:06d}.png"
        Image.fromarray(after).save(output)
        tick = milliseconds()
        residual = residual_engine(str(output))
        qc_ms += milliseconds() - tick
        after_count = len(residual.txts) if residual.txts is not None else 0
        residual_count += after_count
        frames.append({"at_ms": frame["at_ms"], "image": output.name, "sha256": sha256_file(output),
                       "ocr_residual_text_count": after_count,
                       "unmasked_pixels_unchanged": bool(np.array_equal(before[binary_mask == 0], after[binary_mask == 0]))})
    target = folder / "inpaint.json"
    atomic_json(target, {"schema_version": 1, "source_sha256": ocr["source_sha256"], "roi": ocr["roi"],
                         "frames": frames, "render_scope": "CROPPED_FRAME_BENCHMARK", "final_video": False})
    processed_ms = (ocr["end_ms"] - ocr["start_ms"]
                    if ocr.get("temporal_coverage") == "DENSE_SOURCE_FRAMES" else 0)
    return {**identity([manifest, residual_manifest]), "processed_media_ms": processed_ms,
            "metrics": {"model_load_ms": load_ms, "inference_ms": inference_ms},
            "quality_metrics": {"processed_frames": len(frames), "masked_frames": masked_frames,
                                "unmasked_pixels_unchanged": all(f["unmasked_pixels_unchanged"] for f in frames),
                                "ocr_residual_text_count": residual_count, "ocr_qc_ms": qc_ms,
                                "inference_ms_per_frame": inference_ms / len(frames) if frames else None,
                                "flow_warp_flicker": None},
            "quality_evidence": {"status": "REVIEW_REQUIRED", "missing": ["motion_flicker_review", "residual_false_positive_review"]},
            "artifacts": [str(target), *[str(folder / frame["image"]) for frame in frames]]}
