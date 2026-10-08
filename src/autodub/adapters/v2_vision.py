"""Event OCR and evidence-bound clean-frame recovery, with one final encode."""
from __future__ import annotations

import json
import subprocess
import threading
import time
from bisect import bisect_right
from fractions import Fraction
from pathlib import Path

from autodub.adapters.ocr import build_engine
from autodub.adapters.runtime_paths import output_folder, run_file
from autodub.storage import atomic_json, sha256_file


def text_signature(image):
    import cv2

    gray = cv2.cvtColor(cv2.resize(image, (256, 64)), cv2.COLOR_BGR2GRAY)
    # Bright subtitle strokes against a local dark outline, not raw background motion.
    background = cv2.GaussianBlur(gray, (9, 9), 0)
    return (gray.astype("int16") - background.astype("int16") > 25).astype("uint8")


def signature_changed(previous, current, boxes, image_shape):
    """Watch known text rectangles; heartbeat still discovers newly placed text."""
    import cv2
    import numpy as np

    if previous is None:
        return True
    region = np.ones_like(current)
    if boxes:
        region.fill(0)
        scale = np.array([current.shape[1] / image_shape[1], current.shape[0] / image_shape[0]])
        for box in boxes:
            polygon = np.round(np.asarray(box) * scale).astype("int32")
            cv2.fillPoly(region, [polygon], 1)
        region = cv2.dilate(region, np.ones((3, 3), dtype="uint8"))
    changed = np.count_nonzero((current != previous) & (region > 0))
    strokes = np.count_nonzero((current | previous) & region)
    return changed / max(16, strokes) > 0.15


def glyph_mask(image, boxes):
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


def classify_text(lines: list[dict], crop_height: int, crop_width: int = 640) -> str:
    if not lines:
        return "NONE"
    text = "".join(line["text"] for line in lines)
    if any(marker in text for marker in ("♪", "♫", "演唱", "作曲", "作词")):
        return "LYRIC"
    centers = [sum(point[0] for point in line["box"]) / len(line["box"]) for line in lines]
    if not any(crop_width * 0.2 <= center <= crop_width * 0.8 for center in centers):
        return "LOGO"
    # Location is evidence only; long duration alone never proves a lyric.
    return "DIALOGUE" if any("\u4e00" <= char <= "\u9fff" for char in text) else "CAPTION"


def scan(source: Path, config: dict) -> dict:
    from autodub.adapters.subtitle_events import scan as scan_subtitles

    return scan_subtitles(source, config, engine_factory=build_engine)

def recover_crop(image, donor, mask):
    """Register observed clean pixels; insufficient evidence returns the original crop."""
    import cv2
    import numpy as np

    if donor.shape != image.shape or not np.any(mask):
        return image, False
    original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clean = cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY)
    outside = cv2.bitwise_not(cv2.dilate(mask, np.ones((5, 5), dtype="uint8")))
    points = cv2.goodFeaturesToTrack(clean, maxCorners=80, qualityLevel=0.01, minDistance=6, mask=outside)
    transform = np.eye(2, 3, dtype="float32")
    if points is not None and len(points) >= 6:
        moved, valid, _ = cv2.calcOpticalFlowPyrLK(clean, original, points, None,
                                               winSize=(21, 21), maxLevel=2)
        if moved is None or valid is None or int(valid.sum()) < 6:
            return image, False
        keep = valid.reshape(-1) > 0
        transform, inliers = cv2.estimateAffinePartial2D(points[keep], moved[keep], method=cv2.RANSAC,
                                                        ransacReprojThreshold=2.0)
        if transform is None or inliers is None or float(inliers.mean()) < 0.7:
            return image, False
        scale = float(np.hypot(transform[0, 0], transform[0, 1]))
        if not 0.8 <= scale <= 1.25:
            return image, False
    elif np.count_nonzero(outside) < 64 or float(np.median(np.abs(clean.astype("float32") - original)[outside > 0])) > 12:
        return image, False
    h, w = image.shape[:2]
    warped = cv2.warpAffine(donor, transform, (w, h))
    valid = cv2.warpAffine(np.full((h, w), 255, dtype="uint8"), transform, (w, h), flags=cv2.INTER_NEAREST)
    if np.any((mask > 0) & (valid < 255)):
        return image, False
    error = np.mean(np.abs(warped.astype("float32") - image), axis=2)
    if not np.any(outside) or float(np.median(error[outside > 0])) > 15:
        return image, False
    recovered = image.copy()
    recovered[mask > 0] = warped[mask > 0]
    return recovered, True


def restoration_plan(source: Path, config: dict) -> dict:
    folder = output_folder(config)
    ocr = json.loads(run_file(config["ocr_path"]).read_text(encoding="utf-8"))
    if ocr["source_sha256"] != sha256_file(source):
        raise ValueError("OCR source changed")
    selected = [event for event in ocr["events"] if event["kind"] == "DIALOGUE"]
    target = folder / "restoration-plan.json"
    issues = [{"event_id": event["id"], "code": "NO_CLEAN_FRAME_DONOR", "action": "KEEP_ORIGINAL_PIXELS"}
              for event in selected if not event["donors"]]
    atomic_json(target, {**ocr, "events": selected, "issues": issues,
                        "restoration_method": "CLEAN_FRAME_LK_AFFINE", "generative_model": None})
    return {"plan": str(target), "artifacts": [str(target)],
            "stage_status": "DEGRADED" if issues else "SUCCESS", "quality_evidence": {"issues": issues}}


def encode(source: Path, config: dict) -> dict:

    folder = output_folder(config)
    audio = source.resolve() if Path(config["audio_path"]) == source else run_file(config["audio_path"])
    subtitles = run_file(config["subtitles_path"])
    output = folder / "final.mp4"
    encoder = config.get("video_encoder", "h264_nvenc")
    if config.get("production", False) and encoder != "h264_nvenc":
        raise ValueError("Production encoder must be NVENC")
    options = ["-c:v", encoder, "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-movflags", "+faststart"]
    if encoder == "h264_nvenc":
        options += ["-preset", "p5", "-cq", "23"]
    graph = "subtitles=subtitles.srt" if subtitles.stat().st_size > 1 else "null"
    local_subtitles = folder / "subtitles.srt"
    local_subtitles.write_bytes(subtitles.read_bytes())
    plan = json.loads(run_file(config["restoration_plan"]).read_text()) if config.get("restoration_plan") else None
    issues, repaired, untouched = [], 0, 0
    tick = time.perf_counter()
    if not plan or not plan["events"] or config.get("subtitle_mode") != "replace":
        subprocess.run([config.get("ffmpeg_bin", "ffmpeg"), "-hide_banner", "-loglevel", "error",
                        "-i", str(source), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
                        "-vf", graph, *options, "-y", str(output)], cwd=folder,
                       capture_output=True, check=True, timeout=int(config.get("media_timeout_seconds", 7200)))
    else:
        if plan["source_sha256"] != sha256_file(source):
            raise ValueError("Restoration plan source changed")
        _encode_restored(source, audio, folder, output, graph, options, plan, config, issues)
        repaired = sum(event.get("restored_frames", 0) for event in plan["events"])
        untouched = sum(event.get("unrestored_frames", 0) for event in plan["events"])
    from autodub.media import probe_media
    duration = probe_media(output, config.get("ffprobe_bin", "ffprobe"))["duration_ms"]
    if abs(duration - config["duration_ms"]) > 100:
        raise ValueError("Encoded output timeline drift")
    target = folder / "render.json"
    atomic_json(target, {"schema_version": 1, "video": str(output), "issues": issues,
                        "restored_frames": repaired, "unrestored_frames": untouched,
                        "source_sha256": sha256_file(source), "encode_passes": 1, "encoder": encoder,
                        "generative_model": None, "visual_quality_verified": False})
    return {"video": str(output), "artifacts": [str(target), str(output), str(local_subtitles)],
            "stage_status": "DEGRADED" if issues else "SUCCESS", "quality_evidence": {"issues": issues},
            "metrics": {"combined_processing_ms": (time.perf_counter() - tick) * 1000}}


def _encode_restored(source, audio, folder, output, graph, options, plan, config, issues):
    import cv2

    # Reject unsupported timelines rather than silently turning VFR into CFR.
    probe = subprocess.run([config.get("ffprobe_bin", "ffprobe"), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=r_frame_rate,avg_frame_rate,start_time", "-of", "json", str(source)],
                           check=True, capture_output=True, text=True, timeout=30)
    stream = json.loads(probe.stdout)["streams"][0]
    fps = Fraction(stream["avg_frame_rate"])
    if fps <= 0 or Fraction(stream["r_frame_rate"]) != fps or abs(float(stream.get("start_time", 0))) > 0.002:
        raise ValueError("Temporal restoration requires a zero-origin constant frame timeline")
    width, height = plan["width"], plan["height"]
    command = [config.get("ffmpeg_bin", "ffmpeg"), "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
               "-framerate", str(fps), "-i", "pipe:0", "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
               "-vf", graph, *options, "-y", str(output)]
    capture = cv2.VideoCapture(str(source))
    with (folder / "encoder.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=folder, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        watchdog = threading.Timer(int(config.get("media_timeout_seconds", 7200)), process.kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            index = _write_restored_frames(capture, process.stdin, plan, fps, config["duration_ms"], issues)
            process.stdin.close()
            if process.wait(timeout=60) != 0:
                raise RuntimeError("Final NVENC renderer failed")
            if abs(index * 1000 / float(fps) - config["duration_ms"]) > 100:
                raise ValueError("Decoded frame timeline differs from source duration")
        finally:
            watchdog.cancel()
            capture.release()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


class EventTimeline:
    """Bounded, ordered events; frame lookup does not loop on a JSON time value."""

    def __init__(self, events: list[dict], duration_ms: int):
        if not 0 < duration_ms <= 86400000 or len(events) > 100000:
            raise ValueError("Restoration timeline exceeds supported bounds")
        previous = 0
        for event in events:
            start, end = event["start_ms"], event["end_ms"]
            if not isinstance(start, int) or not isinstance(end, int) or not previous <= start < end <= duration_ms:
                raise ValueError("Restoration events must be ordered and within the source")
            if len(event["donors"]) > 2:
                raise ValueError("A subtitle event may have at most two clean donors")
            previous = end
        self.events = events
        self.ends = tuple(event["end_ms"] for event in events)

    def at(self, at_ms: int) -> dict | None:
        index = bisect_right(self.ends, at_ms)
        if index == len(self.events) or at_ms < self.events[index]["start_ms"]:
            return None
        return self.events[index]


def _write_restored_frames(capture, stream, plan, fps, duration_ms, issues):
    timeline = EventTimeline(plan["events"], duration_ms)
    donors, seen_errors, index, previous = {}, set(), 0, None
    while True:
        ok, image = capture.read()
        if not ok:
            return index
        event = timeline.at(round(index * 1000 / float(fps)))
        index += 1
        if event is not previous:
            donors.clear()
            previous = event
        if event is not None:
            restored = _restore_event(image, event, plan, donors)
            field = "restored_frames" if restored else "unrestored_frames"
            event[field] = event.get(field, 0) + 1
            if not restored and event["id"] not in seen_errors:
                issues.append({"event_id": event["id"], "code": "BACKGROUND_RECOVERY_UNCERTAIN",
                               "action": "KEEP_ORIGINAL_PIXELS"})
                seen_errors.add(event["id"])
        stream.write(image.tobytes())


def _restore_event(image, event, plan, donors):
    import cv2
    top, bottom, scaled_width, scaled_height = plan["crop"]
    crop = cv2.resize(image[top:bottom], (scaled_width, scaled_height))
    mask = glyph_mask(crop, event["boxes"])
    for raw_path in event["donors"]:
        if raw_path not in donors:
            donors[raw_path] = cv2.imread(str(run_file(raw_path)))
        donor = donors[raw_path]
        if donor is None:
            continue
        recovered, restored = recover_crop(crop, donor, mask)
        if restored:
            full = cv2.resize(recovered, (plan["width"], bottom - top))
            full_mask = cv2.resize(mask, (plan["width"], bottom - top), interpolation=cv2.INTER_NEAREST)
            image[top:bottom][full_mask > 0] = full[full_mask > 0]
            return True
    return False
