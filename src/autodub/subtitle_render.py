"""Streamed subtitle muxing and ROI-limited hard-subtitle replacement."""
from __future__ import annotations

import html
import json
import math
import os
import subprocess
import tempfile
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from autodub.adapters.common import identity
from autodub.contracts import Roi
from autodub.media import probe_media
from autodub.storage import atomic_json, sha256_file


class SubtitleRenderReview(RuntimeError):
    """A renderer condition that requires a human review instead of completion."""


def _segment_dict(segment: Any) -> dict[str, Any]:
    if isinstance(segment, dict):
        return segment
    if hasattr(segment, "model_dump"):
        return segment.model_dump()
    raise ValueError("segments must be Segment objects or dictionaries")


def _timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def subtitle_srt(segments: list, duration_ms: int) -> str:
    """Build deterministic SRT with bounded, escaped translated text."""
    if not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms <= 0:
        raise ValueError("duration_ms must be a positive integer")
    entries = []
    for segment in segments:
        item = _segment_dict(segment)
        start, end = item.get("start_ms"), item.get("end_ms")
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)
                or start < 0 or end <= start):
            raise ValueError("subtitle segment has invalid bounds")
        start, end = min(start, duration_ms), min(end, duration_ms)
        text = str(item.get("subtitle_vi") or item.get("dub_vi") or "").strip()
        if not text or end <= start:
            continue
        if len(text) > 2048:
            raise ValueError("subtitle text exceeds the 2048-character limit")
        # Escape SRT/ASS markup, remove control chars, and keep user text from
        # creating extra cues or changing timestamps.
        lines = []
        for line in text.replace("\r", "").split("\n"):
            line = "".join(" " if ord(char) < 32 or 0x7F <= ord(char) <= 0x9F else char
                           for char in line).strip()
            # ASS override blocks use braces rather than HTML angle brackets.
            line = line.replace("{", "｛").replace("}", "｝")
            if line:
                lines.append(html.escape(line, quote=False))
        text = "\n".join(lines)
        if text:
            entries.append((start, end, text))
    entries.sort(key=lambda cue: (cue[0], cue[1], cue[2]))
    blocks = [f"{index}\n{_timestamp(start)} --> {_timestamp(end)}\n{text}"
              for index, (start, end, text) in enumerate(entries, start=1)]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _run(command: list[str], timeout: float) -> None:
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("FFmpeg unavailable or subtitle render timed out") from error
    if result.returncode:
        raise RuntimeError("FFmpeg failed while rendering subtitles")


def _probe_video(source: Path, ffprobe_bin: str) -> tuple[dict, Fraction, int | None]:
    command = [ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-show_streams",
               "-show_format", "-of", "json", str(source)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                timeout=60, check=False)
        if result.returncode:
            raise ValueError
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        nominal = Fraction(stream["r_frame_rate"])
        average = Fraction(stream["avg_frame_rate"])
        if nominal <= 0 or average <= 0:
            raise ValueError
        count = stream.get("nb_frames")
        return payload, nominal, int(count) if count and str(count).isdigit() else None
    except (OSError, subprocess.TimeoutExpired, KeyError, IndexError, ValueError, ZeroDivisionError) as error:
        raise ValueError("Unable to verify source frame timeline") from error


def _probe_preview_audio(path: Path, ffprobe_bin: str) -> dict:
    command = [ffprobe_bin, "-v", "error", "-select_streams", "a:0", "-show_streams",
               "-show_format", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                timeout=60, check=False)
        if result.returncode:
            raise ValueError
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        duration_ms = round(float(Decimal(str(payload["format"]["duration"])) * 1000))
        sample_rate, channels = int(stream["sample_rate"]), int(stream["channels"])
        if duration_ms <= 0:
            raise ValueError
        return {"duration_ms": duration_ms, "sample_rate_hz": sample_rate, "channels": channels}
    except (OSError, subprocess.TimeoutExpired, KeyError, IndexError, ValueError,
            InvalidOperation, TypeError) as error:
        raise ValueError("Preview audio track could not be verified") from error


def _roi_pixels(roi_value: dict, width: int, height: int) -> tuple[int, int, int, int]:
    roi = Roi.model_validate(roi_value)
    left = max(0, min(width, int(math.floor(roi.x * width))))
    top = max(0, min(height, int(math.floor(roi.y * height))))
    right = max(0, min(width, int(math.ceil((roi.x + roi.w) * width))))
    bottom = max(0, min(height, int(math.ceil((roi.y + roi.h) * height))))
    if right <= left or bottom <= top:
        raise ValueError("subtitle ROI is empty at source resolution")
    return left, top, right, bottom


def _filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,")


def _atomic_video_command(source: Path, preview_audio: Path, output_tmp: Path, srt: Path,
                          mode: str, media: dict, config: dict, timeout: float) -> None:
    ffmpeg = config.get("ffmpeg_bin", "ffmpeg")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source),
               "-i", str(preview_audio), "-map", "0:v:0", "-map", "1:a:0"]
    if mode == "off":
        command.extend(["-c:v", "copy"])
    else:
        dimensions = media["video"]
        pixel_format = "yuv420p" if dimensions["width"] % 2 == 0 and dimensions["height"] % 2 == 0 else "yuv444p"
        subtitle_filter = f"subtitles=filename='{_filter_path(srt)}':charenc=UTF-8"
        command.extend(["-vf", subtitle_filter, "-c:v", "libx264", "-preset",
                        str(config.get("preset", "medium")), "-crf", str(config.get("crf", 20)),
                        "-pix_fmt", pixel_format, "-vsync", "0"])
    command.extend(["-c:a", "aac", "-b:a", str(config.get("audio_bitrate", "192k")),
                    "-t", f"{media['duration_ms'] / 1000:.6f}", "-movflags", "+faststart",
                    "-y", str(output_tmp)])
    _run(command, timeout)


def _replace_hardsubs(source: Path, preview_audio: Path, output_tmp: Path, roi: tuple[int, int, int, int],
                      fps: Fraction, expected_frames: int | None, config: dict,
                      deadline: float) -> dict:
    import cv2
    import numpy as np
    import onnxruntime as ort

    from autodub.adapters.common import asset, configure_offline, milliseconds
    from autodub.adapters.inpaint import inpaint_masked
    from autodub.adapters.ocr import build_engine, polygon_mask

    configure_offline(config)
    model_folder, lama_manifest = asset(config, "lama-onnx")
    engine, ocr_manifest = build_engine(config)
    available = ort.get_available_providers()
    requested = config.get("vision_provider", "CPUExecutionProvider")
    providers = [requested] if requested in available else ["CPUExecutionProvider"]
    tick = milliseconds()
    try:
        session = ort.InferenceSession(str(model_folder / "lama_fp32.onnx"), providers=providers)
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_folder / "lama_fp32.onnx"), providers=providers)
    model_load_ms = milliseconds() - tick

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("OpenCV could not decode source video")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count_metadata = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    left, top, right, bottom = roi
    max_frame_pixels = config.get("max_frame_pixels", 8294400)
    if (not isinstance(max_frame_pixels, int) or isinstance(max_frame_pixels, bool)
            or not 1 <= max_frame_pixels <= 8294400):
        capture.release()
        raise ValueError("max_frame_pixels must be between 1 and 8294400")
    if (width <= 0 or height <= 0 or width * height > max_frame_pixels
            or abs(reported_fps - float(fps)) > max(0.02, float(fps) * 0.001)
            or float(fps) > 120 or right > width or bottom > height):
        capture.release()
        raise SubtitleRenderReview("SOURCE_TIMELINE_UNVERIFIED")

    frame_interval_ms = 1000 / float(fps)
    tolerance_ms = max(2.0, frame_interval_ms * 0.05)
    ffmpeg = config.get("ffmpeg_bin", "ffmpeg")
    pixel_format = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-thread_queue_size", "64",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}",
               "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0",
               "-i", str(preview_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
               "-preset", str(config.get("preset", "medium")), "-crf", str(config.get("crf", 20)),
               "-pix_fmt", pixel_format, "-vsync", "0", "-c:a", "aac", "-b:a",
               str(config.get("audio_bitrate", "192k")), "-t",
               f"{config['source_duration_ms'] / 1000:.6f}", "-movflags", "+faststart", "-y", str(output_tmp)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    count = detected = inpainted = 0
    inference_ms = 0
    unmasked_preserved = True
    first_pts = None
    try:
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Subtitle render exceeded its configured wall-time limit")
            ok, frame = capture.read()
            if not ok:
                break
            pts_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            if frame.shape[:2] != (height, width):
                raise SubtitleRenderReview("SOURCE_FRAME_DIMENSION_CHANGED")
            expected_pts = count * frame_interval_ms
            if not math.isfinite(pts_ms) or abs(pts_ms - expected_pts) > tolerance_ms:
                raise SubtitleRenderReview("SOURCE_TIMELINE_UNVERIFIED")
            if first_pts is None:
                first_pts = pts_ms

            crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
            tick = milliseconds()
            result = engine(crop)
            inference_ms += milliseconds() - tick
            boxes = result.boxes.tolist() if result.boxes is not None else []
            if boxes:
                detected += len(boxes)
                dilation = config.get("mask_dilate_px", 4)
                if not isinstance(dilation, int) or isinstance(dilation, bool) or not 0 <= dilation <= 12:
                    raise ValueError("mask_dilate_px must be an integer between 0 and 12")
                local_mask = np.asarray(polygon_mask((right - left, bottom - top), boxes,
                                        dilate_px=dilation), dtype=np.uint8)
                if local_mask.any():
                    full_mask = np.zeros((height, width), dtype=np.uint8)
                    full_mask[top:bottom, left:right] = local_mask
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tick = milliseconds()
                    try:
                        cleaned = inpaint_masked(rgb, full_mask, session,
                            context_px=int(config.get("context_px", 64)), output_range="0_255")
                    except Exception:
                        if "CUDAExecutionProvider" not in providers:
                            raise
                        tick = milliseconds()
                        session = ort.InferenceSession(str(model_folder / "lama_fp32.onnx"),
                                                       providers=["CPUExecutionProvider"])
                        model_load_ms += milliseconds() - tick
                        providers = ["CPUExecutionProvider"]
                        cleaned = inpaint_masked(rgb, full_mask, session,
                            context_px=int(config.get("context_px", 64)), output_range="0_255")
                    inference_ms += milliseconds() - tick
                    if not np.array_equal(rgb[full_mask == 0], cleaned[full_mask == 0]):
                        unmasked_preserved = False
                        raise RuntimeError("Inpaint changed pixels outside detected subtitle polygons")
                    frame = cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR)
                    inpainted += 1
            if process.stdin is None:
                raise RuntimeError("Video encoder input was not created")
            process.stdin.write(frame.tobytes())
            count += 1
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            raise TimeoutError("Subtitle render encoder exceeded its wall-time limit") from error
        if process.returncode:
            raise RuntimeError("FFmpeg failed while encoding replaced subtitle frames")
    except Exception:
        capture.release()
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        capture.release()
    if count == 0 or (expected_frames and count != expected_frames) or (frame_count_metadata and count != frame_count_metadata):
        raise SubtitleRenderReview("SOURCE_FRAME_COUNT_UNVERIFIED")
    return {"width": width, "height": height, "frame_count": count,
            "detected_text_polygons": detected, "inpainted_frames": inpainted,
            "unmasked_pixels_preserved_before_encode": unmasked_preserved,
            "model_load_ms": model_load_ms, "ocr_inference_ms": inference_ms,
            "providers": providers, "model_manifests": [lama_manifest, ocr_manifest],
            "fps": f"{fps.numerator}/{fps.denominator}", "first_pts_ms": first_pts}


def _review_result(source: Path, output: Path, srt_path: Path, report_path: Path,
                   source_hash: str, mode: str, reason: str,
                   frame_result: dict | None = None) -> dict:
    manifests = frame_result.get("model_manifests", []) if frame_result else []
    provenance = identity(manifests) if manifests else {"model_revision": "", "weights_sha256": None}
    frame_report = ({key: value for key, value in frame_result.items() if key != "model_manifests"}
                    if frame_result else None)
    report = {"schema_version": 1, "status": "NEEDS_REVIEW", "action": "NEEDS_REVIEW",
              "review_required": True, "failure_code": reason, "mode": mode,
              "source_sha256": source_hash, "output_video": None,
              **provenance,
              "models": [{"id": item["id"], "revision": item["model_revision"],
                          "weights_sha256": item["weights_sha256"]} for item in manifests],
              "frame_result": frame_report,
              "technical_render_verified": False,
              "semantic_quality": "UNVERIFIED", "ocr_residual_text_pass": None,
              "quality_evidence": {"status": "REVIEW_REQUIRED", "semantic_quality": "UNVERIFIED",
                                   "ocr_residual_text_pass": None}}
    atomic_json(report_path, report)
    return {"review_required": True, "action": "NEEDS_REVIEW", "failure_code": reason,
            **provenance,
            "processed_media_ms": 0,
            "metrics": ({"model_load_ms": frame_result.get("model_load_ms", 0),
                         "ocr_inference_ms": frame_result.get("ocr_inference_ms", 0)}
                        if frame_result else {}),
            "quality_metrics": {},
            "quality_evidence": report["quality_evidence"],
            "artifacts": [str(srt_path), str(report_path)]}


def render_subtitles(source: Path, preview_audio: Path, output: Path, segments: list,
                     config: dict) -> dict:
    """Mux dubbing audio and optional Vietnamese captions into a new video.

    Modes: ``off`` replaces audio only, ``burn`` overlays translated captions,
    and ``replace`` applies verified local LaMa inpainting to OCR polygons inside
    an explicitly selected ROI before rendering the translated captions.
    """
    source, preview_audio, output = Path(source), Path(preview_audio), Path(output)
    output_dir = Path(config.get("output_dir", output.parent)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    if not output.is_relative_to(output_dir):
        raise ValueError("render output must stay inside output_dir")
    if output in {source.resolve(), preview_audio.resolve()}:
        raise ValueError("render output must not overwrite an input")
    mode = config.get("subtitle_mode", "off")
    if mode not in {"off", "burn", "replace"}:
        raise ValueError("subtitle_mode must be off, burn, or replace")
    timeout = config.get("timeout_seconds", 900)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 900:
        raise ValueError("subtitle render timeout must be between 0 and 900 seconds")
    if output.suffix.lower() != ".mp4":
        raise ValueError("render output must use the .mp4 extension")
    preset = config.get("preset", "medium")
    crf = config.get("crf", 20)
    if preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"}:
        raise ValueError("unsupported H.264 preset")
    if not isinstance(crf, int) or isinstance(crf, bool) or not 0 <= crf <= 51:
        raise ValueError("H.264 CRF must be an integer between 0 and 51")
    started = time.monotonic()
    deadline = started + timeout
    source_hash = sha256_file(source)
    media = probe_media(source, config.get("ffprobe_bin", "ffprobe"))
    srt_path = output.with_name("subs.vi.srt")
    report_path = output.with_name("subtitle_render.qc.json")
    _write_text_atomic(srt_path, subtitle_srt(segments, media["duration_ms"]))
    try:
        audio_info = _probe_preview_audio(preview_audio, config.get("ffprobe_bin", "ffprobe"))
    except ValueError:
        return _review_result(source, output, srt_path, report_path, source_hash, mode,
                              "PREVIEW_AUDIO_UNVERIFIED")
    if audio_info["channels"] != 2 or audio_info["sample_rate_hz"] != 48000:
        return _review_result(source, output, srt_path, report_path, source_hash, mode,
                              "PREVIEW_AUDIO_FORMAT_MISMATCH")
    if abs(audio_info["duration_ms"] - media["duration_ms"]) > 500:
        return _review_result(source, output, srt_path, report_path, source_hash, mode,
                              "PREVIEW_AUDIO_DURATION_MISMATCH")
    if mode == "replace" and not config.get("roi"):
        return _review_result(source, output, srt_path, report_path, source_hash, mode, "SUBTITLE_ROI_REQUIRED")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="subtitle-render-", dir=output.parent) as temp_dir:
        tmp_video = Path(temp_dir) / "rendered.partial.mp4"
        model_result = None
        if mode == "replace":
            ffprobe_bin = config.get("ffprobe_bin", "ffprobe")
            try:
                payload, fps, ffprobe_count = _probe_video(source, ffprobe_bin)
            except ValueError:
                return _review_result(source, output, srt_path, report_path, source_hash,
                                      mode, "SOURCE_TIMELINE_UNVERIFIED")
            stream = payload["streams"][0]
            average = Fraction(stream.get("avg_frame_rate", "0/0"))
            if fps != average:
                return _review_result(source, output, srt_path, report_path, source_hash, mode,
                                      "SOURCE_VARIABLE_FRAME_RATE")
            roi = _roi_pixels(config["roi"], int(stream["width"]), int(stream["height"]))
            replace_config = {**config, "source_duration_ms": media["duration_ms"]}
            try:
                model_result = _replace_hardsubs(source, preview_audio, tmp_video, roi, fps,
                    ffprobe_count, replace_config, deadline)
            except SubtitleRenderReview as error:
                return _review_result(source, output, srt_path, report_path, source_hash,
                                      mode, str(error) or "SOURCE_TIMELINE_UNVERIFIED")
            if model_result["detected_text_polygons"] == 0:
                return _review_result(source, output, srt_path, report_path, source_hash, mode,
                                      "NO_SUBTITLE_POLYGONS_DETECTED", model_result)
            # The streamed frames already contain OCR-limited replacement.
            # Burn translated SRT in a second bounded FFmpeg pass.
            subtitled = Path(temp_dir) / "subtitled.partial.mp4"
            burn_config = {**config, "ffmpeg_bin": config.get("ffmpeg_bin", "ffmpeg")}
            _atomic_video_command(tmp_video, preview_audio, subtitled, srt_path, "burn",
                                 media, burn_config, max(1.0, deadline - time.monotonic()))
            tmp_video = subtitled
        else:
            _atomic_video_command(source, preview_audio, tmp_video, srt_path, mode, media,
                                  config, max(1.0, deadline - time.monotonic()))
        if time.monotonic() > deadline:
            raise TimeoutError("Subtitle render exceeded its configured wall-time limit")
        if not tmp_video.is_file() or tmp_video.stat().st_size <= 0:
            raise RuntimeError("Subtitle render did not produce a video")
        rendered_media = probe_media(tmp_video, config.get("ffprobe_bin", "ffprobe"))
        duration_delta_ms = abs(rendered_media["duration_ms"] - media["duration_ms"])
        if duration_delta_ms > 500:
            return _review_result(source, output, srt_path, report_path, source_hash,
                                  mode, "OUTPUT_DURATION_OUT_OF_BOUNDS", model_result)
        if mode == "replace":
            _payload, rendered_fps, rendered_frames = _probe_video(tmp_video, config.get("ffprobe_bin", "ffprobe"))
            if rendered_frames != model_result["frame_count"] or rendered_fps != fps:
                return _review_result(source, output, srt_path, report_path, source_hash,
                                      mode, "OUTPUT_FRAME_TIMELINE_UNVERIFIED", model_result)
        os.replace(tmp_video, output)

    if sha256_file(source) != source_hash:
        raise RuntimeError("Subtitle render modified the source video")
    elapsed_ms = round((time.monotonic() - started) * 1000)
    technical = {"output_exists": output.is_file(), "output_bytes": output.stat().st_size,
                 "source_duration_ms": media["duration_ms"], "output_duration_ms": rendered_media["duration_ms"],
                 "duration_delta_ms": duration_delta_ms,
                 "duration_pass": duration_delta_ms <= 500,
                 "preview_audio": audio_info,
                 "output_video_codec": rendered_media["video"]["codec_name"],
                 "output_width": rendered_media["video"]["width"],
                 "output_height": rendered_media["video"]["height"],
                 "encoder": "copy" if mode == "off" else "libx264",
                 "source_sha256": source_hash, "output_sha256": sha256_file(output),
                 "render_wall_ms": elapsed_ms, "mode": mode}
    review_required = False
    quality = {"status": "TECHNICAL_ONLY",
               "technical_render_verified": True,
               "semantic_quality": "UNVERIFIED",
               "ocr_residual_text_pass": None,
               "human_listening_required": True,
               "inpaint_preserved_unmasked_pixels_in_memory": (
                   model_result["unmasked_pixels_preserved_before_encode"] if model_result else None),
               "encoded_pixels_outside_roi_bit_exact": None if model_result else (mode == "off")}
    manifests = model_result["model_manifests"] if model_result else []
    provenance = identity(manifests) if manifests else {"model_revision": "", "weights_sha256": None}
    frame_report = ({key: value for key, value in model_result.items() if key != "model_manifests"}
                    if model_result else None)
    report = {"schema_version": 1, "technical_render_verified": True,
              "status": "NEEDS_REVIEW" if review_required else "RENDERED",
              "action": "NEEDS_REVIEW" if review_required else "RENDERED",
              "review_required": review_required, **technical, **provenance,
              "semantic_quality": "UNVERIFIED", "ocr_residual_text_pass": None,
              "models": [{"id": item["id"], "revision": item["model_revision"],
                          "weights_sha256": item["weights_sha256"]} for item in manifests],
              "frame_result": frame_report, "quality_evidence": quality}
    atomic_json(report_path, report)
    return {"review_required": review_required, "action": report["action"],
            "model_revision": provenance["model_revision"], "weights_sha256": provenance["weights_sha256"],
            "processed_media_ms": media["duration_ms"], "metrics": {
                "render_wall_ms": elapsed_ms,
                "ocr_inference_ms": model_result["ocr_inference_ms"] if model_result else 0,
                "model_load_ms": model_result["model_load_ms"] if model_result else 0},
            "quality_metrics": technical, "quality_evidence": quality,
            "artifacts": [str(output), str(srt_path), str(report_path)]}
