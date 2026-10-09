"""Sequential PTS analysis, one OCR owner, bounded frame lifetime, durable evidence."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import threading
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from time import perf_counter

from .domain import Request, digest
from .events import EventBuilder, observation_record
from .media import file_sha256
from .ocr import RapidAdapter, load_manifest

DEFAULT_CONFIG = {
    "schema_version": 1,
    "provider": "CUDAExecutionProvider",
    "device_id": 0,
    "cudnn_conv_algo_search": "HEURISTIC",
    "native_threads": 2,
    "recognition_batch": 8,
    "detector_side": 960,
    "watchdog_s": 0.25,
    "change_min_interval_s": 0.08,
    "tile_change_threshold": 8.0,
    "scene_change_threshold": 45.0,
    "review_confidence": 0.68,
}


def write_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def validate_config(data: dict) -> dict:
    if set(data) != set(DEFAULT_CONFIG):
        raise ValueError("Config must have exactly the documented fields")
    result = dict(data)
    if result["schema_version"] != 1:
        raise ValueError("Unsupported config schema")
    for key, lower, upper in (
        ("native_threads", 1, 16),
        ("recognition_batch", 1, 32),
        ("detector_side", 320, 2048),
        ("device_id", 0, 16),
    ):
        if type(result[key]) is not int or not lower <= result[key] <= upper:
            raise ValueError(f"Invalid config field: {key}")
    for key, lower, upper in (
        ("watchdog_s", 0.03, 0.25),
        ("change_min_interval_s", 0.01, 0.25),
        ("tile_change_threshold", 0.1, 255),
        ("scene_change_threshold", 0.1, 255),
        ("review_confidence", 0, 1),
    ):
        value = float(result[key])
        if not lower <= value <= upper:
            raise ValueError(f"Invalid config field: {key}")
        result[key] = value
    if result["provider"] not in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        raise ValueError("Choose CUDA explicitly, or an explicitly labeled CPU profile")
    if result["cudnn_conv_algo_search"] not in ("EXHAUSTIVE", "HEURISTIC", "DEFAULT"):
        raise ValueError("Invalid CUDA convolution search policy")
    return result


def code_fingerprint() -> dict:
    root = Path(__file__).parent
    files = {
        str(p.relative_to(root)): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".html", ".js", ".css")
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"commit": commit, "source_sha256": digest(files), "files": files}


class Cancelled(Exception):
    pass


def analyze(
    video: Path,
    media: dict,
    request: Request,
    output: Path,
    model_manifest: Path,
    config: dict | None = None,
    progress=None,
    cancel: threading.Event | None = None,
):
    started = perf_counter()
    config = validate_config(DEFAULT_CONFIG if config is None else config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence").mkdir()
    report = {
        "schema_version": 1,
        "stage": "discovery",
        "status": "RUNNING",
        "quality_verified": False,
        "sla_verified": False,
        "translation": "NOT_IMPLEMENTED",
        "render": "NOT_IMPLEMENTED",
        "resources": "NOT_MEASURED",
        "config": config,
        "config_sha256": digest(config),
        "code": code_fingerprint(),
        "request": asdict(request),
        "media": media,
        "scope": {
            "status": "EXCLUDED_BY_USER",
            "exclusions": [asdict(x) for x in request.exclusions],
        },
        "limitations": [
            "Bounding times are sampling brackets; no boundary refinement yet",
            "No semantic importance, translation, render, perspective tracking or quality oracle",
            "Rectangular ROI only; text crossing tile boundaries requires review",
            "4 Hz watchdog is time coverage, not proof of recognition recall",
            "Detector resizes within its configured side limit; small text recall is unmeasured",
            "Angle classifier disabled; 180-degree text is not validated",
            "CPU decode/CV and possible ORT CPU partitions are reported separately",
        ],
        "counters": {},
        "timings": {},
        "error": None,
    }
    tracker = EventBuilder(request.start_s, config["review_confidence"])
    counters = Counter()
    timing = Counter()
    engine = None
    container = None
    last_time = None
    last_frame_end = None
    reached_end = False

    def notify(stage, time_s=0):
        if progress:
            progress(
                {
                    "stage": stage,
                    "time_s": time_s,
                    "events": tracker.counter,
                    "counters": dict(counters),
                    "wall_s": perf_counter() - started,
                }
            )

    try:
        notify("INPUT_AND_MODEL_HASH")
        phase = perf_counter()
        report["input_sha256"] = file_sha256(video)
        manifest = load_manifest(model_manifest)
        report["models"] = manifest
        timing["hash_and_manifest_s"] = perf_counter() - phase
        for name in (
            "av",
            "numpy",
            "opencv-python-headless",
            "rapidocr",
            "onnxruntime-gpu",
            "onnxruntime",
            "nvidia-cudnn-cu12",
            "nvidia-cublas-cu12",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cuda-nvrtc-cu12",
        ):
            try:
                report.setdefault("runtime", {})[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        write_json(output / "run-report.json", report)
        notify("MODEL_LOAD")
        phase = perf_counter()
        engine = RapidAdapter(manifest, config, output)
        timing["model_load_s"] = perf_counter() - phase
        import av
        import cv2
        import numpy as np

        container = av.open(str(video))
        stream = next(s for s in container.streams.video if s.index == media["stream_index"])
        stream.codec_context.thread_count = config["native_threads"]
        origin = Fraction(media["origin"]["num"], media["origin"]["den"])
        previous_gray = None
        previous_roi = None
        last_candidate = None
        last_notify = 0.0
        frames = iter(container.decode(stream))
        with (output / "observations.jsonl").open("w", encoding="utf-8") as raw:
            while True:
                if cancel and cancel.is_set():
                    raise Cancelled("Analysis cancelled; partial artifacts retained")
                phase = perf_counter()
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                timing["decode_s"] += perf_counter() - phase
                counters["decoded_frames"] += 1
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(
                        "Frame lacks presentation timestamp; index/fps fallback is disabled"
                    )
                tb = Fraction(frame.time_base)
                time_s = float(frame.pts * tb - origin)
                if last_time is not None and time_s < last_time:
                    raise ValueError("Non-monotonic decoded PTS")
                last_time = time_s
                duration_pts = getattr(frame, "duration", None)
                if duration_pts and duration_pts > 0:
                    last_frame_end = time_s + float(duration_pts * tb)
                else:
                    last_frame_end = None
                if time_s < request.start_s:
                    continue
                if time_s >= request.end_s:
                    reached_end = True
                    break
                if frame.width != media["width"] or frame.height != media["height"]:
                    raise ValueError("Source raster dimensions changed during analysis")
                counters["analyzed_frames"] += 1
                phase = perf_counter()
                image = frame.to_ndarray(format="bgr24")
                exclusions = request.at(time_s)
                # Blank before thumbnail resize: excluded pixels cannot leak through interpolation.
                scan = image.copy()
                for rect in exclusions:
                    x0, y0, x1, y1 = rect.pixels(frame.width, frame.height)
                    scan[y0:y1, x0:x1] = 0
                gray = cv2.resize(cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY), (256, 144))
                roi_changed = exclusions != previous_roi
                scene = False
                changed = True
                if previous_gray is not None and not roi_changed:
                    diff = cv2.absdiff(gray, previous_gray)
                    scene = float(np.mean(diff)) >= config["scene_change_threshold"]
                    # Every tile participates; no global averaging away a local update.
                    tiles = diff.reshape(9, 16, 16, 16).mean(axis=(1, 3))
                    changed = float(tiles.max()) >= config["tile_change_threshold"]
                elapsed = float("inf") if last_candidate is None else time_s - last_candidate
                due = (
                    roi_changed
                    or scene
                    or elapsed >= config["watchdog_s"]
                    or (changed and elapsed >= config["change_min_interval_s"])
                )
                previous_gray, previous_roi = gray, exclusions
                timing["conversion_and_scan_s"] += perf_counter() - phase
                if due:
                    phase = perf_counter()
                    obs, evidence = engine.read(
                        image,
                        exclusions,
                        frame.pts,
                        {"num": tb.numerator, "den": tb.denominator},
                        time_s,
                    )
                    timing["ocr_s"] += perf_counter() - phase
                    counters["candidate_frames"] += 1
                    counters["observations"] += len(obs)
                    if last_candidate is not None:
                        report["max_sample_gap_s"] = max(
                            report.get("max_sample_gap_s", 0),
                            time_s - last_candidate,
                        )
                    last_candidate = time_s
                    phase = perf_counter()
                    for row in obs:
                        raw.write(json.dumps(observation_record(row), ensure_ascii=False) + "\n")
                    assignments = tracker.update(time_s, obs, cut=scene or roi_changed)
                    for event_id, row, save in assignments:
                        if save:
                            destination = output / "evidence" / f"{event_id}.png"
                            ok, encoded = cv2.imencode(".png", evidence[row.id])
                            if not ok:
                                raise RuntimeError("Evidence image encoding failed")
                            destination.write_bytes(encoded.tobytes())
                    timing["events_and_evidence_s"] += perf_counter() - phase
                if perf_counter() - last_notify >= 1:
                    notify("ANALYZING", time_s)
                    last_notify = perf_counter()
        if not counters["analyzed_frames"]:
            raise ValueError("No frames in the requested interval")
        tracker.boundary(request.end_s, "ANALYSIS_END")
        report["last_decoded_time_s"] = last_time
        report["decoded_tail_end_s"] = last_frame_end
        report["interval_coverage"] = (
            "DECODED"
            if reached_end or (last_frame_end is not None and last_frame_end >= request.end_s)
            else "NEEDS_REVIEW"
        )
        if report["interval_coverage"] == "NEEDS_REVIEW":
            report["coverage_issue"] = (
                "EOF tail coverage unknown or shorter than requested; inspect VFR frame durations"
            )
        report["status"] = "COMPLETED_UNVERIFIED"
    except Cancelled as exc:
        report["status"] = "CANCELLED"
        report["error"] = str(exc)
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if container:
            container.close()
        if engine:
            try:
                report["ocr_execution"] = engine.finish()
            except Exception as exc:
                report["profile_error"] = str(exc)
                if report["status"] == "COMPLETED_UNVERIFIED":
                    report["status"] = "FAILED"
                    report["error"] = "OCR provider trace could not be finalized"
        events = tracker.snapshot()
        report["counters"] = dict(counters, events=len(events))
        report["timings"] = dict(timing, total_wall_s=perf_counter() - started)
        report["rtf"] = report["timings"]["total_wall_s"] / (request.end_s - request.start_s)
        report["api_cost"] = 0
        write_json(
            output / "events.json",
            {"schema_version": 1, "roi_hash": request.roi_hash, "events": events},
        )
        write_json(output / "run-report.json", report)
        lines = [
            "# VNLE discovery run",
            f"Status: {report['status']}",
            f"Wall: {report['timings']['total_wall_s']:.3f}s; events: {len(events)}",
            "Quality and complete VNLE SLA: unverified.",
            "Subtitle exclusion: EXCLUDED_BY_USER before both OCR models.",
        ]
        if report["error"]:
            lines.append(f"Error: {report['error']}")
        (output / "run-report.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
        notify(report["status"], last_time or 0)
    return report
