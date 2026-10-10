"""Sequential PTS analysis, one OCR owner, bounded frame lifetime, durable evidence."""

from __future__ import annotations

import importlib.metadata
import json
import os
import queue
import subprocess
import threading
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from time import perf_counter

from .domain import Request, digest
from .events import EventBuilder, observation_record
from .cache import contains_chinese
from .media import file_sha256
from .ocr import RapidAdapter, load_manifest

DEFAULT_CONFIG = {
    "schema_version": 1,
    "provider": "CUDAExecutionProvider",
    "device_id": 0,
    "cudnn_conv_algo_search": "HEURISTIC",
    "native_threads": 2,
    "recognition_batch": 8,
    "stable_ocr_shapes": True,
    "recognition_canvas_width": 1536,
    "detector_side": 960,
    "detector_backend": "ppocr",
    "enable_temporal_cache": True,
    "enable_motion_gated_det": True,
    "chinese_only": False,
    "detector_watchdog_s": 0.5,
    "cache_similarity_threshold": 0.82,
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
    merged = dict(DEFAULT_CONFIG, **data)
    if set(merged) != set(DEFAULT_CONFIG):
        raise ValueError("Config must have exactly the documented fields")
    result = dict(merged)
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
    if type(result["stable_ocr_shapes"]) is not bool:
        raise ValueError("stable_ocr_shapes must be a boolean")
    canvas_width = result["recognition_canvas_width"]
    if type(canvas_width) is not int or not 320 <= canvas_width <= 4096 or canvas_width % 32:
        raise ValueError("Recognition canvas width must be 320..4096 and divisible by 32")
    if result["detector_backend"] not in ("ppocr", "chinese_detector"):
        raise ValueError("detector_backend must be either 'ppocr' or 'chinese_detector'")
    for key in ("enable_temporal_cache", "enable_motion_gated_det", "chinese_only"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    for key, lower, upper in (
        ("detector_watchdog_s", 0.05, 5.0),
        ("cache_similarity_threshold", 0.5, 1.0),
    ):
        value = float(result[key])
        if not lower <= value <= upper:
            raise ValueError(f"Invalid config field: {key}")
        result[key] = value
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
    producer_thread = None
    ev_queue = None
    ev_thread = None
    stop_producer = threading.Event()
    stop_ev = threading.Event()
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
        decode_thread_type = os.environ.get("VNLE_DECODE_THREAD_TYPE", "").strip().upper()
        if decode_thread_type in ("AUTO", "FRAME", "SLICE"):
            stream.codec_context.thread_type = decode_thread_type
        origin = Fraction(media["origin"]["num"], media["origin"]["den"])
        last_notify = 0.0
        prof = getattr(engine, "profiler", None)
        prof_enabled = bool(prof is not None and getattr(prof, "enabled", False))
        queue_capacity = int(os.environ.get("VNLE_PIPELINE_QUEUE", "4"))
        use_async_ev = (
            queue_capacity > 0
            and os.environ.get("VNLE_ASYNC_EVIDENCE", "1").strip() not in ("0", "false", "no")
        )
        use_prod_det_prep = (
            queue_capacity > 0
            and os.environ.get("VNLE_PRODUCER_DET_PREP", "1").strip() not in ("0", "false", "no")
        )
        ev_queue: queue.Queue | None = queue.Queue(maxsize=64) if use_async_ev else None
        ev_errors: list[BaseException] = []
        ev_thread: threading.Thread | None = None

        if ev_queue is not None:
            def _ev_worker():
                while not stop_ev.is_set():
                    try:
                        item = ev_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is None:
                        ev_queue.task_done()
                        break
                    destination, crop_img = item
                    try:
                        t_enc0 = perf_counter()
                        ok, encoded = cv2.imencode(".png", crop_img)
                        if prof_enabled:
                            prof.record("evidence_png_encode", perf_counter() - t_enc0)
                        if not ok:
                            raise RuntimeError("Evidence image encoding failed")
                        t_w0 = perf_counter()
                        destination.write_bytes(encoded.tobytes())
                        if prof_enabled:
                            prof.record("evidence_disk_write", perf_counter() - t_w0)
                    except BaseException as exc:
                        ev_errors.append(exc)
                    finally:
                        ev_queue.task_done()

            ev_thread = threading.Thread(target=_ev_worker, name="vnle-evidence-writer", daemon=True)
            ev_thread.start()

        def _write_completed(raw_fp, completed_frames):
            phase_ev = perf_counter()
            for f_time_s, obs, evidence, cut in completed_frames:
                t_ev0 = perf_counter()
                counters["observations"] += len(obs)
                for row in obs:
                    raw_fp.write(json.dumps(observation_record(row), ensure_ascii=False) + "\n")
                target_obs = (
                    [row for row in obs if contains_chinese(row.text)]
                    if config.get("chinese_only", True)
                    else obs
                )
                assignments = tracker.update(f_time_s, target_obs, cut=cut)
                if prof_enabled:
                    prof.record("event_builder", perf_counter() - t_ev0)
                for event_id, row, save in assignments:
                    if save:
                        if ev_errors:
                            raise ev_errors[0]
                        destination = output / "evidence" / f"{event_id}.png"
                        if ev_queue is not None:
                            while not (cancel and cancel.is_set()):
                                try:
                                    ev_queue.put((destination, evidence[row.id]), timeout=0.1)
                                    break
                                except queue.Full:
                                    continue
                            if cancel and cancel.is_set():
                                raise Cancelled("Analysis cancelled; partial artifacts retained")
                        else:
                            t_enc0 = perf_counter()
                            ok, encoded = cv2.imencode(".png", evidence[row.id])
                            if prof_enabled:
                                prof.record("evidence_png_encode", perf_counter() - t_enc0)
                            if not ok:
                                raise RuntimeError("Evidence image encoding failed")
                            t_w0 = perf_counter()
                            destination.write_bytes(encoded.tobytes())
                            if prof_enabled:
                                prof.record("evidence_disk_write", perf_counter() - t_w0)
            timing["events_and_evidence_s"] += perf_counter() - phase_ev

        with (output / "observations.jsonl").open("w", encoding="utf-8") as raw:
            if queue_capacity <= 0:
                previous_gray = None
                previous_roi = None
                last_candidate = None
                frames = iter(container.decode(stream))
                while True:
                    if cancel and cancel.is_set():
                        raise Cancelled("Analysis cancelled; partial artifacts retained")
                    phase = perf_counter()
                    try:
                        frame = next(frames)
                    except StopIteration:
                        break
                    dt_decode = perf_counter() - phase
                    timing["decode_s"] += dt_decode
                    if prof_enabled:
                        prof.record("frame_decode", dt_decode)
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
                    t_conv0 = perf_counter()
                    image = frame.to_ndarray(format="bgr24")
                    if prof_enabled:
                        prof.record("frame_conversion", perf_counter() - t_conv0)
                    t_scan0 = perf_counter()
                    exclusions = request.at(time_s)
                    for rect in exclusions:
                        x0, y0, x1, y1 = rect.pixels(frame.width, frame.height)
                        image[y0:y1, x0:x1] = 0
                    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (256, 144))
                    roi_changed = exclusions != previous_roi
                    scene = False
                    changed = True
                    diff = None
                    if previous_gray is not None and not roi_changed:
                        diff = cv2.absdiff(gray, previous_gray)
                        scene = float(np.mean(diff)) >= config["scene_change_threshold"]
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
                    if prof_enabled:
                        prof.record("frame_scan", perf_counter() - t_scan0)
                    timing["conversion_and_scan_s"] += perf_counter() - phase
                    if due:
                        phase = perf_counter()
                        if hasattr(engine, "feed"):
                            completed_frames = engine.feed(
                                image,
                                exclusions,
                                frame.pts,
                                {"num": tb.numerator, "den": tb.denominator},
                                time_s,
                                cut=scene or roi_changed,
                                diff=diff,
                                image_premasked=True,
                            )
                        else:
                            obs, evidence = engine.read(
                                image,
                                exclusions,
                                frame.pts,
                                {"num": tb.numerator, "den": tb.denominator},
                                time_s,
                            )
                            completed_frames = [(time_s, obs, evidence, scene or roi_changed)]
                        timing["ocr_s"] += perf_counter() - phase
                        counters["candidate_frames"] += 1
                        if last_candidate is not None:
                            report["max_sample_gap_s"] = max(
                                report.get("max_sample_gap_s", 0),
                                time_s - last_candidate,
                            )
                        last_candidate = time_s
                        _write_completed(raw, completed_frames)
                    if perf_counter() - last_notify >= 1:
                        notify("ANALYZING", time_s)
                        last_notify = perf_counter()
                report["pipeline_parallel"] = {
                    "mode": "sequential",
                    "queue_capacity": 0,
                    "decode_thread_type": decode_thread_type or "DEFAULT",
                }
            else:
                cand_queue: queue.Queue = queue.Queue(maxsize=queue_capacity)
                prod_stats = {
                    "decode_s": 0.0,
                    "conversion_and_scan_s": 0.0,
                    "det_prep_s": 0.0,
                    "producer_blocked_s": 0.0,
                    "decoded_frames": 0,
                    "analyzed_frames": 0,
                    "last_time": None,
                    "last_frame_end": None,
                    "reached_end": False,
                    "max_sample_gap_s": 0.0,
                }
                consumer_idle_s = 0.0
                queue_occupancy_samples: list[int] = []
                det_prep_fn = (
                    getattr(getattr(engine, "detector", None), "prepare_tensor", None)
                    if use_prod_det_prep
                    else None
                )
                num_slots = queue_capacity + 4

                def _put_with_wait(item):
                    while not stop_producer.is_set():
                        if cancel and cancel.is_set():
                            return False
                        t_put0 = perf_counter()
                        try:
                            cand_queue.put(item, timeout=0.05)
                            prod_stats["producer_blocked_s"] += perf_counter() - t_put0
                            return True
                        except queue.Full:
                            prod_stats["producer_blocked_s"] += perf_counter() - t_put0
                    return False

                def _producer():
                    p_prev_gray = None
                    p_prev_roi = None
                    p_last_cand = None
                    p_slot = 0
                    try:
                        p_frames = iter(container.decode(stream))
                        while not stop_producer.is_set():
                            if cancel and cancel.is_set():
                                _put_with_wait(
                                    ("ERR", Cancelled("Analysis cancelled; partial artifacts retained"))
                                )
                                return
                            t_dec0 = perf_counter()
                            try:
                                frame = next(p_frames)
                            except StopIteration:
                                break
                            dt_decode = perf_counter() - t_dec0
                            prod_stats["decode_s"] += dt_decode
                            if prof_enabled:
                                prof.record("frame_decode", dt_decode)
                            prod_stats["decoded_frames"] += 1
                            if frame.pts is None or frame.time_base is None:
                                raise ValueError(
                                    "Frame lacks presentation timestamp; index/fps fallback is disabled"
                                )
                            tb = Fraction(frame.time_base)
                            time_s = float(frame.pts * tb - origin)
                            if prod_stats["last_time"] is not None and time_s < prod_stats["last_time"]:
                                raise ValueError("Non-monotonic decoded PTS")
                            prod_stats["last_time"] = time_s
                            duration_pts = getattr(frame, "duration", None)
                            if duration_pts and duration_pts > 0:
                                prod_stats["last_frame_end"] = time_s + float(duration_pts * tb)
                            else:
                                prod_stats["last_frame_end"] = None
                            if time_s < request.start_s:
                                continue
                            if time_s >= request.end_s:
                                prod_stats["reached_end"] = True
                                break
                            if frame.width != media["width"] or frame.height != media["height"]:
                                raise ValueError("Source raster dimensions changed during analysis")
                            prod_stats["analyzed_frames"] += 1
                            t_cs0 = perf_counter()
                            t_conv0 = perf_counter()
                            image = frame.to_ndarray(format="bgr24")
                            if prof_enabled:
                                prof.record("frame_conversion", perf_counter() - t_conv0)
                            t_scan0 = perf_counter()
                            exclusions = request.at(time_s)
                            for rect in exclusions:
                                x0, y0, x1, y1 = rect.pixels(frame.width, frame.height)
                                image[y0:y1, x0:x1] = 0
                            gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (256, 144))
                            roi_changed = exclusions != p_prev_roi
                            scene = False
                            changed = True
                            diff = None
                            if p_prev_gray is not None and not roi_changed:
                                diff = cv2.absdiff(gray, p_prev_gray)
                                scene = float(np.mean(diff)) >= config["scene_change_threshold"]
                                tiles = diff.reshape(9, 16, 16, 16).mean(axis=(1, 3))
                                changed = float(tiles.max()) >= config["tile_change_threshold"]
                            elapsed = float("inf") if p_last_cand is None else time_s - p_last_cand
                            due = (
                                roi_changed
                                or scene
                                or elapsed >= config["watchdog_s"]
                                or (changed and elapsed >= config["change_min_interval_s"])
                            )
                            p_prev_gray, p_prev_roi = gray, exclusions
                            if prof_enabled:
                                prof.record("frame_scan", perf_counter() - t_scan0)
                            prod_stats["conversion_and_scan_s"] += perf_counter() - t_cs0
                            if due:
                                if p_last_cand is not None:
                                    prod_stats["max_sample_gap_s"] = max(
                                        prod_stats["max_sample_gap_s"],
                                        time_s - p_last_cand,
                                    )
                                p_last_cand = time_s
                                det_tensor = None
                                if det_prep_fn is not None:
                                    t_dp0 = perf_counter()
                                    det_tensor = det_prep_fn(image, slot=p_slot)
                                    prod_stats["det_prep_s"] += perf_counter() - t_dp0
                                    p_slot = (p_slot + 1) % num_slots
                                payload = (
                                    image,
                                    exclusions,
                                    frame.pts,
                                    {"num": tb.numerator, "den": tb.denominator},
                                    time_s,
                                    scene or roi_changed,
                                    diff,
                                    det_tensor,
                                )
                                if not _put_with_wait(("ITEM", payload)):
                                    return
                        _put_with_wait(("EOF", None))
                    except BaseException as exc:
                        _put_with_wait(("ERR", exc))

                producer_thread = threading.Thread(
                    target=_producer, name="vnle-decode-producer", daemon=True
                )
                producer_thread.start()

                while True:
                    if cancel and cancel.is_set():
                        stop_producer.set()
                        raise Cancelled("Analysis cancelled; partial artifacts retained")
                    queue_occupancy_samples.append(cand_queue.qsize())
                    t_get0 = perf_counter()
                    try:
                        kind, payload = cand_queue.get(timeout=0.05)
                        consumer_idle_s += perf_counter() - t_get0
                    except queue.Empty:
                        consumer_idle_s += perf_counter() - t_get0
                        continue
                    if kind == "EOF":
                        break
                    if kind == "ERR":
                        stop_producer.set()
                        raise payload
                    image, exclusions, pts, tb_dict, time_s, cut_flag, diff, det_tensor = payload
                    phase = perf_counter()
                    if hasattr(engine, "feed"):
                        completed_frames = engine.feed(
                            image,
                            exclusions,
                            pts,
                            tb_dict,
                            time_s,
                            cut=cut_flag,
                            diff=diff,
                            image_premasked=True,
                            det_tensor=det_tensor,
                        )
                    else:
                        obs, evidence = engine.read(
                            image,
                            exclusions,
                            pts,
                            tb_dict,
                            time_s,
                        )
                        completed_frames = [(time_s, obs, evidence, cut_flag)]
                    timing["ocr_s"] += perf_counter() - phase
                    counters["candidate_frames"] += 1
                    _write_completed(raw, completed_frames)
                    if perf_counter() - last_notify >= 1:
                        counters["decoded_frames"] = prod_stats["decoded_frames"]
                        counters["analyzed_frames"] = prod_stats["analyzed_frames"]
                        notify("ANALYZING", time_s)
                        last_notify = perf_counter()

                producer_thread.join(timeout=5.0)
                timing["decode_s"] = prod_stats["decode_s"]
                timing["conversion_and_scan_s"] = prod_stats["conversion_and_scan_s"]
                timing["producer_det_prep_s"] = prod_stats["det_prep_s"]
                timing["producer_blocked_s"] = prod_stats["producer_blocked_s"]
                timing["consumer_idle_s"] = consumer_idle_s
                counters["decoded_frames"] = prod_stats["decoded_frames"]
                counters["analyzed_frames"] = prod_stats["analyzed_frames"]
                last_time = prod_stats["last_time"]
                last_frame_end = prod_stats["last_frame_end"]
                reached_end = prod_stats["reached_end"]
                if prod_stats["max_sample_gap_s"] > 0:
                    report["max_sample_gap_s"] = prod_stats["max_sample_gap_s"]
                occ_arr = np.array(queue_occupancy_samples, dtype=np.float64) if queue_occupancy_samples else np.zeros(1)
                report["pipeline_parallel"] = {
                    "mode": "producer_consumer",
                    "queue_capacity": queue_capacity,
                    "decode_thread_type": decode_thread_type or "DEFAULT",
                    "producer_det_prep": bool(det_prep_fn is not None),
                    "async_evidence": bool(ev_queue is not None),
                    "producer_decode_s": round(prod_stats["decode_s"], 6),
                    "producer_scan_s": round(prod_stats["conversion_and_scan_s"], 6),
                    "producer_det_prep_s": round(prod_stats["det_prep_s"], 6),
                    "producer_active_s": round(
                        prod_stats["decode_s"]
                        + prod_stats["conversion_and_scan_s"]
                        + prod_stats["det_prep_s"],
                        6,
                    ),
                    "producer_blocked_s": round(prod_stats["producer_blocked_s"], 6),
                    "consumer_idle_s": round(consumer_idle_s, 6),
                    "queue_occupancy_mean": round(float(occ_arr.mean()), 3),
                    "queue_occupancy_p50": round(float(np.percentile(occ_arr, 50)), 3),
                    "queue_occupancy_p95": round(float(np.percentile(occ_arr, 95)), 3),
                    "queue_occupancy_max": int(occ_arr.max()),
                }

            phase = perf_counter()
            remaining_frames = engine.flush_remaining() if hasattr(engine, "flush_remaining") else []
            timing["ocr_s"] += perf_counter() - phase
            _write_completed(raw, remaining_frames)
            if ev_queue is not None and ev_thread is not None:
                phase_ev_flush = perf_counter()
                ev_queue.put(None)
                ev_thread.join(timeout=30.0)
                timing["events_and_evidence_s"] += perf_counter() - phase_ev_flush
                if ev_thread.is_alive():
                    stop_ev.set()
                    raise TimeoutError("Evidence writer timed out after 30s; queue not drained")
                if ev_errors:
                    raise ev_errors[0]
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
        stop_producer.set()
        stop_ev.set()
        if producer_thread is not None and producer_thread.is_alive():
            producer_thread.join(timeout=5.0)
        if ev_queue is not None and ev_thread is not None and ev_thread.is_alive():
            try:
                ev_queue.put_nowait(None)
            except (queue.Full, ValueError):
                pass
            ev_thread.join(timeout=5.0)
        if container:
            container.close()
        if engine:
            try:
                report["ocr_execution"] = engine.finish()
                if (
                    isinstance(report["ocr_execution"], dict)
                    and report["ocr_execution"].get("resources")
                ):
                    report["resources"] = report["ocr_execution"]["resources"]
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
