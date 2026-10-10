"""Pinned RapidOCR adapter with explicit artifacts and profiled ORT sessions.

No SubAI dependency, model download, or automatic provider/model replacement.
Native RapidOCR handles detector preprocessing and batched crop recognition.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from collections import Counter
from pathlib import Path
from time import perf_counter

from .domain import Observation, allowed_tiles
from .media import file_sha256


class AttrMap(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


ALLOWED_FAMILIES = ("PP-OCRv6-small", "Chinese-PPOCRv4-small")


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("family") not in ALLOWED_FAMILIES:
        raise ValueError(
            f"Expected an explicit model manifest v1 with family in {ALLOWED_FAMILIES}"
        )
    resolved = {}
    for name in ("detector", "recognizer"):
        row = data[name]
        artifact = (path.parent / row["path"]).resolve()
        expected = row.get("sha256", "")
        if len(expected) != 64 or file_sha256(artifact) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
        if row.get("license") != "Apache-2.0" or not row.get("source"):
            raise ValueError(f"Artifact provenance missing: {name}")
        resolved[name] = dict(row, path=str(artifact))
    if data.get("dictionary"):
        row = data["dictionary"]
        artifact = (path.parent / row["path"]).resolve()
        if file_sha256(artifact) != row["sha256"]:
            raise ValueError("Dictionary hash mismatch")
        resolved["dictionary"] = dict(row, path=str(artifact))
    return {"schema_version": 1, "family": data["family"], **resolved}


class ChineseDetector:
    """Dedicated Chinese-script Text Detector backend.

    Wraps the Chinese-targeted DBNet detection model (e.g. ch_PP-OCRv4)
    with standardized bounding box outputs compatible with EventBuilder.
    """

    def __init__(self, cfg):
        from rapidocr.ch_ppocr_det import TextDetector

        self._inner = TextDetector(cfg)

    def __call__(self, img):
        return self._inner(img)


class RapidAdapter:
    def __init__(self, manifest: dict, config: dict, output: Path):
        if importlib.metadata.version("rapidocr") != "3.10.0":
            raise ValueError("This adapter requires RapidOCR 3.10.0; API compatibility is frozen")
        import cv2
        import numpy as np
        import onnxruntime as ort
        from rapidocr import RapidOCR
        from rapidocr.ch_ppocr_det import TextDetector
        from rapidocr.ch_ppocr_rec import TextRecognizer
        from rapidocr.utils.process_img import get_rotate_crop_image

        self.cv2, self.np = cv2, np
        self.crop = get_rotate_crop_image
        self.sessions = {}
        self.config = config
        self.manifest = manifest
        self.counter = 0
        self.timings = Counter()
        self.calls = Counter()
        self.batch_size = config["recognition_batch"]
        self.pending_crops = []
        self.pending_frames = []
        self.dll_directories = []
        self.runtime_dll_paths = []
        provider = config["provider"]
        if provider == "CUDAExecutionProvider":
            if os.name == "nt":
                # cuDNN dynamically loads secondary engines after ORT preload.
                # Keep wheel directories searchable for this process's lifetime.
                nvidia = Path(ort.__file__).parent.parent / "nvidia"
                for folder in sorted(nvidia.glob("*/bin")):
                    if folder.is_dir():
                        self.runtime_dll_paths.append(str(folder.resolve()))
                        self.dll_directories.append(os.add_dll_directory(str(folder.resolve())))
                os.environ["PATH"] = os.pathsep.join(
                    [*self.runtime_dll_paths, os.environ.get("PATH", "")]
                )
            # Search the explicitly installed NVIDIA runtime wheels, not SubAI DLLs.
            ort.preload_dlls(directory="")
        if provider not in ort.get_available_providers():
            raise ValueError(f"Requested provider is unavailable: {provider}")
        if provider not in ("CUDAExecutionProvider", "CPUExecutionProvider"):
            raise ValueError("Prototype supports explicit CUDA or explicit CPU only")
        cv2.setNumThreads(config["native_threads"])
        self.available_providers = ort.get_available_providers()
        for name in ("detector", "recognizer"):
            options = ort.SessionOptions()
            options.intra_op_num_threads = config["native_threads"]
            options.inter_op_num_threads = 1
            options.enable_profiling = True
            options.profile_file_prefix = str(output / f"ort-{name}")
            provider_options = {"device_id": config["device_id"]}
            if provider == "CUDAExecutionProvider":
                provider_options["cudnn_conv_algo_search"] = config["cudnn_conv_algo_search"]
            providers = [(provider, provider_options)]
            if provider == "CUDAExecutionProvider":
                # CPU partitions are observable, not a second model or hidden retry.
                providers.append(("CPUExecutionProvider", {}))
            session = ort.InferenceSession(
                manifest[name]["path"],
                sess_options=options,
                providers=providers,
            )
            session.disable_fallback()
            if session.get_providers()[0] != provider:
                raise ValueError(f"ORT did not activate {provider} for {name}")
            self.sessions[name] = session

        # RapidOCR constructor only parses its configuration. Own sessions bypass
        # its downloader and warning-only provider fallback behavior.
        engine = RapidOCR(
            params={
                "Global.use_cls": False,
                "Global.use_preprocess_img": False,
                "Global.use_vertical_padding": False,
                "Det.model_path": manifest["detector"]["path"],
                "Rec.model_path": manifest["recognizer"]["path"],
                "Rec.rec_batch_num": self.batch_size,
                "Det.limit_side_len": config["detector_side"],
                "Det.limit_type": "max",
            }
        )
        det_cfg = AttrMap(dict(engine.cfg.Det))
        det_cfg["session"] = self.sessions["detector"]
        rec_cfg = AttrMap(dict(engine.cfg.Rec))
        rec_cfg["session"] = self.sessions["recognizer"]
        rec_cfg["font_path"] = engine.cfg.Global.font_path
        if "dictionary" in manifest:
            rec_cfg["rec_keys_path"] = manifest["dictionary"]["path"]
        elif "character" not in self.sessions["recognizer"].get_modelmeta().custom_metadata_map:
            raise ValueError(
                "Recognizer has no embedded dictionary; provide a hash-pinned dictionary"
            )

        detector_backend = config.get("detector_backend", "ppocr")
        if detector_backend == "chinese_detector":
            self.detector = ChineseDetector(det_cfg)
        else:
            self.detector = TextDetector(det_cfg)

        if config["stable_ocr_shapes"]:
            canvas_width = config["recognition_canvas_width"]
            calls = self.calls

            class StableRecognizer(TextRecognizer):
                def resize_norm_img(self, img, max_wh_ratio):
                    # Native resize is unchanged. Add normalized blank pixels only.
                    normalized = super().resize_norm_img(img, max_wh_ratio)
                    width = max(canvas_width, math.ceil(normalized.shape[2] / 32) * 32)
                    if width > canvas_width:
                        calls["recognition_canvas_overflow_images"] += 1
                    return np.pad(normalized, ((0, 0), (0, 0), (0, width - normalized.shape[2])))

            self.recognizer = StableRecognizer(rec_cfg)
        else:
            self.recognizer = TextRecognizer(rec_cfg)

        from .cache import TemporalROICache
        self.cache = TemporalROICache(
            similarity_threshold=config.get("cache_similarity_threshold", 0.82),
            chinese_only=config.get("chinese_only", True),
        )
        self.last_detector_time = -999.0

    def _next_id(self) -> int:
        self.counter += 1
        return self.counter

    @staticmethod
    def _box_overlap(poly_a, poly_b) -> float:
        a_x0 = min(p[0] for p in poly_a)
        a_y0 = min(p[1] for p in poly_a)
        a_x1 = max(p[0] for p in poly_a)
        a_y1 = max(p[1] for p in poly_a)
        b_x0 = float(poly_b[:, 0].min())
        b_y0 = float(poly_b[:, 1].min())
        b_x1 = float(poly_b[:, 0].max())
        b_y1 = float(poly_b[:, 1].max())
        ix0, iy0 = max(a_x0, b_x0), max(a_y0, b_y0)
        ix1, iy1 = min(a_x1, b_x1), min(a_y1, b_y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        union = (a_x1 - a_x0) * (a_y1 - a_y0) + (b_x1 - b_x0) * (b_y1 - b_y0) - inter
        return inter / union if union > 0 else 0.0

    def feed(
        self,
        image,
        exclusions,
        pts: int,
        time_base: dict,
        time_s: float,
        cut: bool = False,
        diff: np.ndarray | None = None,
    ) -> list[tuple[float, list[Observation], dict, bool]]:
        from rapidocr.ch_ppocr_rec import TextRecInput

        height, width = image.shape[:2]
        detector_input = image.copy()
        for rect in exclusions:
            x0, y0, x1, y1 = rect.pixels(width, height)
            detector_input[y0:y1, x0:x1] = 0

        if not hasattr(self, "cache"):
            from .cache import TemporalROICache
            self.cache = TemporalROICache(
                similarity_threshold=getattr(self, "config", {}).get("cache_similarity_threshold", 0.82),
                chinese_only=getattr(self, "config", {}).get("chinese_only", False),
            )
        if not hasattr(self, "last_detector_time"):
            self.last_detector_time = -999.0
        if cut:
            self.cache.reset()

        use_cache = self.config.get("enable_temporal_cache", True)
        cached_obs, cached_ev = [], {}
        all_reliable = True

        if use_cache:
            cached_obs, cached_ev, all_reliable = self.cache.track_and_emit(
                detector_input, exclusions, pts, time_base, time_s, self._next_id
            )

        enable_motion_gating = self.config.get("enable_motion_gated_det", True)
        elapsed_since_det = time_s - self.last_detector_time
        watchdog_s = self.config.get("detector_watchdog_s", 0.5)

        has_ext_motion = (
            self.cache.has_external_motion(
                diff,
                (width, height),
                exclusions,
                threshold=self.config.get("tile_change_threshold", 8.0) * 1.5,
                min_motion_pixels=25,
            )
            if diff is not None
            else True
        )

        need_detector = (
            cut
            or not self.cache.active_rois
            or not all_reliable
            or (
                not enable_motion_gating
                or has_ext_motion
                or elapsed_since_det >= watchdog_s
            )
        )

        candidate_boxes = []
        frame_cached_obs = list(cached_obs)
        frame_cached_ev = dict(cached_ev)

        if not need_detector and use_cache:
            self.calls["detector_skipped_frames"] += 1
        else:
            self.last_detector_time = time_s
            started = perf_counter()
            detection = self.detector(detector_input)
            self.timings["detection_s"] += perf_counter() - started
            self.calls["detector_frames"] += 1
            self.calls["detector_tiles"] += 1

            if detection.boxes is not None:
                for box in detection.boxes:
                    local = box.copy()
                    bx0 = float(self.np.min(local[:, 0]))
                    by0 = float(self.np.min(local[:, 1]))
                    bx1 = float(self.np.max(local[:, 0]))
                    by1 = float(self.np.max(local[:, 1]))
                    if bx1 <= bx0 or by1 <= by0 or (bx1 - bx0) < 1 or (by1 - by0) < 1:
                        self.calls["degenerate_proposals"] += 1
                        continue
                    local[:, 0] = self.np.clip(local[:, 0], 0, width - 1)
                    local[:, 1] = self.np.clip(local[:, 1], 0, height - 1)

                    overlaps_exclusion = False
                    for rect in exclusions:
                        rx0, ry0, rx1, ry1 = rect.pixels(width, height)
                        ix0, iy0 = max(bx0, rx0), max(by0, ry0)
                        ix1, iy1 = min(bx1, rx1), min(by1, ry1)
                        if ix0 < ix1 and iy0 < iy1:
                            inter_area = (ix1 - ix0) * (iy1 - iy0)
                            box_area = (bx1 - bx0) * (by1 - by0)
                            if box_area > 0 and (inter_area / box_area) > 0.05:
                                overlaps_exclusion = True
                                break
                    if overlaps_exclusion:
                        continue

                    # Check if this box matches an already confirmed cached ROI
                    matched_cache_obs, matched_cache_ev = (
                        self.cache.match_box_with_cache(
                            local, detector_input, pts, time_base, time_s, self._next_id
                        )
                        if use_cache
                        else (None, None)
                    )

                    if matched_cache_obs is not None:
                        already_emitted = any(
                            self._box_overlap(obs.polygon, local) >= 0.5
                            for obs in frame_cached_obs
                        )
                        if not already_emitted:
                            frame_cached_obs.append(matched_cache_obs)
                            frame_cached_ev[matched_cache_obs.id] = matched_cache_ev
                            self.calls["cache_recognition_hits"] += 1
                        continue

                    crop = self.crop(detector_input, local)
                    polygon = [[float(p[0]), float(p[1])] for p in local]
                    edge = bool(
                        self.np.any(local[:, 0] < 2)
                        or self.np.any(local[:, 1] < 2)
                        or self.np.any(local[:, 0] > width - 3)
                        or self.np.any(local[:, 1] > height - 3)
                    )
                    candidate_boxes.append((crop, polygon, (0, 0, width, height), edge))
                    self.calls["cache_recognition_misses"] += 1

        frame_obj = {
            "pts": pts,
            "time_base": time_base,
            "time_s": time_s,
            "cut": cut,
            "total_boxes": len(candidate_boxes) + len(frame_cached_obs),
            "observations": list(frame_cached_obs),
            "evidence": dict(frame_cached_ev),
            "detector_input": detector_input,
        }
        self.pending_frames.append(frame_obj)
        for crop, polygon, tile, edge in candidate_boxes:
            self.pending_crops.append(
                (frame_obj, crop, polygon, tile, edge, pts, time_base, time_s)
            )

        while len(self.pending_crops) >= self.batch_size:
            self._run_recognition_batch(self.batch_size)

        return self._drain_completed_frames()

    def _drain_completed_frames(self) -> list[tuple[float, list[Observation], dict, bool]]:
        completed = []
        while (
            self.pending_frames
            and len(self.pending_frames[0]["observations"])
            == self.pending_frames[0]["total_boxes"]
        ):
            head = self.pending_frames.pop(0)
            head.pop("detector_input", None)
            completed.append(
                (head["time_s"], head["observations"], head["evidence"], head["cut"])
            )
        return completed

    def _run_recognition_batch(self, count: int):
        from rapidocr.ch_ppocr_rec import TextRecInput

        batch_items = [self.pending_crops.pop(0) for _ in range(count)]
        images = [item[1] for item in batch_items]
        if self.config["stable_ocr_shapes"]:
            extra = self.batch_size - len(images)
            if extra > 0:
                images.extend(
                    self.np.full((48, 1, 3), 127, dtype=self.np.uint8) for _ in range(extra)
                )
                self.calls["recognition_dummy_rows"] += extra
        started = perf_counter()
        result = self.recognizer(TextRecInput(img=images, return_word_box=False))
        self.timings["recognition_s"] += perf_counter() - started
        self.calls["recognition_batches"] += 1
        if result.txts is None or len(result.txts) != len(images):
            raise RuntimeError("Recognition output length differs from crop batch")
        for i, (f_obj, crop, polygon, tile, edge, pts, time_base, time_s) in enumerate(
            batch_items
        ):
            self.counter += 1
            score = float(result.scores[i]) if result.scores is not None else None
            if score is not None and not math.isfinite(score):
                score = None
            obs = Observation(
                self.counter,
                pts,
                time_base,
                time_s,
                polygon,
                result.txts[i],
                score,
                hashlib.sha256(crop.tobytes()).hexdigest(),
                tile,
                edge,
            )
            f_obj["observations"].append(obs)
            f_obj["evidence"][obs.id] = crop
            if (
                self.config.get("enable_temporal_cache", True)
                and "detector_input" in f_obj
            ):
                self.cache.register_observations([obs], f_obj["detector_input"])

    def flush_remaining(self) -> list[tuple[float, list[Observation], dict, bool]]:
        while self.pending_crops:
            cur_count = min(len(self.pending_crops), self.batch_size)
            self._run_recognition_batch(cur_count)
        return self._drain_completed_frames()

    def read(
        self, image, exclusions, pts: int, time_base: dict, time_s: float
    ) -> tuple[list[Observation], dict]:
        completed = self.feed(image, exclusions, pts, time_base, time_s)
        remaining = self.flush_remaining()
        all_frames = completed + remaining
        obs = []
        ev = {}
        for _, f_obs, f_ev, _ in all_frames:
            obs.extend(f_obs)
            ev.update(f_ev)
        return obs, ev

    def finish(self) -> dict:
        profiles = {}
        for name, session in self.sessions.items():
            path = Path(session.end_profiling())
            trace = json.loads(path.read_text(encoding="utf-8"))
            nodes = Counter()
            durations = Counter()
            for item in trace:
                provider = item.get("args", {}).get("provider")
                if provider:
                    nodes[provider] += 1
                    durations[provider] += item.get("dur", 0) / 1_000_000
            profiles[name] = {
                "session_providers": session.get_providers(),
                "session_provider_options": session.get_provider_options(),
                "trace": path.name,
                "node_executions_by_provider": dict(nodes),
                "node_duration_s": dict(durations),
                "gpu_only": None if not nodes else set(nodes) == {"CUDAExecutionProvider"},
            }
        return {
            "requested_provider": self.config["provider"],
            "runtime_dll_paths": self.runtime_dll_paths,
            "available_providers": self.available_providers,
            "profiles": profiles,
            "timings": dict(self.timings),
            "calls": dict(self.calls),
            "cache_stats": dict(self.cache.stats),
        }
