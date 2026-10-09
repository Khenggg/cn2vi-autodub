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


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("family") != "PP-OCRv6-small":
        raise ValueError("Expected an explicit PP-OCRv6-small model manifest v1")
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
        self.detector = TextDetector(det_cfg)
        self.recognizer = TextRecognizer(rec_cfg)

    def read(self, image, exclusions, pts: int, time_base: dict, time_s: float):
        from rapidocr.ch_ppocr_rec import TextRecInput

        observations = []
        evidence = {}
        pending = []
        pixels = 0

        def flush():
            nonlocal pixels
            if not pending:
                return
            started = perf_counter()
            result = self.recognizer(
                TextRecInput(img=[p[0] for p in pending], return_word_box=False)
            )
            self.timings["recognition_s"] += perf_counter() - started
            self.calls["recognition_batches"] += 1
            if result.txts is None or len(result.txts) != len(pending):
                raise RuntimeError("Recognition output length differs from crop batch")
            for i, (crop, polygon, tile, edge) in enumerate(pending):
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
                observations.append(obs)
                evidence[obs.id] = crop
            pending.clear()
            pixels = 0

        height, width = image.shape[:2]
        for x0, y0, x1, y1 in allowed_tiles(width, height, exclusions):
            tile = (x0, y0, x1, y1)
            tile_img = self.np.ascontiguousarray(image[y0:y1, x0:x1])
            started = perf_counter()
            detection = self.detector(tile_img)
            self.timings["detection_s"] += perf_counter() - started
            self.calls["detector_tiles"] += 1
            if detection.boxes is None:
                continue
            for box in detection.boxes:
                local = box.copy()
                local[:, 0] = self.np.clip(local[:, 0], 0, x1 - x0 - 1)
                local[:, 1] = self.np.clip(local[:, 1], 0, y1 - y0 - 1)
                if self.np.ptp(local[:, 0]) < 1 or self.np.ptp(local[:, 1]) < 1:
                    self.calls["degenerate_proposals"] += 1
                    continue
                crop = self.crop(tile_img, local)
                polygon = [[float(p[0]) + x0, float(p[1]) + y0] for p in local]
                edge = bool(
                    self.np.any(local[:, 0] < 2)
                    or self.np.any(local[:, 1] < 2)
                    or self.np.any(local[:, 0] > x1 - x0 - 3)
                    or self.np.any(local[:, 1] > y1 - y0 - 3)
                )
                if pending and (
                    len(pending) >= self.batch_size or pixels + crop.nbytes > 8 * 1024**2
                ):
                    flush()
                pending.append((crop, polygon, tile, edge))
                pixels += crop.nbytes
        flush()
        return observations, evidence

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
        }
