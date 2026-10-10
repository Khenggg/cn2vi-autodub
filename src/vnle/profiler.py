"""Granular non-overlapping profiler for VNLE OCR and pipeline stages.

Collects per-call latency distributions (calls, sum, mean, p50, p95, min, max),
tensor shape distributions, recognition crop width distributions, and resource
utilization without modifying site-packages or altering RapidOCR outputs.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
from collections import Counter
from time import perf_counter
from typing import Any

import cv2
import numpy as np


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_vals[low]
    weight = pos - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight


def sample_resources() -> dict[str, Any]:
    res: dict[str, Any] = {"cpu_logical_cores": os.cpu_count()}
    if os.name == "nt":
        try:
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                res["ram_working_set_mb"] = round(pmc.WorkingSetSize / (1024 * 1024), 2)
                res["ram_peak_working_set_mb"] = round(pmc.PeakWorkingSetSize / (1024 * 1024), 2)
        except Exception:
            pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        if out:
            first_line = out.splitlines()[0]
            used_str, total_str = [x.strip() for x in first_line.split(",")[:2]]
            res["vram_used_mb"] = float(used_str)
            res["vram_total_mb"] = float(total_str)
    except Exception:
        pass
    return res


class GranularProfiler:
    METRIC_NAMES = (
        # Detector leaf metrics + envelope
        "det_preprocess_resize",
        "det_preprocess_normalize",
        "det_preprocess_tensor",
        "det_runtime_call",
        "det_post_threshold",
        "det_post_contour",
        "det_post_polygon",
        "det_post_filter",
        "det_total",
        # Recognizer leaf metrics + envelope
        "rec_crop_warp",
        "rec_resize_normalize",
        "rec_padding",
        "rec_batch_build",
        "rec_runtime_call",
        "rec_ctc_decode",
        "rec_total",
        # Other pipeline stages
        "frame_decode",
        "frame_conversion",
        "frame_scan",
        "roi_tracking",
        "motion_gate",
        "event_builder",
        "evidence_png_encode",
        "evidence_disk_write",
    )

    SUB_METRICS = (
        "det_post_polygon_minbox",
        "det_post_polygon_score",
        "det_post_polygon_unclip",
    )

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.samples: dict[str, list[float]] = {k: [] for k in self.METRIC_NAMES}
        self.sub_samples: dict[str, list[float]] = {k: [] for k in self.SUB_METRICS}
        self.det_shapes: Counter[str] = Counter()
        self.rec_batch_shapes: Counter[str] = Counter()
        self.rec_batch_sizes: Counter[int] = Counter()
        self.raw_crop_widths: list[int] = []
        self.raw_crop_heights: list[int] = []
        self.resized_48h_widths: list[int] = []
        self.padded_widths: list[int] = []
        self.width_overflow_count: int = 0
        self.peak_vram_mb: float = 0.0
        self.peak_ram_mb: float = 0.0

    def record(self, name: str, duration_s: float):
        if self.enabled:
            self.samples[name].append(duration_s)

    def record_sub(self, name: str, duration_s: float):
        if self.enabled:
            self.sub_samples[name].append(duration_s)

    def record_det_shape(self, shape: tuple[int, ...]):
        if self.enabled:
            self.det_shapes[str(tuple(int(x) for x in shape))] += 1

    def record_rec_batch(self, shape: tuple[int, ...], actual_crops: int):
        if self.enabled:
            self.rec_batch_shapes[str(tuple(int(x) for x in shape))] += 1
            self.rec_batch_sizes[int(actual_crops)] += 1

    def record_crop_dimensions(self, raw_h: int, raw_w: int, resized_w: int, padded_w: int):
        if self.enabled:
            self.raw_crop_heights.append(int(raw_h))
            self.raw_crop_widths.append(int(raw_w))
            self.resized_48h_widths.append(int(resized_w))
            self.padded_widths.append(int(padded_w))

    @staticmethod
    def _summarize_series(vals_s: list[float]) -> dict[str, Any]:
        n = len(vals_s)
        if n == 0:
            return {
                "calls": 0,
                "sum_s": 0.0,
                "sum_ms": 0.0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }
        vals_ms = sorted(v * 1000.0 for v in vals_s)
        total_s = sum(vals_s)
        total_ms = total_s * 1000.0
        return {
            "calls": n,
            "sum_s": round(total_s, 6),
            "sum_ms": round(total_ms, 3),
            "mean_ms": round(total_ms / n, 4),
            "p50_ms": round(_percentile(vals_ms, 0.50), 4),
            "p95_ms": round(_percentile(vals_ms, 0.95), 4),
            "min_ms": round(vals_ms[0], 4),
            "max_ms": round(vals_ms[-1], 4),
        }

    @staticmethod
    def _summarize_widths(widths: list[int]) -> dict[str, Any]:
        n = len(widths)
        if n == 0:
            return {"count": 0}
        s = sorted(float(w) for w in widths)
        buckets = {}
        for bound in (320, 512, 768, 1024, 1536):
            cnt = sum(1 for w in widths if w <= bound)
            buckets[f"le_{bound}"] = {"count": cnt, "ratio": round(cnt / n, 4)}
        gt_1536 = sum(1 for w in widths if w > 1536)
        buckets["gt_1536"] = {"count": gt_1536, "ratio": round(gt_1536 / n, 4)}
        return {
            "count": n,
            "min": int(s[0]),
            "p50": round(_percentile(s, 0.50), 1),
            "p90": round(_percentile(s, 0.90), 1),
            "p95": round(_percentile(s, 0.95), 1),
            "max": int(s[-1]),
            "mean": round(sum(s) / n, 2),
            "cumulative_buckets": buckets,
        }

    def export(self, ort_profiles: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics = {k: self._summarize_series(v) for k, v in self.samples.items()}
        sub_metrics = {k: self._summarize_series(v) for k, v in self.sub_samples.items()}

        leaf_keys = [k for k in self.METRIC_NAMES if k not in ("det_total", "rec_total")]
        hotspots = sorted(
            ({"metric": k, **metrics[k]} for k in leaf_keys),
            key=lambda x: x["sum_s"],
            reverse=True,
        )

        total_resized = sum(self.resized_48h_widths)
        total_padded = sum(self.padded_widths)
        effective_ratio = (
            round(total_resized / total_padded, 4) if total_padded > 0 else 0.0
        )

        res = sample_resources()
        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "metrics": metrics,
            "detector_polygon_sub_metrics": sub_metrics,
            "top_hotspots_leaf": hotspots,
            "tensor_shapes": {
                "detector_input": dict(self.det_shapes),
                "recognizer_batch_input": dict(self.rec_batch_shapes),
            },
            "recognizer_batch_sizes": {str(k): v for k, v in sorted(self.rec_batch_sizes.items())},
            "recognition_crop_widths": {
                "raw_crop_width": self._summarize_widths(self.raw_crop_widths),
                "resized_at_48h_width": self._summarize_widths(self.resized_48h_widths),
                "effective_width_vs_padded_ratio": effective_ratio,
                "padding_waste_ratio": round(1.0 - effective_ratio, 4) if total_padded > 0 else 0.0,
                "width_overflow_count": self.width_overflow_count,
            },
            "ort_cuda_comparison": ort_profiles or {},
            "resources": res,
        }


def make_profiled_detector(base_detector, profiler: GranularProfiler):
    """Wrap a RapidOCR TextDetector (or ChineseDetector) instance with fast LUT preprocessing and stage profiling."""
    from rapidocr.ch_ppocr_det.utils import TextDetOutput

    inner = getattr(base_detector, "_inner", base_detector)
    inner_pre = inner.get_preprocess()
    inner_post = inner.postprocess_op

    det_luts = [
        (
            (np.arange(256, dtype=np.float32) * np.float32(inner_pre.scale) - float(inner_pre.mean[c]))
            / float(inner_pre.std[c])
        ).astype(np.float32)
        for c in range(3)
    ]
    tensor_buffers: dict[tuple[int, int], np.ndarray] = {}
    u8_buffers: dict[tuple[int, int], list[np.ndarray]] = {}

    def profiled_call(img: np.ndarray) -> TextDetOutput:
        prof_on = profiler.enabled
        t_total = perf_counter() if prof_on else 0.0
        if img is None:
            raise ValueError("img is None")

        ori_img_shape = img.shape[0], img.shape[1]

        t0 = perf_counter() if prof_on else 0.0
        resized_img = inner_pre.resize(img)
        if prof_on:
            profiler.record("det_preprocess_resize", perf_counter() - t0)
        if resized_img is None:
            if prof_on:
                profiler.record("det_total", perf_counter() - t_total)
            return TextDetOutput()

        t0 = perf_counter() if prof_on else 0.0
        rh, rw = resized_img.shape[:2]
        shape_key = (rh, rw)
        prepro_img = tensor_buffers.get(shape_key)
        if prepro_img is None:
            prepro_img = np.empty((1, 3, rh, rw), dtype=np.float32)
            tensor_buffers[shape_key] = prepro_img
        u8_chs = u8_buffers.get(shape_key)
        if u8_chs is None:
            u8_chs = [np.empty((rh, rw), dtype=np.uint8) for _ in range(3)]
            u8_buffers[shape_key] = u8_chs
        if prof_on:
            profiler.record("det_preprocess_tensor", perf_counter() - t0)
            t0 = perf_counter()

        cv2.split(resized_img, u8_chs)
        cv2.LUT(u8_chs[0], det_luts[0], dst=prepro_img[0, 0])
        cv2.LUT(u8_chs[1], det_luts[1], dst=prepro_img[0, 1])
        cv2.LUT(u8_chs[2], det_luts[2], dst=prepro_img[0, 2])
        if prof_on:
            profiler.record("det_preprocess_normalize", perf_counter() - t0)
            profiler.record_det_shape(prepro_img.shape)
            t0 = perf_counter()

        preds = inner.session(prepro_img)
        if prof_on:
            profiler.record("det_runtime_call", perf_counter() - t0)
            t0 = perf_counter()

        src_h, src_w = ori_img_shape
        pred_map = preds[:, 0, :, :]
        segmentation = pred_map > inner_post.thresh
        mask_u8 = segmentation[0].astype(np.uint8)
        if inner_post.dilation_kernel is not None:
            mask_u8 = cv2.dilate(mask_u8, inner_post.dilation_kernel)
        mask_u8 *= 255
        if prof_on:
            profiler.record("det_post_threshold", perf_counter() - t0)
            t0 = perf_counter()

        height, width = mask_u8.shape
        outs = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = outs[1] if len(outs) == 3 else outs[0]
        if prof_on:
            profiler.record("det_post_contour", perf_counter() - t0)
            t0 = perf_counter()

        num_contours = min(len(contours), inner_post.max_candidates)
        boxes_list, scores_list = [], []
        pred_single = pred_map[0]
        for index in range(num_contours):
            contour = contours[index]
            ts = perf_counter() if prof_on else 0.0
            points, sside = inner_post.get_mini_boxes(contour)
            if prof_on:
                profiler.record_sub("det_post_polygon_minbox", perf_counter() - ts)
            if sside < inner_post.min_size:
                continue

            ts = perf_counter() if prof_on else 0.0
            if inner_post.score_mode == "fast":
                score = inner_post.box_score_fast(pred_single, points.reshape(-1, 2))
            else:
                score = inner_post.box_score_slow(pred_single, contour)
            if prof_on:
                profiler.record_sub("det_post_polygon_score", perf_counter() - ts)

            if inner_post.box_thresh > score:
                continue

            ts = perf_counter() if prof_on else 0.0
            box = inner_post.unclip(points)
            if prof_on:
                profiler.record_sub("det_post_polygon_unclip", perf_counter() - ts)

            ts = perf_counter() if prof_on else 0.0
            box, sside = inner_post.get_mini_boxes(box)
            if prof_on:
                profiler.record_sub("det_post_polygon_minbox", perf_counter() - ts)
            if sside < inner_post.min_size + 2:
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * src_w), 0, src_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * src_h), 0, src_h)
            boxes_list.append(box.astype(np.int32))
            scores_list.append(score)

        boxes = np.array(boxes_list, dtype=np.int32)
        scores = scores_list
        if prof_on:
            profiler.record("det_post_polygon", perf_counter() - t0)
            t0 = perf_counter()

        boxes, scores = inner_post.filter_det_res(boxes, scores, src_h, src_w)
        if len(boxes) < 1:
            if prof_on:
                profiler.record("det_post_filter", perf_counter() - t0)
                profiler.record("det_total", perf_counter() - t_total)
            return TextDetOutput()
        boxes = inner.sorted_boxes(boxes)
        if prof_on:
            profiler.record("det_post_filter", perf_counter() - t0)
            elapse = perf_counter() - t_total
            profiler.record("det_total", elapse)
        else:
            elapse = 0.0
        return TextDetOutput(img, boxes, scores, elapse=elapse)

    return profiled_call


_REC_LUT = (
    ((np.arange(256, dtype=np.float32) / np.float32(255.0)) - np.float32(0.5))
    / np.float32(0.5)
).astype(np.float32)


def fast_ctc_decode(postprocess_op, preds: np.ndarray, return_word_box: bool = False, **kwargs):
    """Exact CTC greedy decode using softmax blank-dominance property (p[0] > 0.5 => argmax == 0)."""
    if return_word_box or preds.ndim != 3 or preds.shape[2] == 0:
        return postprocess_op(preds, return_word_box, **kwargs)

    first_sum = float(preds[0, 0].sum()) if preds.shape[0] > 0 and preds.shape[1] > 0 else 0.0
    if not (0.98 <= first_sum <= 1.02 and float(preds[0, 0, 0]) >= 0.0):
        return postprocess_op(preds, return_word_box, **kwargs)

    non_blank = preds[:, :, 0] <= 0.5
    preds_idx = np.zeros(preds.shape[:2], dtype=np.int64)
    if np.any(non_blank):
        preds_idx[non_blank] = preds[non_blank].argmax(axis=1)

    ignored_tokens = postprocess_op.get_ignored_tokens()
    character = postprocess_op.character
    batch_size = len(preds_idx)
    result_list: list[tuple[str, float]] = []

    for b in range(batch_size):
        token_indices = preds_idx[b]
        if not np.any(token_indices):
            result_list.append(("", 0.0))
            continue
        selection = np.ones(len(token_indices), dtype=bool)
        selection[1:] = token_indices[1:] != token_indices[:-1]
        for ignored_token in ignored_tokens:
            selection &= token_indices != ignored_token
        sel_pos = np.flatnonzero(selection)
        if len(sel_pos) == 0:
            result_list.append(("", 0.0))
            continue
        sel_tokens = token_indices[sel_pos]
        probs = preds[b, sel_pos, sel_tokens]
        conf_list = [round(float(c), 5) for c in probs]
        text = "".join(character[int(t)] for t in sel_tokens)
        result_list.append((text, float(np.mean(conf_list).round(5))))

    return result_list, []


def select_stable_canvas_width(required_w: int, max_canvas_width: int, base_width: int = 512) -> int:
    """Select a static bucket width that avoids excessive zero-padding while preventing per-batch re-tuning."""
    low = min(base_width, max_canvas_width)
    if required_w <= low:
        return low
    mid = min(768, max_canvas_width)
    if required_w <= mid:
        return mid
    if required_w <= max_canvas_width:
        return max_canvas_width
    return int(math.ceil(required_w / 32.0) * 32)


def make_profiled_recognizer(
    base_recognizer,
    profiler: GranularProfiler,
    stable_shapes: bool,
    canvas_width: int,
    calls_counter: Counter,
    base_canvas_width: int = 512,
):
    """Wrap a RapidOCR TextRecognizer instance with fused single-allocation preprocessing and optional profiling."""
    from rapidocr.ch_ppocr_rec.main import RTL_LANGS
    from rapidocr.ch_ppocr_rec.typings import TextRecInput, TextRecOutput
    from rapidocr.utils.model_resolver import normalize_lang
    from rapidocr.utils.utils import reorder_bidi_for_display
    from rapidocr.utils.vis_res import VisRes

    cached_viser = VisRes(
        lang_type=base_recognizer.cfg.lang_type,
        font_path=base_recognizer.cfg.font_path,
    )
    batch_buffers: dict[tuple[int, int, int, int], np.ndarray] = {}
    u8_scratch: dict[tuple[int, int], list[np.ndarray]] = {}

    def profiled_call(args: TextRecInput, real_count: int | None = None) -> TextRecOutput:
        prof_on = profiler.enabled
        t_total = perf_counter() if prof_on else 0.0
        t_b0 = perf_counter() if prof_on else 0.0

        img_list = [args.img] if isinstance(args.img, np.ndarray) else args.img
        return_word_box = args.return_word_box
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        img_num = len(img_list)
        rec_res = [("", 0.0)] * img_num
        batch_num = base_recognizer.rec_batch_num
        if prof_on:
            profiler.record("rec_batch_build", perf_counter() - t_b0)

        actual_n = img_num if real_count is None else real_count
        imgC, imgH, imgW = base_recognizer.rec_image_shape[:3]

        for beg_img_no in range(0, img_num, batch_num):
            t_b1 = perf_counter() if prof_on else 0.0
            end_img_no = min(img_num, beg_img_no + batch_num)
            cur_batch_len = end_img_no - beg_img_no
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            max_resized_w = imgW
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)
                rw = int(math.ceil(imgH * wh_ratio))
                if rw > max_resized_w:
                    max_resized_w = rw

            if stable_shapes:
                target_w = select_stable_canvas_width(
                    max_resized_w, canvas_width, base_width=base_canvas_width
                )
            else:
                target_w = int(imgH * max_wh_ratio)

            buf_key = (cur_batch_len, imgC, imgH, target_w)
            norm_img_batch_arr = batch_buffers.get(buf_key)
            if norm_img_batch_arr is None:
                norm_img_batch_arr = np.zeros(buf_key, dtype=np.float32)
                batch_buffers[buf_key] = norm_img_batch_arr

            if prof_on:
                profiler.record("rec_batch_build", perf_counter() - t_b1)

            for rno, ino in enumerate(range(beg_img_no, end_img_no)):
                orig_idx = int(indices[ino])
                crop_img = img_list[orig_idx]
                h, w = crop_img.shape[:2]
                ratio = w / float(h)
                raw_resized_w = int(math.ceil(imgH * ratio))
                if stable_shapes and raw_resized_w > canvas_width:
                    calls_counter["recognition_canvas_overflow_images"] += 1
                    if prof_on:
                        profiler.width_overflow_count += 1
                resized_w = min(raw_resized_w, target_w)

                t_rn = perf_counter() if prof_on else 0.0
                resized_image = cv2.resize(crop_img, (resized_w, imgH))
                sc_key = (imgH, resized_w)
                chs = u8_scratch.get(sc_key)
                if chs is None:
                    chs = [np.empty((imgH, resized_w), dtype=np.uint8) for _ in range(3)]
                    u8_scratch[sc_key] = chs
                cv2.split(resized_image, chs)
                cv2.LUT(chs[0], _REC_LUT, dst=norm_img_batch_arr[rno, 0, :, :resized_w])
                cv2.LUT(chs[1], _REC_LUT, dst=norm_img_batch_arr[rno, 1, :, :resized_w])
                cv2.LUT(chs[2], _REC_LUT, dst=norm_img_batch_arr[rno, 2, :, :resized_w])
                if prof_on:
                    profiler.record("rec_resize_normalize", perf_counter() - t_rn)

                t_pad = perf_counter() if prof_on else 0.0
                if resized_w < target_w:
                    norm_img_batch_arr[rno, :, :, resized_w:target_w] = 0.0
                if prof_on:
                    profiler.record("rec_padding", perf_counter() - t_pad)
                    if orig_idx < actual_n:
                        profiler.record_crop_dimensions(h, w, raw_resized_w, target_w)

            if prof_on:
                profiler.record_rec_batch(norm_img_batch_arr.shape, actual_n)

            t_run = perf_counter() if prof_on else 0.0
            preds = base_recognizer.session(norm_img_batch_arr)
            if prof_on:
                profiler.record("rec_runtime_call", perf_counter() - t_run)

            t_dec = perf_counter() if prof_on else 0.0
            line_results, word_results = fast_ctc_decode(
                base_recognizer.postprocess_op,
                preds,
                return_word_box,
                wh_ratio_list=wh_ratio_list,
                max_wh_ratio=max_wh_ratio,
            )
            for rno, one_res in enumerate(line_results):
                if return_word_box:
                    rec_res[indices[beg_img_no + rno]] = (one_res, word_results[rno])
                    continue
                rec_res[indices[beg_img_no + rno]] = (one_res, None)
            if prof_on:
                profiler.record("rec_ctc_decode", perf_counter() - t_dec)

        t_b4 = perf_counter() if prof_on else 0.0
        all_line_results, all_word_results = list(zip(*rec_res))
        txts, scores = list(zip(*all_line_results))
        if normalize_lang(base_recognizer.cfg.lang_type) in RTL_LANGS:
            txts = reorder_bidi_for_display(txts)
        if prof_on:
            profiler.record("rec_batch_build", perf_counter() - t_b4)
            elapse = perf_counter() - t_total
            profiler.record("rec_total", elapse)
        else:
            elapse = 0.0

        return TextRecOutput(
            img_list,
            txts,
            scores,
            all_word_results,
            elapse,
            viser=cached_viser,
        )

    profiled_call._is_profiled = True
    return profiled_call

