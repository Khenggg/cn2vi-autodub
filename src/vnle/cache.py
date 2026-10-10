"""Temporal ROI Cache & Lightweight Region Tracking.

Avoids duplicate OCR detections and recognitions for static or slightly moving text regions.
Tracks confirmed text ROIs using normalized cross-correlation and gates full-frame detection
based on frame differences outside tracked ROIs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .domain import Observation, Rect


def contains_chinese(text: str) -> bool:
    """Check if string contains any CJK Unified Ideographs or extensions."""
    for ch in text:
        code = ord(ch)
        if (
            (0x4E00 <= code <= 0x9FFF)
            or (0x3400 <= code <= 0x4DBF)
            or (0xF900 <= code <= 0xFAFF)
            or (0x20000 <= code <= 0x2A6DF)
        ):
            return True
    return False


@dataclass
class CachedROI:
    roi_id: int
    polygon: list[list[float]]
    bbox: tuple[int, int, int, int]
    text: str
    score: float | None
    crop_sha256: str
    tile: tuple[int, int, int, int]
    touches_tile_edge: bool
    template_gray: np.ndarray
    first_seen_s: float
    last_seen_s: float
    consecutive_hits: int = 1
    consecutive_misses: int = 0
    status: str = "CONFIRMED"


class TemporalROICache:
    def __init__(
        self,
        similarity_threshold: float = 0.82,
        max_motion_padding: int = 8,
        chinese_only: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_motion_padding = max_motion_padding
        self.chinese_only = chinese_only
        self.active_rois: dict[int, CachedROI] = {}
        self.next_roi_id = 1
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_revalidations": 0,
            "tracked_disappeared": 0,
            "skipped_recognitions": 0,
        }

    def reset(self):
        """Clear active cache on shot boundary, scene cut, or exclusion change."""
        self.active_rois.clear()

    def register_observations(
        self,
        observations: list[Observation],
        image: np.ndarray,
    ):
        """Register newly recognized text observations into the temporal cache."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        for obs in observations:
            if self.chinese_only and not contains_chinese(obs.text):
                continue

            bx0 = int(np.floor(obs.bbox[0]))
            by0 = int(np.floor(obs.bbox[1]))
            bx1 = min(w, int(np.ceil(obs.bbox[2])))
            by1 = min(h, int(np.ceil(obs.bbox[3])))
            if bx1 <= bx0 or by1 <= by0:
                continue

            template = gray[by0:by1, bx0:bx1].copy()
            if template.size == 0 or template.shape[0] < 4 or template.shape[1] < 4:
                continue

            # Check if this overlaps closely with an existing active ROI
            matched_id = None
            for rid, roi in self.active_rois.items():
                ix0 = max(bx0, roi.bbox[0])
                iy0 = max(by0, roi.bbox[1])
                ix1 = min(bx1, roi.bbox[2])
                iy1 = min(by1, roi.bbox[3])
                if ix1 > ix0 and iy1 > iy0:
                    inter_area = (ix1 - ix0) * (iy1 - iy0)
                    area_a = (bx1 - bx0) * (by1 - by0)
                    area_b = (roi.bbox[2] - roi.bbox[0]) * (roi.bbox[3] - roi.bbox[1])
                    iou = inter_area / float(area_a + area_b - inter_area)
                    if iou >= 0.45:
                        matched_id = rid
                        break

            if matched_id is not None:
                roi = self.active_rois[matched_id]
                roi.polygon = [list(p) for p in obs.polygon]
                roi.bbox = (bx0, by0, bx1, by1)
                roi.text = obs.text
                roi.score = obs.score
                roi.crop_sha256 = obs.crop_sha256
                roi.template_gray = template
                roi.last_seen_s = obs.time_s
                roi.status = "CONFIRMED"
                roi.consecutive_misses = 0
            else:
                rid = self.next_roi_id
                self.next_roi_id += 1
                self.active_rois[rid] = CachedROI(
                    roi_id=rid,
                    polygon=[list(p) for p in obs.polygon],
                    bbox=(bx0, by0, bx1, by1),
                    text=obs.text,
                    score=obs.score,
                    crop_sha256=obs.crop_sha256,
                    tile=obs.tile,
                    touches_tile_edge=obs.touches_tile_edge,
                    template_gray=template,
                    first_seen_s=obs.time_s,
                    last_seen_s=obs.time_s,
                    consecutive_hits=1,
                    consecutive_misses=0,
                    status="CONFIRMED",
                )

    def track_and_emit(
        self,
        image: np.ndarray,
        exclusions: tuple[Rect, ...],
        pts: int,
        time_base: dict,
        time_s: float,
        next_id_func: Callable[[], int],
    ) -> tuple[list[Observation], dict[int, np.ndarray], bool]:
        """Track active ROIs using light template matching and emit cached observations."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        hits = []
        evidence = {}
        pad = self.max_motion_padding
        all_reliable = True
        to_remove = []

        for rid, roi in list(self.active_rois.items()):
            rx0, ry0, rx1, ry1 = roi.bbox
            th, tw = roi.template_gray.shape[:2]
            if th < 4 or tw < 4:
                to_remove.append(rid)
                continue

            sx0 = max(0, rx0 - pad)
            sy0 = max(0, ry0 - pad)
            sx1 = min(w, rx1 + pad)
            sy1 = min(h, ry1 + pad)

            overlaps_exclusion = False
            for rect in exclusions:
                ex0, ey0, ex1, ey1 = rect.pixels(w, h)
                if max(sx0, ex0) < min(sx1, ex1) and max(sy0, ey0) < min(sy1, ey1):
                    overlaps_exclusion = True
                    break
            if overlaps_exclusion:
                to_remove.append(rid)
                continue

            search_area = gray[sy0:sy1, sx0:sx1]
            if search_area.shape[0] < th or search_area.shape[1] < tw:
                to_remove.append(rid)
                continue

            res = cv2.matchTemplate(search_area, roi.template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val >= self.similarity_threshold:
                dx = (sx0 + max_loc[0]) - rx0
                dy = (sy0 + max_loc[1]) - ry0

                new_polygon = [[float(p[0] + dx), float(p[1] + dy)] for p in roi.polygon]
                new_bbox = (rx0 + dx, ry0 + dy, rx1 + dx, ry1 + dy)
                roi.polygon = new_polygon
                roi.bbox = new_bbox
                roi.last_seen_s = time_s
                roi.consecutive_hits += 1
                roi.consecutive_misses = 0
                roi.status = "CONFIRMED"

                crop_img = image[new_bbox[1] : new_bbox[3], new_bbox[0] : new_bbox[2]].copy()
                if crop_img.size == 0:
                    continue

                obs_id = next_id_func()
                obs = Observation(
                    id=obs_id,
                    pts=pts,
                    time_base=time_base,
                    time_s=time_s,
                    polygon=new_polygon,
                    text=roi.text,
                    score=roi.score,
                    crop_sha256=hashlib.sha256(crop_img.tobytes()).hexdigest(),
                    tile=roi.tile,
                    touches_tile_edge=roi.touches_tile_edge,
                )
                hits.append(obs)
                evidence[obs_id] = crop_img
                self.stats["cache_hits"] += 1
            elif max_val >= 0.58:
                roi.status = "SUSPECTED_CHANGE"
                all_reliable = False
                self.stats["cache_revalidations"] += 1
            else:
                roi.consecutive_misses += 1
                all_reliable = False
                if roi.consecutive_misses >= 2:
                    to_remove.append(rid)
                    self.stats["tracked_disappeared"] += 1

        for rid in to_remove:
            self.active_rois.pop(rid, None)

        return hits, evidence, all_reliable

    def match_box_with_cache(
        self,
        candidate_box: np.ndarray,
        detector_input: np.ndarray,
        pts: int,
        time_base: dict,
        time_s: float,
        next_id_func: Callable[[], int],
    ) -> tuple[Observation | None, np.ndarray | None]:
        """Check if a detector-proposed box matches an already confirmed cached ROI.

        If matched, returns (cached_observation, crop_evidence) avoiding recognizer call.
        """
        bx0 = float(np.min(candidate_box[:, 0]))
        by0 = float(np.min(candidate_box[:, 1]))
        bx1 = float(np.max(candidate_box[:, 0]))
        by1 = float(np.max(candidate_box[:, 1]))

        for roi in self.active_rois.values():
            if roi.status != "CONFIRMED":
                continue
            rx0, ry0, rx1, ry1 = roi.bbox
            ix0, iy0 = max(bx0, rx0), max(by0, ry0)
            ix1, iy1 = min(bx1, rx1), min(by1, ry1)
            if ix1 > ix0 and iy1 > iy0:
                inter_area = (ix1 - ix0) * (iy1 - iy0)
                box_area = (bx1 - bx0) * (by1 - by0)
                roi_area = (rx1 - rx0) * (ry1 - ry0)
                union_area = box_area + roi_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou >= 0.45:
                    # Match found! Use cached recognition
                    cx0, cy0 = max(0, int(bx0)), max(0, int(by0))
                    cx1, cy1 = min(detector_input.shape[1], int(bx1)), min(
                        detector_input.shape[0], int(by1)
                    )
                    crop_img = detector_input[cy0:cy1, cx0:cx1].copy()
                    if crop_img.size == 0:
                        return None, None
                    obs_id = next_id_func()
                    obs = Observation(
                        id=obs_id,
                        pts=pts,
                        time_base=time_base,
                        time_s=time_s,
                        polygon=[[float(p[0]), float(p[1])] for p in candidate_box],
                        text=roi.text,
                        score=roi.score,
                        crop_sha256=hashlib.sha256(crop_img.tobytes()).hexdigest(),
                        tile=roi.tile,
                        touches_tile_edge=roi.touches_tile_edge,
                    )
                    self.stats["skipped_recognitions"] += 1
                    return obs, crop_img
        return None, None

    def has_external_motion(
        self,
        diff_thumbnail: np.ndarray,
        frame_size: tuple[int, int],
        exclusions: tuple[Rect, ...],
        threshold: float = 12.0,
        min_motion_pixels: int = 30,
    ) -> bool:
        """Return True if there is significant pixel change outside tracked ROIs and exclusions."""
        if diff_thumbnail is None:
            return True
        th_h, th_w = diff_thumbnail.shape[:2]
        fw, fh = frame_size

        motion_mask = (diff_thumbnail >= threshold).astype(np.uint8)

        # Mask out exclusions
        for rect in exclusions:
            ex0, ey0, ex1, ey1 = rect.pixels(th_w, th_h)
            motion_mask[ey0:ey1, ex0:ex1] = 0

        # Mask out all tracked ROIs (with slight margin)
        for roi in self.active_rois.values():
            rx0, ry0, rx1, ry1 = roi.bbox
            tx0 = max(0, int((rx0 / fw) * th_w) - 2)
            ty0 = max(0, int((ry0 / fh) * th_h) - 2)
            tx1 = min(th_w, int((rx1 / fw) * th_w) + 2)
            ty1 = min(th_h, int((ry1 / fh) * th_h) + 2)
            motion_mask[ty0:ty1, tx0:tx1] = 0

        moving_pixels = int(np.count_nonzero(motion_mask))
        return moving_pixels >= min_motion_pixels
