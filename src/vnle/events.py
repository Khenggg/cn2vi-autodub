"""Conservative association. Never fuzzy-merge changed numbers or changed text."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict

from .domain import Observation


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def iou(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0


class EventBuilder:
    def __init__(self, start_s: float, confidence: float = 0.68):
        self.previous_s = start_s
        self.confidence = confidence
        self.active: dict[str, dict] = {}
        self.events: list[dict] = []
        self.shot = 0
        self.counter = 0

    def _close(self, event, end_s, bracket, reason):
        event["end_s"] = max(event["start_s"], end_s)
        event["end_bracket_s"] = bracket
        event["close_reason"] = reason
        event["state"] = "NEEDS_REVIEW" if event["review_reasons"] else "CLOSED"
        self.events.append(event)

    def update(self, time_s: float, observations: list[Observation], cut=False):
        if time_s < self.previous_s:
            raise ValueError("Presentation timestamps must be monotonic")
        if cut:
            self.boundary(time_s, "SHOT_OR_EXCLUSION_CHANGE")
            self.shot += 1
        available = dict(self.active)
        next_active = {}
        assignments = []
        for obs in observations:
            text = normalized(obs.text)
            choices = [
                (iou(event["bbox"], obs.bbox), key, event) for key, event in available.items()
            ]
            choices = [c for c in choices if c[0] >= 0.25]
            exact = [c for c in choices if normalized(c[2]["text"]) == text]
            match = max(exact or choices, key=lambda c: c[0], default=None)
            parent = None
            revision = 0
            if match:
                _, key, old = match
                available.pop(key)
                if normalized(old["text"]) == text:
                    event = old
                else:
                    parent = old["id"]
                    revision = old["revision"] + 1
                    midpoint = (old["last_seen_s"] + time_s) / 2
                    self._close(old, midpoint, [old["last_seen_s"], time_s], "TEXT_REVISION")
                    event = None
            else:
                event = None
            if event is None:
                self.counter += 1
                event = {
                    "id": f"e{self.counter:06d}",
                    "shot_id": self.shot,
                    "appearance_id": self.counter,
                    "revision": revision,
                    "parent_event_id": parent,
                    "text": obs.text,
                    "start_s": (self.previous_s + time_s) / 2,
                    "start_bracket_s": [self.previous_s, time_s],
                    "first_pts": obs.pts,
                    "time_base": obs.time_base,
                    "bbox": list(obs.bbox),
                    "geometry_keyframes": [],
                    "observations": [],
                    "best_score": obs.score,
                    "best_observation_id": obs.id,
                    "review_reasons": [],
                    "selected": None,
                    "decision": "PENDING_IMPORTANCE_REVIEW",
                }
            event["last_seen_s"] = time_s
            event["last_pts"] = obs.pts
            event["bbox"] = list(obs.bbox)
            event["observations"].append(obs.id)
            event["geometry_keyframes"].append(
                {
                    "pts": obs.pts,
                    "time_s": time_s,
                    "polygon": obs.polygon,
                }
            )
            reasons = event["review_reasons"]
            if (
                obs.score is None or obs.score < self.confidence
            ) and "LOW_CONFIDENCE" not in reasons:
                reasons.append("LOW_CONFIDENCE")
            if obs.touches_tile_edge and "TILE_EDGE" not in reasons:
                reasons.append("TILE_EDGE")
            if not text and "EMPTY_TEXT" not in reasons:
                reasons.append("EMPTY_TEXT")
            improved = obs.score is not None and (
                event["best_score"] is None or obs.score >= event["best_score"]
            )
            if improved:
                event["best_score"] = obs.score
                event["best_observation_id"] = obs.id
            next_active[event["id"]] = event
            assignments.append((event["id"], obs, improved or len(event["observations"]) == 1))
        for event in available.values():
            midpoint = (event["last_seen_s"] + time_s) / 2
            self._close(event, midpoint, [event["last_seen_s"], time_s], "NOT_OBSERVED")
        self.active = next_active
        self.previous_s = time_s
        return assignments

    def boundary(self, time_s: float, reason: str):
        for event in self.active.values():
            self._close(event, time_s, [event["last_seen_s"], time_s], reason)
        self.active = {}
        self.previous_s = time_s

    def snapshot(self):
        return self.events + [dict(e, state="ACTIVE") for e in self.active.values()]


def observation_record(obs: Observation):
    return asdict(obs)
