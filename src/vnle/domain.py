"""Small durable contracts; coordinates refer to source raster pixels."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}: must be finite")
    return result


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def parse(cls, data: dict) -> Rect:
        rect = cls(*(finite(data[k], k) for k in ("x", "y", "width", "height")))
        if (
            rect.x < 0
            or rect.y < 0
            or rect.width <= 0
            or rect.height <= 0
            or rect.x + rect.width > 1
            or rect.y + rect.height > 1
        ):
            raise ValueError("ROI must be a nonempty rectangle inside the video")
        return rect

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        # Outward rounding protects all pixels touched by the user's rectangle.
        return (
            math.floor(self.x * width),
            math.floor(self.y * height),
            min(width, math.ceil((self.x + self.width) * width)),
            min(height, math.ceil((self.y + self.height) * height)),
        )


@dataclass(frozen=True)
class Exclusion:
    rect: Rect
    start_s: float
    end_s: float

    def active(self, time_s: float) -> bool:
        return self.start_s <= time_s < self.end_s


@dataclass(frozen=True)
class Request:
    exclusions: tuple[Exclusion, ...]
    start_s: float
    end_s: float
    roi_hash: str

    @classmethod
    def parse(cls, data: dict, duration: float) -> Request:
        if data.get("roi_confirmed") is not True:
            raise ValueError("Confirm the subtitle exclusion before analysis")
        start = finite(data.get("start_s", 0), "start_s")
        end = finite(data.get("end_s", duration), "end_s")
        if not 0 <= start < end <= duration:
            raise ValueError("Analysis interval is outside the video")
        raw = data.get("exclusions")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 16:
            raise ValueError("Draw between 1 and 16 subtitle exclusion rectangles")
        items = []
        for row in raw:
            a = finite(row.get("start_s", 0), "ROI start")
            b = finite(row.get("end_s", duration), "ROI end")
            if not 0 <= a < b <= duration:
                raise ValueError("ROI interval is outside the video")
            items.append(Exclusion(Rect.parse(row), a, b))
        frozen = tuple(items)
        return cls(frozen, start, end, digest([asdict(x) for x in frozen]))

    def at(self, time_s: float) -> tuple[Rect, ...]:
        return tuple(e.rect for e in self.exclusions if e.active(time_s))


def allowed_tiles(width: int, height: int, exclusions: tuple[Rect, ...]):
    """Partition the exact complement. No excluded pixel reaches either OCR model.

    Tiles share no pixels. Horizontal strips are merged vertically when possible.
    Text crossing a tile edge may need review; the prototype reports this limitation.
    """
    blocked = [r.pixels(width, height) for r in exclusions]
    xs = sorted({0, width, *(v for r in blocked for v in (r[0], r[2]))})
    ys = sorted({0, height, *(v for r in blocked for v in (r[1], r[3]))})
    previous = {}
    done = []
    for y0, y1 in zip(ys, ys[1:]):
        spans = []
        for x0, x1 in zip(xs, xs[1:]):
            covered = any(a <= x0 and x1 <= c and b <= y0 and y1 <= d for a, b, c, d in blocked)
            if covered:
                continue
            if spans and spans[-1][1] == x0:
                spans[-1] = (spans[-1][0], x1)
            else:
                spans.append((x0, x1))
        current = {}
        for span in spans:
            old = previous.pop(span, None)
            current[span] = (span[0], old[1] if old else y0, span[1], y1)
        done.extend(previous.values())
        previous = current
    done.extend(previous.values())
    return sorted(done, key=lambda r: (r[1], r[0]))


@dataclass
class Observation:
    id: int
    pts: int
    time_base: dict
    time_s: float
    polygon: list[list[float]]
    text: str
    score: float | None
    crop_sha256: str
    tile: tuple[int, int, int, int]
    touches_tile_edge: bool = False

    @property
    def bbox(self):
        xs, ys = zip(*self.polygon)
        return min(xs), min(ys), max(xs), max(ys)
