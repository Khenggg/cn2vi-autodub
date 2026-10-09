"""ffprobe is used only on the execution machine, never during module import."""

from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def parse_probe(data: dict) -> dict:
    streams = [
        s
        for s in data.get("streams", [])
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
    ]
    if not streams:
        raise ValueError("Video stream not found")
    stream = streams[0]
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            rotation = float(side["rotation"])
    sar_raw = stream.get("sample_aspect_ratio", "1:1")
    sar = Fraction(sar_raw.replace(":", "/")) if sar_raw not in ("N/A", "0:1") else Fraction(1)
    if rotation % 360 or sar != 1:
        raise ValueError(
            "Prototype requires rotation=0 and square pixels; ROI mapping is not supported yet"
        )
    duration = float(stream.get("duration", data.get("format", {}).get("duration", 0)))
    if not 0 < duration < float("inf"):
        raise ValueError("Video has no finite duration")
    time_base = Fraction(stream["time_base"])
    start_time = Fraction(stream.get("start_time", "0"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration_s": duration,
        "codec": stream["codec_name"],
        "pixel_format": stream.get("pix_fmt"),
        "stream_index": stream["index"],
        "time_base": {"num": time_base.numerator, "den": time_base.denominator},
        "start_pts": stream.get("start_pts"),
        "origin": {"num": start_time.numerator, "den": start_time.denominator},
        "rotation": rotation,
        "sar": [sar.numerator, sar.denominator],
        "fps_hint": stream.get("avg_frame_rate"),
    }


def probe(path: Path, ffprobe: str = "ffprobe") -> dict:
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
            check=True,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "ffprobe is missing; configure --ffprobe on the execution machine"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"ffprobe rejected the video: {exc.stderr[:500]}") from exc
    return parse_probe(json.loads(result.stdout))
