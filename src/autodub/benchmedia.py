"""FFmpeg preprocessing helpers for benchmark clips and frames."""
from __future__ import annotations

import math
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Iterator

from autodub.media import probe_media


def _seconds(ms: int) -> str:
    if not isinstance(ms, int) or isinstance(ms, bool) or ms < 0:
        raise ValueError("timestamps must be non-negative integer milliseconds")
    return format(Decimal(ms) / Decimal(1000), ".3f")


def _run(command: list[str], timeout: int = 180) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("FFmpeg unavailable or timed out; configure FFMPEG_BIN") from error
    if result.returncode:
        raise RuntimeError("FFmpeg could not create the requested benchmark media")


def iter_chunks(duration_ms: int, chunk_ms: int = 240_000,
                overlap_ms: int = 1_000) -> Iterator[tuple[int, int]]:
    """Yield [start, end) chunks with overlap, including the final short chunk."""
    if (not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms <= 0
            or not isinstance(chunk_ms, int) or isinstance(chunk_ms, bool) or chunk_ms <= 0
            or not isinstance(overlap_ms, int) or isinstance(overlap_ms, bool)
            or overlap_ms < 0 or overlap_ms >= chunk_ms):
        raise ValueError("duration/chunk must be positive integers and 0 <= overlap < chunk")
    start = 0
    while start < duration_ms:
        end = min(duration_ms, start + chunk_ms)
        yield start, end
        if end == duration_ms:
            break
        start = end - overlap_ms


def make_audio_chunk(source: Path, output: Path, start_ms: int, end_ms: int, *,
                     ffmpeg_bin: str = "ffmpeg") -> Path:
    """Extract a bounded mono PCM16 16 kHz WAV segment."""
    if not isinstance(end_ms, int) or isinstance(end_ms, bool) or end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-ss", _seconds(start_ms), "-i",
          str(source), "-t", _seconds(end_ms - start_ms), "-vn", "-ac", "1", "-ar", "16000",
          "-c:a", "pcm_s16le", "-y", str(output)])
    return output


def extract_audio(source: Path, output: Path, start_ms: int = 0, end_ms: int | None = None,
                  sample_rate: int = 16_000, channels: int = 1,
                  ffmpeg_bin: str = "ffmpeg", *, ffprobe_bin: str = "ffprobe") -> Path:
    """Extract an optionally bounded PCM16 WAV with adapter-selectable format."""
    if (not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0
            or not isinstance(channels, int) or isinstance(channels, bool) or channels not in (1, 2)):
        raise ValueError("sample_rate must be positive and channels must be 1 or 2")
    if end_ms is None:
        metadata = probe_media(Path(source), ffprobe_bin)
        end_ms = metadata["duration_ms"]
    if not isinstance(end_ms, int) or isinstance(end_ms, bool) or end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-ss", _seconds(start_ms), "-i",
          str(source), "-t", _seconds(end_ms - start_ms), "-vn", "-ac", str(channels),
          "-ar", str(sample_rate), "-c:a", "pcm_s16le", "-y", str(output)])
    return output


def make_production_audio(source: Path, output: Path, *, ffmpeg_bin: str = "ffmpeg") -> Path:
    """Extract full-source production audio as stereo PCM16 48 kHz WAV."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", str(source), "-vn",
          "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", "-y", str(output)])
    return output


def crop_roi_frame(source: Path, output: Path, time_ms: int, roi: dict, *,
                   ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> Path:
    """Write a single PNG cropped from a normalized ROI after strict source bounds checks."""
    if not isinstance(time_ms, int) or isinstance(time_ms, bool) or time_ms < 0:
        raise ValueError("time_ms must be a non-negative integer")
    if not isinstance(roi, dict):
        raise ValueError("roi must contain normalized x/y/w/h")
    try:
        x, y, w, h = (roi[key] for key in ("x", "y", "w", "h"))
    except KeyError as error:
        raise ValueError("roi must contain normalized x/y/w/h") from error
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (x, y, w, h))
            or x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1 or y + h > 1):
        raise ValueError("roi must stay within normalized frame bounds")
    metadata = probe_media(Path(source), ffprobe_bin)
    if time_ms >= metadata["duration_ms"]:
        raise ValueError("time_ms exceeds source duration")
    video = metadata["video"]
    width, height = int(video["width"]), int(video["height"])
    left, top = int(x * width), int(y * height)
    crop_width, crop_height = int(w * width), int(h * height)
    if crop_width <= 0 or crop_height <= 0 or left + crop_width > width or top + crop_height > height:
        raise ValueError("roi is too small or outside pixel bounds")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exact luma coordinates keep odd ROI bounds consistent with PNG masks on YUV420 sources.
    vf = f"crop={crop_width}:{crop_height}:{left}:{top}:exact=1"
    _run([ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-ss", _seconds(time_ms), "-i",
          str(source), "-vf", vf, "-frames:v", "1", "-y", str(output)])
    return output


def extract_roi_frame(source: Path, output: Path, roi: dict, at_ms: int,
                      ffmpeg_bin: str = "ffmpeg", *, ffprobe_bin: str = "ffprobe") -> Path:
    """Adapter-facing alias with ROI before timestamp, matching the corpus contract."""
    return crop_roi_frame(source, output, at_ms, roi, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)


def chunk_windows(duration_ms: int, chunk_ms: int = 240_000,
                  overlap_ms: int = 1_000) -> list[tuple[int, int]]:
    """Materialize overlapping chunk intervals for adapter preprocessing loops."""
    return list(iter_chunks(duration_ms, chunk_ms, overlap_ms))
