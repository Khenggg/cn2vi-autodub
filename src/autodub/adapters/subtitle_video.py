"""Model-worker adapter for audio muxing, Vietnamese captions, and ROI cleanup."""
from pathlib import Path

from autodub.contracts import Segment
from autodub.subtitle_render import render_subtitles


def run(source: Path, config: dict) -> dict:
    preview_audio = config.get("preview_audio")
    output = config.get("target_output")
    raw_segments = config.get("segments")
    if not isinstance(preview_audio, str) or not isinstance(output, str):
        raise ValueError("VISION_RENDER requires preview_audio and target_output paths")
    if not isinstance(raw_segments, list):
        raise ValueError("VISION_RENDER segments must be a list")
    segments = [Segment.model_validate(item) for item in raw_segments]
    return render_subtitles(source, Path(preview_audio), Path(output), segments, config)
