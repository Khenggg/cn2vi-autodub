import json
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path


class MediaError(RuntimeError):
    pass


def probe_media(path: Path, executable: str = "ffprobe") -> dict:
    try:
        result = subprocess.run(
            [executable, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MediaError("FFprobe unavailable or timed out; install FFmpeg / configure FFPROBE_BIN") from error
    if result.returncode != 0:
        raise MediaError("Video could not be decoded by FFprobe")
    try:
        payload = json.loads(result.stdout)
        streams = payload["streams"]
        video = next(stream for stream in streams if stream.get("codec_type") == "video"
                     and not stream.get("disposition", {}).get("attached_pic", 0))
        audio = next(stream for stream in streams if stream.get("codec_type") == "audio")
        seconds = Decimal(str(payload["format"]["duration"]))
        if not seconds.is_finite() or seconds <= 0 or not video.get("width") or not video.get("height"):
            raise ValueError("Invalid media duration or dimensions")
        return {"schema_version": 1, "duration_ms": int(seconds * 1000),
                "video": {key: video.get(key) for key in ("codec_name", "width", "height", "avg_frame_rate")},
                "audio": {key: audio.get(key) for key in ("codec_name", "sample_rate", "channels")}}
    except (KeyError, StopIteration, ValueError, InvalidOperation, TypeError) as error:
        raise MediaError("Input must contain a valid video and audio track with a known duration") from error
