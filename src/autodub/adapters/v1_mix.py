"""Blockwise three-stem mix; only timed lexical speech with usable TTS is replaced."""
from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

from autodub.adapters.common import output_folder
from autodub.storage import atomic_json, safe_path

RATE = 48000


def _ffmpeg(arguments: list[str], config: dict) -> None:
    binary = shutil.which(os.environ.get("FFMPEG_BIN", "ffmpeg"))
    if not binary:
        raise RuntimeError("FFmpeg is unavailable")
    subprocess.run([binary, "-hide_banner", "-loglevel", "error",
                    *arguments], shell=False, check=True, capture_output=True, timeout=3600)


def run(source: Path, config: dict) -> dict:
    del source
    import numpy as np

    folder = output_folder(config)
    stems = config["stems"]
    segments = config["segments"]
    clips, masks, issues = [], [], []
    for index, segment in enumerate(segments):
        raw = config.get("clips", {}).get(segment["id"])
        if not raw or segment["action"] != "DUB":
            continue
        words = segment.get("words", [])
        if not words or segment.get("speech_kind") != "lexical":
            issues.append({"segment_id": segment["id"], "code": "NO_LEXICAL_TIMING_KEEP_ORIGINAL"})
            continue
        start = min(w["s"] for w in words)
        slot = segment["end_ms"] - start
        run_root = Path(config.get("run_root", folder.parent)).resolve()
        raw_path = safe_path(run_root, raw)
        converted = safe_path(folder, f"clip-{index:06d}.wav")
        _ffmpeg(["-i", str(raw_path), "-ac", "2", "-ar", str(RATE), "-c:a", "pcm_s16le", "-y", str(converted)], config)
        with wave.open(str(converted), "rb") as audio:
            frames = audio.getnframes()
        actual = frames * 1000 / RATE
        factor = max(1.0, actual / slot)
        if factor > 1.2:
            issues.append({"segment_id": segment["id"], "code": "VOICE_EXCEEDS_FROZEN_DURATION_POLICY",
                           "actual_ms": actual, "slot_ms": slot, "action": "KEEP_ORIGINAL"})
            continue
        if factor > 1:
            fitted = folder / f"fit-{index:06d}.wav"
            _ffmpeg(["-i", str(converted), "-af", f"atempo={factor:.9f}", "-c:a", "pcm_s16le",
                     "-y", str(fitted)], config)
            converted = fitted
        with wave.open(str(converted), "rb") as audio:
            values = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").reshape(-1, 2).astype("float32")
        left = round(start * RATE / 1000)
        right = left + len(values)
        if right > round(segment["end_ms"] * RATE / 1000) + RATE // 100:
            issues.append({"segment_id": segment["id"], "code": "FIT_OVERFLOW_KEEP_ORIGINAL"})
            continue
        # Protect all KEEP events, including another speaker's nonverbal event.
        if any(s["action"] != "DUB" and s["start_ms"] < right * 1000 / RATE and s["end_ms"] > start
               for s in segments if s["id"] != segment["id"]):
            issues.append({"segment_id": segment["id"], "code": "PROTECTED_EVENT_COLLISION_KEEP_ORIGINAL"})
            continue
        clips.append((left, right, values))
        masks.extend((round(w["s"] * RATE / 1000), round(w["e"] * RATE / 1000)) for w in words)
    run_root = Path(config.get("run_root", folder.parent)).resolve()
    readers = [wave.open(str(safe_path(run_root, stems[name])), "rb") for name in ("speech", "music", "effects")]
    output = folder / "mixed.wav"
    clipped = 0
    try:
        counts = []
        for audio in readers:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (RATE, 2, 2):
                raise ValueError("Three stems must share stereo PCM16 48 kHz format")
            counts.append(audio.getnframes())
        if len(set(counts)) != 1:
            raise ValueError("Three stem lengths differ")
        with wave.open(str(output), "wb") as writer:
            writer.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
            offset = 0
            while offset < counts[0]:
                count = min(RATE, counts[0] - offset)
                blocks = [np.frombuffer(a.readframes(count), dtype="<i2").reshape(-1, 2).astype("float32")
                          for a in readers]
                for left, right in masks:
                    a, b = max(offset, left), min(offset + count, right)
                    if a < b:
                        blocks[0][a - offset:b - offset] = 0
                mixed = sum(blocks)
                for left, right, values in clips:
                    a, b = max(offset, left), min(offset + count, right)
                    if a < b:
                        mixed[a - offset:b - offset] += values[a - left:b - left]
                clipped += int(np.count_nonzero(np.abs(mixed) > 32767))
                writer.writeframes(np.clip(mixed, -32768, 32767).astype("<i2").tobytes())
                offset += count
    finally:
        for audio in readers:
            audio.close()
    if clipped:
        issues.append({"code": "MIX_CLIPPING", "clipped_samples": clipped})
    report = folder / "audio-mix.json"
    atomic_json(report, {"schema_version": 1, "issues": issues, "mixed": str(output),
                         "replaced_clips": len(clips), "lexical_masks": masks,
                         "nonverbal_policy": "retain speech outside approved lexical word spans"})
    return {"audio": str(output), "stage_status": "DEGRADED" if issues else "SUCCESS",
            "quality_evidence": {"issues": issues}, "artifacts": [str(report), str(output)]}
