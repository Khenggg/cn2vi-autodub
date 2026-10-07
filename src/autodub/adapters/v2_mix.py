"""Deterministic multi-band voice-over; no source separation or lexical muting."""
from __future__ import annotations

import math
import subprocess
import time
import wave
from pathlib import Path

from autodub.adapters.runtime_paths import output_folder, run_file
from autodub.storage import atomic_json, sha256_file

RATE = 48000


def ducking_graph() -> str:
    # A unit control signal yields nominal -14/-3dB steady-state reduction.
    # Attack/release follow actual TTS activity; quiet pauses do not hold the duck.
    mid = 10 ** (-14 / (20 * (1 - 1 / 8)))
    side = 10 ** (-3 / (20 * (1 - 1 / 2)))
    return (
        "[0:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS,"
        "acrossover=split='500 3500':order=4th[low][mid][high];"
        "[2:a]asplit=3[sc_low][sc_mid][sc_high];"
        f"[low][sc_low]sidechaincompress=threshold={side:.9f}:ratio=2:attack=100:release=300:"
        "knee=1:detection=peak:link=maximum:makeup=1[low_duck];"
        f"[mid][sc_mid]sidechaincompress=threshold={mid:.9f}:ratio=8:attack=100:release=300:"
        "knee=1:detection=peak:link=maximum:makeup=1[mid_duck];"
        f"[high][sc_high]sidechaincompress=threshold={side:.9f}:ratio=2:attack=100:release=300:"
        "knee=1:detection=peak:link=maximum:makeup=1[high_duck];"
        "[low_duck][mid_duck][high_duck]amix=inputs=3:normalize=0:duration=first[background];"
        "[1:a]pan=stereo|c0=c0|c1=c0,volume=1.25[voice];"
        "[background][voice]amix=inputs=2:normalize=0:duration=first,"
        "alimiter=limit=0.95:level=0:latency=1[out]"
    )


def _write_bus(path: Path, data, control: bool = False) -> None:
    import numpy as np

    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        for start in range(0, len(data), RATE * 10):
            block = data[start:start + RATE * 10]
            if control:
                # The sidechain follows 20ms activity frames, not segment padding.
                envelope = np.zeros(len(block), dtype="float32")
                for offset in range(0, len(block), RATE // 50):
                    frame = block[offset:offset + RATE // 50]
                    if len(frame) and float(np.sqrt(np.mean(frame * frame))) > 0.003:
                        envelope[offset:offset + len(frame)] = 1.0
                block = envelope
            stream.writeframes((np.clip(block, -1, 1) * 32767).astype("<i2").tobytes())


def run(source: Path, config: dict) -> dict:
    import numpy as np
    import soundfile as sf

    folder = output_folder(config)
    samples = round(config["duration_ms"] * RATE / 1000)
    backing = folder / "voice-bus.f32"
    voice = np.memmap(backing, dtype="float32", mode="w+", shape=(samples,))
    voice[:] = 0
    issues, added, used = [], [], []
    tick = time.perf_counter()
    try:
        for segment in sorted(config["segments"], key=lambda s: (s["start_ms"], s["id"])):
            sid = segment["id"]
            if segment["action"] != "DUB" or sid not in config["clips"]:
                continue
            path = run_file(config["clips"][sid])
            clip, rate = sf.read(path, dtype="float32")
            if rate != RATE or clip.ndim != 1 or not len(clip) or not np.isfinite(clip).all():
                raise ValueError("TTS clip is not mono 48kHz finite audio")
            left, right = round(segment["start_ms"] * RATE / 1000), round(segment["end_ms"] * RATE / 1000)
            if not 0 <= left < right <= samples:
                raise ValueError("Voice slot exceeds source timeline")
            if any(left < end and begin < right for begin, end in used):
                issues.append({"segment_id": sid, "code": "OVERLAPPING_VOICE_SLOT_PRESERVED"})
                continue
            factor = len(clip) / (right - left)
            if factor > 1.25:
                issues.append({"segment_id": sid, "code": "TTS_TOO_LONG_NO_TRUNCATION", "duration_ratio": factor})
                continue
            if factor > 1:
                fitted = folder / f"fitted-{len(added):06d}.wav"
                _fit_clip(path, fitted, factor)
                clip, _ = sf.read(fitted, dtype="float32")
                # atempo can have a small sample-count difference; never chop a syllable.
                if len(clip) > right - left + RATE // 100:
                    issues.append({"segment_id": sid, "code": "FITTED_VOICE_EXCEEDS_SLOT"})
                    continue
                if left + len(clip) > samples:
                    issues.append({"segment_id": sid, "code": "FITTED_VOICE_EXCEEDS_SOURCE"})
                    continue
            end = left + len(clip)
            voice[left:end] += clip
            used.append((left, max(right, end)))
            added.append({"segment_id": sid, "start_ms": segment["start_ms"],
                          "end_ms": round(end * 1000 / RATE)})
        voice.flush()
        voice_path, control_path = folder / "vietnamese.wav", folder / "sidechain.wav"
        _write_bus(voice_path, voice)
        _write_bus(control_path, voice, control=True)
    finally:
        del voice
        backing.unlink(missing_ok=True)
    output = folder / "mastered.wav"
    graph_path = folder / "ducking.ffgraph"
    graph_path.write_text(ducking_graph(), encoding="utf-8")
    subprocess.run([config.get("ffmpeg_bin", "ffmpeg"), "-hide_banner", "-loglevel", "error",
                    "-i", str(source), "-i", str(voice_path), "-i", str(control_path),
                    "-filter_complex_script", str(graph_path), "-map", "[out]", "-t",
                    str(config["duration_ms"] / 1000), "-ar", str(RATE), "-ac", "2",
                    "-c:a", "pcm_s16le", "-y", str(output)],
                   check=True, capture_output=True, timeout=int(config.get("media_timeout_seconds", 7200)))
    with wave.open(str(output), "rb") as mastered:
        drift = abs(mastered.getnframes() * 1000 / RATE - config["duration_ms"])
    if not math.isfinite(drift) or drift > 50:
        raise ValueError("Mastered audio timeline drift")
    target = folder / "mix.json"
    atomic_json(target, {"schema_version": 1, "audio": str(output), "source_sha256": sha256_file(source),
                        "issues": issues, "voice_added": added, "voice_added_count": len(added),
                        "soundtrack_policy": "ORIGINAL_SOUNDTRACK_MULTIBAND_VOICEOVER",
                        "nominal_duck_db": {"low": -3, "mid": -14, "high": -3},
                        "attack_ms": 100, "release_ms": 300, "separation_used": False,
                        "duration_drift_ms": drift, "speed_claim": "UNMEASURED_ON_CLOUD"})
    return {"audio": str(output), "voice_added_count": len(added),
            "artifacts": [str(target), str(output), str(voice_path), str(control_path), str(graph_path)],
            "stage_status": "DEGRADED" if issues else "SUCCESS", "quality_evidence": {"issues": issues},
            "metrics": {"combined_processing_ms": (time.perf_counter() - tick) * 1000}}


def _fit_clip(source: Path, target: Path, speed: float) -> None:
    """The filter accepts a bounded number; job JSON cannot choose an executable."""
    if not isinstance(speed, (float, int)) or not math.isfinite(speed) or not 1 < speed <= 1.25:
        raise ValueError("Invalid voice fitting speed")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
                    "-af", f"atempo={float(speed):.9f}", "-ar", str(RATE), "-y", str(target)],
                   shell=False, check=True, capture_output=True, timeout=60)
