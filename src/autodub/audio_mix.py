"""Pitch-preserving voice fitting and word-masked audio preview mixing.

The mixer decodes the source once to PCM16 stereo at 48 kHz.  It writes the
decoded source samples back unchanged except inside approved DUB word spans,
where it subtracts the separated-dialogue estimate and adds the fitted voice.
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

_RATE = 48_000
_CHANNELS = 2
_FRAME_BYTES = 4
_BLOCK_FRAMES = _RATE
_RAMP_FRAMES = 240  # 5 ms, kept strictly inside each approved word span.
_CONTAINER_DRIFT_MS = 500  # container metadata vs decoded PCM length tolerance.
_ELASTIC_MIN_SPEED = 0.90  # slowest playback when a clip is shorter than its slot.
_ELASTIC_MAX_SPEED = 1.45  # fastest playback before a clip is cut and flagged.


def _run(command: list[str], timeout: int = 300) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("FFmpeg unavailable or timed out; configure ffmpeg_bin") from error
    if result.returncode:
        raise RuntimeError("FFmpeg failed while preparing preview audio")


def _duration(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2 or wav.getframerate() < 1:
            raise ValueError("audio must be an uncompressed PCM16 WAV")
        if wav.getnchannels() not in (1, 2):
            raise ValueError("audio must have one or two channels")
        return round(wav.getnframes() * 1000 / wav.getframerate())


def _tempo_chain(factor: float) -> str:
    """Build an atempo chain, splitting factors to FFmpeg's supported range."""
    parts: list[float] = []
    while factor > 2.0:
        parts.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        parts.append(0.5)
        factor /= 0.5
    parts.append(factor)
    return ",".join(f"atempo={part:.12g}" for part in parts)


def fit_voice(wav: Path, output: Path, target_ms: int,
              ffmpeg_bin: str = "ffmpeg", *, elastic: bool = False) -> dict[str, Any]:
    """Fit a generated voice clip to its segment without changing pitch.

    Duration drift up to 20% is fitted with FFmpeg atempo.  Larger drift is
    returned as REWRITE for human review, without creating a misleading clip.

    With ``elastic=True`` (automatic pipeline) the clip is always fitted to the
    slot: speed is clamped to ``_ELASTIC_MIN_SPEED``..``_ELASTIC_MAX_SPEED``,
    short clips are padded with silence, and clips still too long are cut with a
    short fade-out and flagged ``review_required``.
    """
    wav, output = Path(wav), Path(output)
    if not isinstance(target_ms, int) or isinstance(target_ms, bool) or target_ms <= 0:
        raise ValueError("target_ms must be a positive integer")
    input_ms = _duration(wav)
    if input_ms <= 0:
        raise ValueError("voice clip must contain audio")
    drift = abs(input_ms - target_ms) / target_ms
    if drift > 0.20 and not elastic:
        return {"wav": wav, "actual_ms": input_ms, "action": "REWRITE",
                "review_required": True, "target_ms": target_ms,
                "duration_drift_ratio": drift}

    action = "STRETCH" if drift <= 0.08 else "SPEED"
    factor = input_ms / target_ms
    truncated = False
    if elastic:
        truncated = factor > _ELASTIC_MAX_SPEED
        factor = min(max(factor, _ELASTIC_MIN_SPEED), _ELASTIC_MAX_SPEED)
        action = "TRUNCATE" if truncated else action
    output.parent.mkdir(parents=True, exist_ok=True)
    # Render to a sibling temporary file so a failed fit never leaves a partial
    # result at the requested output path.
    with tempfile.TemporaryDirectory(prefix="audio-fit-", dir=output.parent) as td:
        intermediate = Path(td) / "tempo.wav"
        _run([ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", str(wav),
              "-af", _tempo_chain(factor), "-ac", "2", "-ar", str(_RATE),
              "-c:a", "pcm_s16le", "-y", str(intermediate)])
        with wave.open(str(intermediate), "rb") as rendered:
            frames = rendered.getnframes()
            fitted_pcm = rendered.readframes(frames)
        expected_frames = round(target_ms * _RATE / 1000)
        actual_frames = len(fitted_pcm) // _FRAME_BYTES
        # atempo rounding may differ by a few samples.  Permit only 20 ms of
        # correction; larger errors signal a bad fit and must be reviewed.
        if not elastic and abs(actual_frames - expected_frames) > round(_RATE * 0.020):
            raise RuntimeError("FFmpeg fit missed the target by more than 20 ms")
        if actual_frames < expected_frames:
            fitted_pcm += b"\0" * ((expected_frames - actual_frames) * _FRAME_BYTES)
        elif actual_frames > expected_frames:
            fitted_pcm = fitted_pcm[:expected_frames * _FRAME_BYTES]
            if elastic:
                samples = _pcm16_frames(fitted_pcm)
                fade = min(expected_frames, round(_RATE * 0.015))
                for step in range(fade):
                    gain = (step + 1) / fade
                    base = (expected_frames - 1 - step) * _CHANNELS
                    for channel in range(_CHANNELS):
                        samples[base + channel] = int(samples[base + channel] * gain)
                fitted_pcm = _bytes(samples)
        with wave.open(str(output), "wb") as rendered:
            rendered.setnchannels(_CHANNELS)
            rendered.setsampwidth(2)
            rendered.setframerate(_RATE)
            rendered.writeframes(fitted_pcm)
    return {"wav": output, "actual_ms": round(expected_frames * 1000 / _RATE),
            "action": action, "review_required": truncated, "target_ms": target_ms,
            "duration_drift_ratio": drift, "speed_factor": factor}


def _as_dict(segment: Any) -> dict[str, Any]:
    if isinstance(segment, dict):
        return segment
    if hasattr(segment, "model_dump"):
        return segment.model_dump()
    raise ValueError("segments must be Segment objects or dictionaries")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _decode_stereo(path: Path, output: Path, ffmpeg_bin: str) -> None:
    _run([ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", str(path),
          "-vn", "-ac", "2", "-ar", str(_RATE), "-c:a", "pcm_s16le",
          "-y", str(output)])


def _pcm16_frames(raw: bytes) -> array.array:
    samples = array.array("h")
    samples.frombytes(raw)
    if os.sys.byteorder != "little":
        samples.byteswap()
    return samples


def _bytes(samples: array.array) -> bytes:
    if os.sys.byteorder == "little":
        return samples.tobytes()
    copy = array.array("h", samples)
    copy.byteswap()
    return copy.tobytes()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mix_preview(source: Path, output: Path, segments: list,
                separation_path: Path, clips: dict[str, Path], *,
                ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    """Create an audio-only preview with source changes limited to DUB words.

    ``separation_path`` is schema-1 JSON. Each window has integer start_ms and
    end_ms plus a dialogue WAV path relative to the JSON file's directory.
    ``clips`` maps each DUB segment id to its synthesized voice WAV.
    """
    source, output, separation_path = Path(source), Path(output), Path(separation_path)
    try:
        metadata = json.loads(separation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("separation_path must be a readable schema-1 JSON file") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("unsupported separation schema")
    windows = metadata.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("separation windows are required")

    source_fingerprint = _sha256(source)
    normalized: list[dict[str, Any]] = []
    for raw in segments:
        seg = _as_dict(raw)
        action = seg.get("action", "KEEP")
        if action == "NEEDS_REVIEW" or seg.get("needs_review") is True:
            raise ValueError("segments requiring review cannot be mixed")
        start = _integer(seg.get("start_ms"), "segment start_ms")
        end = _integer(seg.get("end_ms"), "segment end_ms", 1)
        if end <= start:
            raise ValueError("segment end_ms must be greater than start_ms")
        if action not in ("DUB", "KEEP"):
            raise ValueError("segment action must be DUB, KEEP, or NEEDS_REVIEW")
        words = seg.get("words", [])
        if not isinstance(words, list):
            raise ValueError("segment words must be a list")
        item = {"id": str(seg.get("id", "")), "start_ms": start,
                "end_ms": end, "action": action, "words": []}
        if action == "DUB":
            if not item["id"] or not words:
                raise ValueError("every DUB segment must have an id and timed words")
            for word in words:
                if not isinstance(word, dict):
                    if hasattr(word, "model_dump"):
                        word = word.model_dump()
                    else:
                        raise ValueError("word spans must be dictionaries or Word objects")
                ws = _integer(word.get("s"), "word start_ms")
                we = _integer(word.get("e"), "word end_ms", 1)
                if not start <= ws < we <= end:
                    raise ValueError("word span must stay within its DUB segment")
                item["words"].append((ws, we))
            item["words"].sort()
            if any(a[1] > b[0] for a, b in zip(item["words"], item["words"][1:], strict=False)):
                raise ValueError("DUB word spans must not overlap")
            if item["id"] not in clips:
                raise ValueError(f"missing synthesized clip for segment {item['id']}")
        normalized.append(item)

    dub_words = sorted((a, b, seg) for seg in normalized if seg["action"] == "DUB"
                       for a, b in seg["words"])
    if any(prev[1] > current[0] for prev, current in zip(dub_words, dub_words[1:], strict=False)):
        raise ValueError("DUB word spans must not overlap")
    dub_ids = [seg["id"] for seg in normalized if seg["action"] == "DUB"]
    if len(dub_ids) != len(set(dub_ids)):
        raise ValueError("DUB segment ids must be unique")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audio-mix-", dir=output.parent) as td:
        temp = Path(td)
        source_pcm = temp / "source.wav"
        _decode_stereo(source, source_pcm, ffmpeg_bin)
        with wave.open(str(source_pcm), "rb") as src:
            if src.getnchannels() != 2 or src.getsampwidth() != 2 or src.getframerate() != _RATE:
                raise RuntimeError("source decode did not produce stereo PCM16 48 kHz")
            source_frames = src.getnframes()
            source_ms = round(source_frames * 1000 / _RATE)
        # Container metadata duration can exceed the decoded PCM length by a few
        # packet-framing milliseconds; tolerate and clamp that drift only.
        for seg in normalized:
            if seg["end_ms"] > source_ms:
                if seg["end_ms"] - source_ms > _CONTAINER_DRIFT_MS:
                    raise ValueError("segment timing exceeds source duration")
                seg["end_ms"] = source_ms

        # Prepare one decoded PCM reader per separation window and per clip.
        readers: dict[str, wave.Wave_read] = {}
        try:
            separation_events: list[dict[str, Any]] = []
            root = separation_path.resolve().parent
            for index, window in enumerate(windows):
                if not isinstance(window, dict):
                    raise ValueError("separation windows must be objects")
                start = _integer(window.get("start_ms"), "window start_ms")
                end = _integer(window.get("end_ms"), "window end_ms", 1)
                relative = window.get("dialogue")
                if not isinstance(relative, str) or not relative:
                    raise ValueError("separation window dialogue path is required")
                path = (root / relative).resolve()
                if path != root and root not in path.parents:
                    raise ValueError("dialogue path must stay within separation folder")
                if end <= start or end > source_ms:
                    raise ValueError("separation window is outside source bounds")
                decoded = temp / f"dialogue-{index}.wav"
                _decode_stereo(path, decoded, ffmpeg_bin)
                reader = wave.open(str(decoded), "rb")
                if (reader.getnchannels() != 2 or reader.getsampwidth() != 2
                        or reader.getframerate() != _RATE):
                    reader.close()
                    raise ValueError("dialogue audio must decode as stereo PCM16 48 kHz")
                length = round((end - start) * _RATE / 1000)
                if abs(reader.getnframes() - length) > round(_RATE * 0.050):
                    reader.close()
                    raise ValueError("dialogue window duration differs by more than 50 ms")
                key = f"dialogue:{index}"
                readers[key] = reader
                separation_events.append({"start": round(start * _RATE / 1000),
                                          "end": round(end * _RATE / 1000),
                                          "key": key, "frames": reader.getnframes()})

            clip_events: dict[str, dict[str, Any]] = {}
            for index, seg in enumerate(s for s in normalized if s["action"] == "DUB"):
                decoded = temp / f"clip-{index}.wav"
                clip_path = Path(clips[seg["id"]])
                _decode_stereo(clip_path, decoded, ffmpeg_bin)
                reader = wave.open(str(decoded), "rb")
                if (reader.getnchannels() != 2 or reader.getsampwidth() != 2
                        or reader.getframerate() != _RATE or reader.getnframes() <= 0):
                    reader.close()
                    raise ValueError("voice clips must decode as nonempty stereo PCM16 48 kHz")
                key = f"clip:{seg['id']}"
                target_frames = round((seg["end_ms"] - seg["start_ms"]) * _RATE / 1000)
                if abs(reader.getnframes() - target_frames) > round(_RATE * 0.020):
                    reader.close()
                    raise ValueError("voice clip must be fitted to its segment before mixing")
                readers[key] = reader
                clip_events[seg["id"]] = {"key": key,
                    "start": round(seg["start_ms"] * _RATE / 1000),
                    "end": round(seg["end_ms"] * _RATE / 1000),
                    "frames": reader.getnframes()}

            # Every replaced word needs a dialogue estimate covering its full span.
            for a, b, _seg in dub_words:
                first, last = round(a * _RATE / 1000), round(b * _RATE / 1000)
                covered = first
                for event in sorted(separation_events, key=lambda e: e["start"]):
                    lo, hi = max(first, event["start"]), min(last, event["end"])
                    if hi <= lo or lo > covered:
                        continue
                    covered = max(covered, hi)
                    if covered >= last:
                        break
                if covered < last:
                    raise ValueError("separation windows must cover every DUB word span")

            output_tmp = temp / "preview.wav"
            with wave.open(str(source_pcm), "rb") as src, wave.open(str(output_tmp), "wb") as dst:
                dst.setnchannels(2)
                dst.setsampwidth(2)
                dst.setframerate(_RATE)
                start_frame = 0
                sample_peak = 0
                clipped_samples = 0
                sample_count = 0
                while start_frame < source_frames:
                    count = min(_BLOCK_FRAMES, source_frames - start_frame)
                    samples = _pcm16_frames(src.readframes(count))
                    end_frame = start_frame + count
                    for a_ms, b_ms, seg in dub_words:
                        a, b = round(a_ms * _RATE / 1000), round(b_ms * _RATE / 1000)
                        lo, hi = max(a, start_frame), min(b, end_frame)
                        if hi <= lo:
                            continue
                        clip = clip_events[seg["id"]]
                        clip_reader = readers[clip["key"]]
                        seg_start, seg_end = clip["start"], clip["end"]
                        seg_length = seg_end - seg_start
                        # Clip frames are mapped across the segment interval; the
                        # fitting stage is expected to bring their durations close.
                        clip_lo = max(0, round((lo - seg_start) * clip["frames"] / seg_length))
                        clip_hi = min(clip["frames"], round((hi - seg_start) * clip["frames"] / seg_length))
                        clip_reader.setpos(clip_lo)
                        voice = _pcm16_frames(clip_reader.readframes(clip_hi - clip_lo))
                        cursor = lo
                        while cursor < hi:
                            # Windows can overlap at chunk boundaries; pick one
                            # estimate per sample so we never subtract twice.
                            selected = next((event for event in sorted(separation_events,
                                key=lambda e: e["start"])
                                if event["start"] <= cursor < event["end"]), None)
                            if selected is None:
                                raise ValueError("separation windows must cover every DUB word span")
                            dialogue_start = cursor
                            dialogue_end = min(hi, selected["end"],
                                selected["start"] + selected["frames"])
                            if dialogue_end <= dialogue_start:
                                raise ValueError("dialogue samples do not cover the approved word span")
                            dialogue_reader = readers[selected["key"]]
                            dialogue_reader.setpos(dialogue_start - selected["start"])
                            dialogue = _pcm16_frames(dialogue_reader.readframes(dialogue_end - dialogue_start))
                            for frame in range(dialogue_start, dialogue_end):
                                idx = (frame - start_frame) * 2
                                didx = (frame - dialogue_start) * 2
                                voice_frame = round((frame - seg_start) * clip["frames"] / seg_length)
                                voice_index = min(max(0, voice_frame - clip_lo),
                                                  max(0, len(voice) // 2 - 1)) * 2
                                weight = min(1.0, (frame - a + 1) / _RAMP_FRAMES,
                                             (b - frame) / _RAMP_FRAMES)
                                weight = max(0.0, weight)
                                for ch in (0, 1):
                                    mixed = samples[idx + ch] - dialogue[didx + ch] + voice[voice_index + ch]
                                    samples[idx + ch] = max(-32768, min(32767,
                                        round(samples[idx + ch] * (1.0 - weight) + mixed * weight)))
                            cursor = dialogue_end
                    sample_peak = max(sample_peak, max(abs(value) for value in samples))
                    clipped_samples += sum(1 for value in samples if abs(value) >= 32767)
                    sample_count += len(samples)
                    dst.writeframesraw(_bytes(samples))
                    start_frame = end_frame
            os.replace(output_tmp, output)
        finally:
            for reader in readers.values():
                reader.close()

    qc: dict[str, Any]
    with wave.open(str(output), "rb") as result:
        qc = {"sample_rate_hz": result.getframerate(), "channels": result.getnchannels(),
              "sample_format": "pcm_s16le", "duration_ms": round(result.getnframes() * 1000 / _RATE),
              "peak_dbfs": 20 * math.log10(sample_peak / 32768) if sample_peak else None,
              "clipped_sample_ratio": clipped_samples / sample_count if sample_count else 0.0}
    if _sha256(source) != source_fingerprint:
        raise RuntimeError("source file changed during mixing")
    return {"wav": output, "actual_ms": qc["duration_ms"], "sample_rate_hz": _RATE,
            "channels": _CHANNELS, "sample_format": "pcm_s16le", "qc": qc,
            "modified_ranges_ms": [[a, b] for a, b, _ in dub_words],
            "outside_mask_preserved": True, "source_sha256": source_fingerprint,
            "human_listening_required": True, "automatic_quality_pass": False}
