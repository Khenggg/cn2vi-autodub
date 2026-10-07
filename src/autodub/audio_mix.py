"""Pitch-preserving voice fitting and word-masked audio preview mixing.

The mixer decodes the source to PCM16 stereo at 48 kHz. It preserves the original
component everywhere, subtracts estimated dialogue only under aligned DUB words,
and adds fitted Vietnamese continuously over each DUB utterance slot.
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
_ELASTIC_MAX_SPEED = 1.45  # legacy benchmark speed clamp; never cuts speech.


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

    ``elastic`` remains for old benchmark callers only.  It is never allowed to
    cut speech: an overlong clip is returned for rewrite in either mode.
    """
    wav, output = Path(wav), Path(output)
    if not isinstance(target_ms, int) or isinstance(target_ms, bool) or target_ms <= 0:
        raise ValueError("target_ms must be a positive integer")
    input_ms = _duration(wav)
    if input_ms <= 0:
        raise ValueError("voice clip must contain audio")
    drift = abs(input_ms - target_ms) / target_ms
    base = {"wav": wav, "actual_ms": input_ms, "raw_ms": input_ms,
            "target_ms": target_ms, "duration_drift_ratio": drift}
    if drift > 0.20:
        return {**base, "action": "REWRITE", "review_required": True,
                "failure_code": "DURATION_DRIFT_OVER_20_PERCENT"}

    action = "STRETCH" if drift <= 0.08 else "SPEED"
    factor = input_ms / target_ms
    if elastic:
        # Historical tests/benchmarks opt into elastic fitting. Clamp playback
        # speed but reject any result that would require cutting source speech.
        factor = min(max(factor, _ELASTIC_MIN_SPEED), _ELASTIC_MAX_SPEED)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Render to a sibling temporary file so a failed fit never leaves a partial
    # result at the requested output path.
    with tempfile.TemporaryDirectory(prefix="audio-fit-", dir=output.parent) as td:
        intermediate = Path(td) / "tempo.wav"
        try:
            _run([ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-i", str(wav),
                  "-af", _tempo_chain(factor), "-ac", "2", "-ar", str(_RATE),
                  "-c:a", "pcm_s16le", "-y", str(intermediate)])
        except RuntimeError as error:
            return {**base, "wav": wav, "action": "NEEDS_REVIEW",
                    "review_required": True, "failure_code": "FFMPEG_FIT_FAILED",
                    "review_reason": str(error)}
        with wave.open(str(intermediate), "rb") as rendered:
            frames = rendered.getnframes()
            fitted_pcm = rendered.readframes(frames)
        expected_frames = round(target_ms * _RATE / 1000)
        actual_frames = len(fitted_pcm) // _FRAME_BYTES
        # atempo rounding may differ by a few samples.  Permit only 20 ms of
        # correction; larger errors signal a bad fit and must be reviewed.
        if abs(actual_frames - expected_frames) > round(_RATE * 0.020):
            return {**base, "wav": wav, "action": "NEEDS_REVIEW",
                    "review_required": True, "failure_code": "FFMPEG_FIT_MISSED_TARGET",
                    "fitted_ms": round(actual_frames * 1000 / _RATE)}
        if actual_frames < expected_frames:
            fitted_pcm += b"\0" * ((expected_frames - actual_frames) * _FRAME_BYTES)
        elif actual_frames > expected_frames:
            fitted_pcm = fitted_pcm[:expected_frames * _FRAME_BYTES]
        with wave.open(str(output), "wb") as rendered:
            rendered.setnchannels(_CHANNELS)
            rendered.setsampwidth(2)
            rendered.setframerate(_RATE)
            rendered.writeframes(fitted_pcm)
    return {**base, "wav": output, "actual_ms": round(expected_frames * 1000 / _RATE),
            "action": action, "review_required": False, "speed_factor": factor}


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
    """Mix voice across each DUB slot and remove source dialogue only on words.

    The original component is retained everywhere.  The separated dialogue is
    subtracted only under aligned source-word masks; synthesized Vietnamese is
    added continuously over the complete fitted utterance slot.
    """
    source, output, separation_path = Path(source), Path(output), Path(separation_path)
    input_paths = [source, separation_path, *(Path(path) for path in clips.values())]
    if output.resolve() in {path.resolve() for path in input_paths}:
        raise ValueError("preview output must not overwrite an input file")
    normalized: list[dict[str, Any]] = []
    for raw in segments:
        seg = _as_dict(raw)
        action = seg.get("action", "KEEP")
        if action == "NEEDS_REVIEW" or seg.get("needs_review") is True:
            raise ValueError("segments requiring review cannot be mixed")
        if action not in ("DUB", "KEEP"):
            raise ValueError("segment action must be DUB, KEEP, or NEEDS_REVIEW")
        start = _integer(seg.get("start_ms"), "segment start_ms")
        end = _integer(seg.get("end_ms"), "segment end_ms", 1)
        if end <= start:
            raise ValueError("segment end_ms must be greater than start_ms")
        item = {"id": str(seg.get("id", "")), "start_ms": start,
                "end_ms": end, "action": action, "words": []}
        if action == "DUB":
            words = seg.get("words")
            if not item["id"] or not isinstance(words, list) or not words:
                raise ValueError("every DUB segment needs an id and real aligned words")
            for word in words:
                word = _as_dict(word) if not isinstance(word, dict) else word
                ws, we = _integer(word.get("s"), "word start_ms"), _integer(word.get("e"), "word end_ms", 1)
                if not start <= ws < we <= end:
                    raise ValueError("word span must stay within its DUB segment")
                item["words"].append((ws, we))
            item["words"].sort()
            if any(a[1] > b[0] for a, b in zip(item["words"], item["words"][1:], strict=False)):
                raise ValueError("DUB word spans must not overlap")
        normalized.append(item)

    dubs = [seg for seg in normalized if seg["action"] == "DUB"]
    dub_ids = [seg["id"] for seg in dubs]
    if len(dub_ids) != len(set(dub_ids)):
        raise ValueError("DUB segment ids must be unique")
    ordered = sorted(normalized, key=lambda item: (item["start_ms"], item["end_ms"]))
    if any(a["end_ms"] > b["start_ms"] for a, b in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("segment timelines must not overlap")
    if set(clips) != set(dub_ids):
        raise ValueError("clip ids must match DUB segment ids exactly")
    # Validate segment/clip relationships before opening any media files.
    source_fingerprint = _sha256(source)
    dub_words = sorted((a, b, seg) for seg in dubs for a, b in seg["words"])
    if any(a[1] > b[0] for a, b in zip(dub_words, dub_words[1:], strict=False)):
        raise ValueError("DUB word spans must not overlap")
    metadata = None
    if dubs:
        try:
            metadata = json.loads(separation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("separation_path must be a readable schema-1 JSON file") from error
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("unsupported separation schema")
        windows = metadata.get("windows", [])
        if not isinstance(windows, list):
            raise ValueError("separation windows must be a list")
        if not windows:
            # A missing/empty separation result produces a valid decoded-source
            # preview. It does not claim that any DUB audio was applied.
            dubs, dub_words = [], []

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audio-mix-", dir=output.parent) as td:
        temp = Path(td)
        source_pcm = temp / "source.wav"
        _decode_stereo(source, source_pcm, ffmpeg_bin)
        with wave.open(str(source_pcm), "rb") as src:
            if (src.getnchannels(), src.getsampwidth(), src.getframerate()) != (2, 2, _RATE):
                raise RuntimeError("source decode did not produce stereo PCM16 48 kHz")
            source_frames = src.getnframes()
            source_ms = round(source_frames * 1000 / _RATE)

        for seg in normalized:
            if seg["end_ms"] > source_ms:
                if seg["end_ms"] - source_ms > _CONTAINER_DRIFT_MS:
                    raise ValueError("segment timing exceeds source duration")
                seg["end_ms"] = source_ms
                if seg["start_ms"] >= source_ms:
                    raise ValueError("segment timing falls outside decoded source audio")
        dub_words = sorted((a, b, seg) for seg in dubs for a, b in seg["words"])

        # With no dubbed slots the decoded source is already the required
        # stereo/48 kHz pass-through; separation files and windows are irrelevant.
        if not dubs:
            os.replace(source_pcm, output)
            readers: dict[str, wave.Wave_read] = {}
            separation_events: list[dict[str, Any]] = []
            clip_events: dict[str, dict[str, Any]] = {}
            sample_peak = clipped_samples = sample_count = 0
        else:
            readers = {}
            separation_events = []
            clip_events = {}
            try:
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
                    if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (2, 2, _RATE):
                        reader.close()
                        raise ValueError("dialogue audio must decode as stereo PCM16 48 kHz")
                    expected = round((end - start) * _RATE / 1000)
                    if abs(reader.getnframes() - expected) > round(_RATE * 0.050):
                        reader.close()
                        raise ValueError("dialogue window duration differs by more than 50 ms")
                    key = f"dialogue:{index}"
                    readers[key] = reader
                    separation_events.append({"start": round(start * _RATE / 1000),
                        "end": round(end * _RATE / 1000), "key": key, "frames": reader.getnframes()})
                separation_events.sort(key=lambda event: (event["start"], event["end"]))

                for index, seg in enumerate(dubs):
                    decoded = temp / f"clip-{index}.wav"
                    _decode_stereo(Path(clips[seg["id"]]), decoded, ffmpeg_bin)
                    reader = wave.open(str(decoded), "rb")
                    if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (2, 2, _RATE) or reader.getnframes() <= 0:
                        reader.close()
                        raise ValueError("voice clips must decode as nonempty stereo PCM16 48 kHz")
                    seg_start, seg_end = round(seg["start_ms"] * _RATE / 1000), round(seg["end_ms"] * _RATE / 1000)
                    slot_frames = seg_end - seg_start
                    if abs(reader.getnframes() - slot_frames) > round(_RATE * 0.020):
                        reader.close()
                        raise ValueError("voice clip must be fitted to its complete DUB slot within 20 ms")
                    key = f"clip:{seg['id']}"
                    readers[key] = reader
                    clip_events[seg["id"]] = {"key": key, "start": seg_start,
                        "end": seg_end, "frames": reader.getnframes()}

                # Each actual aligned word must be covered by decoded dialogue
                # samples. Overlap uses the earliest-starting window consistently.
                for a_ms, b_ms, _seg in dub_words:
                    first, last = round(a_ms * _RATE / 1000), round(b_ms * _RATE / 1000)
                    cursor = first
                    while cursor < last:
                        selected = next((event for event in separation_events
                            if event["start"] <= cursor < min(event["end"], event["start"] + event["frames"])), None)
                        if selected is None:
                            raise ValueError("separation windows must cover every DUB word span")
                        cursor = min(last, selected["end"], selected["start"] + selected["frames"])

                output_tmp = temp / "preview.wav"
                with wave.open(str(source_pcm), "rb") as src, wave.open(str(output_tmp), "wb") as dst:
                    dst.setnchannels(2)
                    dst.setsampwidth(2)
                    dst.setframerate(_RATE)
                    start_frame = 0
                    sample_peak = clipped_samples = sample_count = 0
                    word_frames = [(round(a * _RATE / 1000), round(b * _RATE / 1000), seg)
                                   for a, b, seg in dub_words]
                    while start_frame < source_frames:
                        count = min(_BLOCK_FRAMES, source_frames - start_frame)
                        samples = _pcm16_frames(src.readframes(count))
                        mixed_values = [float(value) for value in samples]
                        end_frame = start_frame + count

                        # Add each fitted Vietnamese clip across its full slot,
                        # including gaps between source-language words.
                        for seg in dubs:
                            clip = clip_events[seg["id"]]
                            lo, hi = max(clip["start"], start_frame), min(clip["end"], end_frame)
                            if hi <= lo:
                                continue
                            slot = clip["end"] - clip["start"]
                            clip_lo = max(0, (lo - clip["start"]) * clip["frames"] // slot)
                            clip_hi = min(clip["frames"], ((hi - clip["start"]) * clip["frames"] + slot - 1) // slot)
                            voice_reader = readers[clip["key"]]
                            voice_reader.setpos(clip_lo)
                            voice = _pcm16_frames(voice_reader.readframes(clip_hi - clip_lo))
                            for frame in range(lo, hi):
                                vi = min(clip["frames"] - 1,
                                    (frame - clip["start"]) * clip["frames"] // slot)
                                voice_idx = (vi - clip_lo) * 2
                                gain = min(1.0, (frame - clip["start"] + 1) / _RAMP_FRAMES,
                                           (clip["end"] - frame) / _RAMP_FRAMES)
                                idx = (frame - start_frame) * 2
                                for ch in (0, 1):
                                    mixed_values[idx + ch] += voice[voice_idx + ch] * max(0.0, gain)

                        # Subtract the separated source dialogue only under true
                        # aligned source words. The window choice is stable and
                        # each sample receives at most one subtraction.
                        for a, b, _seg in word_frames:
                            lo, hi = max(a, start_frame), min(b, end_frame)
                            cursor = lo
                            while cursor < hi:
                                selected = next((event for event in separation_events
                                    if event["start"] <= cursor < min(event["end"], event["start"] + event["frames"])), None)
                                if selected is None:
                                    raise ValueError("dialogue samples do not cover the approved word span")
                                stop = min(hi, selected["end"], selected["start"] + selected["frames"])
                                dialogue_reader = readers[selected["key"]]
                                dialogue_reader.setpos(cursor - selected["start"])
                                dialogue = _pcm16_frames(dialogue_reader.readframes(stop - cursor))
                                for frame in range(cursor, stop):
                                    index = (frame - start_frame) * 2
                                    didx = (frame - cursor) * 2
                                    gain = min(1.0, (frame - a + 1) / _RAMP_FRAMES, (b - frame) / _RAMP_FRAMES)
                                    for ch in (0, 1):
                                        mixed_values[index + ch] -= dialogue[didx + ch] * max(0.0, gain)
                                cursor = stop

                        samples = array.array("h", (max(-32768, min(32767, round(value)))
                                                      for value in mixed_values))
                        sample_peak = max(sample_peak, max((abs(value) for value in samples), default=0))
                        clipped_samples += sum(1 for value in samples if abs(value) >= 32767)
                        sample_count += len(samples)
                        dst.writeframesraw(_bytes(samples))
                        start_frame = end_frame
                os.replace(output_tmp, output)
            finally:
                for reader in readers.values():
                    reader.close()

    sample_peak = clipped_samples = sample_count = 0
    with wave.open(str(output), "rb") as result:
        actual_ms = round(result.getnframes() * 1000 / _RATE)
        duration_drift_ms = abs(actual_ms - source_ms)
        while True:
            block = result.readframes(_BLOCK_FRAMES)
            if not block:
                break
            values = _pcm16_frames(block)
            sample_peak = max(sample_peak, max((abs(value) for value in values), default=0))
            clipped_samples += sum(1 for value in values if abs(value) >= 32767)
            sample_count += len(values)
    clipping_ratio = clipped_samples / sample_count if sample_count else 0.0
    qc = {"sample_rate_hz": _RATE, "channels": _CHANNELS, "sample_format": "pcm_s16le",
          "duration_ms": actual_ms, "duration_drift_ms": duration_drift_ms,
          "peak_dbfs": 20 * math.log10(sample_peak / 32768) if sample_peak else None,
          "clipped_sample_ratio": clipping_ratio,
          "duration_pass": duration_drift_ms <= 20,
          "clipping_pass": clipping_ratio <= 0.00001,
          "technical_quality_pass": duration_drift_ms <= 20 and clipping_ratio <= 0.00001}
    if _sha256(source) != source_fingerprint:
        raise RuntimeError("source file changed during mixing")
    return {"wav": output, "actual_ms": actual_ms, "sample_rate_hz": _RATE,
            "channels": _CHANNELS, "sample_format": "pcm_s16le", "qc": qc,
            "modified_ranges_ms": [[a, b] for a, b, _ in dub_words],
            "voice_added_ranges_ms": [[s["start_ms"], s["end_ms"]] for s in dubs],
            "outside_mask_preserved": True,
            "source_component_preserved_outside_removal_mask": True,
            "output_differs_outside_removal_mask_due_to_voice_addition": bool(dubs),
            "source_sha256": source_fingerprint, "human_listening_required": True,
            "automatic_quality_pass": False, "technical_quality_pass": qc["technical_quality_pass"]}
