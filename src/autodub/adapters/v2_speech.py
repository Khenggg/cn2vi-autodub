"""Voice candidates and batched FireRed recognition on the original soundtrack."""
from __future__ import annotations

import json
import math
import sys
import time
import wave
from pathlib import Path

from autodub.adapters.common import configure_offline, identity
from autodub.adapters.runtime_paths import asset, output_folder
from autodub.benchmedia import extract_audio
from autodub.domain import merge_windows
from autodub.storage import atomic_json, sha256_file

RATE = 16000
FRAME_MS = 30


def candidate_windows(flags: list[bool], duration_ms: int, *, padding_ms: int = 180,
                      gap_ms: int = 240, max_window_ms: int = 15000) -> list[dict]:
    """Keep short speech; padding and gaps never impose a quota on dialogue coverage."""
    if duration_ms <= 0 or not 300 <= max_window_ms <= 30000:
        raise ValueError("Invalid candidate timeline")
    spans, start = [], None
    for index, active in enumerate([*flags, False]):
        at = min(duration_ms, index * FRAME_MS)
        if active and start is None:
            start = at
        elif not active and start is not None:
            if at > start:
                spans.append((start, at))
            start = None
    result = []
    for left, right in merge_windows(spans, duration_ms, padding_ms, gap_ms):
        for begin in range(left, right, max_window_ms):
            result.append({"start_ms": begin, "end_ms": min(right, begin + max_window_ms),
                           "vocal_run_ms": right - left})
    return result


def detect(source: Path, config: dict) -> dict:
    import webrtcvad

    folder = output_folder(config)
    audio = folder / "original-16k.wav"
    extract_audio(source, audio, end_ms=config["duration_ms"], sample_rate=RATE, channels=1,
                  ffmpeg_bin=config.get("ffmpeg_bin", "ffmpeg"))
    detector = webrtcvad.Vad(int(config.get("vad_aggressiveness", 0)))
    tick = time.perf_counter()
    flags = []
    with wave.open(str(audio), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (RATE, 1, 2):
            raise ValueError("VAD requires mono 16-bit 16kHz PCM")
        size = RATE * FRAME_MS // 1000
        while block := stream.readframes(size):
            block += b"\0" * (size * 2 - len(block))
            flags.append(detector.is_speech(block, RATE))
    windows = candidate_windows(flags, config["duration_ms"],
                                max_window_ms=int(config.get("asr_window_ms", 15000)))
    target = folder / "candidates.json"
    atomic_json(target, {"schema_version": 1, "audio": str(audio), "windows": windows,
                        "source_sha256": sha256_file(source), "detector": "WebRTC-VAD",
                        "candidate_duration_ms": sum(w["end_ms"] - w["start_ms"] for w in windows),
                        "candidate_is_dialogue": False})
    return {"artifacts": [str(target), str(audio)], "audio": str(audio), "windows": windows,
            "metrics": {"inference_ms": (time.perf_counter() - tick) * 1000}}


def vocal_features(samples, rate: int = RATE) -> dict:
    """Weak DSP evidence, never a calibrated singing classifier."""
    import numpy as np

    pitches, periodicity = [], []
    size = rate // 20
    for start in range(0, len(samples) - size, size * 2):
        block = np.asarray(samples[start:start + size], dtype="float64")
        block -= block.mean()
        energy = float(np.dot(block, block))
        if energy < size * 1e-6:
            continue
        spectrum = np.fft.rfft(block * np.hanning(size), n=size * 2)
        corr = np.fft.irfft(spectrum * np.conj(spectrum))[:size]
        lo, hi = rate // 400, rate // 80
        lag = lo + int(np.argmax(corr[lo:hi]))
        strength = float(corr[lag] / max(corr[0], 1e-12))
        periodicity.append(strength)
        if strength > 0.65:
            pitches.append(rate / lag)
    span = 0.0
    if len(pitches) >= 5:
        span = float(12 * np.log2(np.percentile(pitches, 90) / np.percentile(pitches, 10)))
    return {"pitched_fraction": len(pitches) / max(1, len(periodicity)),
            "pitch_span_semitones": span, "singing_classifier_calibrated": False}


def recognize(source: Path, config: dict) -> dict:
    import numpy as np

    code, code_manifest = asset(config, "firered-code")
    weights, manifest = asset(config, "firered-asr2-aed")
    sys.path.insert(0, str(code / "source"))
    from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config

    configure_offline(config)
    candidates = json.loads(Path(config["candidates_path"]).read_text(encoding="utf-8"))
    windows = candidates["windows"]
    folder = output_folder(config)
    if candidates["source_sha256"] != config["source_sha256"]:
        raise ValueError("Candidate source changed")
    batch_size = int(config.get("asr_batch_size", 4))
    if not 1 <= batch_size <= 16:
        raise ValueError("ASR batch size outside memory budget")
    segments, issues, load_ms, inference_ms = [], [], 0.0, 0.0
    if windows:
        tick = time.perf_counter()
        model = FireRedAsr2.from_pretrained("aed", str(weights), FireRedAsr2Config(
            use_gpu=config.get("device", "cuda:0").startswith("cuda"), use_half=True,
            return_timestamp=True, beam_size=3))
        native_flags = []
        original = model._get_and_fix_timestamp

        def timestamps(hypothesis, ids, duration):
            present = hypothesis.get("timestamp") is not None
            native_flags.append(present)
            return original(hypothesis, ids, duration) if present else []

        model._get_and_fix_timestamp = timestamps
        load_ms = (time.perf_counter() - tick) * 1000
        ordered = sorted(enumerate(windows), key=lambda x: x[1]["end_ms"] - x[1]["start_ms"])
        with wave.open(str(source), "rb") as audio:
            for start in range(0, len(ordered), batch_size):
                group = ordered[start:start + batch_size]
                ids, paths, features = [], [], []
                for index, window in group:
                    left, right = window["start_ms"], window["end_ms"]
                    audio.setpos(round(left * RATE / 1000))
                    raw = audio.readframes(round((right - left) * RATE / 1000))
                    wav = folder / f"candidate-{index:06d}.wav"
                    with wave.open(str(wav), "wb") as clip:
                        clip.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
                        clip.writeframes(raw)
                    features.append(vocal_features(np.frombuffer(raw, dtype="<i2").astype("float32") / 32768))
                    ids.append(f"seg_{left:012d}_{index:06d}")
                    paths.append(str(wav))
                tick = time.perf_counter()
                rows = model.transcribe(ids, paths)
                inference_ms += (time.perf_counter() - tick) * 1000
                # FireRed's feature extractor sorts the batch; returned utterance
                # IDs, not the original position, own the absolute time window.
                returned_ids = [row.get("uttid") for row in rows]
                if len(rows) != len(group) or set(returned_ids) != set(ids) or len(set(returned_ids)) != len(ids):
                    raise ValueError("ASR batch coverage mismatch")
                row_by_id = dict(zip(returned_ids, rows, strict=True))
                flags_by_id = dict(zip(returned_ids, native_flags, strict=True)) if len(native_flags) == len(rows) else {}
                for position, ((_, window), sid) in enumerate(zip(group, ids, strict=True)):
                    row = row_by_id[sid]
                    text = str(row.get("text", "")).strip()
                    if not text:
                        issues.append({"segment_id": sid, "code": "EMPTY_VOCAL_CANDIDATE"})
                    words = []
                    if flags_by_id.get(sid):
                        invalid_timing = False
                        for token, left, right in row.get("timestamp", []):
                            s = window["start_ms"] + round(float(left) * 1000)
                            e = window["start_ms"] + round(float(right) * 1000)
                            if window["start_ms"] <= s < e <= window["end_ms"] and e - s >= 20:
                                words.append({"t": token, "s": s, "e": e})
                            else:
                                invalid_timing = True
                        if invalid_timing:
                            words = []
                            issues.append({"segment_id": sid, "code": "INVALID_NATIVE_WORD_TIME", "action": "USE_VAD_WINDOW"})
                    confidence = row.get("confidence")
                    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        confidence = None
                    segments.append({"id": sid, "start_ms": window["start_ms"], "end_ms": window["end_ms"],
                        "zh_text": text, "asr_text": text, "words": words, "action": "KEEP",
                        "timing_source": "NATIVE_WORDS" if words else "VAD_WINDOW",
                        "confidence": {"asr": confidence}, "confidence_calibrated": False,
                        "dialogue_evidence": {**features[position], "vocal_run_ms": window["vocal_run_ms"]}})
                native_flags.clear()
                for path in paths:
                    Path(path).unlink()
    segments.sort(key=lambda s: (s["start_ms"], s["id"]))
    target = folder / "transcript.json"
    atomic_json(target, {"schema_version": 1, "segments": segments, "issues": issues,
                        "source_sha256": config["source_sha256"], "recognition_input": "ORIGINAL_SOUNDTRACK",
                        "word_times_fabricated": False})
    return {**identity([code_manifest, manifest]), "artifacts": [str(target)],
            "stage_status": "DEGRADED" if issues else "SUCCESS", "quality_evidence": {"issues": issues},
            "metrics": {"model_load_ms": load_ms, "inference_ms": inference_ms}}
