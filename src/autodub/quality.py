"""Small, dependency-free metrics for speech and generated audio quality review."""
from __future__ import annotations

import array
import difflib
import math
import re
import statistics
import unicodedata
import wave
from pathlib import Path
from typing import Iterable

_PUNCTUATION = str.maketrans({c: " " for c in "，。！？；：、（）【】《》〈〉…—－,.!?;:()[]{}<>\"'`~—–-"})


def normalize_chinese(text: str, *, normalize_punctuation: bool = True,
                      ignore_spaces: bool = True) -> str:
    """Apply NFC and optional punctuation/whitespace normalization."""
    normalized = unicodedata.normalize("NFC", text)
    if normalize_punctuation:
        normalized = normalized.translate(_PUNCTUATION)
    if ignore_spaces:
        normalized = re.sub(r"\s+", "", normalized)
    else:
        normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def cer(reference: str, hypothesis: str, *, normalize_punctuation: bool = True,
        ignore_spaces: bool = True) -> float | None:
    """Return character error rate using Levenshtein edit distance."""
    ref = normalize_chinese(reference, normalize_punctuation=normalize_punctuation,
                            ignore_spaces=ignore_spaces)
    hyp = normalize_chinese(hypothesis, normalize_punctuation=normalize_punctuation,
                            ignore_spaces=ignore_spaces)
    if not ref:
        return None
    previous = list(range(len(hyp) + 1))
    for row, ref_char in enumerate(ref, 1):
        current = [row]
        for col, hyp_char in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[col] + 1,
                               previous[col - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1] / len(ref)


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def word_boundary_error(reference: Iterable[dict], hypothesis: Iterable[dict]) -> dict:
    """Compare matched word boundaries; reports timing error only, with no confidence score."""
    ref = list(reference)
    hyp = list(hypothesis)
    ref_tokens = [normalize_chinese(str(item.get("t", ""))) for item in ref]
    hyp_tokens = [normalize_chinese(str(item.get("t", ""))) for item in hyp]
    errors: list[int] = []
    matched_words = 0
    for a, b, size in difflib.SequenceMatcher(a=ref_tokens, b=hyp_tokens, autojunk=False).get_matching_blocks():
        for offset in range(size):
            expected, observed = ref[a + offset], hyp[b + offset]
            if all(isinstance(x, int) and not isinstance(x, bool) for x in
                   (expected.get("s"), expected.get("e"), observed.get("s"), observed.get("e"))):
                errors.extend((abs(expected["s"] - observed["s"]), abs(expected["e"] - observed["e"])))
                matched_words += 1
    return {
        "matched_words": matched_words,
        "compared_boundaries": len(errors),
        "median_abs_error_ms": statistics.median(errors) if errors else None,
        "p95_abs_error_ms": _percentile(errors, 0.95),
    }


def audio_qc(path: Path, *, silence_threshold_dbfs: float = -50.0) -> dict:
    """Measure duration, peak/clipping and low-RMS silence ratio for PCM16 WAV."""
    if not math.isfinite(silence_threshold_dbfs) or silence_threshold_dbfs > 0:
        raise ValueError("silence_threshold_dbfs must be finite and <= 0")
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        sample_width = stream.getsampwidth()
        frames = stream.getnframes()
        if sample_width != 2 or channels < 1 or sample_rate < 1:
            raise ValueError("audio_qc requires uncompressed PCM16 WAV")
        pcm = stream.readframes(frames)
    samples = array.array("h")
    samples.frombytes(pcm[:len(pcm) - len(pcm) % 2])
    if not samples:
        raise ValueError("audio_qc requires at least one sample")
    peak = max(abs(value) for value in samples)
    clipping = sum(1 for value in samples if abs(value) >= 32767) / len(samples)
    window_samples = max(1, sample_rate * channels // 10)
    threshold = 32768 * 10 ** (silence_threshold_dbfs / 20)
    silence_windows = 0
    window_count = 0
    for start in range(0, len(samples), window_samples):
        chunk = samples[start:start + window_samples]
        rms = math.sqrt(sum(value * value for value in chunk) / len(chunk))
        silence_windows += rms <= threshold
        window_count += 1
    return {
        "duration_ms": round(frames * 1000 / sample_rate),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_format": "pcm_s16le",
        "peak_dbfs": 20 * math.log10(peak / 32768) if peak else None,
        "clipped_sample_ratio": clipping,
        "silence_window_ratio": silence_windows / window_count,
        "silence_window_ms": 100,
        "silence_threshold_dbfs": silence_threshold_dbfs,
    }
