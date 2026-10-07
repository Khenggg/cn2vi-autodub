"""Multilingual Faster-Whisper ASR and pinned WhisperX forced alignment."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import autodub.media as media
from autodub.adapters.common import asset, configure_offline, identity, output_folder
from autodub.contracts import Segment
from autodub.storage import atomic_json, sha256_file
from autodub.utterances import build_utterances

LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}$")


def _device(config: dict) -> tuple[str, int]:
    value = str(config.get("device", "cuda:0"))
    if value.startswith("cuda"):
        suffix = value.partition(":")[2]
        return "cuda", int(suffix) if suffix.isdigit() else 0
    return value, 0


def _milliseconds(value: object) -> int:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError("Model returned an invalid timestamp") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Model returned an invalid timestamp")
    return round(seconds * 1000)


def run_asr(source: Path, config: dict) -> dict:
    from faster_whisper import WhisperModel

    source = Path(source).resolve(strict=True)
    configure_offline(config)
    model_path, manifest = asset(config, "whisper-asr")
    device, device_index = _device(config)
    compute_type = config.get("compute_type", "int8_float16" if device == "cuda" else "int8")
    started = time.perf_counter()
    model = WhisperModel(str(model_path), device=device, device_index=device_index,
                         compute_type=compute_type, local_files_only=True,
                         download_root=str(model_path.parent))
    load_ms = round((time.perf_counter() - started) * 1000)
    started = time.perf_counter()
    raw_segments, info = model.transcribe(
        str(source), language=config.get("language"), beam_size=int(config.get("beam_size", 5)),
        vad_filter=True, word_timestamps=False,
    )
    raw_segments = list(raw_segments)
    inference_ms = round((time.perf_counter() - started) * 1000)
    language = str(getattr(info, "language", "") or "").lower()
    if not LANGUAGE_CODE.fullmatch(language):
        raise ValueError("Whisper did not return a supported language code")
    duration_ms = media.probe_media(source, config.get("ffprobe_bin", "ffprobe"))["duration_ms"]
    segments = []
    for index, item in enumerate(raw_segments):
        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        start_ms, end_ms = _milliseconds(getattr(item, "start", None)), _milliseconds(getattr(item, "end", None))
        if start_ms >= duration_ms or end_ms > duration_ms or end_ms <= start_ms:
            raise ValueError("Whisper returned a segment outside the source timeline")
        segments.append(Segment(
            id=f"seg_{start_ms:012d}_{index:04d}", start_ms=start_ms, end_ms=end_ms,
            zh_text=text, confidence={"asr": None, "alignment": None},
            action="NEEDS_REVIEW", needs_review=True,
        ))
    folder = output_folder(config)
    target = folder / "transcript.zh.json"
    atomic_json(target, {
        "schema_version": 1, "source_sha256": sha256_file(source), "language": language,
        "language_probability": getattr(info, "language_probability", None),
        "segments": [segment.model_dump() for segment in segments],
        "confidence_policy": "upstream_has_no_calibrated_confidence",
        "word_timing_available": False,
    })
    return {
        **identity([manifest]), "language": language,
        "quality_metrics": {"segment_count": len(segments), "word_timing_available": False},
        "quality_evidence": {"status": "REVIEW_REQUIRED",
                             "missing": ["word_alignment", "language_review", "transcript_review"]},
        "processed_media_ms": duration_ms,
        "metrics": {"model_load_ms": load_ms, "inference_ms": inference_ms,
                    "device": device, "compute_type": compute_type, "segment_count": len(segments)},
        "artifacts": [str(target)],
    }


def _review_segments(segments: list[Segment]) -> list[dict]:
    return [segment.model_copy(update={
        "words": [], "confidence": {"asr": None, "alignment": None},
        "action": "NEEDS_REVIEW", "needs_review": True,
    }).model_dump() for segment in segments]


def _alignment_asset(config: dict, language: str) -> tuple[Path, dict] | None:
    if not LANGUAGE_CODE.fullmatch(language):
        return None
    try:
        return asset(config, f"alignment-{language}")
    except FileNotFoundError:
        return None


def _word_groups(aligned: list[dict], source_segments: list[Segment]) -> list[list[dict]]:
    if len(aligned) != len(source_segments):
        return [[] for _ in source_segments]
    result = []
    for item, source in zip(aligned, source_segments, strict=True):
        words, invalid = [], False
        for word in item.get("words", []):
            token = str(word.get("word", "")).strip()
            if not token or word.get("start") is None or word.get("end") is None:
                continue
            start, end = _milliseconds(word["start"]), _milliseconds(word["end"])
            if not source.start_ms <= start < end <= source.end_ms:
                invalid = True
                break
            words.append({"t": token, "s": start, "e": end})
        result.append([] if invalid else words)
    return result


def run_alignment(source: Path, config: dict) -> dict:
    source = Path(source).resolve(strict=True)
    transcript_path = Path(config["transcript_path"]).resolve(strict=True)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    source_hash = sha256_file(source)
    if transcript.get("schema_version") != 1 or transcript.get("source_sha256") != source_hash:
        raise ValueError("Transcript source hash mismatch")
    segments = [Segment.model_validate(item) for item in transcript.get("segments", [])]
    language = str(transcript.get("language", "")).lower()
    folder = output_folder(config)
    _asr_path, asr_manifest = asset(config, "whisper-asr")
    selected = _alignment_asset(config, language)
    target = folder / "alignment.json"

    if selected is None:
        required = f"alignment-{language}" if LANGUAGE_CODE.fullmatch(language) else "language-code"
        output = _review_segments(segments)
        atomic_json(target, {
            "schema_version": 1, "source_sha256": source_hash, "language": language or None,
            "alignment_status": "UNSUPPORTED_LANGUAGE_NO_WORD_TIMES", "required_asset": required,
            "segments": output, "word_timing_available": False,
        })
        return {
            **identity([asr_manifest]), "language": language or None,
            "quality_metrics": {"aligned_segment_count": 0, "word_timing_available": False},
            "quality_evidence": {"status": "REVIEW_REQUIRED",
                                 "missing": [f"pinned_alignment_asset:{required}", "human_timing_review"]},
            "processed_media_ms": sum(item.end_ms - item.start_ms for item in segments),
            "metrics": {}, "artifacts": [str(target)],
        }

    align_path, align_manifest = selected
    metrics = {"model_load_ms": 0, "inference_ms": 0}
    output = []
    if segments:
        import whisperx

        configure_offline(config)
        device, _ = _device(config)
        cache_dir = Path(config.get("cache_root", "/data/cache")).resolve() / "alignment"
        cache_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        model, metadata = whisperx.load_align_model(
            language_code=language, device=device, model_name=str(align_path), model_dir=str(cache_dir))
        metrics["model_load_ms"] = round((time.perf_counter() - started) * 1000)
        audio = whisperx.load_audio(str(source))
        inputs = [{"start": item.start_ms / 1000, "end": item.end_ms / 1000, "text": item.zh_text}
                  for item in segments]
        started = time.perf_counter()
        result = whisperx.align(inputs, model, metadata, audio, device, return_char_alignments=False)
        metrics["inference_ms"] = round((time.perf_counter() - started) * 1000)
        groups = _word_groups(result.get("segments", []) if isinstance(result, dict) else [], segments)
        for source_segment, words in zip(segments, groups, strict=True):
            utterances = build_utterances(source_segment.zh_text, words,
                                          (source_segment.start_ms, source_segment.end_ms)) if words else []
            if not utterances:
                output.extend(_review_segments([source_segment]))
                continue
            for index, utterance in enumerate(utterances):
                output.append(Segment.model_validate({
                    "id": f"seg_{utterance['start_ms']:012d}_{index:04d}",
                    "start_ms": utterance["start_ms"], "end_ms": utterance["end_ms"],
                    "zh_text": utterance["text"], "words": utterance["words"],
                    "confidence": {"asr": None, "alignment": None},
                    "action": "NEEDS_REVIEW", "needs_review": True,
                }).model_dump())
        metrics["aligned_segment_count"] = sum(bool(group) for group in groups)

    timing_available = bool(output) and all(item["words"] for item in output)
    atomic_json(target, {
        "schema_version": 1, "source_sha256": source_hash, "language": language,
        "alignment_status": "ALIGNED" if timing_available else "PARTIAL_REVIEW_REQUIRED",
        "segments": output, "word_timing_available": timing_available,
    })
    return {
        **identity([asr_manifest, align_manifest]), "language": language or None,
        "quality_metrics": {"aligned_segment_count": metrics.get("aligned_segment_count", 0),
                            "word_timing_available": timing_available},
        "quality_evidence": {"status": "REVIEW_REQUIRED",
                             "missing": ["language_review", "word_timing_review",
                                         *([] if timing_available else ["complete_word_alignment"])]},
        "processed_media_ms": sum(item.end_ms - item.start_ms for item in segments),
        "metrics": metrics, "artifacts": [str(target)],
    }
