"""The V1 audio models: Bandit v2, Community-1, FireRed AED and FireRedPunc."""
from __future__ import annotations

import json
import math
import sys
import time
import unicodedata
from pathlib import Path

from autodub.adapters.common import configure_offline, identity
from autodub.adapters.runtime_paths import asset, output_folder
from autodub.benchmedia import extract_audio
from autodub.storage import atomic_json, sha256_file


def _result(config: dict, name: str, payload: dict, manifests: list, metrics: dict,
            *, degraded: bool = False, extra: list[Path] | None = None) -> dict:
    target = output_folder(config) / name
    atomic_json(target, payload)
    return {**identity(manifests), "stage_status": "DEGRADED" if degraded else "SUCCESS",
            "quality_evidence": {"issues": payload.get("issues", [])},
            "metrics": metrics, "artifacts": [str(target), *map(str, extra or [])]}


def separate(source: Path, config: dict) -> dict:
    import numpy as np
    import soundfile as sf
    code, code_manifest = asset(config, "bandit-infer-code")
    sys.path.insert(0, str(code / "source" / "src"))
    from bandit_infer import BanditSession

    weights, manifest = asset(config, "bandit-v2-cinematic")
    configure_offline(config)
    folder = output_folder(config)
    wav = folder / "source-48k.wav"
    started = time.perf_counter()
    extract_audio(source, wav, end_ms=config["duration_ms"], sample_rate=48000, channels=2,
                  ffmpeg_bin=config.get("ffmpeg_bin", "ffmpeg"))
    audio_info = sf.info(wav)
    rate = audio_info.samplerate
    preprocess_ms = (time.perf_counter() - started) * 1000
    checkpoint = weights / "checkpoint-multi.ckpt"
    tick = time.perf_counter()
    session = BanditSession("v2-multi", family="v2", device=config.get("device", "cuda:0"),
                           checkpoint_path=checkpoint, checkpoint_sha256=sha256_file(checkpoint))
    session.load()
    load_ms = (time.perf_counter() - tick) * 1000
    paths = [folder / f"{name}.wav" for name in ("speech", "music", "effects")]
    writers = {}
    inference_ms, postprocess_ms = 0.0, 0.0
    chunk_frames, context_frames = 60 * rate, 8 * rate
    try:
        writers = {path.stem: sf.SoundFile(path, mode="w", samplerate=rate, channels=2, subtype="PCM_16")
                   for path in paths}
        with sf.SoundFile(wav) as reader:
            for start in range(0, audio_info.frames, chunk_frames):
                end = min(audio_info.frames, start + chunk_frames)
                left, right = max(0, start - context_frames), min(audio_info.frames, end + context_frames)
                reader.seek(left)
                audio = reader.read(right - left, dtype="float32", always_2d=True).T
                original_frames = audio.shape[1]
                if original_frames < 8 * rate:
                    audio = np.pad(audio, ((0, 0), (0, 8 * rate - original_frames)))
                tick = time.perf_counter()
                stems = session.infer(audio, sample_rate=rate)
                inference_ms += (time.perf_counter() - tick) * 1000
                if set(stems) != set(writers):
                    raise ValueError("Bandit v2 must return exactly speech, music and effects")
                tick = time.perf_counter()
                for name, values in stems.items():
                    if values.shape != audio.shape:
                        raise ValueError("Bandit v2 stem changed the input timeline or channels")
                    writers[name].write(values[:, start - left:end - left].T)
                postprocess_ms += (time.perf_counter() - tick) * 1000
    finally:
        session.close()
        for writer in writers.values():
            writer.close()
    return _result(config, "separation.json", {
        "schema_version": 1, "source_sha256": sha256_file(source), "sample_rate": rate,
        "duration_ms": round(audio_info.frames * 1000 / rate),
        "chunk_seconds": 60, "context_seconds": 8, "short_input_zero_padding_seconds": 8,
        "stems": {p.stem: str(p) for p in paths}, "model": "Bandit-v2-Multi-CASS",
    }, [manifest, code_manifest], {"preprocess_ms": preprocess_ms, "model_load_ms": load_ms,
                   "inference_ms": inference_ms, "postprocess_ms": postprocess_ms}, extra=paths)


def diarize(source: Path, config: dict) -> dict:
    import torch
    from pyannote.audio import Pipeline

    folder, manifest = asset(config, "pyannote-community-1")
    configure_offline(config)
    tick = time.perf_counter()
    model = Pipeline.from_pretrained(str(folder))
    model.to(torch.device(config.get("device", "cuda:0")))
    load_ms = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    output = model(str(source))
    annotation = output.speaker_diarization
    turns = [{"start_ms": round(turn.start * 1000), "end_ms": round(turn.end * 1000),
              "speaker_id": speaker, "track_id": str(track)}
             for turn, track, speaker in annotation.itertracks(yield_label=True)]
    overlaps = []
    for index, left in enumerate(turns):
        for right in turns[index + 1:]:
            start, end = max(left["start_ms"], right["start_ms"]), min(left["end_ms"], right["end_ms"])
            if start < end and left["speaker_id"] != right["speaker_id"]:
                overlaps.append({"start_ms": start, "end_ms": end,
                                 "speakers": [left["speaker_id"], right["speaker_id"]]})
    return _result(config, "diarization.json", {"schema_version": 1, "turns": turns,
        "overlaps": overlaps, "speaker_ids_are_characters": False,
        "source_sha256": sha256_file(source)}, [manifest],
        {"model_load_ms": load_ms, "inference_ms": (time.perf_counter() - tick) * 1000})


def speech_windows(turns: list[dict], duration_ms: int, chunk_ms: int = 30000) -> list[dict]:
    """ASR each union once; overlapping speakers remain explicitly unresolved."""
    merged = []
    for turn in sorted(turns, key=lambda item: item["start_ms"]):
        start, end = max(0, turn["start_ms"]), min(duration_ms, turn["end_ms"])
        if start >= end:
            continue
        if merged and start < merged[-1]["end_ms"]:
            merged[-1]["end_ms"] = max(end, merged[-1]["end_ms"])
            merged[-1]["speaker_ids"].add(turn["speaker_id"])
        else:
            merged.append({"start_ms": start, "end_ms": end, "speaker_ids": {turn["speaker_id"]}})
    windows = []
    for group in merged:
        for start in range(group["start_ms"], group["end_ms"], chunk_ms):
            windows.append({"start_ms": start, "end_ms": min(group["end_ms"], start + chunk_ms),
                            "speaker_ids": sorted(group["speaker_ids"])})
    return windows


def recognize(source: Path, config: dict) -> dict:
    code, code_manifest = asset(config, "firered-code")
    sys.path.insert(0, str(code / "source"))
    from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config

    weights, manifest = asset(config, "firered-asr2-aed")
    configure_offline(config)
    turns = json.loads(Path(config["diarization_path"]).read_text(encoding="utf-8"))["turns"]
    windows = speech_windows(turns, config["duration_ms"], config.get("asr_chunk_ms", 30000))
    folder = output_folder(config)
    tick = time.perf_counter()
    model = FireRedAsr2.from_pretrained("aed", str(weights), FireRedAsr2Config(
        use_gpu=config.get("device", "cuda:0").startswith("cuda"), use_half=True,
        return_timestamp=True, beam_size=3))
    upstream_timestamps = model._get_and_fix_timestamp
    native_timestamp_flags = []

    def native_timestamps_only(hypothesis, token_ids, duration):
        present = hypothesis.get("timestamp") is not None
        native_timestamp_flags.append(present)
        # The upstream helper otherwise invents equally spaced times. Missing
        # native evidence remains missing; it must not become a lexical mask.
        return upstream_timestamps(hypothesis, token_ids, duration) if present else []

    model._get_and_fix_timestamp = native_timestamps_only
    load_ms = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    segments, raw, issues = [], [], []
    inference_ms, preprocess_ms = 0.0, 0.0
    for index, window in enumerate(windows):
        wav = folder / f"speech-{index:06d}.wav"
        tick = time.perf_counter()
        extract_audio(source, wav, window["start_ms"], window["end_ms"], sample_rate=16000,
                      channels=1, ffmpeg_bin=config.get("ffmpeg_bin", "ffmpeg"))
        preprocess_ms += (time.perf_counter() - tick) * 1000
        sid = f"seg_{window['start_ms']:012d}_{index:04d}"
        tick = time.perf_counter()
        results = model.transcribe([sid], [str(wav)])
        inference_ms += (time.perf_counter() - tick) * 1000
        if not results:
            issues.append({"segment_id": sid, "code": "ASR_EMPTY_RESULT", **window})
            continue
        result = results[0]
        raw.append({"window": window, "output": result,
                    "native_timestamp_available": native_timestamp_flags[-1] if native_timestamp_flags else False})
        native_timestamp_flags.clear()
        text = str(result.get("text", "")).strip()
        if not text:
            issues.append({"segment_id": sid, "code": "ASR_EMPTY_TEXT", **window})
            continue
        words = []
        for token, left, right in result.get("timestamp", []):
            start = window["start_ms"] + round(float(left) * 1000)
            end = window["start_ms"] + round(float(right) * 1000)
            if window["start_ms"] <= start < end <= window["end_ms"]:
                words.append({"t": token, "s": start, "e": end})
                if end - start < 20:
                    issues.append({"segment_id": sid, "code": "VERY_SHORT_NATIVE_WORD_TIME",
                                   "token": token, "start_ms": start, "end_ms": end})
            else:
                issues.append({"segment_id": sid, "code": "INVALID_NATIVE_WORD_TIME",
                               "raw_time": [left, right]})
        overlap = len(window["speaker_ids"]) != 1
        if not words:
            issues.append({"segment_id": sid, "code": "NATIVE_WORD_TIMING_UNAVAILABLE"})
        nonverbal = bool(text) and all(char in "啊呃嗯哎哈呵！？!?,，。 " for char in text)
        confidence = result.get("confidence")
        if confidence is not None and not (math.isfinite(confidence) and 0 <= confidence <= 1):
            confidence = None
        if overlap:
            issues.append({"segment_id": sid, "code": "UNRESOLVED_OVERLAPPING_SPEAKERS", **window})
        segments.append({"id": sid, "start_ms": window["start_ms"], "end_ms": window["end_ms"],
            "zh_text": text, "words": words, "speaker_id": window["speaker_ids"][0] if not overlap else None,
            "track_id": window["speaker_ids"][0] if not overlap else "overlap",
            "action": "KEEP" if overlap or nonverbal or not words else "DUB", "needs_review": bool(overlap or not words),
            "confidence": {"asr": confidence}, "confidence_calibrated": False,
            "speech_kind": "nonverbal" if nonverbal else "lexical", "subtitle_vi": "", "dub_vi": ""})
    return _result(config, "transcript.json", {"schema_version": 1, "source_sha256": sha256_file(source),
        "language": "zh", "segments": segments, "raw_results": raw, "issues": issues,
        "native_timestamp_policy": "FireRed upstream adjusts/clamps timestamps; not independent timing ground truth"},
        [manifest, code_manifest], {"model_load_ms": load_ms, "preprocess_ms": preprocess_ms,
                                    "inference_ms": inference_ms},
        degraded=bool(issues))


def punctuate(source: Path, config: dict) -> dict:
    del source
    code, code_manifest = asset(config, "firered-code")
    sys.path.insert(0, str(code / "source"))
    from fireredasr2s.fireredpunc.punc import FireRedPunc, FireRedPuncConfig

    weights, manifest = asset(config, "firered-punc")
    configure_offline(config)
    value = json.loads(Path(config["transcript_path"]).read_text(encoding="utf-8"))
    segments = value["segments"]
    tick = time.perf_counter()
    model = FireRedPunc.from_pretrained(str(weights), FireRedPuncConfig(use_gpu=False))
    load_ms = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    rows = model.process([s["zh_text"] for s in segments], [s["id"] for s in segments])
    if len(rows) != len(segments):
        raise ValueError("Punctuation output does not cover the input segments")
    split_segments = []
    for segment, row in zip(segments, rows, strict=True):
        segment["asr_text"] = segment["zh_text"]
        segment["zh_text"] = row["punc_text"]
        if config.get('split_native_utterances'):
            segment['dialogue_evidence'] = {**segment.get('dialogue_evidence', {}), 'asr_parent': {
                'id': segment['id'], 'zh_text': segment['zh_text'],
                'start_ms': segment['start_ms'], 'end_ms': segment['end_ms']}}
        split_segments.extend(sentence_segments(segment, split_commas=bool(config.get('split_native_utterances', False))))
    value["segments"] = split_segments
    return _result(config, "punctuation.json", value, [manifest, code_manifest],
        {"model_load_ms": load_ms, "inference_ms": (time.perf_counter() - tick) * 1000},
        degraded=bool(value.get("issues")))


def sentence_segments(segment: dict, *, split_commas: bool = False) -> list[dict]:
    """Split punctuation at native word boundaries only; never invent word times."""
    words = segment.get("words", [])
    text = segment["zh_text"]
    def normalize(value):
        return "".join(char for char in unicodedata.normalize("NFKC", value) if char.isalnum())
    if not words or normalize(text) != normalize("".join(w["t"] for w in words)):
        return [segment]
    groups, current, cursor, cursor_start = [], [], 0, 0
    for word in words:
        needed = len(normalize(word["t"]))
        count, end = 0, cursor
        while end < len(text) and count < needed:
            count += len(normalize(text[end]))
            end += 1
        while end < len(text) and not normalize(text[end]):
            end += 1
        piece = text[cursor:end]
        current.append(word)
        boundaries = "。！？!?；;，," if split_commas else "。！？!?；;"
        if any(char in piece for char in boundaries):
            groups.append((current, text[:end] if not groups else text[cursor_start:end]))
            current = []
            cursor_start = end
        elif len(current) == 1:
            cursor_start = cursor
        cursor = end
    if current:
        groups.append((current, text[cursor_start:]))
    if len(groups) <= 1:
        if split_commas and words:
            return [{**segment, 'start_ms': words[0]['s'], 'end_ms': words[-1]['e']}]
        return [segment]
    return [{**segment, "id": f"{segment['id']}:s{index}", "start_ms": group[0]["s"],
             "end_ms": group[-1]["e"], "words": group, "zh_text": wording,
             "asr_text": "".join(w["t"] for w in group)}
            for index, (group, wording) in enumerate(groups)]
