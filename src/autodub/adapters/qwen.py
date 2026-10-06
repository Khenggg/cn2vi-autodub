"""Qwen3-ASR/ForcedAligner adapters against qwen-asr 0.0.6, using pinned local snapshots."""
import json
import wave
from decimal import Decimal
from pathlib import Path

from autodub.adapters.common import (
    allocator_metrics,
    asset,
    configure_offline,
    identity,
    milliseconds,
    output_folder,
    synchronize,
    torch_device,
)
from autodub.benchmedia import chunk_windows, extract_audio
from autodub.contracts import Segment
from autodub.media import probe_media
from autodub.quality import cer, word_boundary_error
from autodub.storage import atomic_json, sha256_file


def bounds(source: Path, config: dict) -> tuple[int, int]:
    if source.suffix.lower() == ".wav":
        with wave.open(str(source), "rb") as audio:
            duration = audio.getnframes() * 1000 // audio.getframerate()
    else:
        duration = probe_media(source, config.get("ffprobe_bin", "ffprobe"))["duration_ms"]
    start, end = config.get("start_ms", 0), config.get("end_ms", duration)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (start, end)) or not 0 <= start < end <= duration:
        raise ValueError("Invalid benchmark source interval")
    return start, end


def aligned_words(result, offset_ms: int, duration_ms: int) -> list[dict]:
    words = []
    for item in result:
        start = offset_ms + round(Decimal(str(item.start_time)) * 1000)
        end = offset_ms + round(Decimal(str(item.end_time)) * 1000)
        if not offset_ms <= start < end <= offset_ms + duration_ms:
            raise ValueError("Aligner returned a word outside the source chunk")
        words.append({"t": item.text, "s": start, "e": end})
    return words


class QwenAsrProvider:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.manifest = None
        self.metrics = {}

    def load(self):
        import torch
        from qwen_asr import Qwen3ASRModel
        configure_offline(self.config)
        path, self.manifest = asset(self.config, "qwen-asr")
        self.device, dtype = torch_device(torch, self.config)
        started = milliseconds()
        self.model = Qwen3ASRModel.from_pretrained(str(path), dtype=dtype, device_map=self.device,
                                                 max_inference_batch_size=1,
                                                 max_new_tokens=int(self.config.get("max_new_tokens", 4096)),
                                                 local_files_only=True)
        synchronize(torch, self.device)
        self.metrics["model_load_ms"] = milliseconds() - started

    def transcribe(self, audio_path: str) -> list[Segment]:
        import torch
        source = Path(audio_path)
        start, end = bounds(source, self.config)
        windows = list(chunk_windows(end - start,
                       int(self.config.get("chunk_ms", 240000)), int(self.config.get("overlap_ms", 1000))))
        if self.model is None:
            self.load()
        folder = output_folder(self.config)
        segments = []
        inference_ms = 0
        for index, (left, right) in enumerate(windows):
            left, right = left + start, right + start
            wav = extract_audio(source, folder / f"asr_{index:04d}.wav", left, right,
                                ffmpeg_bin=self.config.get("ffmpeg_bin", "ffmpeg"))
            synchronize(torch, self.device)
            tick = milliseconds()
            result = self.model.transcribe(audio=str(wav), language="Chinese", return_time_stamps=False)
            synchronize(torch, self.device)
            inference_ms += milliseconds() - tick
            if len(result) != 1:
                raise ValueError("ASR output count mismatch")
            segments.append(Segment(id=f"seg_{left:012d}", start_ms=left, end_ms=right,
                                    zh_text=result[0].text, confidence={"asr": None, "alignment": None},
                                    action="NEEDS_REVIEW", needs_review=True))
        self.metrics.update(inference_ms=inference_ms, **allocator_metrics(torch, self.device))
        return segments


def run_asr(source: Path, config: dict) -> dict:
    provider = QwenAsrProvider(config)
    segments = provider.transcribe(str(source))
    folder = output_folder(config)
    transcript = folder / "transcript.zh.json"
    atomic_json(transcript, {"schema_version": 1, "source_sha256": sha256_file(source),
                            "segments": [segment.model_dump() for segment in segments],
                            "confidence_policy": "upstream_has_no_calibrated_confidence", "overlap_not_reconciled": True})
    reference = config.get("reference_text", "")
    start, end = bounds(source, config)
    return {**identity([provider.manifest]), "quality_metrics": {
                "cer": cer(reference, "".join(segment.zh_text for segment in segments)),
                "nonverbal_preservation": None, "overlap_boundary_review_required": True},
            "quality_evidence": {"status": "REVIEW_REQUIRED", "missing": ["nonverbal_preservation_review",
                                  "overlap_boundary_review", *([] if reference.strip() else ["zh_reference"])]},
            "processed_media_ms": end - start, "metrics": provider.metrics, "artifacts": [str(transcript)]}


def run_alignment(source: Path, config: dict) -> dict:
    import torch
    from qwen_asr import Qwen3ForcedAligner
    configure_offline(config)
    path, manifest = asset(config, "qwen-aligner")
    folder = output_folder(config)
    transcript = json.loads(Path(config["transcript_path"]).read_text(encoding="utf-8"))
    if transcript.get("schema_version") != 1 or transcript["source_sha256"] != sha256_file(source):
        raise ValueError("Transcript source hash mismatch")
    source_duration = probe_media(source, config.get("ffprobe_bin", "ffprobe"))["duration_ms"]
    validated = [Segment.model_validate(payload) for payload in transcript["segments"]]
    if any(segment.end_ms > source_duration or segment.end_ms - segment.start_ms > 300000
           for segment in validated):
        raise ValueError("Alignment chunk exceeds source/300 second limit")
    device, dtype = torch_device(torch, config)
    started = milliseconds()
    model = Qwen3ForcedAligner.from_pretrained(str(path), dtype=dtype, device_map=device, local_files_only=True)
    synchronize(torch, device)
    load_ms = milliseconds() - started
    segments, inference_ms, processed_ms = [], 0, 0
    for index, segment in enumerate(validated):
        if not segment.zh_text.strip():
            segments.append(segment)
            continue
        wav = extract_audio(source, folder / f"align_{index:04d}.wav", segment.start_ms, segment.end_ms,
                            ffmpeg_bin=config.get("ffmpeg_bin", "ffmpeg"))
        synchronize(torch, device)
        tick = milliseconds()
        result = model.align(audio=str(wav), text=segment.zh_text, language="Chinese")
        synchronize(torch, device)
        inference_ms += milliseconds() - tick
        if len(result) != 1:
            raise ValueError("Alignment output count mismatch")
        words = aligned_words(result[0], segment.start_ms, segment.end_ms - segment.start_ms)
        # Timing is evidence, not proof that screams/overlap are lexical speech.
        segment = Segment.model_validate({**segment.model_dump(), "words": words,
                                          "action": "NEEDS_REVIEW", "needs_review": True})
        segments.append(segment)
        processed_ms += segment.end_ms - segment.start_ms
    target = folder / "alignment.json"
    atomic_json(target, {"schema_version": 1, "source_sha256": sha256_file(source),
                         "segments": [segment.model_dump() for segment in segments],
                         "overlap_not_reconciled": True, "lexical_mask_approved": False})
    reference_words = config.get("reference_words", [])
    quality = word_boundary_error(reference_words, [word.model_dump() for segment in segments for word in segment.words])
    return {**identity([manifest]), "quality_metrics": quality,
            "quality_evidence": {"status": "REVIEW_REQUIRED", "lexical_mask_approved": False,
                                 "missing": ["lexical_mask_human_approval", "overlap_boundary_review",
                                             *([] if reference_words else ["reference_words"])]},
            "processed_media_ms": processed_ms, "metrics": {"model_load_ms": load_ms,
                "inference_ms": inference_ms, **allocator_metrics(torch, device)}, "artifacts": [str(target)]}
