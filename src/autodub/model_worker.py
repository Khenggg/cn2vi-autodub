"""Child-process entry point for isolated audio and video model stages."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from autodub.config import DEFAULT_VOICE_ID
from autodub.storage import atomic_json


def _run_tts(source: Path, config: dict) -> dict:
    del source
    from autodub.adapters.vieneu import VieNeuProvider
    segments = config.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("TTS segments are required")
    provider = VieNeuProvider(config)
    clips: dict[str, str] = {}
    quality: dict[str, dict] = {}
    artifacts: list[str] = []
    processed_ms = 0
    inference_ms = 0
    output_root = Path(config["output_dir"]).resolve()
    default_voice = config.get("voice_id", DEFAULT_VOICE_ID)

    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Invalid TTS segment")
        segment_id = segment.get("id")
        text = segment.get("text", segment.get("dub_vi", ""))
        if not segment_id or not text.strip():
            continue
        target_ms = segment.get("target_ms", segment.get("end_ms", 0) - segment.get("start_ms", 0))
        if target_ms <= 0:
            target_ms = 1000

        # Support voice selection based on segment or fallback to default
        voice_id = segment.get("voice_id") or default_voice
        emotion = segment.get("emotion", "neutral")

        seg_output = output_root / "clips" / f"{segment_id.replace(':', '_')}.wav"
        seg_output.parent.mkdir(parents=True, exist_ok=True)
        seg_config = {**config, "output_dir": str(seg_output.parent)}
        provider.config = seg_config
        synth = provider.synthesize(text, voice_id=voice_id, emotion_hint=emotion, target_ms=target_ms)

        # Move/copy to clean segment path
        synth_wav = Path(synth["wav"])
        if synth_wav != seg_output:
            synth_wav.replace(seg_output)
        clips[segment_id] = str(seg_output)
        quality[segment_id] = synth["qc"]
        artifacts.append(str(seg_output))
        processed_ms += synth["actual_ms"]
        inference_ms += provider.metrics.get("inference_ms", 0)

    report_path = output_root / "tts_clips.json"
    atomic_json(report_path, {"schema_version": 1, "clips": clips, "quality": quality})
    artifacts.append(str(report_path))
    from autodub.adapters.common import identity
    return {
        **identity(provider.manifests),
        "clips": clips,
        "processed_media_ms": processed_ms,
        "metrics": {"inference_ms": inference_ms, **provider.metrics},
        "quality_metrics": {"clip_count": len(clips)},
        "artifacts": artifacts,
    }


def _run_separating(source: Path, config: dict) -> dict:
    from autodub.adapters.roformer import is_roformer_available
    from autodub.adapters.roformer import run as run_roformer
    if is_roformer_available() or config.get("separation_provider") == "roformer":
        return run_roformer(source, config)

    from autodub.adapters.bandit import run
    from autodub.media import probe_media
    # Ensure all windows meet the BandIt standard window requirement (>= 30s)
    duration_ms = probe_media(source, config.get("ffprobe_bin", "ffprobe"))["duration_ms"]
    raw_windows = config.get("windows", [])
    if not raw_windows:
        # Default full-range window
        raw_windows = [{"start_ms": 0, "end_ms": min(duration_ms, 30000)}]

    padded_windows = []
    for w in raw_windows:
        start = max(0, w["start_ms"])
        end = min(duration_ms, w["end_ms"])
        if end - start < 30000:
            # Pad to 30s if within duration
            padding_needed = 30000 - (end - start)
            start = max(0, start - padding_needed // 2)
            end = min(duration_ms, start + 30000)
            if end - start < 30000 and start > 0:
                start = max(0, end - 30000)
        padded_windows.append({"start_ms": start, "end_ms": end})

    padded_config = {**config, "windows": padded_windows}
    return run(source, padded_config)


def run_stage(stage: str, source: Path, config: dict) -> dict:
    if stage == "ASR":
        from autodub.adapters.whisper import run_asr
        return run_asr(source, config)
    if stage == "ALIGNING":
        from autodub.adapters.whisper import run_alignment
        return run_alignment(source, config)
    if stage == "TRANSLATING":
        from autodub.adapters.local_translation import run
        return run(source, config)
    if stage == "SEPARATING":
        return _run_separating(source, config)
    if stage == "TTS":
        return _run_tts(source, config)
    if stage == "PROPAINTER":
        from autodub.adapters.propainter import run
        return run(source, config)
    if stage == "VISION_RENDER":
        from autodub.adapters.subtitle_video import run
        return run(source, config)
    raise ValueError(f"Unsupported model stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    request_path, result_path = Path(args.request), Path(args.result)
    stage = None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if (not isinstance(request, dict) or request.get("schema_version") != 1
                or request.get("stage") not in {"ASR", "ALIGNING", "TRANSLATING", "SEPARATING", "TTS", "PROPAINTER", "VISION_RENDER"}
                or not isinstance(request.get("config"), dict)):
            return 2
        stage = request["stage"]
        source = Path(request["source"]).resolve(strict=True)
        result = run_stage(stage, source, request["config"])
        if not isinstance(result, dict):
            return 3
        atomic_json(result_path, {"schema_version": 1, "stage": stage, "result": result})
        return 0
    except Exception as error:
        error_code = error.__class__.__name__
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", error_code):
            error_code = "WorkerError"
        if stage:
            try:
                atomic_json(result_path, {"schema_version": 1, "stage": stage, "error_code": error_code})
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
