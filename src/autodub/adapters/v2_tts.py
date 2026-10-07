"""One VieNeu v3 Turbo model and one fixed voice enrollment per job; batched utterances."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from autodub.adapters.common import configure_offline, identity, local_hub_files
from autodub.adapters.runtime_paths import asset, output_folder
from autodub.config import DEFAULT_VOICE_ID
from autodub.storage import atomic_json, sha256_file


def run(source: Path, config: dict) -> dict:
    del source
    import numpy as np
    import soundfile as sf
    from vieneu import Vieneu

    if config.get("voice_id", DEFAULT_VOICE_ID) != DEFAULT_VOICE_ID:
        raise ValueError("Ngọc Huyền is the fixed production voice")
    folder = output_folder(config)
    eligible = [s for s in config["segments"] if s["action"] == "DUB" and s.get("dub_vi", "").strip()]
    clips, issues, generated_ms = {}, [], 0
    load_ms = inference_ms = conditioning_ms = 0.0
    manifests = []
    if eligible:
        reference = Path(config["voice_reference"]).resolve(strict=True)
        if sha256_file(reference) != config["voice_reference_sha256"]:
            raise ValueError("Fixed voice reference changed")
        backbone, model_meta = asset(config, "vieneu-turbo")
        codec, codec_meta = asset(config, "moss-torch")
        manifests = [model_meta, codec_meta]
        configure_offline(config)
        batch_size = int(config.get("tts_batch_size", 8))
        if not 1 <= batch_size <= 32:
            raise ValueError("TTS batch exceeds configured memory ceiling")
        tick = time.perf_counter()
        with local_hub_files({"OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX": codec,
                              "pnnbao-ump/VieNeu-TTS-v3-Turbo": backbone}):
            model = Vieneu(mode="v3turbo", backend="pytorch", device=config.get("device", "cuda:0"),
                           backbone_repo=str(backbone), moss_tokenizer=str(codec),
                           dtype=config.get("tts_dtype", "bfloat16"), max_batch_size=batch_size,
                           max_streams=1, babble_retries=0)
        load_ms = (time.perf_counter() - tick) * 1000
        try:
            if model.sample_rate != 48000:
                raise ValueError("Production voice must produce 48kHz audio")
            tick = time.perf_counter()
            model.add_voice(DEFAULT_VOICE_ID, reference, denoise=False, save=False)
            conditioning_ms = (time.perf_counter() - tick) * 1000
            # Small bounded batches keep generated waveforms out of hour-long RAM buffers.
            ordered = sorted(eligible, key=lambda s: len(s["dub_vi"]))
            for start in range(0, len(ordered), batch_size):
                group = ordered[start:start + batch_size]
                tick = time.perf_counter()
                rows = model.infer_batch([s["dub_vi"] for s in group], voice=DEFAULT_VOICE_ID,
                                         batch_size=batch_size, apply_watermark=False)
                inference_ms += (time.perf_counter() - tick) * 1000
                if len(rows) != len(group):
                    raise ValueError("TTS batch coverage mismatch")
                for segment, audio in zip(group, rows, strict=True):
                    audio = np.asarray(audio, dtype="float32")
                    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-6:
                        issues.append({"segment_id": segment["id"], "code": "EMPTY_OR_INVALID_TTS"})
                        continue
                    identifier = hashlib.sha256(segment["id"].encode()).hexdigest()[:24]
                    path = folder / f"voice-{identifier}.wav"
                    sf.write(path, audio, 48000, subtype="PCM_16")
                    clips[segment["id"]] = str(path)
                    generated_ms += round(len(audio) * 1000 / 48000)
        finally:
            model.close()
    target = folder / "tts.json"
    atomic_json(target, {"schema_version": 1, "clips": clips, "issues": issues,
                        "voice_id": DEFAULT_VOICE_ID, "voice_reference_sha256": config.get("voice_reference_sha256"),
                        "eligible_count": len(eligible), "generated_count": len(clips),
                        "generated_audio_ms": generated_ms, "conditioning_enrollments": 1 if eligible else 0})
    return {**(identity(manifests) if manifests else {}), "clips": clips,
            "artifacts": [str(target), *clips.values()],
            "stage_status": "DEGRADED" if issues else "SUCCESS",
            "quality_evidence": {"issues": issues},
            "metrics": {"model_load_ms": load_ms, "conditioning_ms": conditioning_ms, "inference_ms": inference_ms}}
