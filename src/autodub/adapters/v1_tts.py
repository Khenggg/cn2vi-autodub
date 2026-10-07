"""Fixed-reference Vietnamese IndexTTS2 with pinned, offline auxiliary models."""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from autodub.adapters.common import asset, configure_offline, identity, local_hub_files, output_folder
from autodub.benchmedia import extract_audio
from autodub.config import DEFAULT_VOICE_ID
from autodub.storage import atomic_json, sha256_file


def stage_code(source: Path, target: Path, replacements: dict[str, dict[str, str]]) -> None:
    """Make declared path-only adaptations in a run copy; never mutate pinned assets."""
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    for relative, mappings in replacements.items():
        path = target / relative
        text = path.read_text(encoding="utf-8")
        for old, new in mappings.items():
            if text.count(old) != 1:
                raise ValueError("Pinned upstream local-path adaptation does not match")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def run(source: Path, config: dict) -> dict:
    from omegaconf import OmegaConf

    reference = Path(config["voice_reference"]).resolve(strict=True)
    reference_hash = sha256_file(reference)
    if reference_hash != config["voice_reference_sha256"]:
        raise ValueError("Fixed Ngọc Huyền reference changed during the run")
    if config.get("voice_id", DEFAULT_VOICE_ID) != DEFAULT_VOICE_ID:
        raise ValueError("Voice changes require an explicitly versioned voice configuration")
    ids = ("indextts-code", "indextts2-vi", "indextts-w2v", "indextts-codec",
           "indextts-speaker", "indextts-vocoder")
    assets = {key: asset(config, key) for key in ids}
    folder = output_folder(config)
    configure_offline(config)
    os.environ["INDEXTTS_DISABLE_AUTO_DOWNLOAD"] = "1"
    os.environ["HF_HUB_CACHE"] = str(folder / "offline-cache")
    w2v = str(assets["indextts-w2v"][0])
    local_code = folder / "runtime-source"
    stage_code(assets["indextts-code"][0] / "source", local_code, {
        "indextts/infer_v2.py": {'from_pretrained("facebook/w2v-bert-2.0")':
                                f"from_pretrained({w2v!r}, local_files_only=True)"},
        "indextts/utils/maskgct_utils.py": {'from_pretrained("facebook/w2v-bert-2.0")':
                                f"from_pretrained({w2v!r}, local_files_only=True)"},
    })
    sys.path.insert(0, str(local_code))
    weights = assets["indextts2-vi"][0]
    derived = OmegaConf.load(weights / "config.yaml")
    derived.qwen_emo_path = ""
    derived.vocoder.name = str(assets["indextts-vocoder"][0])
    cfg_path = folder / "runtime-config.yaml"
    OmegaConf.save(derived, cfg_path)
    tick = time.perf_counter()
    with local_hub_files({"amphion/MaskGCT": assets["indextts-codec"][0],
                          "funasr/campplus": assets["indextts-speaker"][0]}):
        from indextts.infer_v2 import IndexTTS2
        model = IndexTTS2(cfg_path=str(cfg_path), model_dir=str(weights), use_fp16=True,
                         device=config.get("device", "cuda:0"), use_cuda_kernel=False,
                         use_deepspeed=False, use_accel=False)
    if model.campplus_model is None:
        raise RuntimeError("Pinned speaker guidance failed to load")
    load_ms = (time.perf_counter() - tick) * 1000
    clips, failures, inference_ms, preprocess_ms = {}, [], 0.0, 0.0
    for segment in config["segments"]:
        if segment["action"] != "DUB" or not segment.get("dub_vi", "").strip():
            continue
        sid = segment["id"]
        tick = time.perf_counter()
        emo = folder / f"emotion-{len(clips):06d}.wav"
        extract_audio(source, emo, segment["start_ms"], segment["end_ms"], sample_rate=24000,
                      channels=1, ffmpeg_bin=config.get("ffmpeg_bin", "ffmpeg"))
        preprocess_ms += (time.perf_counter() - tick) * 1000
        output = folder / f"voice-{len(clips):06d}.wav"
        tick = time.perf_counter()
        try:
            model.infer(spk_audio_prompt=str(reference), text=segment["dub_vi"], output_path=str(output),
                        emo_audio_prompt=str(emo), emo_alpha=1.0, use_emo_text=False, do_sample=False)
            if not output.is_file():
                raise RuntimeError("IndexTTS2 produced no audio")
            clips[sid] = str(output)
        except Exception as error:
            failures.append({"segment_id": sid, "code": "TTS_CLIP_UNAVAILABLE",
                             "error_type": type(error).__name__})
        inference_ms += (time.perf_counter() - tick) * 1000
    report = folder / "tts.json"
    atomic_json(report, {"schema_version": 1, "clips": clips, "issues": failures,
        "voice_id": DEFAULT_VOICE_ID, "voice_reference_sha256": reference_hash,
        "emotion_source": "original separated Chinese speech", "model_fallbacks": [],
        "derived_config_sha256": sha256_file(cfg_path)})
    return {**identity([item[1] for item in assets.values()]), "clips": clips,
        "stage_status": "DEGRADED" if failures else "SUCCESS", "quality_evidence": {"issues": failures},
        "metrics": {"model_load_ms": load_ms, "preprocess_ms": preprocess_ms, "inference_ms": inference_ms},
        "artifacts": [str(report), str(cfg_path), *clips.values()]}
