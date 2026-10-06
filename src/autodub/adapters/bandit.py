"""BandIt ERB48 selective-window inference with explicit checkpoint/config provenance."""
import importlib
import os
import sys
from pathlib import Path

from autodub.adapters.common import (
    allocator_metrics,
    asset,
    identity,
    milliseconds,
    output_folder,
    synchronize,
    torch_device,
)
from autodub.benchmedia import extract_audio
from autodub.media import probe_media
from autodub.storage import atomic_json

PINNED_CODE_REVISION = "840d5eb9ede59d64569c423244547e58cb00f647"


def run(source: Path, config: dict) -> dict:
    import torch
    import torchaudio
    code, code_manifest = asset(config, "bandit-code")
    weights, weights_manifest = asset(config, "bandit-erb48")
    if code_manifest.get("model_revision") != PINNED_CODE_REVISION:
        raise ValueError("BandIt code asset is not the pinned upstream commit")
    repository = code / "source"
    device, _ = torch_device(torch, config)
    os.environ["PROJECT_ROOT"] = str(repository)
    # Keep imports from creating ignored bytecode inside the immutable model checkout.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repository))
    # The upstream uses generic module names core/utils; suite gives this adapter its own process.
    LightningSystem = importlib.import_module("core").LightningSystem
    read_config = importlib.import_module("utils.config").read_nested_yaml
    configuration = read_config(str(repository / "expt" / "dnr-3s-erb48-l1snr.yaml"))
    configuration["system"]["inference"]["fader"]["kwargs"]["batch_size"] = int(config.get("batch_size", 1))
    if not 1 <= configuration["system"]["inference"]["fader"]["kwargs"]["batch_size"] <= 8:
        raise ValueError("Bandit inference batch size must be between one and eight")
    folder = output_folder(config)
    duration = probe_media(source, config.get("ffprobe_bin", "ffprobe"))["duration_ms"]
    windows = config["windows"]
    for window in windows:
        if not 0 <= window["start_ms"] < window["end_ms"] <= duration:
            raise ValueError("Bandit window outside source")
    tick = milliseconds()
    # This checkpoint is trusted only after upstream checksum verification in asset().
    checkpoint = torch.load(weights / "model.ckpt", map_location="cpu", weights_only=False)
    model = LightningSystem(config=configuration["system"], attach_fader=True)
    loaded = model.load_state_dict(checkpoint["state_dict"], strict=False)
    missing = [k for k in loaded.missing_keys if k != "fader.standard_window"]
    if missing or loaded.unexpected_keys:
        raise ValueError("Bandit checkpoint does not match the pinned ERB48 model")
    model.to(device).eval()
    synchronize(torch, device)
    load_ms = milliseconds() - tick
    manifests, artifacts, inference_ms = [], [], 0
    for index, window in enumerate(windows):
        wav = extract_audio(source, folder / f"bandit_{index:04d}.wav", window["start_ms"], window["end_ms"],
                            sample_rate=44100, channels=1, ffmpeg_bin=config.get("ffmpeg_bin", "ffmpeg"))
        audio, fs = torchaudio.load(str(wav))
        if fs != model.fs:
            raise ValueError("Bandit sample rate does not match checkpoint")
        stem_folder = folder / f"window_{index:04d}"
        model.set_predict_output_path(str(stem_folder))
        synchronize(torch, device)
        tick = milliseconds()
        with torch.inference_mode():
            model.predict_step({"audio": {"mixture": audio[None].to(device)}, "track": [f"window_{index:04d}"]},
                               include_track_name=False, get_residual=False, get_no_vox_combinations=False, fs=fs)
        synchronize(torch, device)
        inference_ms += milliseconds() - tick
        stems = sorted(stem_folder.glob("*.wav"))
        if len(stems) != 3:
            raise ValueError("Bandit must output dialogue, music and effects stems")
        artifacts.extend(str(path) for path in stems)
        manifests.append({**window, "stems": [path.relative_to(folder).as_posix() for path in stems], "input_channels": 1})
    target = folder / "separation.json"
    atomic_json(target, {"schema_version": 1, "windows": manifests, "production_mix": False,
                         "stereo_preservation_validated": False})
    return {**identity([code_manifest, weights_manifest]), "processed_media_ms": sum(w["end_ms"] - w["start_ms"] for w in windows),
            "metrics": {"model_load_ms": load_ms, "inference_ms": inference_ms, **allocator_metrics(torch, device)},
            "quality_metrics": {"dialogue_leakage": None, "sfx_damage": None, "separated_windows": len(windows)},
            "quality_evidence": {"status": "REVIEW_REQUIRED", "missing": ["reference_stems", "nonverbal_listening", "stereo_review"]},
            "artifacts": [str(target), *artifacts]}
