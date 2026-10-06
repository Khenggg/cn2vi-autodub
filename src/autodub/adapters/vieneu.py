"""VieNeu v3 Turbo female preset synthesis, with explicit pinned CPU/GPU assets."""
from pathlib import Path

from autodub.adapters.common import (
    allocator_metrics,
    asset,
    configure_offline,
    identity,
    local_hub_files,
    milliseconds,
    output_folder,
    synchronize,
    torch_device,
)
from autodub.domain import fitting_action
from autodub.quality import audio_qc


class VieNeuProvider:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.manifests = []
        self.metrics = {}

    def load(self):
        from vieneu import Vieneu
        configure_offline(self.config)
        self.device = self.config.get("device", "cuda:0")
        backbone, meta = asset(self.config, "vieneu-turbo-onnx" if self.device == "cpu" else "vieneu-turbo")
        self.manifests = [meta]
        kwargs = {"mode": "v3turbo", "device": self.device, "backbone_repo": str(backbone),
                  "backend": "onnx" if self.device == "cpu" else "pytorch", "max_batch_size": 1,
                  "max_streams": 1, "babble_retries": 2, "precision": "fp32"}
        codec_id = "moss-onnx" if self.device == "cpu" else "moss-torch"
        codec, codec_meta = asset(self.config, codec_id)
        self.manifests.append(codec_meta)
        if self.device == "cpu":
            kwargs["onnx_dir"] = str(backbone / "onnx_update")
        else:
            import torch
            torch_device(torch, self.config)
            kwargs.update(moss_tokenizer=str(codec), dtype=self.config.get("dtype", "bfloat16"))
        tick = milliseconds()
        with local_hub_files({"OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX": codec,
                              "pnnbao-ump/VieNeu-TTS-v3-Turbo": backbone}):
            self.model = Vieneu(**kwargs)
        self.metrics["model_load_ms"] = milliseconds() - tick

    def synthesize(self, text: str, voice_id: str = "Mai Anh", emotion_hint: str = "neutral",
                   target_ms: int = 1000) -> dict:
        import soundfile
        if not text.strip() or target_ms <= 0:
            raise ValueError("TTS needs non-empty text and positive target_ms")
        if self.model is None:
            self.load()
        if voice_id not in [name for _, name in self.model.list_preset_voices()]:
            raise ValueError("Configured preset voice is unavailable")
        folder = output_folder(self.config)
        if self.device != "cpu":
            import torch
            synchronize(torch, self.device)
        tick = milliseconds()
        waveform = self.model.infer(text=text, voice=voice_id, batch_size=1, apply_watermark=False)
        if self.device != "cpu":
            synchronize(torch, self.device)
        self.metrics["inference_ms"] = milliseconds() - tick
        if self.device != "cpu":
            self.metrics.update(allocator_metrics(torch, self.device))
        wav = folder / "tts.wav"
        soundfile.write(str(wav), waveform, self.model.sample_rate, subtype="PCM_16")
        qc = audio_qc(wav)
        return {"wav": str(wav), "actual_ms": qc["duration_ms"], "qc": qc,
                "duration_action": fitting_action(qc["duration_ms"], target_ms),
                "emotion_hint": emotion_hint, "emotion_control": "punctuation_only"}


def run(source: Path, config: dict) -> dict:
    # TTS RTF uses generated audio duration, not the source video's length.
    provider = VieNeuProvider(config)
    result = provider.synthesize(config["text"], config.get("voice_id", "Mai Anh"),
                                 config.get("emotion_hint", "neutral"), int(config["target_ms"]))
    return {**identity(provider.manifests), "quality_metrics": {**result["qc"],
                "target_ms": config["target_ms"], "duration_action": result["duration_action"],
                "pronunciation_human_rating": None}, "quality_evidence": {"status": "REVIEW_REQUIRED", "missing": ["pronunciation_human_rating"]},
            "processed_media_ms": result["actual_ms"], "metrics": provider.metrics,
            "artifacts": [result["wav"]]}
