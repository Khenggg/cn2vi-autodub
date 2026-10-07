"""Fresh-cloud V2 API, actual OCR CUDA-session and NVENC smoke checks."""
import argparse
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from autodub.adapters.common import configure_offline
from autodub.adapters.runtime_paths import asset


def check(profile: str, models_root: Path):
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["AUTODUB_WORKER_MODELS"] = str(models_root.resolve())
    config = {"models_root": str(models_root)}
    configure_offline(config)
    if profile == "asr":
        import webrtcvad
        code, _ = asset(config, "firered-code")
        sys.path.insert(0, str(code / "source"))
        from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
        from fireredasr2s.fireredpunc.punc import FireRedPunc
        assert "return_timestamp" in inspect.signature(FireRedAsr2Config).parameters
        assert callable(FireRedAsr2.transcribe) and callable(FireRedPunc.process)
        assert webrtcvad.Vad(0).is_speech(b"\0" * 960, 16000) is False
    elif profile == "tts":
        from vieneu.v3turbo import V3TurboVieNeuTTS
        for name in ("infer_batch", "add_voice", "close"):
            assert callable(getattr(V3TurboVieNeuTTS, name))
        assert "batch_size" in inspect.signature(V3TurboVieNeuTTS.infer_batch).parameters
    elif profile == "vision":
        import numpy as np

        from autodub.adapters.ocr import build_engine
        engine, _ = build_engine({**config, "ocr_asset_id": "rapidocr-v6-medium", "ocr_require_cuda": True},
                                resolve_asset=asset)
        assert all(session.get_providers()[0] == "CUDAExecutionProvider" for session in engine.autodub_cuda_sessions)
        engine(np.zeros((64, 128, 3), dtype="uint8"))
        with tempfile.TemporaryDirectory(prefix="v2-nvenc-") as directory:
            target = Path(directory) / "encoder.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=640x360:rate=10", "-t", "1", "-c:v", "h264_nvenc",
                            "-y", str(target)], check=True, capture_output=True, timeout=30)
            assert target.stat().st_size > 0
    else:
        raise ValueError("Unknown V2 production profile")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("asr", "tts", "vision"), required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    args = parser.parse_args()
    check(args.profile, args.models_root)
    print(json.dumps({"profile": args.profile, "api_check": "VERIFIED",
                      "full_video_quality_and_sla": "UNVERIFIED"}))


if __name__ == "__main__":
    main()
