"""Cloud-only import/API checks; no model construction or inference."""
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


def check(profile: str, models_root: Path) -> None:
    # ASR and punctuation share a pinned checkout; imports must not mutate it.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["AUTODUB_WORKER_MODELS"] = str(models_root.resolve())
    config = {"models_root": str(models_root)}
    configure_offline(config)
    if profile == "separation":
        code, _ = asset(config, "bandit-infer-code")
        sys.path.insert(0, str(code / "source/src"))
        from bandit_infer import BanditSession
        assert "checkpoint_sha256" in inspect.signature(BanditSession).parameters
    elif profile in {"asr", "punctuation"}:
        code, _ = asset(config, "firered-code")
        sys.path.insert(0, str(code / "source"))
        if profile == "asr":
            from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
            assert "return_timestamp" in inspect.signature(FireRedAsr2Config).parameters
            assert callable(FireRedAsr2.transcribe)
        else:
            from fireredasr2s.fireredpunc.punc import FireRedPunc
            assert callable(FireRedPunc.process)
    elif profile == "diarization":
        import torchcodec
        from pyannote.audio import Pipeline
        assert callable(Pipeline.from_pretrained) and torchcodec is not None
    elif profile == "indextts":
        code, _ = asset(config, "indextts-code")
        sys.path.insert(0, str(code / "source"))
        from indextts.infer_v2 import IndexTTS2
        assert "emo_audio_prompt" in inspect.signature(IndexTTS2.infer).parameters
    elif profile == "vision":
        from autodub.adapters.ocr import build_engine
        engine, _ = build_engine({**config, "ocr_asset_id": "rapidocr-v6-medium"}, resolve_asset=asset)
        assert callable(engine)
        code, _ = asset(config, "propainter-code")
        sys.path.insert(0, str(code / "source"))
        from model.propainter import InpaintGenerator
        assert callable(InpaintGenerator)
        with tempfile.TemporaryDirectory(prefix="nvenc-preflight-") as directory:
            target = Path(directory) / "encoder.mp4"
            # Normal video dimensions avoid hardware minimum-size rejection.
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=640x360:rate=10", "-t", "1", "-c:v", "h264_nvenc",
                            "-y", str(target)], check=True, capture_output=True, timeout=30)
            assert target.stat().st_size > 0
    else:
        raise ValueError("Unknown V1 profile")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    args = parser.parse_args()
    check(args.profile, args.models_root)
    print(json.dumps({"profile": args.profile, "imports": "VERIFIED", "inference": "UNVERIFIED"}))


if __name__ == "__main__":
    main()
