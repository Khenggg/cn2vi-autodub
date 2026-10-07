"""V1 filesystem authority belongs to the launcher, independently of job JSON."""
import json
import os
from pathlib import Path

from autodub.model_assets import _verify_download, load_manifest
from autodub.storage import safe_path

MODEL_IDS = frozenset({"bandit-v2-cinematic", "bandit-infer-code", "pyannote-community-1",
    "firered-code", "firered-asr2-aed", "firered-punc", "indextts-code", "indextts2-vi",
    "indextts-w2v", "indextts-codec", "indextts-speaker", "indextts-vocoder",
    "rapidocr-v6-medium", "propainter-code", "propainter-weights",
    "vieneu-turbo", "moss-torch", "ngoc-huyen-reference"})


def worker_run_root() -> Path:
    return Path(os.environ["AUTODUB_WORKER_RUN_ROOT"]).resolve()


def output_folder(config: dict) -> Path:
    del config
    folder = Path(os.environ["AUTODUB_WORKER_OUTPUT"]).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def asset(config: dict, identifier: str) -> tuple[Path, dict]:
    del config
    if identifier not in MODEL_IDS:
        raise ValueError("Unsafe or unselected V1 model asset identifier")
    root = Path(os.environ["AUTODUB_WORKER_MODELS"]).resolve()
    path = safe_path(root, identifier)
    manifest = load_manifest(path)
    if os.environ.get("AUTODUB_WORKER_RUN_ROOT"):
        lock_path = worker_run_root() / "snapshot/benchmarks/models.lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        expected = next((m for m in lock["models"] if m["id"] == identifier), None)
        if expected is None or expected["revision"] != manifest["model_revision"]:
            raise ValueError("Model asset differs from the frozen lock")
        for item in expected.get("files", []):
            _verify_download(safe_path(path, item["path"]), item)
    return path, manifest
