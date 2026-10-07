import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from autodub.model_assets import load_manifest


def asset(config: dict, identifier: str) -> tuple[Path, dict]:
    root = Path(config.get("models_root", "/data/models")).resolve()
    path = root / identifier
    return path, load_manifest(path)


def output_folder(config: dict) -> Path:
    folder = Path(config["output_dir"]).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def identity(manifests: list[dict]) -> dict:
    joined = [{"id": m["id"], "revision": m["model_revision"], "hash": m["weights_sha256"]}
              for m in manifests]
    return {"model_revision": "+".join(m["model_revision"] for m in manifests),
            "weights_sha256": hashlib.sha256(json.dumps(joined, sort_keys=True).encode()).hexdigest()}


def configure_offline(config: dict) -> None:
    cache = Path(config.get("cache_root", "/data/cache")).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(cache / "huggingface"),
                      HF_HUB_CACHE=str(cache / "huggingface" / "hub"),
                      HF_MODULES_CACHE=str(cache / "huggingface" / "modules"))


def torch_device(torch, config: dict):
    device = config.get("device", "cuda:0")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Configured CUDA device is unavailable")
        free, _ = torch.cuda.mem_get_info(device)
        if free < int(config.get("gpu_free_safety_mb", 1800)) * 1024**2:
            raise RuntimeError("GPU safety reserve unavailable")
        torch.cuda.reset_peak_memory_stats(device)
    dtype = getattr(torch, config.get("dtype", "float32" if device == "cpu" else "bfloat16"))
    return device, dtype


def synchronize(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def allocator_metrics(torch, device: str) -> dict:
    if not device.startswith("cuda"):
        return {"gpu_allocator_peak_bytes": None}
    return {"gpu_allocator_peak_bytes": torch.cuda.max_memory_allocated(device),
            "gpu_reserved_peak_bytes": torch.cuda.max_memory_reserved(device)}


@contextmanager
def local_hub_files(repo_folders: dict[str, Path]):
    """Resolve SDK codec helper calls to checksum-verified local snapshots; reject unpinned pulls.

    VieNeu 3.8.1 does not expose codec_dir through its factory, so construction needs this resolver.
    Benchmark suite owns one model process at a time. The patch is restored after construction.
    """
    import huggingface_hub
    original = huggingface_hub.hf_hub_download

    def resolve(repo_id, filename, *, subfolder=None, **kwargs):
        from autodub.storage import safe_path
        if repo_id not in repo_folders:
            raise ValueError("SDK requested an unpinned model repository")
        path = safe_path(repo_folders[repo_id], f"{subfolder}/{filename}" if subfolder else filename)
        if not path.is_file():
            raise FileNotFoundError("SDK requested a missing pinned asset")
        return str(path)
    huggingface_hub.hf_hub_download = resolve
    try:
        yield
    finally:
        huggingface_hub.hf_hub_download = original


def milliseconds() -> int:
    return round(time.perf_counter() * 1000)
