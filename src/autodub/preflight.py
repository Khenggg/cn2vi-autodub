"""Read-only environment checks for reproducible cloud benchmark runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # Keep local diagnostics useful in a partially installed checkout.
    psutil = None


MIN_RAM_BYTES = 15_000_000_000
MIN_DISK_BYTES = 100_000_000_000
MIN_GPU_TOTAL_MIB = 11_500
MIN_GPU_FREE_MIB = 1_800
EXPECTED_TORCH = "2.8.0"
EXPECTED_TORCHAUDIO = "2.8.0"
EXPECTED_CUDA = "12.8"
PROFILES = ("asr", "tts", "vision", "bandit", "separation", "diarization", "punctuation", "indextts")


def _check(name: str, status: str, detail: str, **values: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, **values}


def _conditional_status(ok: bool | None, require_cloud: bool) -> str:
    if ok is True:
        return "PASS"
    if ok is False:
        return "FAIL" if require_cloud else "PENDING"
    return "FAIL" if require_cloud else "PENDING"


def _run(command: list[str], timeout: int = 8) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _command_check(name: str, executable: str, args: list[str], *, strict: bool) -> tuple[dict[str, Any], str | None]:
    path = shutil.which(executable)
    if path is None:
        return _check(name, _conditional_status(False, strict), f"{executable} executable not found"), None
    result = _run([path, *args])
    if result is None or result.returncode != 0:
        return _check(name, _conditional_status(False, strict), f"{executable} did not complete successfully"), path
    first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "available")
    return _check(name, "PASS", first_line), path


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key in {"ID", "VERSION_ID", "PRETTY_NAME"}:
                values[key] = value.strip().strip('"')
    except OSError:
        pass
    return values


def _nvenc_probe(ffmpeg_path: str | None, strict: bool) -> dict[str, Any]:
    if not ffmpeg_path:
        return _check("ffmpeg_nvenc", _conditional_status(False, strict), "FFmpeg unavailable")
    result = _run([ffmpeg_path, "-hide_banner", "-encoders"], timeout=10)
    if result is None or result.returncode != 0:
        return _check("ffmpeg_nvenc", _conditional_status(False, strict), "Could not inspect FFmpeg encoder list")
    listed = any("_nvenc" in line for line in result.stdout.splitlines())
    return _check(
        "ffmpeg_nvenc",
        _conditional_status(listed, strict),
        "NVENC encoder is listed; runtime encoding was not tested" if listed else "NVENC encoder is not listed",
        encoder_listed=listed,
        runtime_encoding_tested=False,
    )


def _ffmpeg_features(ffmpeg_path: str | None, strict: bool) -> dict[str, Any]:
    if not ffmpeg_path:
        return _check("ffmpeg_libass", _conditional_status(False, strict), "FFmpeg unavailable")
    result = _run([ffmpeg_path, "-hide_banner", "-filters"], timeout=10)
    if result is None or result.returncode != 0:
        return _check("ffmpeg_libass", _conditional_status(False, strict), "Could not inspect FFmpeg filters")
    available = any(" subtitles " in f" {line} " for line in result.stdout.splitlines())
    return _check("ffmpeg_libass", _conditional_status(available, strict),
                  "subtitles filter listed" if available else "subtitles filter not listed", available=available)


def _gpu_info(strict: bool) -> tuple[dict[str, Any], dict[str, int] | None]:
    path = shutil.which("nvidia-smi")
    if not path:
        return _check("nvidia_smi", _conditional_status(False, strict), "nvidia-smi not found"), None
    result = _run([path, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
    if result is None or result.returncode != 0:
        return _check("nvidia_smi", _conditional_status(False, strict), "GPU query failed"), None
    line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    try:
        name, total, free = line.rsplit(",", 2)
        metrics = {"gpu_name": name.strip(), "total_mib": int(total), "free_mib": int(free)}
    except (ValueError, TypeError):
        return _check("nvidia_smi", _conditional_status(False, strict), "Could not parse first GPU memory row"), None
    return _check("nvidia_smi", "PASS", "GPU reported by nvidia-smi", **metrics), metrics


def _torch_checks(strict: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    versions: dict[str, Any] = {"torch": None, "torchaudio": None, "cuda_runtime": None, "device": None,
                                "capability": None, "arch_list": []}
    try:
        import torch
    except ImportError:
        checks.append(_check("torch", _conditional_status(False, strict), "torch is not installed in this environment"))
        checks.extend([
            _check("torch_versions", _conditional_status(False, strict), "Torch/CUDA version checks unavailable"),
            _check("cuda_device", _conditional_status(False, strict), "CUDA device smoke test unavailable"),
            _check("blackwell_sm_120", _conditional_status(False, strict), "Torch architecture list unavailable"),
        ])
        return checks, versions

    versions["torch"] = str(torch.__version__).split("+", 1)[0]
    versions["cuda_runtime"] = torch.version.cuda
    checks.append(_check("torch", "PASS", f"torch {torch.__version__}"))
    version_messages = []
    version_ok = versions["torch"] == EXPECTED_TORCH and torch.version.cuda == EXPECTED_CUDA
    try:
        versions["torchaudio"] = importlib.metadata.version("torchaudio").split("+", 1)[0]
        version_ok = version_ok and versions["torchaudio"] == EXPECTED_TORCHAUDIO
        version_messages.append(f"torchaudio {versions['torchaudio']}")
    except importlib.metadata.PackageNotFoundError:
        version_ok = False
        version_messages.append("torchaudio missing")
    version_messages.append(f"CUDA runtime {torch.version.cuda or 'unavailable'}")
    checks.append(_check("torch_versions", _conditional_status(version_ok, strict),
                         ", ".join(version_messages), expected_torch=EXPECTED_TORCH,
                         expected_torchaudio=EXPECTED_TORCHAUDIO, expected_cuda=EXPECTED_CUDA))

    if not torch.cuda.is_available():
        checks.append(_check("cuda_device", _conditional_status(False, strict), "torch.cuda.is_available() is false"))
        checks.append(_check("blackwell_sm_120", _conditional_status(False, strict), "CUDA device architecture unavailable"))
        return checks, versions

    try:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        capability = torch.cuda.get_device_capability(device)
        versions.update(device=properties.name, capability=f"{capability[0]}.{capability[1]}")
        smoke = torch.zeros((1024, 1024), device=f"cuda:{device}")
        smoke.add_(1)
        torch.cuda.synchronize(device)
        del smoke
        torch.cuda.empty_cache()
        checks.append(_check("cuda_device", "PASS", "CUDA allocation, tensor operation, and synchronize succeeded",
                             device=properties.name, capability=versions["capability"]))
        try:
            arch_list = list(torch.cuda.get_arch_list())
        except (AttributeError, RuntimeError):
            arch_list = []
        versions["arch_list"] = arch_list
        supports_blackwell = ("sm_120" in arch_list or "compute_120" in arch_list) and capability == (12, 0)
        # Require Blackwell-specific kernels only on a Blackwell device.
        architecture_ok = supports_blackwell if capability == (12, 0) else True
        checks.append(_check("blackwell_sm_120", _conditional_status(architecture_ok, strict),
                             "CUDA smoke passed on a non-Blackwell GPU" if capability != (12, 0) else
                             "GPU reports compute capability 12.0 and Torch includes sm_120" if supports_blackwell else
                             "GPU capability 12.0 plus Torch sm_120 support are both required",
                             sm_120_supported=supports_blackwell,
                             not_required=capability != (12, 0),
                             arch_list_available=bool(arch_list)))
    except Exception as error:
        # Exception text may include paths or environment details; report only its type.
        checks.append(_check("cuda_device", _conditional_status(False, strict),
                             f"CUDA smoke test failed ({type(error).__name__})"))
        checks.append(_check("blackwell_sm_120", _conditional_status(False, strict),
                             "Blackwell support not established because CUDA smoke failed"))
    return checks, versions


def _onnx_cpu_check(strict: bool) -> dict[str, Any]:
    try:
        import onnxruntime
    except ImportError:
        return _check("onnxruntime_cpu", _conditional_status(False, strict),
                      "onnxruntime is not installed in this environment")
    try:
        providers = list(onnxruntime.get_available_providers())
    except Exception as error:
        return _check("onnxruntime_cpu", _conditional_status(False, strict),
                      f"ONNX Runtime provider query failed ({type(error).__name__})")
    available = "CPUExecutionProvider" in providers
    return _check("onnxruntime_cpu", _conditional_status(available, strict),
                  "CPUExecutionProvider is available" if available else "CPUExecutionProvider is unavailable",
                  version=getattr(onnxruntime, "__version__", None), cpu_provider_available=available,
                  available_providers=providers)


def _onnx_cuda_check(strict: bool) -> dict[str, Any]:
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        providers = list(ort.get_available_providers())
    except Exception as error:
        return _check("onnxruntime_cuda", _conditional_status(False, strict),
                      f"CUDA provider unavailable ({type(error).__name__})")
    return _check("onnxruntime_cuda", _conditional_status("CUDAExecutionProvider" in providers, strict),
                  "CUDA provider must also pass the model-session smoke check", available_providers=providers,
                  runtime_session_verified=False)


def _voiceover_filters(ffmpeg_path: str | None, strict: bool) -> dict[str, Any]:
    result = _run([ffmpeg_path, "-hide_banner", "-filters"]) if ffmpeg_path else None
    names = {"acrossover", "sidechaincompress", "amix", "alimiter", "atempo"}
    available = {line.split()[1] for line in result.stdout.splitlines() if len(line.split()) >= 2} if result else set()
    missing = sorted(names - available)
    return _check("ffmpeg_voiceover_dsp", _conditional_status(not missing, strict),
                  "Multi-band DSP filters available" if not missing else "Required DSP filters missing", missing=missing)


def _disk_check(name: str, path: Path, strict: bool) -> dict[str, Any]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return _check(name, _conditional_status(False, strict), f"Cannot inspect storage for {path}", path=str(path))
    status = _conditional_status(free >= MIN_DISK_BYTES, strict)
    return _check(name, status, f"{free} bytes free; need {MIN_DISK_BYTES} bytes", path=str(path),
                  free_bytes=free, minimum_bytes=MIN_DISK_BYTES)


def run_preflight(*, profile: str | None = None, require_cloud: bool = False,
                  models_root: Path = Path("/data/models"), cache_root: Path = Path("/data/cache"),
                  model_manifest: Path | None = None, torch_for_vision: bool = False,
                  onnx_cuda: bool = False, voiceover: bool = False) -> dict[str, Any]:
    """Collect environment diagnostics. This function does not install or download anything."""
    strict = require_cloud
    checks: list[dict[str, Any]] = []
    os_release = _os_release()
    os_ok = os_release.get("ID") == "ubuntu" and os_release.get("VERSION_ID") in {"22.04", "24.04"}
    checks.append(_check("os", _conditional_status(os_ok, strict),
                         os_release.get("PRETTY_NAME", platform.platform()), expected="Ubuntu 22.04 or 24.04"))
    py_ok = sys.version_info[:2] == (3, 12)
    checks.append(_check("python", _conditional_status(py_ok, strict), platform.python_version(),
                         expected="3.12.x", executable=sys.executable))

    ffmpeg, ffmpeg_path = _command_check("ffmpeg", "ffmpeg", ["-version"], strict=strict)
    ffprobe, _ = _command_check("ffprobe", "ffprobe", ["-version"], strict=strict)
    checks.extend((ffmpeg, ffprobe, _ffmpeg_features(ffmpeg_path, strict), _nvenc_probe(ffmpeg_path, strict)))
    if voiceover:
        checks.append(_voiceover_filters(ffmpeg_path, strict))

    gpu_check, gpu = _gpu_info(strict)
    checks.append(gpu_check)
    if gpu is None:
        checks.append(_check("gpu_memory", _conditional_status(False, strict), "GPU memory metrics unavailable"))
    else:
        checks.append(_check("gpu_memory", _conditional_status(
            gpu["total_mib"] >= MIN_GPU_TOTAL_MIB and gpu["free_mib"] >= MIN_GPU_FREE_MIB, strict),
            f"{gpu['total_mib']} MiB total, {gpu['free_mib']} MiB free; need at least "
            f"{MIN_GPU_TOTAL_MIB} MiB total and {MIN_GPU_FREE_MIB} MiB free",
            total_mib=gpu["total_mib"], free_mib=gpu["free_mib"],
            minimum_total_mib=MIN_GPU_TOTAL_MIB, minimum_free_mib=MIN_GPU_FREE_MIB))

    if psutil is None:
        checks.append(_check("ram", _conditional_status(False, strict), "psutil is not installed"))
    else:
        total_ram = psutil.virtual_memory().total
        checks.append(_check("ram", _conditional_status(total_ram >= MIN_RAM_BYTES, strict),
                             f"{total_ram} bytes total; need {MIN_RAM_BYTES} bytes", total_bytes=total_ram,
                             minimum_bytes=MIN_RAM_BYTES))
    checks.extend((_disk_check("models_disk", models_root, strict), _disk_check("cache_disk", cache_root, strict)))

    if profile == "vision" and not torch_for_vision:
        torch_info = {"torch": None, "torchaudio": None, "cuda_runtime": None, "device": None,
                      "capability": None, "arch_list": [], "not_required": True}
        checks.append(_onnx_cuda_check(strict) if onnx_cuda else _onnx_cpu_check(strict))
    else:
        torch_checks, torch_info = _torch_checks(strict)
        checks.extend(torch_checks)
        if profile == "vision":
            checks.append(_onnx_cuda_check(strict) if onnx_cuda else _onnx_cpu_check(strict))
    if profile:
        required_distributions = {
            "asr": ("kaldi-native-fbank", "kaldiio", "transformers"),
            "tts": ("transformers",),
            "vision": (),
            "bandit": (),
            "separation": ("soundfile",),
            "diarization": ("pyannote.audio", "torchcodec"),
            "punctuation": ("transformers",),
            "indextts": ("transformers", "audioread", "descript-audiotools", "modelscope"),
        }[profile]
        for package in required_distributions:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                checks.append(_check(f"profile_package:{package}", _conditional_status(False, strict),
                                     f"{package} distribution is not installed"))
            else:
                checks.append(_check(f"profile_package:{package}", "PASS", f"{package} {version}", version=version))
    if model_manifest is not None:
        exists = model_manifest.is_file()
        checks.append(_check("model_manifest", _conditional_status(exists, strict),
                             "manifest file exists; model asset contents are validated by model_assets fetch"
                             if exists else "manifest file not found", path=str(model_manifest)))

    cloud_failures = [c["name"] for c in checks if c["status"] == "FAIL"]
    pending = [c["name"] for c in checks if c["status"] == "PENDING"]
    ready = not cloud_failures and not pending
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "require_cloud": require_cloud,
        "readiness": "READY" if ready else "BLOCKED" if cloud_failures else "PENDING",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "executable": sys.executable, "torch": torch_info},
        "checks": checks,
        "summary": {"passed": sum(c["status"] == "PASS" for c in checks),
                    "failed": len(cloud_failures), "pending": len(pending),
                    "blocking_checks": cloud_failures},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("preflight-report.json"))
    parser.add_argument("--require-cloud", action="store_true",
                        help="return exit code 2 unless all cloud checks pass")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--models-root", type=Path, default=Path("/data/models"))
    parser.add_argument("--cache-root", type=Path, default=Path("/data/cache"))
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--torch-for-vision", action="store_true")
    parser.add_argument("--onnx-cuda", action="store_true")
    parser.add_argument("--voiceover", action="store_true")
    args = parser.parse_args(argv)
    report = run_preflight(profile=args.profile, require_cloud=args.require_cloud,
                           models_root=args.models_root, cache_root=args.cache_root,
                           model_manifest=args.model_manifest, torch_for_vision=args.torch_for_vision,
                           onnx_cuda=args.onnx_cuda, voiceover=args.voiceover)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Preflight {report['readiness']}: {report['summary']['passed']} passed, "
          f"{report['summary']['failed']} failed, {report['summary']['pending']} pending; {args.output}")
    return 2 if args.require_cloud and report["readiness"] != "READY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
