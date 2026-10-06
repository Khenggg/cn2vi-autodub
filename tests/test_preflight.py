import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from autodub import preflight


class FakeCuda:
    def __init__(self, available=True, arch_list=None, arch_error=False):
        self.available = available
        self.arch_list = ["sm_120"] if arch_list is None else arch_list
        self.arch_error = arch_error

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def get_device_properties(self, _device):
        return SimpleNamespace(name="NVIDIA RTX 5060 Ti")

    def get_device_capability(self, _device):
        return (12, 0)

    def synchronize(self, _device):
        return None

    def empty_cache(self):
        return None

    def get_arch_list(self):
        if self.arch_error:
            raise RuntimeError("ARCH_LIST_UNAVAILABLE")
        return self.arch_list


class FakeTensor:
    def add_(self, _value):
        return self


def setup_cloud_env(monkeypatch, *, total_mib=16_384, free_mib=8_000, arch_list=None,
                    arch_error=False, cuda_available=True, tensor_error=None):
    monkeypatch.setattr(preflight, "_os_release", lambda: {
        "ID": "ubuntu", "VERSION_ID": "24.04", "PRETTY_NAME": "Ubuntu 24.04 LTS"})
    monkeypatch.setattr(preflight.sys, "version_info", (3, 12, 0, "final", 0))
    monkeypatch.setattr(preflight.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(preflight.shutil, "which", lambda executable: f"/usr/bin/{executable}")
    def command(command, timeout=8):
        del timeout
        if command[0].endswith("nvidia-smi"):
            return subprocess.CompletedProcess(command, 0,
                f"NVIDIA RTX 5060 Ti, {total_mib}, {free_mib}\n", "")
        if "-filters" in command:
            return subprocess.CompletedProcess(command, 0, " ... subtitles V->V Subtitle overlay\n", "")
        if "-encoders" in command:
            return subprocess.CompletedProcess(command, 0, " V..... h264_nvenc NVIDIA NVENC H.264 encoder\n", "")
        return subprocess.CompletedProcess(command, 0, "ffmpeg version fixture\n", "")
    monkeypatch.setattr(preflight, "_run", command)
    monkeypatch.setattr(preflight.psutil, "virtual_memory", lambda: SimpleNamespace(total=32_000_000_000))
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _path: SimpleNamespace(free=200_000_000_000))
    cuda = FakeCuda(cuda_available, arch_list, arch_error)
    def zeros(*_args, **_kwargs):
        if tensor_error:
            raise RuntimeError(tensor_error)
        return FakeTensor()
    fake_torch = SimpleNamespace(__version__="2.8.0+cu128", version=SimpleNamespace(cuda="12.8"),
                                 cuda=cuda, zeros=zeros)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    def version(package):
        return {"torchaudio": "2.8.0+cu128", "transformers": "4.57.6",
                "diffusers": "0.35.2"}[package]
    monkeypatch.setattr(preflight.importlib.metadata, "version", version)


def check(report, name):
    return next(item for item in report["checks"] if item["name"] == name)


def test_cuda_preflight_can_pass_all_cloud_gates(monkeypatch, tmp_path):
    setup_cloud_env(monkeypatch)
    report = preflight.run_preflight(profile="asr", require_cloud=True,
                                     models_root=tmp_path / "models", cache_root=tmp_path / "cache",
                                     model_manifest=tmp_path / "models.lock")
    # The optional manifest check remains the only blocker until a real lock file exists.
    assert check(report, "model_manifest")["status"] == "FAIL"
    (tmp_path / "models.lock").write_text("{}", encoding="utf-8")
    report = preflight.run_preflight(profile="asr", require_cloud=True,
                                     models_root=tmp_path / "models", cache_root=tmp_path / "cache",
                                     model_manifest=tmp_path / "models.lock")
    assert report["readiness"] == "READY", [
        (item["name"], item["status"], item["detail"]) for item in report["checks"]
        if item["status"] != "PASS"
    ]
    assert check(report, "cuda_device")["status"] == "PASS"
    assert check(report, "blackwell_sm_120")["sm_120_supported"] is True
    assert check(report, "ffmpeg_nvenc")["runtime_encoding_tested"] is False


def test_gpu_free_memory_below_guard_blocks_cloud_readiness(monkeypatch, tmp_path):
    setup_cloud_env(monkeypatch, free_mib=1_700)
    report = preflight.run_preflight(require_cloud=True, models_root=tmp_path, cache_root=tmp_path)
    assert report["readiness"] == "BLOCKED"
    assert check(report, "gpu_memory")["status"] == "FAIL"


def test_torch_arch_list_without_sm120_does_not_claim_blackwell_ready(monkeypatch, tmp_path):
    setup_cloud_env(monkeypatch, arch_list=["sm_90"])
    report = preflight.run_preflight(require_cloud=True, models_root=tmp_path, cache_root=tmp_path)
    assert check(report, "cuda_device")["status"] == "PASS"
    assert check(report, "blackwell_sm_120")["status"] == "FAIL"
    assert report["readiness"] == "BLOCKED"


def test_unavailable_torch_arch_list_is_explicitly_unconfirmed(monkeypatch, tmp_path):
    setup_cloud_env(monkeypatch, arch_error=True)
    report = preflight.run_preflight(require_cloud=True, models_root=tmp_path, cache_root=tmp_path)
    arch = check(report, "blackwell_sm_120")
    assert arch["status"] == "FAIL"
    assert arch["arch_list_available"] is False
    assert arch["sm_120_supported"] is False


def test_tensor_exception_text_is_not_serialized(monkeypatch, tmp_path):
    setup_cloud_env(monkeypatch, tensor_error="HF_TOKEN=DO_NOT_REPORT")
    report = preflight.run_preflight(require_cloud=True, models_root=tmp_path, cache_root=tmp_path)
    encoded = json.dumps(report)
    assert "DO_NOT_REPORT" not in encoded
    assert "HF_TOKEN" not in encoded
    assert check(report, "cuda_device")["status"] == "FAIL"


def test_local_non_gpu_mode_keeps_gpu_checks_pending(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "_os_release", lambda: {})
    monkeypatch.setattr(preflight.shutil, "which", lambda _executable: None)
    monkeypatch.setattr(preflight, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(preflight.psutil, "virtual_memory", lambda: SimpleNamespace(total=8_000_000_000))
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _path: SimpleNamespace(free=200_000_000_000))
    cuda = FakeCuda(available=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.8.0+cu128",
                         version=SimpleNamespace(cuda="12.8"), cuda=cuda, zeros=lambda *_a, **_kw: FakeTensor()))
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda _package: "2.8.0+cu128")
    report = preflight.run_preflight(models_root=tmp_path, cache_root=tmp_path)
    assert report["readiness"] == "PENDING"
    assert check(report, "nvidia_smi")["status"] == "PENDING"
    assert check(report, "cuda_device")["status"] == "PENDING"
    output = tmp_path / "local-preflight.json"
    assert preflight.main(["--output", str(output), "--models-root", str(tmp_path),
                           "--cache-root", str(tmp_path)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["readiness"] == "PENDING"
    assert preflight.main(["--require-cloud", "--output", str(output), "--models-root", str(tmp_path),
                           "--cache-root", str(tmp_path)]) == 2


def test_nvenc_listing_is_not_reported_as_runtime_encode_success(monkeypatch, tmp_path):
    setup_cloud_env(monkeypatch)
    report = preflight.run_preflight(require_cloud=True, models_root=tmp_path, cache_root=tmp_path)
    nvenc = check(report, "ffmpeg_nvenc")
    assert nvenc["encoder_listed"] is True
    assert nvenc["runtime_encoding_tested"] is False
    assert "was not tested" in nvenc["detail"]


def test_vision_profile_checks_onnx_cpu_without_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "_torch_checks",
                        lambda *_args, **_kwargs: pytest.fail("vision must not import/check torch"))
    fake_onnx = SimpleNamespace(__version__="1.22.0",
                                get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)
    monkeypatch.setattr(preflight, "_os_release", lambda: {})
    monkeypatch.setattr(preflight.shutil, "which", lambda _executable: None)
    monkeypatch.setattr(preflight, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(preflight.psutil, "virtual_memory", lambda: SimpleNamespace(total=8_000_000_000))
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _path: SimpleNamespace(free=200_000_000_000))
    report = preflight.run_preflight(profile="vision", models_root=tmp_path, cache_root=tmp_path)
    assert check(report, "onnxruntime_cpu")["status"] == "PASS"
    assert not any(item["name"] == "torch" for item in report["checks"])
    assert report["environment"]["torch"]["not_required"] is True
