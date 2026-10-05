import os
import platform
import shutil
import subprocess
import time

import psutil

from autodub.config import Settings
from autodub.domain import Budget


def gpu_status() -> dict:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=3, check=False)
        if result.returncode == 0 and result.stdout.strip():
            name, total, free = result.stdout.splitlines()[0].rsplit(",", 2)
            return {"available": True, "name": name.strip(), "total_mb": int(total), "free_mb": int(free)}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return {"available": False, "name": None, "total_mb": None, "free_mb": None}


def status(settings: Settings, started: float, worker_state: str, active: str | None) -> dict:
    ram = psutil.virtual_memory()
    disk = shutil.disk_usage(settings.data_dir)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    gpu = gpu_status()
    return {"version": "0.1.0", "phase": "CORE_FOUNDATION", "worker_state": worker_state,
            "active_episode_id": active, "gpu": gpu,
            "gpu_heavy_stage_admissible": bool(gpu["available"] and gpu["free_mb"] >= settings.gpu_safety_mb),
            "ram": {"total_bytes": ram.total, "free_bytes": ram.available},
            "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
            "network": {"throughput_mbps": None, "measured_by": "browser_upload"},
            "ffprobe_available": bool(shutil.which(settings.ffprobe_bin)),
            "models": [{"name": name, "ready": False, "reason": "Benchmark and integration pending"}
                       for name in ("ASR", "Aligner", "Bandit", "TTS", "OCR", "Inpainting")],
            "api_key_configured": bool(os.getenv("DASHSCOPE_API_KEY")), "api_health": "NOT_VERIFIED",
            "platform": platform.system(), "session_elapsed_ms": elapsed_ms,
            "cost": {**Budget(cloud_rate=settings.cloud_rate).project(elapsed_ms),
                     "basis": "process_uptime_only", "is_estimate": True,
                     "cloud_rate_vnd_per_hour": settings.cloud_rate},
            "limits": {"chunk_bytes": settings.max_chunk_bytes, "max_upload_bytes": settings.max_upload_bytes}}
