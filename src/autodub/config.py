import os
from dataclasses import dataclass, field
from pathlib import Path

# User-selected default. Change only after an explicit user request.
DEFAULT_VOICE_ID = "Ngọc Huyền"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")).resolve())
    admin_token: str = field(default_factory=lambda: os.getenv("ADMIN_TOKEN", ""))
    cookie_secure: bool = field(default_factory=lambda: os.getenv("COOKIE_SECURE", "false").lower() == "true")
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", 20 * 1024**3)))
    workspace_quota_bytes: int = field(default_factory=lambda: int(os.getenv("WORKSPACE_QUOTA_BYTES", 80 * 1024**3)))
    max_chunk_bytes: int = 8 * 1024**2
    ffprobe_bin: str = field(default_factory=lambda: os.getenv("FFPROBE_BIN", "ffprobe"))
    ffmpeg_bin: str = field(default_factory=lambda: os.getenv("FFMPEG_BIN", "ffmpeg"))
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "/data/models")).resolve())
    venvs_dir: Path = field(default_factory=lambda: Path(os.getenv("VENVS_DIR", "/opt/autodub/venvs")).resolve())
    enable_pipeline: bool = field(default_factory=lambda: os.getenv("ENABLE_PIPELINE", "true" if Path("/data/models").is_dir() else "false").lower() in ("1", "true", "yes"))
    voice_id: str = field(default_factory=lambda: os.getenv("VOICE_ID", DEFAULT_VOICE_ID))
    pipeline_generation: str = field(default_factory=lambda: os.getenv("PIPELINE_GENERATION", "v1"))
    voice_reference: Path = field(default_factory=lambda: Path(os.getenv("VOICE_REFERENCE", "/data/voices/ngoc-huyen.wav")))
    subtitle_mode: str = field(default_factory=lambda: os.getenv("SUBTITLE_MODE", "off"))
    cloud_rate: int = field(default_factory=lambda: int(os.getenv("CLOUD_RATE_VND_PER_HOUR", "6000")))
    gpu_safety_mb: int = field(default_factory=lambda: int(os.getenv("GPU_FREE_VRAM_SAFETY_MB", "1800")))
    frontend_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "frontend" / "dist")

    def prepare(self) -> None:
        for folder in ("app-state", "uploads", "work", "outputs", "checkpoints", "logs"):
            (self.data_dir / folder).mkdir(parents=True, exist_ok=True)
