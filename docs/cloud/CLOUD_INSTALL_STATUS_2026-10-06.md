# Cloud installation status — 2026-10-06

## Completed

- Connected to the authorized Ubuntu 22.04.5 cloud container and verified the SSH host key against the pinned local known-host entry.
- Cloned `https://github.com/Khenggg/cn2vi-autodub.git` to `/opt/autodub` at commit `01d9e7c45a4e8ecae01aed76a302dcaa2f4fa275` (`main`). The checkout is clean and tracks `origin/main`.
- Confirmed an NVIDIA RTX 3060 with 12 GB VRAM and about 366 GB free on the cloud root filesystem.
- Built the frontend with `npm ci` and `npm run build`; generated `frontend/dist/index.html`.
- Created `/opt/autodub/venvs/core` with managed CPython 3.12.12 and installed the core project dependencies. `pip check` reported no broken requirements.
- Confirmed FFmpeg 4.4.2 starts and includes the libass `subtitles` filter.
- Started the web service on `127.0.0.1:8080`. `GET /healthz` returned HTTP 200 with `{"status":"ok","version":"0.1.0"}`. Admin configuration is protected at `/etc/autodub/admin.env` with mode `0600`; the baseline web service remains in preparation-only mode pending real model/provider validation.

## Restored assets and environment checks

- Reused the already-authorized protected Drive profile at `/root/.config/rclone/rclone.conf` (mode `0600`) and downloaded `gdrive:Autodub_Backup/autodub-backup.tar` to `/data/.restore/autodub-backup.tar` (29,347,420,160 bytes). The archive and staged assets remain on the host for recovery; no Drive write or delete was performed.
- Validated and staged 1,154,544 selected archive entries under `/data/.restore/staged-assets`, restricted to `data/models` and `opt/autodub/venvs`. Promoted only absent entries and preserved the pre-existing `core` environment. Original venv configuration backups are in `/data/.restore/venv-config-backups`.
- Installed model directories under `/data/models`: `bandit-code` (1.5 MB), `bandit-erb48` (375 MB), `lama-onnx` (199 MB), `moss-torch` (85 MB), `propainter-code` (113 MB), `propainter-weights` (191 MB), `qwen-aligner` (1.8 GB), `qwen-asr` (1.8 GB), `rapidocr-v6` (31 MB), and `vieneu-turbo` (2.6 GB).
- Restored `asr`, `bandit`, `tts`, and `vision` environments using the managed CPython 3.12.12 interpreter; `core` was preserved. Repaired each restored environment’s stale `pyvenv.cfg` interpreter home/version and its project `.pth` entry from the backup host’s path to `/opt/autodub/src`.
- `pip check` passes for all five environments: `core`, `asr`, `tts`, `vision`, and `bandit`. Installed only the locked FastAPI `0.142.2` and Uvicorn `0.54.0` dependency set in `vision` and `bandit`; the resolver preview and post-install package comparison showed no changes to protected Torch/ONNX/NumPy/model-runtime packages.
- One-element CUDA allocations passed in `asr`, `bandit`, and `tts` with PyTorch `2.8.0+cu128`, CUDA `12.8`, on the RTX 3060. No model inference or full-video run was performed, so model quality/readiness remains unverified.
- Restore evidence is retained in `/data/.restore/restore-stage.log`, `/data/.restore/restore-stage.status`, `/data/.restore/restore-promote-venvs.log`, `/data/.restore/final-venv-verification.log`, and the protected `/data/.restore/restore-verified-report.json` (mode `0600`). The initial finalizer exited during model promotion; directory inspection confirmed all ten model directories are present, and the venv promotion log confirms `core` was preserved.

## Pending

- Run model-level functional checks only after the pipeline source is assembled; no model/API inference was part of this restore.
- No apt/dpkg transaction, OS upgrade, NVIDIA driver change, or Docker daemon setup was performed.
