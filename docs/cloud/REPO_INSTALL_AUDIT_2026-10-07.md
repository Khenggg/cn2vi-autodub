# Repo install audit — 2026-10-07

Baseline GitHub main: `01d9e7c`. The development branch is `codex/pipeline-hardening` (PR #1); main has not been merged.

Fixed blockers: Ubuntu 22.04 rejected by outer setup; 12 GB/Ampere rejected by the old Blackwell-only preflight; invalid model downloader commands and swallowed errors in main's new-server script; inconsistent model selections across installers; missing FastAPI/Uvicorn in worker environments; unpinned audio-separator added into a NumPy 1 environment; launcher ignoring custom model/env roots.

Canonical install now uses `config/cloud-runtime.json` and hash-locked runtime dependencies with isolated ASR/separation environments. Model weights and separator parameter JSONs have immutable revision/checksum entries. No Drive restore is invoked. Setup is explicit, repeatable, records step timing, distinguishes environment readiness from asset verification, and leaves the web service stopped.

User selected Faster-Whisper large-v3-turbo + multilingual alignment. Default forced-alignment assets cover English and Chinese; additional languages require pinned assets and review when unavailable. Kim_Vocal_2 is an MDX-Net model, not itself a Mel-RoFormer checkpoint.

Validation boundary: this change was prepared with source/metadata/dependency-resolution/static checks only. No model weights were downloaded locally, no local model/test execution, and no fresh cloud install or model inference was run for this change. New-model memory, quality, end-to-end behavior and dependency import compatibility remain cloud validation tasks.
