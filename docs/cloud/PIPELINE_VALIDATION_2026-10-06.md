# Pipeline hardening: cloud validation, 2026-10-06

Source baseline: `01d9e7c45a4e8ecae01aed76a302dcaa2f4fa275`. Development was performed in an isolated local worktree; every project test, lint invocation, frontend build, and model/runtime check in this work was executed on the authorized cloud host.

## Verified results

- **237 unique test cases in 27 modules passed, zero failures and zero skips.** This is an aggregate of complete module runs, with the latest affected-module results replacing older failing runs; it is not one uninterrupted whole-suite invocation.
- Ruff passed for `src tests benchmarks scripts`.
- Frontend `tsc --noEmit` and Vite production build passed.
- Every `scripts/*.sh` passed Bash syntax checks.
- Synthetic video tests exercised real FFmpeg 4.4.2/libass burn and decode, and real frame transport with mocked OCR/LaMa. Source and unmasked decoded-frame invariants were checked; lossy encoded pixels are not claimed identical.
- API/recovery tests covered edit approval, episode/segment identity, confidence provenance, strict fitting, artifact tamper invalidation, preserved manual edits, queue/drain/restart, ROI gating, and authenticated output preview.
- Restore archive tests covered selected roots, trusted legacy interpreter links, dangling/escaping links, traversal, and link-parent rejection.

Cloud validation used Ubuntu 22.04, CPython 3.12.12, Pytest 9.1.1, Ruff 0.16.10, NumPy 1.26.4, OpenCV 4.11.0.86 and FFmpeg 4.4.2. Model processes were mocked for pipeline tests; no real translation API call or full-video/model inference was performed.

## Host filesystem observation

The first SSH-attached suite lost its output when the server closed the connection. Later checks used detached, bounded jobs with durable logs. Overlay-backed SQLite commits also stalled during two groups. A standalone ten-commit WAL probe took 5.28 seconds; this is a host/filesystem observation, not a CPU or GPU inference benchmark.

The 28 database/API/recovery cases passed in 1.25 seconds with isolated RAM-backed temporary data under `/dev/shm` on the same cloud host. Application durability settings were unchanged. These unit results validate logical recovery and file/hash boundaries; they do not prove power-loss durability or production performance on the overlay filesystem.

## Evidence and limits

Cloud evidence is under `/data/validation-hardening`: the module JUnit files, `validation-summary.json`, `db-final.log`, `affected-final.log`, `ruff-final.log`, `frontend.log`, and `shell-final.log`. Source was tested in `/opt/autodub-validation`; the clean GitHub checkout and baseline web service at `/opt/autodub` were preserved.

Ten model directories and five Python environments were restored; all five `pip check` calls and one-element CUDA allocations in ASR/BandIt/TTS passed. See [installation report](CLOUD_INSTALL_STATUS_2026-10-06.md). This establishes environment/CUDA readiness only. Translation provider selection, missing optional/local translation assets, model accuracy, listening quality, temporal subtitle quality, full-video GPU memory and actual cost remain for the next model discussion and representative benchmark. Docker image checks were not run because this rented container has no Docker daemon.
