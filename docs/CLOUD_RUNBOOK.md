# GPU cloud benchmark runbook

The primary host setup is the self-extracting installer. It installs the pinned environments, fetches/verifies the locked assets and runs cloud preflight; it does not create or terminate a VM or start benchmark jobs. Review [automatic cloud setup](AUTOMATIC_CLOUD_SETUP.md) for package creation, transfer and first run. Check provider billing and stop the VM yourself when the run is complete.

For the current manual transfer workflow, commit all prepared changes on Windows and run `scripts/package_cloud.ps1`. It builds the frontend and creates `.cache/cn2vi-cloud-setup.run` plus its SHA-256 sidecar; the one `.run` contains the Git bundle and built frontend. Transfer that file to the host and optionally verify transfer integrity:

```bash
sha256sum -c cn2vi-cloud-setup.run.sha256
bash cn2vi-cloud-setup.run --dry-run
bash cn2vi-cloud-setup.run
```

There is no public hosting URL yet. Copy the file to the host manually. The embedded digest checks payload integrity but is not a publisher signature. The host must be Ubuntu 24.04 x86_64 with a working NVIDIA driver. Setup installs Git, certificates and Python 3.12, then calls `cloud_setup.sh` to install environments, fetch and verify model assets, and run preflight. It does not install NVIDIA drivers, reboot, create/stop cloud machines, or run benchmarks. Default roots are `/data`, `/opt/autodub/venvs`, and `/data/autodub/project`; see the linked setup guide to override them. Keep at least 28 GB RAM and 100 GB free disk for the RTX 5060 Ti 16 GB reference configuration.

The former separate Git bundle flow remains available for manual diagnostics only. `scripts/package_cloud.ps1 -BundleOnly` creates `.cache/cn2vi-cloud.bundle` and its sidecar; after verifying the transfer, clone it with Git, copy the matching built frontend, and run the documented scripts from that checkout. Do not mix a bundle from one commit with a frontend from another.

The sections below retain the step-by-step manual diagnostics. The installer already fetches and verifies assets and runs each profile preflight; do not repeat these steps when the setup journal reports `READY`.

## Manual diagnostics: fetch pinned model assets

Weight transfer is a separate, potentially large and metered step. Review the lock file and required disk capacity before fetching. Run the repository asset fetcher from the core environment:

```bash
/opt/autodub/venvs/core/bin/python -m autodub.model_assets fetch \
  --lock benchmarks/models.lock.json \
  --root /data/models
```

The fetch command is expected to verify pinned revisions and checksums. Its success confirms files match the lock; it does not establish inference quality or GPU readiness. Keep API credentials in the environment or an approved secret store. Do not paste them into command arguments or reports.

## Manual diagnostics: run preflight in each profile environment

Run the cloud gate from every isolated environment that will execute a model job:

```bash
mkdir -p /data/results/preflight
for profile in asr tts vision bandit; do
  args=(--require-cloud --profile "${profile}" --models-root /data/models --cache-root /data/cache \
        --output "/data/results/preflight/${profile}.json")
  if [[ "${profile}" == asr ]]; then
    args+=(--model-manifest /data/models/qwen-asr/model-manifest.json)
  fi
  "/opt/autodub/venvs/${profile}/bin/python" -m autodub.preflight "${args[@]}"
done
```

Before renting a GPU, the vision environment can be checked independently for its CPU ONNX provider:

```bash
/opt/autodub/venvs/vision/bin/python -m autodub.preflight \
  --profile vision --models-root /data/models --cache-root /data/cache \
  --output /data/results/preflight/vision-local.json
```

Without `--require-cloud`, missing host/GPU capabilities are reported as `PENDING` and the command exits zero so local CPU diagnostics can be saved. Vision uses ONNX Runtime's CPU provider and does not need torch/torchaudio. Local mode never installs CUDA packages or downloads weights.

The ASR manifest is written by the model fetch step and describes the fetched model assets. The global `benchmarks/models.lock.json` is the fetcher's source lock, not a per-model manifest.

Exit code `2` means a cloud gate is blocked. Read each report's `checks` and fix the named condition before submitting the associated job. Reports contain environment diagnostics only and do not serialize environment variables, command output, or exception messages.

Preflight checks Ubuntu 24.04/Python 3.12, FFmpeg/FFprobe and the `subtitles` filter, NVENC encoder listing, GPU memory, system RAM, model/cache disk, pinned PyTorch versions where used, a small CUDA tensor allocation/operation, and both GPU compute capability `12.0` and PyTorch `sm_120` support. A listed NVENC encoder is only a build capability; this preflight does not perform a video encode. CUDA smoke success plus `nvidia-smi` alone does not claim Blackwell readiness.

## Manual diagnostics: run benchmark jobs in the matching environment

Use a suite plan with a per-job `python` path so each process uses the profile environment installed for that model. Copy `benchmarks/suite.example.json`, then set each model job's interpreter to `/opt/autodub/venvs/asr/bin/python`, `tts/bin/python`, `vision/bin/python`, or `bandit/bin/python` as appropriate. Keep input, config, adapter, output paths, and measured model settings explicit in the plan. For example:

```json
{
  "schema_version": 1,
  "jobs": [
    {
      "id": "asr-cloud",
      "stage": "asr",
      "python": "/opt/autodub/venvs/asr/bin/python",
      "input": "/data/inputs/sample.mp4",
      "config": "/data/configs/asr.json",
      "output": "/data/results/asr.json"
    },
    {
      "id": "tts-cloud",
      "stage": "tts",
      "python": "/opt/autodub/venvs/tts/bin/python",
      "input": "/data/inputs/sample.mp4",
      "config": "/data/configs/tts.json",
      "output": "/data/results/tts.json"
    }
  ]
}
```

Set model caches before invoking the suite so libraries keep transient downloads on the data volume:

```bash
export HF_HOME=/data/cache/huggingface
export TORCH_HOME=/data/cache/torch
export XDG_CACHE_HOME=/data/cache
/opt/autodub/venvs/core/bin/python -m autodub.bench_suite \
  --plan /data/configs/suite.json --dry-run
/opt/autodub/venvs/core/bin/python -m autodub.bench_suite \
  --plan /data/configs/suite.json
```

Add vision and bandit jobs with their matching profile interpreter and actual configured benchmark stages. The suite runner executes jobs sequentially in fresh processes, continues after a failed job, and records a summary. Its `MEASURED` status still requires human review of model provenance and quality evidence. Preflight is an environment gate, not a benchmark result. Save source checksums and raw reports with the run artifacts; a model returning metrics does not prove those metrics are valid.

For the bounded ProPainter subtitle-removal frame-sequence job, use the `bandit` interpreter and provide dense OCR masks, an explicit ROI and interval, `fp16: true`, and a crop under 720×480. See [ProPainter benchmark notes](PROPAINTER_BENCHMARK.md) for the manifest contract, config example, output limits, and review requirements.
