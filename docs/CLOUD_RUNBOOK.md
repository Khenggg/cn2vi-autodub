# GPU cloud benchmark runbook

The primary source and setup route is the private GitHub repository `Khenggg/cn2vi-autodub`. Clone it on the cloud host, then run `bash scripts/cloud_setup.sh`; the setup guide has the exact commands and host requirements. Setup installs the pinned environments, fetches/verifies locked assets and runs cloud preflight. It does not create or terminate a VM or start benchmark jobs. Check provider billing and stop the VM yourself when the run is complete.

The private repository requires an authorized GitHub login, credential helper, or SSH key. Do not put a PAT in the clone URL. The optional offline fallback is the self-extracting `.run` created by `scripts/package_cloud.ps1`; it embeds the Git bundle and built frontend. See [automatic cloud setup](AUTOMATIC_CLOUD_SETUP.md). Its SHA-256 sidecar detects transfer corruption but is not a publisher signature.

The sections below retain the step-by-step manual diagnostics. `cloud_setup.sh` already fetches and verifies assets and runs each profile preflight; do not repeat these steps when the setup journal reports `READY`.

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
