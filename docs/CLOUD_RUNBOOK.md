# GPU cloud benchmark runbook

This runbook prepares an Ubuntu 24.04 GPU host for the pinned benchmark profiles. It does not provision or terminate cloud machines, fetch model weights, or start a benchmark automatically. Check provider billing and stop the VM yourself when the run is complete.

## 1. Bootstrap the host

This checkout can also be transferred without a Git hosting remote. After committing the prepared code, run `scripts/package_cloud.ps1` on Windows. It creates `.cache/cn2vi-cloud.bundle` and a SHA-256 sidecar from the committed branch, without model weights, environment folders or local caches. Transfer the bundle and sidecar to the host, then verify and clone there:

```bash
cd /data/transfer
sha256sum -c cn2vi-cloud.bundle.sha256
git clone cn2vi-cloud.bundle ~/cn2vi-autodub
cd ~/cn2vi-autodub
```

The host needs Git before cloning; provider images usually supply it, or install Git through apt first. Video/corpus and model assets are transferred/fetched separately. Rebuild the bundle after any later code change.

Start from a clean Ubuntu 24.04 image with Python 3.12, a Blackwell RTX 5060 Ti 16 GB GPU, at least 28 GB decimal system RAM, and at least 100 GB decimal free disk on the model/cache volume. Clone this repository onto the instance and run:

```bash
cd /path/to/D-Video
bash scripts/cloud_bootstrap.sh
```

The script requests `sudo` for apt and dedicated data/venv directories. It installs FFmpeg, Git, Latin and CJK fonts, `tini`, and Python venv support, then creates a core environment and isolated `asr`, `tts`, `vision`, and `bandit` environments. Core installs the exact-version-pinned `requirements.lock`; this file does not carry hashes. Model profiles install the pinned CUDA 12.8 PyTorch pair first where applicable, then their hash-checked `requirements/bench-<profile>.txt` lock and a source path file for this checkout. Bootstrap checks `pip check` and exact torch 2.8.0, torchaudio 2.8.0, CUDA 12.8 wheel metadata in the torch profiles. Vision stays ONNX CPU and verifies `CPUExecutionProvider` without installing torch. Re-running updates environments and preserves existing data. It does not call cloud APIs or remove anything.

Set `AUTODUB_DATA_ROOT` and/or `AUTODUB_VENV_ROOT` before invoking the script to use different mount points. The defaults are `/data` and `/opt/autodub/venvs`. Models, caches, and reports use `/data/models`, `/data/cache`, and `/data/results` respectively.

## 2. Fetch pinned model assets

Weight transfer is a separate, potentially large and metered step. Review the lock file and required disk capacity before fetching. Run the repository asset fetcher from the core environment:

```bash
/opt/autodub/venvs/core/bin/python -m autodub.model_assets fetch \
  --lock benchmarks/models.lock.json \
  --root /data/models
```

The fetch command is expected to verify pinned revisions and checksums. Its success confirms files match the lock; it does not establish inference quality or GPU readiness. Keep API credentials in the environment or an approved secret store. Do not paste them into command arguments or reports.

## 3. Run preflight in each profile environment

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

## 4. Run benchmark jobs in the matching environment

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
