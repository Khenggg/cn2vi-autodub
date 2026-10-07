# CN2VI V1: fresh ephemeral cloud installation

Every newly rented/restarted host is treated as empty. No previous virtualenv, Docker,
model cache, API file, reference voice or Drive restore is assumed.

Use Ubuntu 22.04 or 24.04 with a working NVIDIA driver, x86_64, at least 12 GB VRAM,
16 GB RAM and 100 GB free disk. The reference benchmark host has 16 GB VRAM and about
28 GB RAM; fitting the complete new stack on 12 GB needs actual GPU measurement.

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates
git clone https://github.com/Khenggg/cn2vi-autodub.git
cd cn2vi-autodub
```

Before the full model download, accept Community-1 access at
https://huggingface.co/pyannote/speaker-diarization-community-1 and create a read token.
Enter it on the cloud terminal without echoing or saving it in shell history:

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
bash install.sh --verify-startup
unset HF_TOKEN
```

The installer provisions system libraries/FFmpeg/fonts, managed Python 3.12,
isolated dependency profiles, CUDA PyTorch, pinned Node/npm and the frontend. It downloads
only the V1 plan, verifies file checksums/commits, checks CUDA/audio imports and model API
imports, and performs a short NVENC encoder check. Native setup does not require Docker
or NVIDIA Container Toolkit and never upgrades a working host driver.

Weights currently total about 11.45 GiB, plus source checkouts, virtualenvs and caches.
Historical model entries remain in the benchmark lock but are not selected by V1 setup.
`bash install.sh --dry-run` prints the plan. `--prepare-only` prepares environments without
weights; it does not establish model readiness. Journals under `/data/results` distinguish
environment/assets/startup verification from unverified model inference and video quality.

Additional job inputs, separate from Git:

- The existing user-selected reference is copied to `/data/voices/ngoc-huyen.wav` before
  the first GPU run. Each run freezes its hash and a private copy. No replacement voice
  is generated automatically. The development copy is outside Git.
- `/data/run/translation.env` contains `DEEPSEEK_API_KEY=...`, permissions 0600.
  `start_web.sh` reads this assignment without executing the file. An absent key does
  not activate a local translation model.
- Source video is uploaded through the web interface.

Start processing after these inputs are available:

```bash
bash scripts/start_web.sh stop
ENABLE_PIPELINE=true PIPELINE_GENERATION=v1 bash scripts/start_web.sh start
```

Forward SSH port 8080 to the local browser. The admin token is in
`/data/run/admin_token.txt`, owner-only; it is not printed in install logs.

## First GPU validation

Run a short real-video sample first. Freeze the commit, configuration, model lock,
reference and source for the whole experiment. Stage failures never invoke another model.
Quality problems remain visible in the exported video/report; fatal dependencies preserve
the independently usable original audio/picture. An encoder failure may leave only the
preview or original source, explicitly labelled as partial.

Artifacts are under `/data/work/<episode>/v1-runs/<run-id>`: preview/final, immutable
source-code/config snapshots, raw ASR evidence, translation usage, voice clips, context,
checkpoint and `run-report.json`/`run-report.md`. Missing API price is unknown, not zero.
Model workers reload weights per stage; host cache warmth is not measured or claimed.

Drain checkpoints after a completed stage. Resume rejects changed code/config/source or
artifact hashes. This is resume within the same restored data layout; workspace `.aidub`
remains metadata-only and is not a portable media/model backup. Preserve the source and
episode run directory separately before destroying an ephemeral host.

Initial experimental limits: unresolved overlapping speakers keep original speech;
unconfirmed character/addressee identities remain explicit; inpainting requires a constant,
zero-origin frame timeline and uses bounded subtitle crops with temporal context. Mask,
emotion/timbre fidelity, full-film timing and cost targets require GPU/video review.
The current scheduler processes episodes serially; overlapping CPU/API work across episodes
and complete portable workspace import are subsequent product work, not measured gains.
