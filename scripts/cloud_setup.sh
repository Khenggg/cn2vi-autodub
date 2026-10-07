#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/cloud_setup.sh [--dry-run] [--prepare-only] [--check-host] [--verify-startup] [--profiles <list>] [--profile <name>] [--with-docker]

Run from the cloned GitHub checkout on the cloud VM. The normal run bootstraps the
Ubuntu host, fetches and verifies the runtime plan assets, then requires every
profile preflight to pass. --prepare-only skips model weights; --check-host is read-only.
--verify-startup tests Web UI startup and healthz. It does not create or stop machines or run benchmarks.

Environment:
  PROJECT_ROOT          Project directory (defaults to this script's parent)
  AUTODUB_DATA_ROOT     Persistent data root (defaults to /data)
  AUTODUB_VENV_ROOT     Virtualenv root (defaults to /opt/autodub/venvs)
  AUTODUB_SETUP_JOURNAL Journal path (defaults to DATA_ROOT/results/cloud-setup-<run>.log)
  AUTODUB_PROFILES      Optional comma-separated profile list
EOF
}

DRY_RUN=0
PREPARE_ONLY=0
CHECK_HOST=0
VERIFY_STARTUP=0
WITH_DOCKER=0
EXPLICIT_PROFILES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --check-host) CHECK_HOST=1; shift ;;
    --verify-startup) VERIFY_STARTUP=1; shift ;;
    --with-docker) WITH_DOCKER=1; shift ;;
    --profiles) EXPLICIT_PROFILES="$2"; shift 2 ;;
    --profile) EXPLICIT_PROFILES="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 64 ;;
  esac
done

if [[ -z "$EXPLICIT_PROFILES" && -n "${AUTODUB_PROFILES:-}" ]]; then
  EXPLICIT_PROFILES="${AUTODUB_PROFILES}"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
AUTODUB_DATA_ROOT="${AUTODUB_DATA_ROOT:-/data}"
AUTODUB_VENV_ROOT="${AUTODUB_VENV_ROOT:-/opt/autodub/venvs}"
LOCK_FILE="$PROJECT_ROOT/benchmarks/models.lock.json"
BOOTSTRAP="$PROJECT_ROOT/scripts/cloud_bootstrap.sh"

PLAN_PYTHON="$(command -v python3 || true)"

PLAN_ARGS=()
if [[ -n "$EXPLICIT_PROFILES" ]]; then
  PLAN_ARGS+=(--profiles "$EXPLICIT_PROFILES")
fi

ASSETS=()
PROFILES=()
if [[ -n "$PLAN_PYTHON" ]]; then
  ASSET_LIST="$("$PLAN_PYTHON" "$PROJECT_ROOT/scripts/cloud_plan.py" "${PLAN_ARGS[@]}" --field assets)"
  PROFILE_LIST="$("$PLAN_PYTHON" "$PROJECT_ROOT/scripts/cloud_plan.py" "${PLAN_ARGS[@]}" --field profiles)"
  mapfile -t ASSETS <<< "$ASSET_LIST"
  mapfile -t PROFILES <<< "$PROFILE_LIST"
elif ((DRY_RUN)); then
  echo 'Install git, ca-certificates and python3 before setup.' >&2
  exit 69
fi

if ((DRY_RUN)); then
  printf 'DRY RUN: no packages, downloads, models, or preflight commands will run.\n'
  printf 'Project: %s\nData root: %s\nVirtualenv root: %s\n' \
    "$PROJECT_ROOT" "$AUTODUB_DATA_ROOT" "$AUTODUB_VENV_ROOT"
  printf '1. Bootstrap host: bash %s\n' "$BOOTSTRAP"
  printf '2. Fetch GPU benchmark assets from %s/models.lock.json:\n   %s\n' \
    "$PROJECT_ROOT/benchmarks" "${ASSETS[*]}"
  printf '3. Verify the same pinned assets.\n'
  printf '4. Run --require-cloud preflight profiles: %s\n' "${PROFILES[*]}"
  printf 'prepare_only=%s; Drive restore is never automatic.\n' "$PREPARE_ONLY"
  printf 'Reports and journal: %s/results\n' "$AUTODUB_DATA_ROOT"
  exit 0
fi

[[ -d "$PROJECT_ROOT" ]] || { printf 'cloud_setup: project directory not found\n' >&2; exit 66; }
[[ -f "$BOOTSTRAP" ]] || { printf 'cloud_setup: bootstrap script not found\n' >&2; exit 66; }
[[ -f "$LOCK_FILE" ]] || { printf 'cloud_setup: model lock not found\n' >&2; exit 66; }

export PROJECT_ROOT AUTODUB_DATA_ROOT AUTODUB_VENV_ROOT
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$AUTODUB_DATA_ROOT/cache/huggingface"
export TORCH_HOME="$AUTODUB_DATA_ROOT/cache/torch"
export XDG_CACHE_HOME="$AUTODUB_DATA_ROOT/cache/xdg"

check_cloud_host() {
  local architecture disk_probe available_bytes gpu_memory max_gpu_mib
  if [[ ! -r /etc/os-release ]] || ! grep -Fxq 'ID=ubuntu' /etc/os-release || \
    ! grep -Eq '^VERSION_ID="(22.04|24.04)"$' /etc/os-release; then
    printf 'cloud_setup: supported host is Ubuntu 22.04/24.04\n' >&2
    return 69
  fi
  architecture="$(uname -m)"
  if [[ "$architecture" != x86_64 ]]; then
    printf 'cloud_setup: supported host architecture is x86_64\n' >&2
    return 69
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'cloud_setup: NVIDIA driver tools are unavailable; use a GPU-ready host image\n' >&2
    return 69
  fi
  if ! gpu_memory="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)"; then
    printf 'cloud_setup: NVIDIA driver cannot query a GPU; no driver installation or reboot is attempted\n' >&2
    return 69
  fi
  max_gpu_mib="$(awk 'BEGIN { max=0 } /^[[:space:]]*[0-9]+[[:space:]]*$/ { value=$1+0; if (value>max) max=value } END { printf "%d", max }' <<< "$gpu_memory")"
  if ((max_gpu_mib < 11500)); then
    printf 'cloud_setup: GPU has %s MiB total; at least 11500 MiB is required\n' "$max_gpu_mib" >&2
    return 69
  fi
  disk_probe="$AUTODUB_DATA_ROOT"
  while [[ ! -e "$disk_probe" && "$disk_probe" != / ]]; do
    disk_probe="$(dirname -- "$disk_probe")"
  done
  available_bytes="$(df -PB1 -- "$disk_probe" | awk 'NR==2 {print $4}')"
  if [[ ! "$available_bytes" =~ ^[0-9]+$ ]] || ((available_bytes < 100000000000)); then
    printf 'cloud_setup: data filesystem needs at least 100 GB decimal free space\n' >&2
    return 69
  fi
  printf 'cloud_setup: host preflight passed (Ubuntu 22.04/24.04, x86_64, GPU >= 11500 MiB, disk >= 100 GB)\n'
}

if ((CHECK_HOST)); then
  check_cloud_host
  exit $?
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
JOURNAL="${AUTODUB_SETUP_JOURNAL:-$AUTODUB_DATA_ROOT/results/cloud-setup-$RUN_ID.log}"
JOURNAL_DIR="$(dirname -- "$JOURNAL")"
if ! mkdir -p -- "$JOURNAL_DIR" 2>/dev/null; then
  if ! command -v sudo >/dev/null 2>&1; then
    printf 'cloud_setup: cannot create journal directory and sudo is unavailable\n' >&2
    exit 1
  fi
  sudo -v || { printf 'cloud_setup: sudo authentication failed while preparing journal\n' >&2; exit 1; }
  RUN_USER="${SUDO_USER:-$(id -un)}"
  RUN_GROUP="$(id -gn "$RUN_USER")"
  sudo install -d -o "$RUN_USER" -g "$RUN_GROUP" -- "$JOURNAL_DIR" || {
    printf 'cloud_setup: cannot create journal directory\n' >&2
    exit 1
  }
fi

printf 'cloud_setup_version=1\nrun_id=%s\nstarted_at=%s\nproject_root=%s\ndata_root=%s\nvenv_root=%s\n' \
  "$RUN_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PROJECT_ROOT" "$AUTODUB_DATA_ROOT" "$AUTODUB_VENV_ROOT" > "$JOURNAL"
CURRENT_STEP=initialization
FINALIZED=0

on_exit() {
  local rc=$?
  if ((rc != 0 && FINALIZED == 0)); then
    printf 'run_status=BLOCKED\nfailed_step=%s\nexit_code=%s\n' "$CURRENT_STEP" "$rc" >> "$JOURNAL" || true
  fi
}
trap on_exit EXIT

run_step() {
  local name="$1"
  shift
  CURRENT_STEP="$name"
  printf 'step=%s status=RUNNING started_at=%s\n' "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$JOURNAL"
  printf 'cloud_setup: %s\n' "$name"
  if "$@"; then
    printf 'step=%s status=PASS finished_at=%s\n' "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$JOURNAL"
  else
    local rc=$?
    printf 'step=%s status=FAIL exit_code=%s finished_at=%s\n' "$name" "$rc" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$JOURNAL"
    printf 'cloud_setup: BLOCKED step=%s exit_code=%s\n' "$name" "$rc" >&2
    printf 'run_status=BLOCKED\nfailed_step=%s\nexit_code=%s\n' "$name" "$rc" >> "$JOURNAL"
    FINALIZED=1
    exit "$rc"
  fi
}

BOOTSTRAP_ARGS=()
if [[ -n "$EXPLICIT_PROFILES" ]]; then
  BOOTSTRAP_ARGS+=(--profiles "$EXPLICIT_PROFILES")
fi
if ((WITH_DOCKER)); then
  BOOTSTRAP_ARGS+=(--with-docker)
fi

run_step host_preflight check_cloud_host
run_step bootstrap bash "$BOOTSTRAP" "${BOOTSTRAP_ARGS[@]}"

CORE_PYTHON="$AUTODUB_VENV_ROOT/core/bin/python"
for executable in "$CORE_PYTHON"; do
  [[ -x "$executable" ]] || {
    CURRENT_STEP=bootstrap_output_validation
    printf 'cloud_setup: bootstrap did not create the core Python environment\n' >&2
    exit 1
  }
done

if [[ ${#PROFILES[@]} -eq 0 || ${#ASSETS[@]} -eq 0 ]]; then
  ASSET_LIST="$("$CORE_PYTHON" "$PROJECT_ROOT/scripts/cloud_plan.py" "${PLAN_ARGS[@]}" --field assets)"
  PROFILE_LIST="$("$CORE_PYTHON" "$PROJECT_ROOT/scripts/cloud_plan.py" "${PLAN_ARGS[@]}" --field profiles)"
  mapfile -t ASSETS <<< "$ASSET_LIST"
  mapfile -t PROFILES <<< "$PROFILE_LIST"
fi

if ((PREPARE_ONLY == 0)); then
  if [[ " ${ASSETS[*]} " == *" pyannote-community-1 "* && -z "${HF_TOKEN:-}" ]]; then
    printf 'cloud_setup: Community-1 requires Hugging Face access. Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-community-1 and export HF_TOKEN on this cloud host.\n' >&2
    exit 69
  fi
  run_step fetch_assets "$CORE_PYTHON" -m autodub.model_assets fetch --lock "$LOCK_FILE" \
    --root "$AUTODUB_DATA_ROOT/models" --only "${ASSETS[@]}"
  run_step verify_assets "$CORE_PYTHON" -m autodub.model_assets verify --lock "$LOCK_FILE" \
    --root "$AUTODUB_DATA_ROOT/models" --only "${ASSETS[@]}"
  if [[ " ${ASSETS[*]} " == *" nltk-tokenizers "* ]]; then
    run_step prepare_tokenizers "$CORE_PYTHON" "$PROJECT_ROOT/scripts/cloud_tokenizers.py" \
      --models-root "$AUTODUB_DATA_ROOT/models" --cache-root "$AUTODUB_DATA_ROOT/cache"
  fi
fi

if [[ -f "$PROJECT_ROOT/scripts/cloud_frontend.sh" && -f "$PROJECT_ROOT/frontend/package.json" ]]; then
  run_step frontend_build bash "$PROJECT_ROOT/scripts/cloud_frontend.sh"
fi

for profile in "${PROFILES[@]}"; do
  report="$AUTODUB_DATA_ROOT/results/preflight-$profile.json"
  args=(--require-cloud --profile "$profile" --models-root "$AUTODUB_DATA_ROOT/models" \
    --cache-root "$AUTODUB_DATA_ROOT/cache" --output "$report")
  if [[ "$profile" == vision && " ${ASSETS[*]} " == *" propainter-code "* ]]; then
    args+=(--torch-for-vision)
  fi
  if [[ "$profile" == asr && "$PREPARE_ONLY" == 0 ]]; then
    if [[ " ${ASSETS[*]} " == *" firered-asr2-aed "* ]]; then
      args+=(--model-manifest "$AUTODUB_DATA_ROOT/models/firered-asr2-aed/model-manifest.json")
    else
      args+=(--model-manifest "$AUTODUB_DATA_ROOT/models/whisper-asr/model-manifest.json")
    fi
  fi
  run_step "preflight_$profile" "$AUTODUB_VENV_ROOT/$profile/bin/python" -m autodub.preflight "${args[@]}"
  if ((PREPARE_ONLY == 0)) && [[ " ${ASSETS[*]} " == *" firered-code "* ]]; then
    run_step "model_import_$profile" "$AUTODUB_VENV_ROOT/$profile/bin/python" \
      "$PROJECT_ROOT/scripts/v1_model_check.py" --profile "$profile" --models-root "$AUTODUB_DATA_ROOT/models"
  fi
done

verify_service_startup() {
  local start_script="$PROJECT_ROOT/scripts/start_web.sh"
  if [[ ! -f "$start_script" ]]; then
    printf 'cloud_setup: start_web.sh not found at %s\n' "$start_script" >&2
    return 1
  fi
  printf 'cloud_setup: verifying web UI startup with ENABLE_PIPELINE=false...\n'
  ENABLE_PIPELINE=false bash "$start_script" start
  local probe_ok=0
  for ((attempt=0; attempt<15; attempt++)); do
    if command -v curl >/dev/null 2>&1; then
      if curl -sf --max-time 2 http://127.0.0.1:8080/healthz >/dev/null 2>&1; then
        probe_ok=1
        break
      fi
    fi
    sleep 1
  done
  if ((probe_ok == 0)); then
    printf 'cloud_setup: web startup probe timed out; check web log\n' >&2
    return 1
  fi
  printf 'cloud_setup: web startup probe passed; server running (PID file exists)\n'
  return 0
}

if ((VERIFY_STARTUP)); then
  run_step verify_startup verify_service_startup
fi

SETUP_STATUS=ASSETS_VERIFIED
if ((PREPARE_ONLY)); then SETUP_STATUS=ENVIRONMENT_READY; fi
if ((VERIFY_STARTUP)); then SETUP_STATUS=STARTUP_VERIFIED; fi
printf 'run_status=%s\ninference_quality=UNVERIFIED\nfinished_at=%s\n' \
  "$SETUP_STATUS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$JOURNAL"
FINALIZED=1
CURRENT_STEP=complete
printf 'cloud_setup: %s journal=%s; model inference and video quality remain unverified.\n' "$SETUP_STATUS" "$JOURNAL"
