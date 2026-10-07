#!/usr/bin/env bash
set -Eeuo pipefail

# Safe to rerun. Installs OS tools and creates/updates environments; never shuts down
# the host, removes existing data, or downloads model weights.
# Native setup does not require Docker. Preserves working NVIDIA driver (verifies, never upgrades).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${AUTODUB_DATA_ROOT:-/data}"
VENV_ROOT="${AUTODUB_VENV_ROOT:-/opt/autodub/venvs}"

EXPLICIT_PROFILES=""
WITH_DOCKER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiles)
      EXPLICIT_PROFILES="$2"
      shift 2
      ;;
    --profile)
      EXPLICIT_PROFILES="$2"
      shift 2
      ;;
    --with-docker)
      WITH_DOCKER=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "${EXPLICIT_PROFILES}" && -n "${AUTODUB_PROFILES:-}" ]]; then
  EXPLICIT_PROFILES="${AUTODUB_PROFILES}"
fi

PIP_VERSION="25.3"
SETUPTOOLS_VERSION="80.9.0"
WHEEL_VERSION="0.45.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

if [[ ! -f /etc/os-release ]]; then
  echo "cloud_bootstrap: /etc/os-release is missing; Ubuntu 22.04 or 24.04 is required." >&2
  exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != ubuntu || ( "${VERSION_ID:-}" != "24.04" && "${VERSION_ID:-}" != "22.04" ) ]]; then
  echo "cloud_bootstrap: expected Ubuntu 22.04 or 24.04; found ${PRETTY_NAME:-unknown OS}." >&2
  exit 1
fi

if (( EUID == 0 )); then
  SUDO=()
else
  if ! command -v sudo >/dev/null 2>&1; then
    echo "cloud_bootstrap: sudo is required to install apt packages and prepare /data and /opt paths." >&2
    exit 1
  fi
  # Explicitly request/validate sudo before making host changes.
  sudo -v
  SUDO=(sudo)
fi

RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_GROUP="$(id -gn "${RUN_USER}")"

# Install missing system tools, FFmpeg/FFprobe, and audio/video shared libraries FIRST
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y --no-install-recommends \
  ca-certificates curl git unzip jq build-essential pkg-config \
  ffmpeg libsndfile1 sox libsox-fmt-all libgl1 libglib2.0-0 \
  fonts-dejavu-core fonts-liberation fonts-noto-core fonts-noto-cjk \
  python3 python3-venv python3-pip xz-utils tini

if (( WITH_DOCKER )); then
  if ! command -v docker >/dev/null 2>&1; then
    echo "cloud_bootstrap: installing Docker as explicitly requested (--with-docker)..."
    "${SUDO[@]}" apt-get install -y --no-install-recommends docker.io
  fi
fi

# Prepare and verify data and venv roots (resolves unreadable /data issues)
if [[ ! -d "${DATA_ROOT}" ]]; then
  "${SUDO[@]}" install -d -m 0775 -o "${RUN_USER}" -g "${RUN_GROUP}" "${DATA_ROOT}"
else
  "${SUDO[@]}" chmod a+rx "${DATA_ROOT}"
  if [[ "${RUN_USER}" != "root" ]]; then
    if ! [ -w "${DATA_ROOT}" ]; then
      "${SUDO[@]}" chown "${RUN_USER}:${RUN_GROUP}" "${DATA_ROOT}" || true
      "${SUDO[@]}" chmod 0775 "${DATA_ROOT}" || true
    fi
  fi
fi

# Own dedicated directories; preserve unrelated files below /data.
"${SUDO[@]}" install -d -m 0775 -o "${RUN_USER}" -g "${RUN_GROUP}" \
  "${DATA_ROOT}/models" "${DATA_ROOT}/cache" "${DATA_ROOT}/results" "${VENV_ROOT}" \
  "${DATA_ROOT}/run" "${DATA_ROOT}/logs" "${DATA_ROOT}/app-state" \
  "${DATA_ROOT}/uploads" "${DATA_ROOT}/work" "${DATA_ROOT}/outputs" "${DATA_ROOT}/checkpoints"
"${SUDO[@]}" install -d -m 0750 -o "${RUN_USER}" -g "${RUN_GROUP}" "${DATA_ROOT}/voices"

# Verify writability for current user; fail clearly on permission errors
PROBE_FILE="${DATA_ROOT}/.probe_write_$$"
if ! touch "${PROBE_FILE}" 2>/dev/null; then
  "${SUDO[@]}" chown -R "${RUN_USER}:${RUN_GROUP}" \
    "${DATA_ROOT}/models" "${DATA_ROOT}/cache" "${DATA_ROOT}/results" \
    "${VENV_ROOT}" "${DATA_ROOT}/run" "${DATA_ROOT}/logs" || true
  if ! touch "${PROBE_FILE}" 2>/dev/null; then
    echo "cloud_bootstrap: PERMISSION ERROR: DATA_ROOT (${DATA_ROOT}) is not writable by ${RUN_USER}." >&2
    echo "Ensure ${DATA_ROOT} is owned or writable by ${RUN_USER} (e.g. sudo chown -R ${RUN_USER}:${RUN_GROUP} ${DATA_ROOT})." >&2
    exit 1
  fi
fi
rm -f "${PROBE_FILE}"

# Ensure managed Python 3.12
if [[ "${VERSION_ID}" == "24.04" ]]; then
  "${SUDO[@]}" apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev
fi

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.12)"
else
  # Ubuntu 22.04 does not provide Python 3.12 in its standard apt repositories.
  TOOL_ENV="${DATA_ROOT}/cache/bootstrap-tools"
  "${SUDO[@]}" install -d -m 0775 -o "${RUN_USER}" -g "${RUN_GROUP}" "${DATA_ROOT}/cache"
  python3 -m venv "${TOOL_ENV}"
  "${TOOL_ENV}/bin/python" -m pip install "uv==0.9.6"
  "${TOOL_ENV}/bin/uv" python install 3.12.12
  PYTHON_BIN="$("${TOOL_ENV}/bin/uv" python find 3.12.12)"
fi
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

export PIP_CACHE_DIR="${DATA_ROOT}/cache/pip"
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export TORCH_HOME="${DATA_ROOT}/cache/torch"
export XDG_CACHE_HOME="${DATA_ROOT}/cache/xdg"
mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
  echo "cloud_bootstrap: project root does not contain pyproject.toml: ${PROJECT_ROOT}" >&2
  exit 1
fi

# Query profile list now that PYTHON_BIN is guaranteed
PLAN_ARGS=()
if [[ -n "${EXPLICIT_PROFILES}" ]]; then
  PLAN_ARGS+=(--profiles "${EXPLICIT_PROFILES}")
fi
PROFILE_LIST="$("${PYTHON_BIN}" "$PROJECT_ROOT/scripts/cloud_plan.py" --field profiles "${PLAN_ARGS[@]}")"
mapfile -t PROFILES <<< "$PROFILE_LIST"

resolve_profile_requirements() {
  local profile="$1"
  if [[ -f "${PROJECT_ROOT}/requirements/runtime-${profile}.txt" ]]; then
    printf '%s\n' "requirements/runtime-${profile}.txt"
  elif [[ -f "${PROJECT_ROOT}/requirements/bench-${profile}.txt" ]]; then
    printf '%s\n' "requirements/bench-${profile}.txt"
  elif [[ -f "${PROJECT_ROOT}/requirements/${profile}.txt" ]]; then
    printf '%s\n' "requirements/${profile}.txt"
  elif [[ -f "${PROJECT_ROOT}/requirements/${profile}.lock" ]]; then
    printf '%s\n' "requirements/${profile}.lock"
  else
    return 1
  fi
}

create_or_update_env() {
  local name="$1"
  local requirements_file="${2:-}"
  local env_path="${VENV_ROOT}/${name}"
  if [[ ! -x "${env_path}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${env_path}"
  fi
  local python="${env_path}/bin/python"
  local st_ver="${SETUPTOOLS_VERSION}"
  if [[ "${name}" == bandit ]]; then
    st_ver="69.5.1"
  fi
  "${python}" -m pip install --upgrade \
    "pip==${PIP_VERSION}" "setuptools==${st_ver}" "wheel==${WHEEL_VERSION}"
  "${python}" -m pip --version
  if [[ "${name}" == core ]]; then
    "${python}" -m pip install --requirement "${PROJECT_ROOT}/requirements.lock"
    "${python}" -m pip install --no-deps --no-build-isolation --editable "${PROJECT_ROOT}"
  else
    if [[ ! -f "${PROJECT_ROOT}/${requirements_file}" ]]; then
      echo "cloud_bootstrap: missing profile requirements: ${PROJECT_ROOT}/${requirements_file}" >&2
      return 1
    fi
    # Install torch only if required by this profile and not vision
    if grep -Eq '(^torch==|^torchaudio==|^torchvision==)' "${PROJECT_ROOT}/${requirements_file}"; then
      "${python}" -m pip install --index-url "${TORCH_INDEX}" \
        "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" "torchvision==0.23.0+cu128"
    fi
    "${python}" -m pip install --require-hashes --requirement "${PROJECT_ROOT}/${requirements_file}"
    "${python}" -m pip install --require-hashes --requirement "${PROJECT_ROOT}/requirements/worker-common.txt"
    local site_packages
    site_packages="$("${python}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    printf '%s\n' "${PROJECT_ROOT}/src" > "${site_packages}/autodub-project-src.pth"
  fi
  "${python}" -m pip check || {
    echo "cloud_bootstrap: INCOMPATIBLE DEPENDENCY ERROR in ${env_path}: pip check failed." >&2
    return 1
  }
  local import_modules=""
  case "${name}" in
    asr|punctuation) import_modules="torch,torchaudio,numpy,transformers,kaldi_native_fbank,soundfile" ;;
    separation) import_modules="torch,torchaudio,numpy,soundfile" ;;
    diarization) import_modules="torch,torchaudio,torchcodec,pyannote.audio" ;;
    indextts) import_modules="torch,torchaudio,librosa,audioread,soundfile,transformers,modelscope" ;;
    vision) import_modules="torch,torchvision,cv2,onnxruntime,rapidocr" ;;
    *) import_modules="" ;;
  esac
  if [[ -n "$import_modules" ]]; then
    "${python}" -c 'import importlib,sys; [importlib.import_module(name) for name in sys.argv[1].split(",")]' "$import_modules"
  fi

  if [[ "${name}" == vision ]]; then
    "${python}" -c 'import onnxruntime as ort; providers=ort.get_available_providers(); assert "CPUExecutionProvider" in providers, providers; print(f"verified ONNX Runtime CPU provider; providers={providers}")'
  elif [[ "${name}" != core ]]; then
    # Verify torch and CUDA if torch is installed in this profile
    if "${python}" -c 'import torch' >/dev/null 2>&1; then
      "${python}" -c 'import importlib.metadata as m, torch; ta=m.version("torchaudio") if "torchaudio" in [d.metadata["Name"] for d in m.distributions()] else "none"; assert (torch.__version__, torch.version.cuda)==("2.8.0+cu128", "12.8"), (torch.__version__, torch.version.cuda); print(f"verified torch={torch.__version__}, torchaudio={ta}, CUDA={torch.version.cuda}")'
    fi
    # Verify PyAV if installed (catches NumPy 2 / PyAV binary incompatibility)
    if "${python}" -c 'import av' >/dev/null 2>&1; then
      "${python}" -c 'import av; print(f"verified PyAV={av.__version__}")'
    fi
    # Verify audioread backend if installed (catches missing FFmpeg/FFprobe backends)
    if "${python}" -c 'import audioread' >/dev/null 2>&1; then
      "${python}" -c 'import audioread; backends=audioread.available_backends(); assert backends, "audioread has no available backend"; print(f"verified audioread backend: {[b.__name__ for b in backends]}")'
    fi
  fi
}

create_or_update_env core
for profile in "${PROFILES[@]}"; do
  if ! requirement="$(resolve_profile_requirements "${profile}")"; then
    echo "cloud_bootstrap: missing profile requirements for '${profile}': could not find requirements/runtime-${profile}.txt, requirements/bench-${profile}.txt, or requirements/${profile}.txt" >&2
    exit 1
  fi
  create_or_update_env "${profile}" "${requirement}"
done

echo "Cloud bootstrap complete. Model downloads are a separate, metered step."
echo "Environments: ${VENV_ROOT}/core and ${PROFILES[*]}"
echo "Models/cache/results: ${DATA_ROOT}/{models,cache,results}"
