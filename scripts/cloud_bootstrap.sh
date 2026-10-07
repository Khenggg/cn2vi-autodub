#!/usr/bin/env bash
set -Eeuo pipefail

# Safe to rerun. Installs OS tools and creates/updates environments; never shuts down
# the host, removes existing data, or downloads model weights.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${AUTODUB_DATA_ROOT:-/data}"
VENV_ROOT="${AUTODUB_VENV_ROOT:-/opt/autodub/venvs}"
PLAN_PYTHON="$(command -v python3 || true)"
[[ -n "$PLAN_PYTHON" ]] || { echo 'Install python3 and git before cloning/setup.' >&2; exit 1; }
PROFILE_LIST="$("$PLAN_PYTHON" "$PROJECT_ROOT/scripts/cloud_plan.py" --field profiles)"
mapfile -t PROFILES <<< "$PROFILE_LIST"
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
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y --no-install-recommends \
  ca-certificates build-essential libsndfile1 ffmpeg git fonts-dejavu-core fonts-liberation fonts-noto-core fonts-noto-cjk \
  python3 python3-venv python3-pip curl xz-utils tini

if [[ "${VERSION_ID}" == "24.04" ]]; then
  "${SUDO[@]}" apt-get install -y --no-install-recommends python3.12 python3.12-venv
fi
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.12)"
else
  # Ubuntu 22.04 does not provide Python 3.12 in its standard apt repositories.
  TOOL_ENV="${DATA_ROOT}/cache/bootstrap-tools"
  "${SUDO[@]}" install -d -o "${RUN_USER}" -g "${RUN_GROUP}" "${DATA_ROOT}/cache"
  python3 -m venv "${TOOL_ENV}"
  "${TOOL_ENV}/bin/python" -m pip install "uv==0.9.6"
  "${TOOL_ENV}/bin/uv" python install 3.12.12
  PYTHON_BIN="$("${TOOL_ENV}/bin/uv" python find 3.12.12)"
fi
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# Own only the dedicated directories; preserve unrelated files below /data.
"${SUDO[@]}" install -d -o "${RUN_USER}" -g "${RUN_GROUP}" \
  "${DATA_ROOT}/models" "${DATA_ROOT}/cache" "${DATA_ROOT}/results" "${VENV_ROOT}"

export PIP_CACHE_DIR="${DATA_ROOT}/cache/pip"
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export TORCH_HOME="${DATA_ROOT}/cache/torch"
export XDG_CACHE_HOME="${DATA_ROOT}/cache/xdg"
mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
  echo "cloud_bootstrap: project root does not contain pyproject.toml: ${PROJECT_ROOT}" >&2
  exit 1
fi

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
    if [[ "${name}" != vision ]]; then
      "${python}" -m pip install --index-url "${TORCH_INDEX}" \
        "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" "torchvision==0.23.0+cu128"
    fi
    "${python}" -m pip install --require-hashes --requirement "${PROJECT_ROOT}/${requirements_file}"
    "${python}" -m pip install --require-hashes --requirement "${PROJECT_ROOT}/requirements/worker-common.txt"
    local site_packages
    site_packages="$("${python}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    printf '%s\n' "${PROJECT_ROOT}/src" > "${site_packages}/autodub-project-src.pth"
  fi
  "${python}" -m pip check
  if [[ "${name}" == vision ]]; then
    "${python}" -c 'import onnxruntime as ort; providers=ort.get_available_providers(); assert "CPUExecutionProvider" in providers, providers; print(f"verified ONNX Runtime CPU provider; providers={providers}")'
  elif [[ "${name}" != core ]]; then
    "${python}" -c 'import importlib.metadata as m, torch; ta=m.version("torchaudio"); assert (torch.__version__,ta,torch.version.cuda)==("2.8.0+cu128","2.8.0+cu128","12.8"), (torch.__version__,ta,torch.version.cuda); print(f"verified torch={torch.__version__}, torchaudio={ta}, CUDA={torch.version.cuda}")'
  fi
}

create_or_update_env core
for profile in "${PROFILES[@]}"; do
  case "$profile" in
    asr|separation) requirement="requirements/runtime-${profile}.txt" ;;
    *) requirement="requirements/bench-${profile}.txt" ;;
  esac
  create_or_update_env "${profile}" "$requirement"
done

echo "Cloud bootstrap complete. Model downloads are a separate, metered step."
echo "Environments: ${VENV_ROOT}/core and ${PROFILES[*]}"
echo "Models/cache/results: ${DATA_ROOT}/{models,cache,results}"
