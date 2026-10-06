#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# CN2VI AutoDub - 1-Click Deployment Script for RTX 3060 12GB (Ubuntu)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${AUTODUB_DATA_ROOT:-/data}"
VENV_ROOT="${AUTODUB_VENV_ROOT:-/opt/autodub/venvs}"

echo "=========================================================="
echo "  CN2VI AutoDub: Deploying on RTX 3060 12GB Ubuntu Server"
echo "=========================================================="

# 1. GPU Check
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] Detected NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[WARNING] nvidia-smi not found. Ensure NVIDIA CUDA drivers are installed."
fi

# 2. Bootstrap system dependencies & Python venvs
echo "[INFO] Running cloud bootstrap..."
bash "${SCRIPT_DIR}/cloud_bootstrap.sh"

# 3. Install SOTA Audio Separator (Mel-RoFormer / Kim_Vocal_2 support)
echo "[INFO] Installing RoFormer audio-separator in separation venv..."
if [[ -x "${VENV_ROOT}/bandit/bin/pip" ]]; then
    "${VENV_ROOT}/bandit/bin/pip" install --no-cache-dir "audio-separator[gpu]" || \
    "${VENV_ROOT}/bandit/bin/pip" install --no-cache-dir "audio-separator" || true
fi

# 4. Model Assets Download (ASR, Forced Aligner, VieNeu-TTS)
# NOTE: Translation model (Qwen 9.3GB) is SKIPPED because DeepSeek-V3 API is used!
echo "[INFO] Fetching required models (ASR, Aligner, TTS)..."
CORE_PYTHON="${VENV_ROOT}/core/bin/python"
if [[ -x "${CORE_PYTHON}" ]]; then
    "${CORE_PYTHON}" -m autodub.model_assets download --profile asr || true
    "${CORE_PYTHON}" -m autodub.model_assets download --profile tts || true
    "${CORE_PYTHON}" -m autodub.model_assets download --profile bandit || true
fi

# 5. Check DeepSeek API key configuration
if [[ -f "${REPO_DIR}/docs/API.txt.txt" ]]; then
    echo "[INFO] Found DeepSeek API key in docs/API.txt.txt. Translation will use DeepSeek-V3 API."
elif [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[INFO] DEEPSEEK_API_KEY environment variable is set."
else
    echo "[WARNING] No DeepSeek API key found. Place your key in docs/API.txt.txt or set DEEPSEEK_API_KEY."
fi

# 6. Start the Web UI
echo "[INFO] Starting AutoDub Web UI..."
bash "${SCRIPT_DIR}/start_web.sh" restart

echo "=========================================================="
echo "  Deployment Complete! System is ready to dub videos."
echo "=========================================================="
