#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VIDEO_PATH="${1:-/data/uploads/3b998daa3f9c400db25035bfb1c6707b/66a0a891f9884195a5782eb39ed4114f/source.mp4}"

echo "========================================================"
echo "    CN2VI AutoDub - Phase 0 GPU Benchmark Runner        "
echo "========================================================"
echo "Target video: ${VIDEO_PATH}"
echo "Running benchmark using RTX 5060 Ti (16GB VRAM)..."

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
/opt/autodub/venvs/core/bin/python "${SCRIPT_DIR}/run_gpu_benchmark.py" --video "${VIDEO_PATH}" "$@"
