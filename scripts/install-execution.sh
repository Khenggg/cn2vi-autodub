#!/usr/bin/env bash
# Run on the authorized cloud only. Does not download weights or start inference.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ "${1:-}" != "--confirm-execution-machine" ]]; then
  echo 'Use only on the execution machine: bash scripts/install-execution.sh --confirm-execution-machine' >&2
  exit 2
fi
if ! command -v python3 >/dev/null || ! command -v ffprobe >/dev/null; then
  if [[ $(id -u) -eq 0 ]]; then elevation=(); else elevation=(sudo); fi
  "${elevation[@]}" apt-get update
  "${elevation[@]}" apt-get install -y python3 python3-venv ffmpeg
fi
python3 -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12+ is required"'
python3 -m venv .venv-vnle
.venv-vnle/bin/python -m pip install --upgrade pip
.venv-vnle/bin/python -m pip install -e '.[analysis,cuda]'
echo 'Environment installed. Next provision the model manifest explicitly; see docs/PROTOTYPE.md.'
