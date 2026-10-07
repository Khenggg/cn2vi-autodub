#!/usr/bin/env bash
set -Eeuo pipefail
# Compatibility entrypoint. All installation rules live in cloud_setup.sh.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$SCRIPT_DIR/cloud_setup.sh" "$@"
