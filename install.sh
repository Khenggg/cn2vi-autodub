#!/usr/bin/env bash
set -Eeuo pipefail

# CN2VI AutoDub root installation entrypoint for fresh Ubuntu 22.04 / 24.04 hosts.
# Forwards options directly to scripts/cloud_setup.sh.
# Supported options:
#   --dry-run          Preview install plan without modifying system
#   --prepare-only     Install OS packages, venvs, and frontend; skip model weights
#   --check-host       Run host hardware and OS preflight checks
#   --verify-startup   Verify Web UI background startup and /healthz endpoint
#   --profiles <list>  Explicit comma-separated profile choices
#   --profile <name>   Explicit single profile choice
#   --with-docker      Install Docker/container-toolkit only if needed for deployment

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SETUP_SCRIPT="$SCRIPT_DIR/scripts/cloud_setup.sh"

if [[ ! -f "$SETUP_SCRIPT" ]]; then
  echo "install.sh: setup script not found at ${SETUP_SCRIPT}" >&2
  exit 1
fi

exec bash "$SETUP_SCRIPT" "$@"
