#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

# Template expanded by autodub.cloud_package; the .run contains code, frontend and checksums.
SOURCE_COMMIT='__SOURCE_COMMIT__'
PAYLOAD_SHA256='__PAYLOAD_SHA256__'
PAYLOAD_LINE=__PAYLOAD_LINE__
DATA_ROOT="${AUTODUB_DATA_ROOT:-/data}"
export AUTODUB_DATA_ROOT="${DATA_ROOT}"
export AUTODUB_VENV_ROOT="${AUTODUB_VENV_ROOT:-/opt/autodub/venvs}"
export PROJECT_ROOT="${AUTODUB_PROJECT_ROOT:-${DATA_ROOT}/autodub/project}"

if [[ "${1:-}" == --dry-run && $# == 1 ]]; then
  echo "Cloud installer plan: source ${SOURCE_COMMIT}"
  echo "1. Check Ubuntu 24.04 x86_64; install Git, certificates and Python 3.12."
  echo "2. Verify embedded payload; extract code and built frontend into ${PROJECT_ROOT}."
  echo "3. Install isolated environments, fetch/verify model assets, and run cloud preflight."
  echo "Data: ${DATA_ROOT}; environments: ${AUTODUB_VENV_ROOT}. No changes made."
  exit 0
fi
if (( $# )); then
  echo "Usage: bash cn2vi-cloud-setup.run [--dry-run]" >&2
  exit 2
fi
if [[ ! -r /etc/os-release ]]; then
  echo "Ubuntu 24.04 is required." >&2; exit 2
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04 || "$(uname -m)" != x86_64 ]]; then
  echo "This installer requires Ubuntu 24.04 x86_64." >&2; exit 2
fi
SETUP_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/autodub-unpack.XXXXXXXX")"
trap 'if [[ -d "${SETUP_TEMP}" && "$(basename -- "${SETUP_TEMP}")" == autodub-unpack.* ]]; then rm -rf -- "${SETUP_TEMP}"; fi' EXIT
tail -n +"${PAYLOAD_LINE}" "$0" > "${SETUP_TEMP}/payload.tar.gz"
printf '%s  %s\n' "${PAYLOAD_SHA256}" "${SETUP_TEMP}/payload.tar.gz" | sha256sum --check --status
# Ubuntu images with stock Python/Git can reject package/checkout conflicts before apt.
if command -v python3 >/dev/null && command -v git >/dev/null; then
  python3 - --archive "${SETUP_TEMP}/payload.tar.gz" --sha256 "${PAYLOAD_SHA256}" \
    --source-commit "${SOURCE_COMMIT}" --staging-dir "${SETUP_TEMP}/validation" \
    --project-root "${PROJECT_ROOT}" --validate-only <<'AUTODUB_VALIDATION_PYTHON'
__INSTALLER_PYTHON__
AUTODUB_VALIDATION_PYTHON
fi
if (( EUID == 0 )); then
  SUDO=()
else
  command -v sudo >/dev/null || { echo "Root or sudo access is required." >&2; exit 2; }
  sudo -v
  SUDO=(sudo)
fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y --no-install-recommends ca-certificates git python3.12 python3.12-venv
python3.12 - --archive "${SETUP_TEMP}/payload.tar.gz" --sha256 "${PAYLOAD_SHA256}" \
  --source-commit "${SOURCE_COMMIT}" --staging-dir "${SETUP_TEMP}/files" \
  --project-root "${PROJECT_ROOT}" <<'AUTODUB_INSTALLER_PYTHON'
__INSTALLER_PYTHON__
AUTODUB_INSTALLER_PYTHON
bash "${PROJECT_ROOT}/scripts/cloud_setup.sh"
exit $?
# AUTODUB_EMBEDDED_PAYLOAD
