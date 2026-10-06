#!/usr/bin/env bash
set -Eeuo pipefail

# Restore only model assets and virtualenv files from the protected Drive backup.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DATA_ROOT="${AUTODUB_DATA_ROOT:-/data}"
VENV_ROOT="${AUTODUB_VENV_ROOT:-/opt/autodub/venvs}"
BACKUP_REMOTE="gdrive:Autodub_Backup/autodub-backup.tar"

fail() { printf 'cloud_restore: %s\n' "$*" >&2; exit 1; }
as_root() {
  if (( EUID == 0 )); then "$@"
  elif command -v sudo >/dev/null 2>&1; then sudo "$@"
  else fail 'root access is required to write the configured restore destinations.'
  fi
}

printf '\nCN2VI AutoDub - Safe cloud restore\n'

command -v rclone >/dev/null 2>&1 || fail 'rclone is required; install it before restoring.'
command -v python3 >/dev/null 2>&1 || fail 'python3 is required for archive validation.'
[[ -f "$PROJECT_ROOT/scripts/cloud_restore_archive.py" ]] || fail 'archive validator is missing.'

# Use a host-owned Python 3.12 runtime to repair only the known stale venv link.
PYTHON312="${AUTODUB_PYTHON312:-}"
if [[ -z "$PYTHON312" ]] && command -v python3.12 >/dev/null 2>&1; then
  PYTHON312="$(command -v python3.12)"
fi
if [[ -z "$PYTHON312" ]] && command -v uv >/dev/null 2>&1; then
  PYTHON312="$(uv python find 3.12 2>/dev/null || true)"
fi
[[ -n "$PYTHON312" && -x "$PYTHON312" ]] || fail 'provide a trusted Python 3.12 executable in AUTODUB_PYTHON312 or install it with uv.'
PYTHON312="$(readlink -f -- "$PYTHON312")"
"$PYTHON312" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' \
  || fail 'the trusted interpreter must be Python 3.12.'

TMP_ROOT="${TMPDIR:-/tmp}"
RESTORE_TMP="$(mktemp -d "$TMP_ROOT/autodub-restore.XXXXXX")"
trap 'rm -rf -- "$RESTORE_TMP"' EXIT
ARCHIVE="$RESTORE_TMP/autodub-backup.tar"
REMOTE_JSON="$RESTORE_TMP/remote.json"

# Keep using the already-configured, protected rclone profile. Do not rewrite it.
printf '[1/4] Reading exact remote file metadata: %s\n' "$BACKUP_REMOTE"
rclone lsjson --stat "$BACKUP_REMOTE" >"$REMOTE_JSON" \
  || fail 'rclone could not read metadata for the configured backup file.'
REMOTE_SIZE="$(python3 - "$REMOTE_JSON" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
size = value.get("Size") if isinstance(value, dict) else None
if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
    raise SystemExit("remote metadata has no valid positive Size")
print(size)
PY
)" || fail 'remote metadata did not contain a valid archive size.'

printf '[2/4] Downloading the verified remote archive.\n'
rclone copyto "$BACKUP_REMOTE" "$ARCHIVE" --progress \
  || fail 'rclone download failed.'
[[ -s "$ARCHIVE" ]] || fail 'downloaded archive is empty.'
LOCAL_SIZE="$(stat -c '%s' -- "$ARCHIVE")"
[[ "$LOCAL_SIZE" == "$REMOTE_SIZE" ]] || fail 'download size differs from remote metadata.'
printf 'remote=%s\nsize=%s\nsha256=%s\n' "$BACKUP_REMOTE" "$REMOTE_SIZE" \
  "$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)" >"$RESTORE_TMP/.download-verified"

printf '[3/4] Validating and staging only data/models and opt/autodub/venvs.\n'
STAGE="$RESTORE_TMP/extracted"
python3 "$PROJECT_ROOT/scripts/cloud_restore_archive.py" "$ARCHIVE" "$STAGE" \
  --python312 "$PYTHON312" || fail 'archive validation or staging failed; nothing was promoted.'

printf '[4/4] Promoting missing model and venv entries without replacing existing data.\n'
as_root mkdir -p -- "$DATA_ROOT/models" "$VENV_ROOT"
shopt -s nullglob dotglob
for staged_root in "$STAGE/data/models" "$STAGE/opt/autodub/venvs"; do
  [[ -d "$staged_root" ]] || continue
  if [[ "$staged_root" == "$STAGE/data/models" ]]; then target_root="$DATA_ROOT/models"
  else target_root="$VENV_ROOT"
  fi
  for item in "$staged_root"/*; do
    [[ -e "$item" || -L "$item" ]] || continue
    target="$target_root/$(basename -- "$item")"
    if [[ -e "$target" || -L "$target" ]]; then
      printf 'Preserved existing path: %s\n' "$target"
      continue
    fi
    as_root mv -n -- "$item" "$target_root/"
    if [[ -e "$item" || -L "$item" ]]; then
      printf 'Preserved concurrent destination conflict: %s\n' "$target"
    else
      printf 'Restored: %s\n' "$target"
    fi
  done
done

cat <<EOF

Restore data was staged with remote size and download checks, and existing model/venv entries were preserved.
Virtualenv portability and inference readiness are NOT verified by this restore. Before use, check each restored environment's ownership and interpreter target, run its Python version and package consistency checks, verify required model manifests/dependencies, and validate GPU/CUDA plus FFmpeg on this host. Rebuild an environment if those checks fail; this script does not alter drivers, OS packages, or reboot the host.

Web startup command:
  bash "$PROJECT_ROOT/scripts/start_web.sh" start
EOF
