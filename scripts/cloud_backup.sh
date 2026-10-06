#!/usr/bin/env bash
set -Eeuo pipefail

# CN2VI AutoDub - Cloud Backup Script
# Nén toàn bộ 10 Model AI và 5 Virtualenvs đẩy lên Google Drive.

DATA_ROOT="${AUTODUB_DATA_ROOT:-/data}"
VENV_ROOT="${AUTODUB_VENV_ROOT:-/opt/autodub/venvs}"
BACKUP_LOCAL="$DATA_ROOT/autodub-backup.tar"
BACKUP_REMOTE="${BACKUP_REMOTE:-gdrive:Autodub_Backup/}"

printf '\n============================================================\n'
printf '   CN2VI AutoDub - SAO LƯU TOÀN BỘ LÊN GOOGLE DRIVE        \n'
printf '============================================================\n\n'

# 1. Kiểm tra tồn tại models và venvs
if [[ ! -d "$DATA_ROOT/models" || ! -d "$VENV_ROOT" ]]; then
  printf 'cloud_backup: Không tìm thấy thư mục models (%s/models) hoặc venvs (%s)!\n' "$DATA_ROOT" "$VENV_ROOT" >&2
  exit 1
fi

# 2. Kiểm tra rclone
if ! command -v rclone >/dev/null 2>&1; then
  printf 'cloud_backup: rclone chưa được cài đặt. Vui lòng cài rclone trước.\n' >&2
  exit 1
fi

if ! rclone listremotes | grep -q '^gdrive:'; then
  printf 'cloud_backup: Chưa cấu hình "gdrive" trong rclone. Vui lòng chạy "rclone config" trước.\n' >&2
  exit 1
fi

# 3. Nén dữ liệu
printf '[1/2] Đang nén toàn bộ 10 Model AI và 5 Môi trường Python vào %s...\n' "$BACKUP_LOCAL"
sudo rm -f "$BACKUP_LOCAL"
sudo tar -cf "$BACKUP_LOCAL" "$DATA_ROOT/models" "$VENV_ROOT"
sudo chown "$USER:$USER" "$BACKUP_LOCAL"

printf 'Đã nén xong! Dung lượng file: %s\n\n' "$(ls -lh "$BACKUP_LOCAL" | awk '{print $5}')"

# 4. Upload lên Google Drive
printf '[2/2] Đang tải file lên Google Drive (%s)...\n' "$BACKUP_REMOTE"
rclone copy "$BACKUP_LOCAL" "$BACKUP_REMOTE" --progress

printf '\n============================================================\n'
printf '   SAO LƯU THÀNH CÔNG LÊN GOOGLE DRIVE!                    \n'
printf '============================================================\n\n'
