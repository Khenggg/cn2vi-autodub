#!/usr/bin/env bash
set -Eeuo pipefail

# CN2VI AutoDub - Fast Cloud Restore Script
# Tự động tải môi trường và 10 model từ Google Drive về máy Cloud mới trong 1-2 phút.

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DATA_ROOT="${AUTODUB_DATA_ROOT:-/data}"
VENV_ROOT="${AUTODUB_VENV_ROOT:-/opt/autodub/venvs}"
BACKUP_REMOTE="${BACKUP_REMOTE:-gdrive:Autodub_Backup/autodub-backup.tar}"
BACKUP_LOCAL="$DATA_ROOT/autodub-backup.tar"

printf '\n============================================================\n'
printf '   CN2VI AutoDub - KHÔI PHỤC TỰ ĐỘNG TỪ GOOGLE DRIVE       \n'
printf '============================================================\n\n'

# 1. Kiểm tra quyền sudo
if ! command -v sudo >/dev/null 2>&1; then
  printf 'cloud_restore: sudo là bắt buộc để chuẩn bị thư mục hệ thống.\n' >&2
  exit 1
fi

# 2. Cài đặt các công cụ hệ thống tối thiểu nếu thiếu
printf '[1/5] Kiểm tra và cài đặt công cụ cần thiết (rclone, ffmpeg, python3.12)...\n'
MISSING_PKGS=()
for cmd in rclone ffmpeg git python3.12; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    MISSING_PKGS+=("$cmd")
  fi
done

if ((${#MISSING_PKGS[@]} > 0)); then
  printf 'Đang cài đặt các gói còn thiếu: %s\n' "${MISSING_PKGS[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y -qq rclone ffmpeg git python3.12 python3.12-venv
fi

# 3. Chuẩn bị thư mục /data và /opt
printf '[2/5] Chuẩn bị thư mục dữ liệu %s và venv %s...\n' "$DATA_ROOT" "$VENV_ROOT"
sudo mkdir -p "$DATA_ROOT" "$VENV_ROOT"
sudo chown -R "$USER:$USER" "$DATA_ROOT" "$VENV_ROOT"

# 4. Kiểm tra cấu hình rclone
printf '[3/5] Kiểm tra kết nối Google Drive (rclone)...\n'
if ! rclone listremotes | grep -q '^gdrive:'; then
  printf '\n[CẢNH BÁO] Chưa tìm thấy cấu hình "gdrive" trong rclone!\n'
  printf 'Vui lòng chạy "rclone config" để kết nối Google Drive trước khi khôi phục.\n'
  exit 1
fi

# 5. Tải file backup từ Google Drive nếu chưa có
printf '[4/5] Tải file backup từ Google Drive (%s)...\n' "$BACKUP_REMOTE"
if [[ -f "$BACKUP_LOCAL" ]]; then
  printf 'Tìm thấy file backup có sẵn tại %s, bỏ qua bước tải.\n' "$BACKUP_LOCAL"
else
  rclone copy "$BACKUP_REMOTE" "$DATA_ROOT" --progress
fi

# 6. Giải nén vào hệ thống
printf '[5/5] Giải nén toàn bộ 10 Model AI và 5 Môi trường Python...\n'
sudo tar -xf "$BACKUP_LOCAL" -C /
sudo chown -R "$USER:$USER" "$DATA_ROOT" "$VENV_ROOT"

# 7. Tự động vá lỗi môi trường (Self-Healing - Đảm bảo hoạt động 100% không cần can thiệp)
printf '\n[Tự động vá lỗi] Kiểm tra môi trường bandit & liên kết mã nguồn...\n'
if [[ -x "$VENV_ROOT/bandit/bin/pip" ]]; then
  # Đảm bảo setuptools < 70 để giữ pkg_resources tương thích với thư viện Bandit
  "$VENV_ROOT/bandit/bin/pip" install -q "setuptools<70" || true
fi

# Đảm bảo mã nguồn mới nhất được liên kết vào tất cả các venv
for venv in "$VENV_ROOT"/*; do
  if [[ -d "$venv" && -x "$venv/bin/python" ]]; then
    site_packages="$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
    if [[ -n "$site_packages" && -d "$site_packages" ]]; then
      printf '%s\n' "$PROJECT_ROOT/src" > "$site_packages/autodub-project-src.pth" || true
    fi
  fi
done

# 8. Kiểm tra frontend dist nếu cần
if [[ ! -f "$PROJECT_ROOT/frontend/dist/index.html" && -f "$PROJECT_ROOT/scripts/cloud_frontend.sh" ]]; then
  printf 'Frontend chưa được build, đang tự động build giao diện web...\n'
  bash "$PROJECT_ROOT/scripts/cloud_frontend.sh"
fi

printf '\n============================================================\n'
printf '   KHÔI PHỤC HOÀN TẤT 100%! HỆ THỐNG ĐÃ SẴN SÀNG!          \n'
printf '============================================================\n\n'
printf 'Khởi chạy Web chạy ngầm daemon (khuyên dùng):\n'
printf '  bash "%s/scripts/start_web.sh" start\n\n' "$PROJECT_ROOT"
printf 'Hoặc khởi chạy trực tiếp:\n'
printf '  cd "%s"\n' "$PROJECT_ROOT"
printf '  DATA_DIR=%s \\\n' "$DATA_ROOT"
printf '  %s/core/bin/python -m uvicorn \\\n' "$VENV_ROOT"
printf '    autodub.main:create_app --factory \\\n'
printf '    --host 127.0.0.1 --port 8080 --workers 1\n\n'
