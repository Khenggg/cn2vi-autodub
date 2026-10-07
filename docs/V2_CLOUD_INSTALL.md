# Cài CN2VI V2 trên cloud mới

Ubuntu 22.04/24.04 x86_64, NVIDIA RTX 3060 12GB hoặc GPU tương đương, driver đang hoạt động, RAM ít nhất 16 GB và ít nhất 100 GB trống. Mỗi lần thuê máy mới chạy lại từ repo; không restore nguyên môi trường/model cũ từ Drive. Nhà cung cấp phải expose GPU và NVENC cho máy/container. Script không thay NVIDIA driver.

## 1. Clone và cài

Lệnh cho nhánh V2 đã công bố (khi merge vào main có thể clone main):

```bash
if ! command -v git >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update && apt-get install -y git ca-certificates
  else
    sudo apt-get update && sudo apt-get install -y git ca-certificates
  fi
fi
git clone --branch codex/v2-smart-voiceover --single-branch https://github.com/Khenggg/cn2vi-autodub.git
cd cn2vi-autodub
bash scripts/cloud_setup.sh --verify-startup
```

Script tự cài Python 3.12, Node/npm cho frontend, Git/curl/jq/unzip, FFmpeg/FFprobe, fonts CJK, libsndfile/SoX/espeak, bộ công cụ build, các môi trường core/asr/tts/vision, tải đúng revision và kiểm tra hash các assets. `setup_new_server.sh` là alias của cùng installer. Không cần Docker; chỉ cài Docker/toolkit khi yêu cầu `--with-docker`.

Production tải **7 assets, khoảng 5.64 GiB** (chưa tính source Git/dependencies/cache). Các assets: FireRed code, AED, punctuation; VieNeu Turbo, MOSS Torch, reference Ngọc Huyền; PP-OCRv6 Medium. Không tải Bandit/Kim/RoFormer/ProPainter/IndexTTS2/pyannote. Không cần token pyannote Community-1 cho V2.

`--prepare-only` chỉ chuẩn bị môi trường, chưa tải weights và chưa chứng minh model sẵn sàng. `--dry-run` chỉ in kế hoạch. Installer có thể chạy lại; không xóa media, model cache hoặc data hiện có. Lỗi dependency/provider/NVENC dừng cài với journal, không fallback CPU để báo GPU-ready.

## 2. Key DeepSeek và bật xử lý

Web được mở trước ở chế độ upload/review. Tạo key riêng trên cloud; không commit vào Git:

```bash
umask 077
mkdir -p /data/run
read -r -s -p 'DeepSeek API key: ' cn2vi_api_key
printf '\n'
printf 'DEEPSEEK_API_KEY=%s\n' "$cn2vi_api_key" > /data/run/translation.env
unset cn2vi_api_key
chmod 600 /data/run/translation.env
PIPELINE_GENERATION=v2 ENABLE_PIPELINE=true bash scripts/start_web.sh restart
```

Cũng hỗ trợ `DEEPSEEK_API_KEY` qua environment hoặc `DEEPSEEK_KEY_FILE` trỏ đến file riêng (mặc định `docs/API.txt.txt`). Nội dung file là một key `sk-...` hoặc assignment `DEEPSEEK_API_KEY=...`; không execute file như shell. Key không nằm trong snapshot/report. Giữ quyền 0600.

Ngọc Huyền được cài tự động ở `/data/voices/ngoc-huyen.wav` và kiểm tra hash. Nếu file hiện có khác, installer dừng để giữ giọng hiện tại; không âm thầm thay reference. Không cần chọn lại giọng mỗi lần thuê máy.

## 3. Mở web từ Windows

Chạy trong terminal Windows, dùng SSH host/user/port của phiên cloud hiện tại:

```powershell
ssh -N -L 18080:127.0.0.1:8080 -p <PORT> <USER>@<HOST>
```

Giữ terminal tunnel mở, vào `http://127.0.0.1:18080`. Admin token nằm tại `/data/run/admin_token.txt`; đọc ở terminal cloud để đăng nhập. Upload video, chọn burn/replace/off và bắt đầu. Endpoint `/healthz` kiểm tra web, không chứng minh inference thành công.

## 4. Bằng chứng và giới hạn

Journal `/data/results/cloud-setup-*.log` phải có PASS cho mọi profile. Vision check dựng ba session OCR thực, yêu cầu provider đầu là CUDA, chạy một crop và thử encode NVENC. ASR/TTS kiểm tra API/import, không thay thế chạy mẫu thật. Native model checksum không bị thay đổi bởi Python bytecode.

Run V2 lưu tại `/data/work/<episode-id>/v2-runs`, report/exports đăng ký trong web. Báo cáo có thời gian từng stage, candidate duration, số câu cần dub/số clip sinh/số câu thực sự mix, lỗi donor và `quality_and_sla_verified=false`. Có clip TTS không đồng nghĩa đã mix đủ câu.

VAD không ép phim chỉ có 25–28 phút thoại. Các ngưỡng dialogue/OST và OCR role là heuristic chưa hiệu chuẩn. Không có subtitle vẫn có thể DUB. Timing không có word alignment dùng cửa sổ VAD thật, không giả mốc từng chữ. Không xác định danh tính nhân vật từ speaker label chưa được xác nhận.

Temporal restoration cần donor sạch cùng cảnh và registration đạt kiểm tra. Không có donor/occlusion thì giữ pixel gốc và báo PARTIAL. Đường restoration hiện yêu cầu video CFR có timeline bắt đầu 0; VFR/offset không hỗ trợ được báo lỗi, không tự đổi thời gian. Render vẫn decode/registration bằng CPU và copy frame vào NVENC; chưa có zero-copy/NVDEC.

Mục tiêu 60 phút → 600 giây **chưa được xác nhận**. Tải/cài/upload báo riêng; thời gian job bao gồm load model, decode, API, synthesis, mix và encode. Cần kiểm tra một clip có thoại/OST/cười khóc trước khi chạy phim dài.
