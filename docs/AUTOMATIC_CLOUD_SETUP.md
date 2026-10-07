# Cài cloud từ repo đang phát triển

Luồng mặc định clone GitHub, đọc `config/cloud-runtime.json`, dựng virtualenv từ dependency lock, tải đúng model trong `benchmarks/models.lock.json` rồi xác minh checksum. Không tự tải backup Drive, không phục hồi venv của máy cũ và không tự chạy video.

## Lấy đúng nhánh

Bản sửa đang ở nhánh `codex/pipeline-hardening` trong PR #1, chưa merge vào main. Trên máy Ubuntu mới có GPU NVIDIA được nhà cung cấp cấp sẵn:

```bash
# Chạy với root; nếu là user thường, thêm sudo trước apt-get.
apt-get update
apt-get install -y --no-install-recommends ca-certificates git python3
git clone --branch codex/pipeline-hardening https://github.com/Khenggg/cn2vi-autodub.git
cd cn2vi-autodub
git rev-parse HEAD
bash scripts/cloud_setup.sh --dry-run
bash scripts/cloud_setup.sh --check-host
```

Host hỗ trợ Ubuntu 22.04/24.04 x86_64, NVIDIA driver hoạt động, GPU tổng VRAM >=11,500 MiB, RAM >=15 GB thập phân và disk trống >=100 GB. Đây là ngưỡng chuẩn bị môi trường cho RTX 3060 12 GB; hiệu năng, chất lượng và mức VRAM inference chưa xác nhận trên bộ model mới. Python 3.12 và Node/npm pin được setup tự cài. Không cần Docker để chạy luồng virtualenv này; setup không cài/nâng cấp driver hay reboot.

## Cài đặt

Chỉ cài công cụ, môi trường và frontend trước, chưa tải model weights:

```bash
bash scripts/cloud_setup.sh --prepare-only
```

Cài đầy đủ từ repo, không dùng Drive:

```bash
bash scripts/cloud_setup.sh
```

`setup_new_server.sh` là alias của cùng luồng, không còn quy tắc tải model riêng. Có thể đổi nơi lưu bằng `AUTODUB_DATA_ROOT=/data AUTODUB_VENV_ROOT=/opt/autodub/venvs`. Journal có mốc bắt đầu/kết thúc mỗi bước ở `/data/results/cloud-setup-*.log`. Lỗi dừng setup và ghi rõ bước lỗi; không báo thành công giả. Chạy lại sẽ kiểm tra/reuse model đúng checksum và dùng cache package.

## Bộ model được chọn

| Bước | Bộ cài mặc định |
|---|---|
| ASR | Faster-Whisper large-v3-turbo, tự phát hiện ngôn ngữ |
| Căn thời gian | WhisperX + weights căn tiếng Anh/Trung được pin |
| Tách âm | Kim_Vocal_2 (MDX-Net), package audio-separator trong venv `separation` |
| TTS | VieNeu v3 Turbo + MOSS codec |
| Phụ đề | RapidOCR + LaMa |

Qwen ASR/Aligner, BandIt và ProPainter vẫn có entries benchmark cũ trong lock nhưng không thuộc danh sách cài mặc định. Các ngôn ngữ ngoài Anh/Trung cần thêm weights `alignment-<language>` đã pin vào lock và plan; nếu thiếu, pipeline giữ transcript để review, không tải model trôi nổi trong lúc inference và không giả lập word timestamps.

ASR mới và bộ tách âm dùng môi trường riêng vì cần NumPy 2; TTS/vision vẫn giữ dependency của chúng. Cấu hình dịch DeepSeek đang có trong code được giữ; không gọi API trong setup. Dịch local và lựa chọn provider là bước cấu hình riêng, không thể coi ASR mới là model dịch.

## Xác minh rồi khởi động

`ENVIRONMENT_READY` chỉ xác nhận môi trường; `ASSETS_VERIFIED` xác nhận thêm model files. Cả hai không chứng minh model inference hoặc phim thật đã qua test. Sau khi thuê cloud, cần chạy tests và smoke model ngắn trước khi dùng phim dài. Không tự bật dịch vụ sau setup. Launcher mặc định tắt pipeline để chuẩn bị/review; chỉ đặt `ENABLE_PIPELINE=true` sau khi smoke model trên cloud đã qua:

```bash
AUTODUB_DATA_ROOT=/data AUTODUB_VENV_ROOT=/opt/autodub/venvs bash scripts/start_web.sh start
```

Web bind `127.0.0.1:8080`; truy cập qua SSH tunnel. Admin token ở `/data/run/admin_token.txt` với quyền 0600; không đưa token/key vào Git. File `.env` không tự được launcher đọc: export biến từ cấu hình riêng đã bảo vệ trước khi start. `MODELS_DIR` và `VENVS_DIR` được launcher suy ra từ data/venv root.

Drive restore vẫn là thao tác riêng bằng `cloud_restore.sh`, chỉ dùng khi chủ động cần lấy dữ liệu cũ. Không chạy restore cùng luồng cài mới.
