# Cài cloud từ GitHub

Luồng chính lấy mã nguồn trực tiếp từ repository GitHub riêng `Khenggg/cn2vi-autodub`, rồi chạy `scripts/cloud_setup.sh`. Setup cài môi trường, tải và kiểm tra model assets đã pin, rồi chạy preflight. Nó không tạo hoặc tắt máy cloud, cài NVIDIA driver, chạy benchmark hay gọi API dịch.

## 1. Clone repository riêng

Host cần kết nối Internet và Git. Đảm bảo tài khoản trên host có quyền đọc repository private. Nếu GitHub yêu cầu xác thực, dùng `gh auth login`, credential helper đã cấu hình hoặc SSH key được cấp quyền; không đặt PAT trong URL hoặc command line.

```bash
git clone https://github.com/Khenggg/cn2vi-autodub.git
cd cn2vi-autodub
```

## 2. Kiểm tra rồi chạy setup

Host cần Ubuntu 24.04 x86_64, driver NVIDIA đã cài và hoạt động (`nvidia-smi`), GPU có ít nhất 15,000 MiB VRAM và tối thiểu 100 GB thập phân còn trống trên filesystem dữ liệu. Cấu hình tham chiếu là RTX 5060 Ti 16 GB, RAM 28 GB. Có quyền root hoặc `sudo` để cài gói hệ thống. Setup không cài/nâng cấp driver hoặc yêu cầu reboot.

Xem kế hoạch trước; `--dry-run` không chạy apt, tải assets, tạo environment hay sửa dữ liệu:

```bash
bash scripts/cloud_setup.sh --dry-run
```

Khi sẵn sàng, chạy:

```bash
bash scripts/cloud_setup.sh
```

Có thể đổi nơi lưu data và virtualenv bằng biến môi trường:

```bash
AUTODUB_DATA_ROOT=/data \
AUTODUB_VENV_ROOT=/opt/autodub/venvs \
bash scripts/cloud_setup.sh
```

Setup cài core cùng các profile ASR, TTS, vision và Bandit; tải rồi xác minh assets từ các nguồn đã khóa; sau đó chạy preflight bắt buộc cho bốn profile. Cần Internet và dung lượng đáng kể. Node.js 24.14.1 được cài vào cache dưới `AUTODUB_DATA_ROOT`; không thay thế Node 18 đang có trên hệ thống. Cấu hình frontend hỗ trợ tự động, không cần tự cài Node/npm toàn hệ thống.

Mặc định data/models/cache/reports ở `/data`; virtualenv ở `/opt/autodub/venvs`. Checkout Git nằm tại thư mục clone. Setup giữ dữ liệu hiện có, có thể chạy lại, không tải hoặc gửi video/media hay venv/cache từ máy phát triển. Model files được tải trực tiếp trên host. Setup thành công báo `READY`; journal nằm dưới `/data/results/cloud-setup-<run>.log`, preflight reports tại `/data/results/preflight-<profile>.json` (theo data root đã cấu hình).

## 3. Bước benchmark sau setup

`READY` xác nhận host và môi trường qua preflight, không xác nhận chất lượng model. Tạo plan, chạy dry-run rồi benchmark theo [cloud runbook](CLOUD_RUNBOOK.md). Giai đoạn này chưa cần video hoặc API key. Benchmark đại diện cần video/corpus có quyền sử dụng. Nếu sau đó chọn provider DashScope, cấu hình `DASHSCOPE_API_KEY` an toàn trên host và chọn rõ `QWEN_TRANSLATION_MODEL`; không đưa key vào URL, command line, repository hay report.

## Ngoại tuyến

Nếu host không thể clone GitHub, có thể dùng file `.run` tự chứa làm fallback offline. Gói bằng `scripts/package_cloud.ps1`, chuyển file thủ công rồi chạy `bash cn2vi-cloud-setup.run`; xem [runbook cài đặt tự động](CLOUD_RUNBOOK.md). SHA-256 sidecar chỉ kiểm tra lỗi truyền file, không phải chữ ký xác thực nhà phát hành.
