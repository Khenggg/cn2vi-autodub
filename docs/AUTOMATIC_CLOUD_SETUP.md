# Cài đặt cloud tự động

Quy trình này đóng gói mã nguồn đã commit và frontend đã build vào một file `cn2vi-cloud-setup.run`. Trên máy Ubuntu, file tự kiểm tra payload, dựng checkout đúng commit, cài môi trường, tải/kiểm tra model assets và chạy preflight. Nó không tạo hoặc tắt máy cloud, cài driver NVIDIA, chạy benchmark hay gọi API dịch.

## Tạo gói trên Windows

Đảm bảo thay đổi đã commit, frontend dependencies cài được bằng `npm ci`, và core Python environment `.venv` đã sẵn sàng. Chạy từ repository:

```powershell
.\scripts\package_cloud.ps1
```

Mặc định script build frontend, rồi ghi ba file dưới `.cache`: Git bundle `cn2vi-cloud.bundle`, installer một file `cn2vi-cloud-setup.run`, và SHA-256 sidecar cho mỗi gói. Installer chứa bundle cùng `frontend/dist`; model weights, video và credentials không được đóng gói. `-BundleOnly` chỉ tạo Git bundle cho quy trình clone thủ công cũ.

Hiện chưa có URL hosting công khai. Sau khi chuẩn bị gói, chuyển duy nhất `cn2vi-cloud-setup.run` lên host Ubuntu bằng phương thức file transfer do bạn chọn. Nếu chuyển kèm sidecar, có thể kiểm tra lỗi truyền file:

```bash
sha256sum -c cn2vi-cloud-setup.run.sha256
```

SHA-256 phát hiện payload hoặc file bị thay đổi so với digest đi kèm; sidecar và digest nhúng trong installer không phải chữ ký xác thực nhà phát hành. Muốn xác thực nguồn phát hành, cần đối chiếu hash toàn file qua một kênh tin cậy riêng.

## Chạy trên host

Host phải là Ubuntu 24.04 x86_64 với driver NVIDIA đã cài và hoạt động. Dùng máy tham chiếu RTX 5060 Ti 16 GB, RAM tối thiểu 28 GB và ít nhất 100 GB đĩa trống trên volume model/cache. Installer cần quyền root hoặc `sudo` để cài Git, certificate, Python 3.12 và virtualenv support; nó không cài, nâng cấp hoặc yêu cầu reboot driver NVIDIA.

Chạy kiểm tra kế hoạch trước:

```bash
bash cn2vi-cloud-setup.run --dry-run
```

Lệnh này chỉ in commit, đường dẫn và các bước dự kiến; không chạy apt, tải file, tạo checkout hay sửa dữ liệu. Khi sẵn sàng, chạy:

```bash
bash cn2vi-cloud-setup.run
```

Installer kiểm tra Ubuntu 24.04 x86_64 và checksum payload trước apt. Nếu host đã có Python 3/Git, nó kiểm tra cả archive, bundle, commit và xung đột checkout trước apt; nếu thiếu, nó cài công cụ tối thiểu rồi kiểm tra trước khi cài model environments. Sau đó frontend được copy vào project root. Mặc định project ở `/data/autodub/project`, dữ liệu/model/cache/reports ở `/data`, còn venv ở `/opt/autodub/venvs`. Có thể đổi các root trước khi chạy:

```bash
AUTODUB_DATA_ROOT=/data \
AUTODUB_VENV_ROOT=/opt/autodub/venvs \
AUTODUB_PROJECT_ROOT=/data/autodub/project \
bash cn2vi-cloud-setup.run
```

Setup tự động cài core cùng các profile ASR, TTS, vision và Bandit; tải và kiểm tra model assets đã pin; cuối cùng chạy preflight cloud cho từng profile. Các bước này cần Internet, có thể tải nhiều dữ liệu và dùng đáng kể dung lượng. Hãy kiểm tra billing và dung lượng host trước khi chạy. Một lần setup thành công không tự chạy suite benchmark.

Rerun cùng installer chỉ tiếp tục khi project hiện có đúng commit đã pin, Git worktree sạch và frontend build khớp payload. Nếu project khác commit hoặc có sửa đổi cục bộ, installer dừng và giữ nguyên dữ liệu; chọn một `AUTODUB_PROJECT_ROOT` mới nếu muốn cài song song. Dữ liệu dưới `AUTODUB_DATA_ROOT` được giữ lại.

Theo dõi journal ở `/data/results/cloud-setup-<run>.log` và preflight reports ở `/data/results/preflight-<profile>.json` (hoặc các root đã cấu hình). Khi setup báo `READY`, kiểm tra report từng profile trước khi tạo plan và chạy benchmark theo [cloud runbook](CLOUD_RUNBOOK.md). Preflight là kiểm tra môi trường, không xác nhận chất lượng model.

## Việc cần chuẩn bị cho benchmark

Sau khi có host sẵn sàng, cần video/corpus có quyền sử dụng và cấu hình suite để đánh giá phim thực tế. Chưa cần truyền video hoặc API key cho installer. Nếu chọn provider dịch DashScope ở giai đoạn sau, cấu hình `DASHSCOPE_API_KEY` an toàn trên host và chọn rõ `QWEN_TRANSLATION_MODEL`; không nhúng key vào gói, command line hay report.
