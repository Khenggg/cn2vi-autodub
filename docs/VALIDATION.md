# Xác minh nền tảng CN2VI AutoDub

Ngày: 06/10/2026, Asia/Saigon. Máy local Windows 11, Python 3.12.14, Node 24.14.1, FFmpeg/FFprobe 8.1.1, GPU GTX 1650 Ti 4 GB. Đây không phải reference cloud RTX 5060 Ti 16 GB.

## Đã chạy

- Script `scripts/check.ps1`: **28 tests passed**, Ruff check passed, TypeScript/Vite production build passed. Một warning từ Starlette TestClient về API httpx cũ; không có test failure. Test temp nằm trong workspace để tránh ACL của thư mục temp dùng chung trên Windows.
- `pip check`: không có dependency lỗi; runtime/dev dependencies khóa exact versions, frontend có package-lock.
- Test media thật: FFmpeg tạo source tổng hợp có video H.264 và audio AAC; upload qua API → queue → FFprobe → source hash → CHECKPOINTED. Tests phần model dùng fixture/contract, không kiểm chứng AI.
- Probe CLI trên video tổng hợp 2 giây: 59 ms stage wall time trong lần đo này; báo cáo raw ở `docs/validation/probe-smoke-report.json`. Không suy diễn từ probe sang hiệu năng ASR/TTS/inpainting hoặc video dài.
- Edge qua skill Playwright: đăng nhập, tạo Series, upload file từ máy, chọn Bắt đầu, SSE cập nhật CHECKPOINTED/ASR/5%, lưu glossary user-locked. Kiểm tra visual desktop 1440×960 và mobile 390×844; bảng tập phim cuộn trong khung trên màn nhỏ.
- Codebase-memory MCP: index trước và sau xây dựng, get_architecture, search_graph/search_code, trace_path và đọc source. Index sau xây dựng có 257 nodes/788 edges. Đã xử lý Git ownership do sandbox tạo repo bằng trust entry chỉ cho `D:/Video`. Sau commit, `detect_changes(base_branch='2407a7b', depth=2)` xác nhận **46 tệp thay đổi** và trả 200 symbol impact entries. Các vùng thay đổi là core API, storage/service, scheduler/media, benchmark, React UI và tests/deployment. Không có mã sản phẩm cũ trong baseline; kết quả này mô tả toàn bộ code mới và các quan hệ trong graph, không phải 200 regression đã xác minh.

## Chưa xác minh

- Docker daemon không chạy: chưa build/run image local. Dockerfile/compose và CI build + health smoke đã được viết, nhưng CI chưa được chạy trên remote.
- Tool trình duyệt tích hợp lỗi kernel lúc khởi động; dùng Edge/Playwright làm fallback. Cổng local 8080 bị Windows từ chối và 8765 đã được dùng; smoke server được Windows cấp cổng tự động. Điều này không xác nhận port/deployment trên cloud.
- Chưa có adapters/model weights, inference hoặc gọi DashScope. Chưa benchmark GPU cloud, thời gian cold-pull, VRAM model, non-verbal preservation, chất lượng tiếng Việt, OCR/inpaint/NVENC hoặc chi phí API.
- Chưa có full Phase A/B/C, import workspace, portable resume snapshot, retry từng segment hay GPU multi-resource scheduler. Metadata export không thay thế backup chứa media hoặc portable recovery.
- Chưa đạt/khóa các mục tiêu 5.000 VND và 35 phút / 120 phút source; các số trong UI là ước tính theo uptime core.
