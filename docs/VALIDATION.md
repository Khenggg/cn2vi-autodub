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
- Chưa gọi DashScope hoặc chạy inference CUDA. Chưa benchmark GPU cloud, thời gian cold-pull, VRAM model, non-verbal preservation, chất lượng tiếng Việt trên corpus thật, temporal inpaint/NVENC hoặc chi phí API. Adapters và CPU model smoke mới được xác minh riêng ở phần dưới.
- Chưa có full Phase A/B/C, import workspace, portable resume snapshot, retry từng segment hay GPU multi-resource scheduler. Metadata export không thay thế backup chứa media hoặc portable recovery.
- Chưa đạt/khóa các mục tiêu 5.000 VND và 35 phút / 120 phút source; các số trong UI là ước tính theo uptime core.

## Chuẩn bị model benchmark trước GPU

- Kiểm tra cuối bằng `scripts/check.ps1`: **141 tests passed**, Ruff (gồm scripts) đạt, TypeScript/Vite production build đạt. Có một warning Starlette/httpx deprecation như baseline; không có test failure. Thêm NumPy/Pillow vào dev lock để tests vision không bị bỏ qua và uv 0.9.6 cho workflow local.

- Đã compile năm dependency profile locks bằng uv 0.9.6, Python 3.12/universal, exact versions và SHA-256 hashes. Lượt cài thực phát hiện bản lock Linux thiếu dependencies Windows; đã sửa sang universal. `local_prepare.ps1 -TtsCpu` đã chạy thực: vision và TTS CPU đều qua dependency check và native imports; `pip check` core không phát hiện lỗi.
- Đã fetch và verify revision/checksums của RapidOCR, LaMa, VieNeu ONNX/MOSS ONNX, ProPainter release weights và pinned source Bandit/ProPainter. Registry có 12 asset groups; không đưa weights hoặc cache lớn vào Git. Weights CUDA còn cần fetch/verify trên host đích. Git assets bị chặn nếu dirty/untracked/ignored files; verification đối chiếu HEAD, ID và revision lock.
- Script CPU smoke đã chạy thực bằng suite, process riêng cho OCR → LaMa → TTS. Cả ba `MEASURED`, giữ synthetic label và quality review gate. Raw reports, PNG, WAV, source tổng hợp và giới hạn phép đo ở [CPU smoke evidence](validation/cpu-smoke/README.md).
- Đã phát hiện bằng visual QC và sửa normalization output của LaMa; có regression test cho output 0–255. Residual OCR count = 0 và unmasked pixels giữ nguyên trên một frame tổng hợp. Kết quả không đại diện chất lượng inpaint chuyển động.
- Preflight local vision: 8 checks PASS, 5 PENDING, 0 FAIL; chưa đạt reference cloud. Các mục pending gồm Ubuntu, GPU memory, RAM và disk theo gate 100 GB. Script bootstrap được kiểm tra cú pháp Bash; chưa thực thi apt/CUDA setup trên Ubuntu/GPU host.
- Graph MCP đã được thử cho architecture/search/trace; sau khi transport server đóng, những file mới chưa vào graph đầy đủ. Đọc exact files và kiểm tra tests/diff được dùng để tiếp tục. Kết quả graph cũ không được coi là blast radius cuối của những adapters mới.

Providers chưa tích hợp vào scheduler/web. Kế hoạch tiếp theo và điều kiện bắt đầu thuê GPU ở [PRE_GPU_READINESS](PRE_GPU_READINESS.md).

## Bộ cài cloud tự động

Bộ cài `.run` chứa source Git bundle và frontend đã build; tự xác minh/giải nén, cài dependencies, tải/verify 10 asset groups GPU và chạy preflight bốn profiles. HTTP downloader có retry/resume. Tests có fixture Git thật và shell giả lập; ghi nhận đầy đủ cả lỗi cách ly test WSL đã xảy ra và đã sửa tại [AUTOMATIC_SETUP_VALIDATION](validation/AUTOMATIC_SETUP_VALIDATION.md). Chưa chạy bộ cài trên host GPU thật của người dùng.

Kiểm tra cuối: **178 tests passed**, một warning Starlette/httpx; Ruff, TypeScript/Vite build và cú pháp Bash/PowerShell đạt. Tests verify checkout/extract thực trong fixture offline và dry-run của file installer tự chứa. Không coi journal READY giả lập là kết quả setup host GPU.
