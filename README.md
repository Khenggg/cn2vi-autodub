# CN2VI AutoDub

Web cá nhân chuyển phim tiếng Trung sang thoại Việt và Vietsub, giữ non-verbal/SFX, xóa subtitle Trung sau review ROI. Yêu cầu gốc ở `docs/CN2VI_AutoDub_Implementation_Guide_v1.0.docx`.

**Phiên bản 0.1 triển khai nền tảng Phase 1 và runner cho Phase 0.** Có tạo Series, glossary, upload tiếp tục theo offset, queue ưu tiên, FFprobe, checkpoint, SSE, drain và xuất metadata workspace. ASR, dịch, tách âm, TTS, xóa/burn subtitle chưa tích hợp; episode dừng ở CHECKPOINTED trước ASR. Không có pipeline/demo AI giả.

## Chạy local

Cần Python 3.12+, Node 24.14.1 (hoặc bản đáp ứng Vite), npm và FFmpeg/FFprobe trong PATH. Dependencies được khóa trong requirements*.lock và frontend/package-lock.json. Không tự load `.env`; export biến môi trường qua shell nếu cần.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
Set-Location frontend
npm ci
npm run build
Set-Location ..
.\scripts\dev.ps1
```

Mở http://127.0.0.1:8080 và nhập admin token được in lúc server boot. Nếu executable FFprobe không trong PATH, đặt `$env:FFPROBE_BIN='đường dẫn đầy đủ tới ffprobe.exe'`. Script dev bind localhost. Dữ liệu mặc định ở `./data`, không bị xóa khi restart. Trình duyệt nhận cookie HttpOnly; không lưu token vào localStorage.

Nếu cổng 8080 bị Windows chặn hoặc đang được dùng, chạy `.\scripts\dev.ps1 -Port 0` để Windows cấp cổng trống, rồi mở URL được in trong terminal. Phiên kiểm thử trên máy hiện tại đã dùng cách này.

Linux:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.lock
.venv/bin/pip install --no-deps -e .
cd frontend && npm ci && npm run build && cd ..
.venv/bin/uvicorn autodub.main:create_app --factory --host 127.0.0.1 --port 8080 --workers 1
```

Chỉ chạy **một uvicorn worker**, vì core sở hữu scheduler và SQLite. Có thể dùng `npm run dev` trong frontend (port 5173, proxy API đến 8080) trong lúc phát triển.

## Docker core

```sh
docker compose up --build
docker compose logs -f core
```

Hoặc:

```sh
docker build -t cn2vi-autodub:core .
docker run --rm -p 127.0.0.1:8080:8080 -v cn2vi-data:/data cn2vi-autodub:core
```

Docker core dùng Ubuntu 24.04 và chạy non-root, named volume `/data`. Nếu bind mount, cấp quyền cho UID 10001 trên thư mục được mount. Image này chưa chứa CUDA/PyTorch/weights và chưa yêu cầu `--gpus all`. Image GPU theo tài liệu sẽ được xây sau khi Phase 0 khóa model/runtime. Docker phải đang chạy để build; kết quả build local được ghi riêng trong VALIDATION.

## Quy trình hiện hoạt động

1. Tạo Series, đặt số ưu tiên (nhỏ hơn chạy trước).
2. Chọn/kéo video. Chunk tối đa 8 MiB, hiện throughput đo được. Upload ngắt thì chọn lại cùng tệp từ cùng browser để tiếp tục; mapping local dựa trên Series, tên, size và lastModified. Không đổi nội dung tệp giữa các lần resume.
3. Chọn Bắt đầu/hàng đợi. FFprobe kiểm tra video + audio, hash source đối chiếu, lưu checkpoint. Video không hợp lệ thành FAILED; tập khác vẫn được chuẩn bị.
4. Xem nguồn hoặc tải checkpoint JSON. CHECKPOINTED/ASR thể hiện đang chờ provider; không phải preview lồng tiếng Việt.
5. Drain: chặn start mới, đợi stage hiện tại, báo có thể tắt. Resume để nhận việc. Xuất workspace tải metadata `.aidub`, không chứa media/credentials. **Import và portable resume đầy đủ chưa có.** Giữ video gốc và volume hiện tại.
6. Xóa episode/Series bằng UI khi muốn dọn tệp; server không tự xóa source/output.

## Kiểm thử và benchmark

```powershell
.\scripts\check.ps1
```

```sh
pytest -q
ruff check src tests benchmarks
cd frontend && npm run build
```

Test bao phủ auth/CSRF, upload resume/offset/hash và crash rename, state machine, scheduler priority, drain đang xử lý, retry, source corruption, ROI gate, glossary user-lock, metadata export không lộ secrets, recovery preparation và benchmark failure reporting. Test FFprobe/media thật được chạy khi FFmpeg đã cài. Model golden tests cần video có quyền sử dụng và GPU cloud.

[Benchmark CLI](benchmarks/README.md), [tiến độ và gates](docs/IMPLEMENTATION_STATUS.md), [kiến trúc](docs/adr/0001-core-foundation.md), [kết quả xác minh](docs/VALIDATION.md).

Tài liệu framework tham khảo: [FastAPI security](https://fastapi.tiangolo.com/tutorial/security/), [Vite guide](https://vite.dev/guide/), [Python SQLite](https://docs.python.org/3/library/sqlite3.html). Candidate ASR cần kiểm tra từ [Qwen3-ASR upstream](https://github.com/QwenLM/Qwen3-ASR) khi triển khai adapter.
