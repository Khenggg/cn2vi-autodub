# CN2VI AutoDub

Web cá nhân chuyển phim tiếng Trung sang thoại Việt và Vietsub, giữ non-verbal/SFX, xóa subtitle Trung sau review ROI. Yêu cầu gốc ở `docs/CN2VI_AutoDub_Implementation_Guide_v1.0.docx`.

**Pipeline Phase 2 đã được nối vào scheduler/web trên main `01d9e7c`.** Nhánh phát triển này bổ sung checkpoint theo từng stage, kiểm duyệt lời thoại/thời lượng, QC âm thanh và xử lý Vietsub/ROI. Chi tiết luồng và giới hạn ở [PIPELINE_RUNTIME](docs/PIPELINE_RUNTIME.md); kết quả cloud được ghi riêng và không thay thế đánh giá chất lượng phim thật.

Trong quy trình làm việc hiện tại, máy local dùng để sửa và đồng bộ mã; cài đặt, test, build và inference chạy trên GPU cloud. Các lệnh local bên dưới là tham khảo cho môi trường phát triển khác.

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
3. Chọn chế độ đầu ra rồi upload: xóa sub Trung + Vietsub, thêm Vietsub, hoặc chỉ lồng tiếng. Bắt đầu/hàng đợi kiểm tra nguồn rồi chạy các stage đã cấu hình; runtime thiếu sẽ báo lỗi thay vì giả định model sẵn sàng.
4. Duyệt lời thoại khi cần, chọn vùng phụ đề khi thay sub Trung. Dòng quá dài cần sửa hoặc chọn giữ âm gốc; không tự cắt đuôi giọng. Xem nguồn/bản Việt và tải artifact khi được tạo.
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

## Chuẩn bị trước GPU cloud

Đã khóa revision/checksum model, dependency profiles độc lập, adapter Qwen ASR/aligner, Bandit, VieNeu-TTS, RapidOCR, LaMa và ProPainter. Bộ suite chạy từng model trong process riêng, ghi thời gian/RTF/RAM/VRAM và phần bằng chứng chất lượng còn thiếu. Provider DashScope có retry và kiểm tra JSON; chưa gọi API thật.

OCR, LaMa và TTS ONNX đã chạy trên CPU với clip tổng hợp. Clip này chỉ xác minh đường chạy; chất lượng phim thật, GPU và mục tiêu chi phí chưa được xác nhận. Xem [CPU smoke](docs/CPU_SMOKE.md), [chuẩn bị corpus](docs/BENCHMARK_CORPUS.md), [sinh suite](docs/BENCHMARK_PLAN.md), [cloud runbook](docs/CLOUD_RUNBOOK.md) và [điểm chuyển sang GPU](docs/PRE_GPU_READINESS.md).

Bootstrap native hỗ trợ Ubuntu 22.04/24.04 x86_64 và Python 3.12; host hiện tại là RTX 3060 12 GB trong container Ubuntu 22.04. Dùng `bash scripts/cloud_restore.sh` khi đã có rclone profile và backup Drive, hoặc `bash scripts/setup_new_server.sh` cho cài mới. Restore không tự xác nhận tính tương thích venv; cần kiểm tra môi trường trên host. Bộ preflight/reference cũ dưới đây dành cho RTX 5060 Ti 16 GB và cần phân biệt với host hiện tại. [Bộ cài một file](docs/AUTOMATIC_CLOUD_SETUP.md) tự giải nén code/giao diện, cài môi trường, tải/kiểm tra weights và chạy preflight. Tạo bằng `scripts/package_cloud.ps1`, chuyển `.cache/cn2vi-cloud-setup.run` sang cloud rồi chạy `bash cn2vi-cloud-setup.run`. `--dry-run` chỉ xem kế hoạch. Tải HTTP có retry/resume và chỉ niêm phong model khi checksum khớp. Không cần Node/npm trên cloud; NVIDIA driver phải hoạt động sẵn.

Tạo plan trên máy sẽ chạy benchmark để đường dẫn media/interpreter đúng. Benchmark, API key và video thật được cấu hình sau khi setup thành công; repo không tự thuê hoặc tắt máy cloud. Nếu đã có checkout, `bash scripts/cloud_setup.sh` chạy cùng chuỗi setup; các lệnh riêng trong runbook vẫn có thể dùng để chẩn đoán.

Tài liệu framework tham khảo: [FastAPI security](https://fastapi.tiangolo.com/tutorial/security/), [Vite guide](https://vite.dev/guide/), [Python SQLite](https://docs.python.org/3/library/sqlite3.html), [Qwen3-ASR upstream](https://github.com/QwenLM/Qwen3-ASR).
