# VNLE prototype 0.1 — mở video, khoanh sub và xuất sự kiện chữ

Ngày 09/10/2026. Người dùng đã yêu cầu bắt đầu triển khai sau nghiên cứu.
Đây là bản mẫu discovery P0/P1; chưa phải MVP Việt hóa hoàn chỉnh.

**Cập nhật máy thực hiện 09/10/2026:** người dùng chỉ định Windows local GTX 1650 Ti 4 GB. Cài qua `scripts/install-execution.ps1 -ConfirmExecutionMachine`, tải model bằng `.venv-vnle\Scripts\python.exe -m vnle.assets --destination models\ppocr-v6-small --confirm-execution-machine`, rồi bật server bằng `.venv-vnle\Scripts\python.exe -m vnle serve --port 18083 --config config\prototype.cuda.json --model-manifest models\ppocr-v6-small\manifest.json --enable-analysis`. Không cần cloud/tunnel cho lượt này. Các ghi chú cloud và local UI-only phía dưới mô tả lượt triển khai trước khi người dùng xác nhận máy thực hiện.

Sau khi cài xong, khởi động Windows bằng `powershell -ExecutionPolicy Bypass -File .\scripts\start-vnle.ps1`. Có thể mở lại kết quả qua `/?video=VIDEO_ID&run=RUN_ID`: trang lấy video, vùng đã xác nhận và dữ liệu của lượt chạy, không tự khởi chạy lại OCR.

## Những phần đã viết

- Giao diện web tiếng Việt, chọn MP4, tua/dừng, kéo chuột khoanh tối đa 16 hình chữ nhật.
- Xác nhận ROI bắt buộc trước phân tích; sửa/xóa vùng yêu cầu xác nhận lại.
- Local UI-only xem trực tiếp file trong trình duyệt, không upload hoặc chạy OCR; xuất `vnle-request.json` để dùng bên máy thực hiện.
- Máy thực hiện nhận upload theo stream, probe video và phục vụ MP4 với HTTP Range.
- Probe, input/code/config/model SHA-256, metadata PTS/time-base và origin.
- Decode PyAV tuần tự; change scan ngoài ROI, watchdog tối đa 250ms trong config (khoảng thực giữa các frame có thể dài hơn với VFR).
- Cắt ảnh thành các tile là phần bù của ROI **trước detector**. Recognizer chỉ nhận crops từ các tile này. Không OCR sub rồi loại kết quả.
- Một worker sở hữu một detector và một recognizer PP-OCRv6 small. RapidOCR batch crop theo cấu hình; ORT sessions load một lần và ghi trace per-node.
- Ghép observation theo chữ và vị trí; đổi số/chữ tạo revision, biến mất rồi quay lại tạo appearance mới. Chữ confidence thấp được giữ và đánh dấu cần xem lại.
- Xuất `events.json`, `observations.jsonl`, ảnh chứng cứ, `run-report.json/.md`; hủy/lỗi giữ phần đã có. Không tự retry.
- UI xem ảnh/chữ/timecode, tua về vị trí chứng cứ, tải báo cáo; danh sách có phân trang để không dựng hàng nghìn DOM node cùng lúc.

## Chưa triển khai hoặc chưa xác minh

Chưa có importance classifier/semantic gate, dịch, glossary/TM, overlay planner, render, full QC, compatibility SubAI, boundary refinement, perspective/occlusion tracking, polygon ROI hay resume inference. Không sửa SubAI.

Rotation khác 0 và sample aspect ratio khác 1 **bị từ chối rõ**, tránh khoanh một nơi và loại trừ một nơi khác. ROI UI hiện áp dụng toàn video; CLI nhận interval riêng cho từng ROI. Outward rounding tối đa dưới một pixel trên mỗi cạnh bảo vệ pixel được user khoanh; không tự thêm margin.

OCR là line-level observations, chưa ghép panel/reading order nhiều dòng. IoU association có thể sai khi camera pan, nhiều chữ trùng hoặc OCR chập chờn. Không coi chữ nhận được là đã xác định quan trọng. `selected=null`, `decision=PENDING_IMPORTANCE_REVIEW`.

Phần bù ROI dùng tile không overlap; chữ vắt qua biên có thể bị cắt, được đánh dấu `TILE_EDGE`. Detector resize tối đa 960 trong cấu hình thử; nhận chữ nhỏ chưa nghiệm thu. Detector upstream có giới hạn proposal; `degenerate_proposals` có counter. Angle classifier không load, hướng 180° chưa được nghiệm thu.

Timecode dùng PTS thật; start/end được ước lượng giữa mẫu có/không có chữ, và lưu bracket. Ngưỡng 300ms chưa được chứng minh trên video thật. Mọi sample gap thật được báo, không suy fps hint thành PTS. Chưa có gold corpus hoặc benchmark 1650Ti. `COMPLETED_UNVERIFIED` nghĩa là hoàn tất phân tích, **không là pass quality hoặc SLA**.

Source RGB còn nguyên để cắt tile; bản scan được blank ROI trước resize thumbnail. Không có pixel ROI nào vào detector/crop recognizer. Batch pending giới hạn 8 crops/8MiB trong profile; một source frame, scan thumbnail và evidence crops của frame hiện tại vẫn có bộ nhớ riêng. Metadata các event tích lũy theo số observation; chưa đo RSS/VRAM, không tuyên bố cap process.

## Xem giao diện trên Windows local

Không cần cài dependencies AI cho bước này:

```powershell
Set-Location -LiteralPath 'D:\Video'
powershell -ExecutionPolicy Bypass -File .\scripts\start-preview.ps1
```

Nếu `python` không trỏ tới Python thật, truyền `-Python 'C:\Users\Ken\AppData\Local\Programs\Python\Python312\python.exe'`.
Mở `http://127.0.0.1:18083`. Chọn video, dừng ở khung có sub, kéo vùng, xác nhận, lưu JSON.
Chế độ này không gọi ffprobe hay upload file; video chỉ được browser đọc để preview. Nút OCR bị khóa.

Trong lượt triển khai hiện tại server UI-only đã được khởi động và trang đã được kiểm tra hiển thị. Chưa chọn/chạy video thực tế.

## Cài trên máy thực hiện Ubuntu 24.04

Chưa thuê/kết nối cloud trong lượt này. Chỉ chạy các lệnh sau trên máy thực hiện được chỉ định, không chạy trên máy local phát triển:

```bash
cd /path/to/vnle
bash scripts/install-execution.sh --confirm-execution-machine
.venv-vnle/bin/python -m vnle.assets --destination models/ppocr-v6-small --confirm-execution-machine
.venv-vnle/bin/python -m vnle serve --port 18083 \
  --data-root data/vnle --config config/prototype.cuda.json \
  --model-manifest models/ppocr-v6-small/manifest.json --enable-analysis
```

NVIDIA driver phải được host cung cấp. Script cài Python venv/FFmpeg khi thiếu, dependencies mới riêng cho VNLE, ORT1.23.2 với NVIDIA CUDA12/cuDNN9 runtime wheels. Đã kiểm tra metadata PyPI: ORT1.31.0 mới dùng extras CUDA13, nên không chọn phiên bản đó cho profile CUDA12 này. ORT preload từ wheel NVIDIA, không đọc DLL/model của SubAI. Version runtime này là candidate chưa qua clean-env cloud verification; CUDA không khởi tạo được sẽ báo lỗi, không tự chạy CPU.

Install script chỉ cài môi trường; tải weights là lệnh riêng có xác nhận execution machine. Downloader khóa cặp chính thức theo URL/hash registry và không đổi nguồn/model khi lỗi. Download/hash/provisioning không được chạy trong lượt local này.

Trên Windows cloud:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-execution.ps1 -ConfirmExecutionMachine
.\.venv-vnle\Scripts\python.exe -m vnle.assets --destination models\ppocr-v6-small --confirm-execution-machine
.\.venv-vnle\Scripts\python.exe -m vnle serve --port 18083 --config config\prototype.cuda.json --model-manifest models\ppocr-v6-small\manifest.json --enable-analysis
```

Windows cần Python3.12+, NVIDIA driver/VC runtime và một build FFmpeg độc lập có `ffprobe` trên PATH. Script báo thiếu thay vì dùng runtime của SubAI. Gói NVIDIA CUDA wheels trên Windows cũng cần kiểm tra trên máy thật.

Server chỉ bind loopback. SSH tunnel từ máy local, thay endpoint bằng máy được cấp mới:

```powershell
ssh -N -L 18084:127.0.0.1:18083 -p PORT USER@HOST
```

Mở `http://127.0.0.1:18084`, upload, khoanh vùng, xác nhận, chạy mặc định 30s đầu. Không dùng port local18083 đang là preview để nhầm với cloud.

## CLI với vùng đã khoanh

Copy video và `vnle-request.json` sang máy thực hiện; dùng đường dẫn thật:

```bash
.venv-vnle/bin/python -m vnle analyze input.mp4 \
  --request vnle-request.json --output data/runs/first-30s \
  --config config/prototype.cuda.json \
  --model-manifest models/ppocr-v6-small/manifest.json
```

Output directory phải mới để không ghi đè run. File `config/request.example.json` chỉ minh họa schema, không là ROI được duyệt cho video của bạn. Không dùng mẫu đó thay thao tác khoanh thật.

## Validation trong lượt này

- Python AST, TOML/JSON và `node --check`: pass.
- Ruff lint/format: pass sau sửa định dạng; chỉ là static inspection.
- UI-only server khởi động, trang/controls được kiểm tra trực tiếp qua browser; không inference.
- CI đã thêm Windows/Ubuntu contract tests và native PyAV/CV integration dùng video synthetic/fake OCR. Bao gồm ROI complement/pixel isolation, VFR PTS, revision/numeral/reappearance, Range/security, partial report và UI-only không bật inference.
- **Pytest chưa chạy** vì quy định dùng CI; nhánh chưa push nên CI chưa được kích hoạt. Native tests không là quality/throughput của OCR thật.
- Chưa cloud install, model download/inference, media benchmark, provider trace thực hoặc video output.

## Nguồn adapter

Adapter gọi thư viện có license, không copy mã SubAI/các repo GPL/AGPL.
[RapidOCR pinned source](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/rapidocr/main.py),
[native batched recognizer](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/rapidocr/ch_ppocr_rec/main.py),
[custom ORT session hook](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/rapidocr/inference_engine/onnxruntime/main.py),
[model registry/license evidence](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/MODEL_LICENSES.md).
PyPI RapidOCR3.10.0 đã được kiểm tra tồn tại; source snapshot chưa là chứng minh adapter hoạt động với installed wheel. CI native adapter contract và cloud run là cổng kiểm tiếp theo.
