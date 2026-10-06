# Chuẩn bị trước GPU cloud — 06/10/2026

Baseline core: `9b4960a`. Phần benchmark được chuẩn bị để khóa runtime và đo từng model trước khi tích hợp pipeline AI vào web.

## Đã chuẩn bị

- Model registry: 12 nhóm assets, pinned source commits/Hugging Face revisions, checksums upstream hoặc SHA-256 đã niêm phong từ release chính thức; kiểm tra SHA/size khi tải và khi nạp. LaMa, RapidOCR, VieNeu ONNX/MOSS ONNX và ProPainter weights đã tải local. SHA của ProPainter được tính sau tải HTTPS và đối chiếu size GitHub release; đây không phải chữ ký tác giả.
- Năm dependency locks có hashes: ASR, TTS CUDA, TTS CPU, vision, Bandit/ProPainter. Tách environment model khỏi core web để tránh xung đột Gradio/Pydantic. Torch CUDA 12.8 được kiểm tra riêng ở cloud bootstrap.
- Adapters thật: Qwen ASR/align, Bandit ERB48, VieNeu, RapidOCR ROI, LaMa tight crop và ProPainter frame sequence có giới hạn. DashScope translation giữ glossary/context, kiểm tra IDs/JSON, retry có giới hạn, không ghi key vào report.
- Corpus validator, metric CER/alignment/audio QC, config/plan generator, suite chạy process riêng theo thứ tự, dry-run, timeout/failure reporting, environment preflight, cloud bootstrap và runbook.
- `scripts/package_cloud.ps1` tạo Git bundle và bộ cài tự giải nén `.run` có code/giao diện/checksums. Chạy một lệnh trên cloud để cài môi trường, tải/verify models rồi preflight; không đóng gói sẵn weights/cache/env. Xem [AUTOMATIC_CLOUD_SETUP](AUTOMATIC_CLOUD_SETUP.md). Tải HTTP tiếp tục phần dở khi server hỗ trợ Range; file hoàn chỉnh đúng hash được dùng lại.
- Local CPU smoke đã chạy OCR → LaMa và VieNeu ONNX, tạo mask/PNG/WAV thật. Raw reports ở [validation](validation/); phép đo này chỉ dùng dữ liệu tổng hợp.

## Điểm cần GPU

Phần tiếp theo cần máy cloud để kiểm chứng Qwen/aligner, Bandit, VieNeu CUDA và ProPainter bằng inference thật, VRAM allocator peak và thời gian cold load. Máy local GTX 1650 Ti 4 GB không đạt reference gate. Cấu hình kiểm thử theo tài liệu: Ubuntu 24.04, Python 3.12, RTX 5060 Ti 16 GB, RAM tối thiểu 28 GB và disk trống tối thiểu 100 GB. Đây là reference dự án, chưa phải cam kết hiệu năng.

Trước khi bắt đầu trả tiền GPU, cần source phim thật có quyền sử dụng, ít nhất 10 phút với coverage các loại cảnh trong [corpus guide](BENCHMARK_CORPUS.md), cùng transcript/timing/ROI để đánh giá. Hiện chưa có source representative được cung cấp. Có thể chạy smoke CUDA ngắn để kiểm tra compatibility khi có host, nhưng chưa thể kết luận chất lượng phim hoặc chi phí mục tiêu.

Khi sẵn sàng, cung cấp đường dẫn video và cách truy cập host/SSH đã thuê. Theo [cloud runbook](CLOUD_RUNBOOK.md): bootstrap → fetch/verify → preflight từng profile → tạo plan trên cloud → dry-run → chạy suite → lưu reports/artifacts. Không cần đưa API key vào chat; DashScope cần biến môi trường riêng trên host và tên model được chọn rõ ràng.

## Những phần vẫn còn sau benchmark

Pipeline lồng tiếng end-to-end trong web, mix/review non-verbal/SFX, ROI editor/tracking/temporal smoothing, final render NVENC/libass, import/portable recovery và batch release tests vẫn chưa xong. Chưa gọi DashScope thật, chưa build Docker trên Linux hoặc benchmark GPU; chưa xác nhận các mục tiêu thời gian/chi phí. Chi tiết ở [IMPLEMENTATION_STATUS](IMPLEMENTATION_STATUS.md).

Codebase-memory đã dùng trước khi sửa; server MCP đóng transport trong vòng chuẩn bị này. Các tệp sửa được đọc trực tiếp và kiểm tra bằng tests/Ruff. Blast-radius graph cuối sẽ ghi trong VALIDATION nếu server phục hồi.
