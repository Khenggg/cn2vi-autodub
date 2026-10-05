# ADR 0001 Core foundation trước khi tích hợp model

Ngày: 06/10/2026. Trạng thái: accepted cho Phase 1.

Nguồn yêu cầu: `docs/CN2VI_AutoDub_Implementation_Guide_v1.0.docx`, đặc biệt mục 0, 4, 8–12, 19–23. Trước khi bắt đầu thư mục chỉ có tài liệu; không có code hoặc model weights.

## Quyết định

FastAPI phục vụ REST, SSE và bundle React/Vite. SQLite WAL lưu workspace, Series, Episode, segment, artifact, issue, job_event, glossary và translation memory. Một process core là owner, không chạy uvicorn nhiều workers. Timestamp dùng integer milliseconds. Số priority nhỏ hơn chạy trước; thứ tự episode theo ordinal. Upload theo offset, chunk tối đa 8 MiB, file nằm trong data root, fsync trước commit offset, SHA-256 và atomic rename khi hoàn tất.

Phase 1 dùng một thread cho stage CPU PREPARING với FFprobe timeout. Đây là bước tăng dần tới topology nhiều process trong tài liệu: khi bổ sung Phase A, scheduler điều phối audio/video worker tách process qua localhost/Unix socket; core vẫn là SQLite owner. Chưa có heavy GPU stage để cấp lease. System báo khả năng admission theo free VRAM nhưng không tuyên bố model ready.

PREPARING xuất checkpoint có schema_version, source hash và media metadata. Sau đó Episode là CHECKPOINTED, next_stage=ASR, progress=0.05. Không tạo transcript/TTS/preview/final giả. Bật provider cần benchmark trên đúng GPU, weights checksum và exact runtime version. Queue tiếp tục chuẩn bị các tập còn lại bằng CPU; chưa có policy overlap nhiều tài nguyên Phase A.

Review gate bắt buộc: API ROI chỉ chấp nhận PREVIEW_READY/AWAITING_ROI, rectangle normalized nằm trong frame. Lưu ROI không khởi chạy removal. Các bước OCR/inpaint/encode chưa tồn tại ở Phase 1.

Auth dùng admin token từ env hoặc sinh khi boot, đổi thành session cookie HttpOnly/SameSite Strict, hạn 24 giờ. Mutation cần header và same-origin; file đọc theo DB ID. Public deployment cần chọn binding/firewall phù hợp; compose mặc định localhost. Không lưu secret trong DB/export/log diagnostic.

Drain chặn job mới, đợi probe đang chạy checkpoint, báo READY_TO_SHUTDOWN. Khởi động lại phục hồi PREPARING bị ngắt thành QUEUED; trạng thái drain giữ lại. Export `.aidub` chứa metadata và checksums, không chứa source/output hoặc env. Import và portable resume snapshot đầy đủ thuộc Phase 4; checkpoint JSON Phase 1 không thay thế portable snapshot.

## Hệ quả

Core có thể phát triển/kiểm thử trên CPU Windows và Linux. Docker core Ubuntu 24.04 chưa chứa CUDA/PyTorch/model weights, chưa phải image GPU release. Chưa khóa SLA, giá API hoặc khả năng NVENC/Blackwell. Các adapter mới không được làm thay đổi Segment contract hoặc tự vượt review gate.
