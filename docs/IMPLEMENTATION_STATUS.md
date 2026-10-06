# CN2VI AutoDub tiến độ triển khai

Baseline: Implementation Guide v1.0 ngày 06/10/2026.

| Giai đoạn | Trạng thái hiện tại | Gate tiếp theo |
| --- | --- | --- |
| 0 Benchmark | Adapters ASR/align/Bandit/TTS/OCR/LaMa/ProPainter, model lock và profile locks, corpus validator/suite/preflight; OCR/LaMa/TTS CPU smoke đã đo | Corpus thật 10 phút, CUDA inference và chất lượng/VRAM/chi phí thực |
| 1 Core | FastAPI, React, SQLite, auth, upload/resume, Series queue, FFprobe, checkpoint, drain, SSE; Docker và CI được viết | Test local và build; kiểm tra image trên Linux |
| 2 Phase A | Segment/provider interfaces, window merge, duration fitting, budget rules; providers Qwen/VieNeu/DashScope có riêng | Tích hợp ASR → align → translation → Bandit → TTS → mix → QC vào scheduler/web |
| 3 Review/B/C | ROI contract/API review gate; benchmark ROI OCR/LaMa và ProPainter bounded crop | ROI canvas, tracking/temporal smoothing, player audio Việt, pipeline render libass/NVENC |
| 4 Recovery | Metadata export và khôi phục preparation khi restart | Import workspace, portable resume zip/hash validation, retry từng segment, resource leases |
| 5 Cost/perf | Ước tính cost theo uptime core, admission VRAM sơ bộ | GPU/API metrics, real billing start, budget escalation enforcement |
| 6 Release | README, ADR, CI nền tảng | Batch 20×5 phút, benchmark 120 phút, signed image |

Phase 1 kết thúc Episode ở CHECKPOINTED với next_stage=ASR. Video nguồn phát trong UI vẫn có audio tiếng Trung. Không có video lồng tiếng Việt/final ở phiên bản này. Checkpoint JSON có checksum metadata; portable resume artifact chưa được triển khai. Export metadata hiện chưa có import tương ứng.

Mục tiêu chi phí và tốc độ trong tài liệu chưa được benchmark. Ước tính trong UI tính từ lúc core process khởi động, có cả thời gian chờ user; chưa gồm cold-pull hoặc API usage.

Kết quả kiểm thử ở [VALIDATION](VALIDATION.md); phần cần chuẩn bị trước khi thuê GPU ở [PRE_GPU_READINESS](PRE_GPU_READINESS.md). Smoke tổng hợp không mở quality gate hoặc tự bật các stage AI trong web.
