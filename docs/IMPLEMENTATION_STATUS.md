# CN2VI AutoDub tiến độ triển khai

Baseline: Implementation Guide v1.0 ngày 06/10/2026.

| Giai đoạn | Trạng thái hiện tại | Gate tiếp theo |
| --- | --- | --- |
| 0 Benchmark | Runner và CLI ASR/Bandit/TTS/inpaint; chưa có adapter model hoặc kết quả GPU | Bộ golden 10 phút, adapter thực, exact weights/runtime và kết quả chất lượng |
| 1 Core | FastAPI, React, SQLite, auth, upload/resume, Series queue, FFprobe, checkpoint, drain, SSE; Docker và CI được viết | Test local và build; kiểm tra image trên Linux |
| 2 Phase A | Segment/provider interfaces, window merge, duration fitting và budget rules đã có | ASR → align → translation → Bandit → TTS → mix → QC thật |
| 3 Review/B/C | ROI contract và API review gate | Player audio Việt, ROI canvas, OCR/TBE/LaMa/ProPainter, libass/NVENC |
| 4 Recovery | Metadata export và khôi phục preparation khi restart | Import workspace, portable resume zip/hash validation, retry từng segment, resource leases |
| 5 Cost/perf | Ước tính cost theo uptime core, admission VRAM sơ bộ | GPU/API metrics, real billing start, budget escalation enforcement |
| 6 Release | README, ADR, CI nền tảng | Batch 20×5 phút, benchmark 120 phút, signed image |

Phase 1 kết thúc Episode ở CHECKPOINTED với next_stage=ASR. Video nguồn phát trong UI vẫn có audio tiếng Trung. Không có video lồng tiếng Việt/final ở phiên bản này. Checkpoint JSON có checksum metadata; portable resume artifact chưa được triển khai. Export metadata hiện chưa có import tương ứng.

Mục tiêu chi phí và tốc độ trong tài liệu chưa được benchmark. Ước tính trong UI tính từ lúc core process khởi động, có cả thời gian chờ user; chưa gồm cold-pull hoặc API usage.

Kết quả kiểm thử cụ thể sẽ ghi vào `docs/VALIDATION.md` sau khi chạy kiểm tra.
