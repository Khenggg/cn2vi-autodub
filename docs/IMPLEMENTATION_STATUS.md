# CN2VI V1: current development status

Updated 2026-10-07. The user architecture and experimental policy in `docs/requirements` supersede the legacy stack below.

- V1 web orchestration, isolated workers and pinned cloud install plan are implemented for Bandit v2, Community-1, FireRedASR2-AED/Punc, PP-OCRv6 Medium, DeepSeek, IndexTTS2 Vietnamese and ProPainter.
- Runs freeze code/config/source/reference provenance, record stage timings/resource measurements, retain partial audio/picture after failed dependencies, and save playable output plus JSON/Markdown reports. Missing native timing is not fabricated.
- Fresh setup installs system audio/video libraries, Python/Node, isolated pinned requirements and model assets, verifies imports and NVENC, and distinguishes environment readiness from model/video quality.
- GitHub CI covers unit/media boundary tests, Ruff, frontend build and Docker health. Actual Linux GPU installation and all new-model inference/quality/cost measurements remain pending.
- Initial execution is serial across episodes. Cross-episode CPU/API overlap and complete portable workspace import remain product work. Character identities and relationships without evidence remain tentative. Original audio/image is preserved when a required branch is unavailable.

See [fresh cloud installation](V1_CLOUD_INSTALL.md) for prerequisites and the first GPU experiment. Legacy reports below are historical and do not verify V1 models.

---

# CN2VI AutoDub tiến độ triển khai

Cập nhật từ main `01d9e7c` và nhánh pipeline hardening ngày 06/10/2026. Trạng thái triển khai nguồn và bằng chứng chạy là hai phần riêng; kết quả cloud mới được ghi trong báo cáo validation của nhánh.

| Phần | Đã triển khai | Còn cần xác nhận / hoàn thiện |
| --- | --- | --- |
| Core | Auth, Series/glossary, upload tiếp tục, hàng đợi, SSE, media probing, tải artifact | Portable import workspace và resource leases |
| Lồng tiếng | Kim_Vocal_2 → ASR trên vocals → align trên vocals → dịch → TTS → fitting → mix dùng lại stem → QC → mux đã nối web | Chất lượng trên phim thật, từ/ngôn ngữ sai và giọng nhân vật |
| Kiểm duyệt | Sửa lời Việt, giữ âm gốc, thời gian từng từ, giọng/nhân vật thủ công; chặn fitting lệch quá 20% | Diarization và tự chọn giọng chưa có trong thay đổi này |
| Recovery | Checkpoint schema 2, hash nguồn/artifact, tái dùng stage hợp lệ, giữ sửa tay, drain/restart/cancel | Mix Python dừng ở ranh giới stage, chưa ngắt giữa phép hòa trộn; portable resume khác host cần xác minh |
| Vietsub/ROI | Chọn chế độ upload, vẽ ROI, SRT/libass, OCR/LaMa trong vùng và worker vision riêng | Chất lượng temporal/residual subtitle cần đánh giá; timeline không hỗ trợ sẽ dừng review; chưa có NVENC tối ưu |
| Cài cloud | Native Ubuntu 22.04/24.04, managed Python, đúng CLI fetch/verify, restore có staging và bảo toàn target cũ | Restore venv khác host phải kiểm tra interpreter/dependencies/CUDA; Docker daemon không có trong container hiện tại |
| Benchmark / chi phí | Locks, adapters, corpus/suite và ước tính UI có sẵn | Không dùng test giả lập thay quality gate, RTF phim thật hay hóa đơn; chưa benchmark batch 20×5 phút/120 phút |

Luồng đầy đủ và chính sách dừng/duyệt được mô tả trong [PIPELINE_RUNTIME](PIPELINE_RUNTIME.md). Các báo cáo [VALIDATION](VALIDATION.md), [CPU_SMOKE](CPU_SMOKE.md) và preflight trước đây là bằng chứng lịch sử cho phạm vi ghi trong từng báo cáo.

Thay đổi này giữ lựa chọn model/provider hiện có. Không chạy API dịch thật trong bước cài đặt hoặc kiểm tra cấu trúc. Technical completion của artifact không chứng minh bản dịch đúng, giọng tự nhiên hoặc chất lượng xóa phụ đề.
