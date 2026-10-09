# Visual Narrative Localization Engine — VNLE

Repo đã chuyển từ CN2VI autodub sang nghiên cứu engine **Việt hóa chữ quan trọng trong hình video**. SubAI tiếp tục xử lý lời thoại và lồng tiếng bên ngoài; không sửa ứng dụng đó.

**Trạng thái ngày 09/10/2026:** prototype discovery đã chạy 30 giây video thật trên Windows GTX 1650 Ti 4 GB trong127.453s, sau sửa DLL/cuDNN và ổn định tensor shapes. Cùng input/model/ROI,900frames/301OCRcandidates; nhanh hơn4.777lần so với lượt608.800s. Kết quả còn false positives/đọc sai/sự kiện trùng, cần xem lại. CI chưa chạy. Chưa có pipeline dịch/render VNLE hoàn chỉnh. SLA 600 giây cho video 600 giây **chưa đạt nghiệm thu**.

[Hướng dẫn bản mẫu và giới hạn đã kiểm tra](docs/PROTOTYPE.md). Xem giao diện local bằng `scripts/start-preview.ps1`; OCR chỉ bật trên máy thực hiện có dependencies/model manifest. Chế độ preview local không tải model hoặc upload video.

Bước đầu theo yêu cầu: **người dùng khoanh vùng text sub trước; VNLE không phát hiện chữ và không OCR vùng đó**. Phần còn lại được quét tìm bảng hệ thống, tên nhân vật, tin nhắn và manh mối. Mode A dựng chữ trước SubAI vẫn là luồng yêu cầu; Mode B chỉ là phương án so sánh.

## Tài liệu

[Báo cáo kỹ thuật A–G](docs/FINAL_RESEARCH_REPORT.md) · [Reset và khôi phục](docs/REPOSITORY_RESET.md)

- [Tầm nhìn](docs/00_PROJECT_VISION.md)
- [Mã nguồn mở](docs/01_OPEN_SOURCE_RESEARCH.md)
- [So sánh kiến trúc và contracts](docs/02_ARCHITECTURE_COMPARISON.md)
- [Thuật toán discovery/events](docs/03_ALGORITHM_RESEARCH.md)
- [OCR và tracking](docs/04_OCR_TRACKING_RESEARCH.md)
- [Importance và dịch](docs/05_SEMANTIC_DECISION_DESIGN.md)
- [Dựng hình native](docs/06_RENDERING_ENGINE_RESEARCH.md)
- [Tính khả thi hiệu năng](docs/07_PERFORMANCE_FEASIBILITY.md)
- [Tương thích SubAI](docs/08_EXTERNAL_TOOL_COMPATIBILITY.md)
- [ADR và stack](docs/09_TECH_STACK_DECISIONS.md)
- [Lộ trình](docs/10_PROJECT_ROADMAP.md)
- [Tiêu chí nghiệm thu](docs/11_ACCEPTANCE_CRITERIA.md)
- [Rủi ro](docs/12_RISK_REGISTER.md)
- [Nguồn và revision](docs/13_RESEARCH_SOURCES.md)

## Bản cũ

Git root `D:\Video`; remote giữ nguyên `https://github.com/Khenggg/cn2vi-autodub.git`. Nhánh mới `codex/vnle-research-reset`; nhánh gốc `codex/v2-smart-voiceover`, HEAD `7c7ed60`.

Kho lưu `D:\Video\.legacy-backup\cn2vi-20261009-152059`, nhánh `codex/backup-cn2vi-before-vnle-20261009-152059`. Có 9.640 tệp / 1.258.417.187 byte được đối chiếu SHA-256, bản gốc lưu riêng và Git bundle đã verify. Kho có thể chứa credentials/media, được Git bỏ qua, không upload. Cache và venv cũ giữ nguyên, không dùng làm dependency VNLE. Có hai lỗi truy cập trong inventory, nên không tuyên bố đã sao lưu hết mọi byte local. Không xóa dữ liệu ngoài repo.

Bước tiếp theo: chạy CI, kiểm tra prototype với video/ROI người dùng trên máy thực hiện, đánh giá events/timecode/recall/provider trace trước khi nối dịch và render. Các tài liệu nghiên cứu giữ nguyên snapshot lúc nghiên cứu; tình trạng triển khai mới được ghi trong `docs/PROTOTYPE.md`.
