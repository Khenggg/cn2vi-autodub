# VNLE development rules

Repo D:\Video đã chuyển từ CN2VI sang VNLE. User đã duyệt xây prototype discovery (preview/manual subtitle exclusion/OCR/events) sau nghiên cứu. Xem docs/PROTOTYPE.md cho tình trạng thật; chưa dịch/render hoàn chỉnh.

Ngày 09/10/2026 user đã chỉ định máy Windows local GTX 1650 Ti 4 GB là máy thực hiện VNLE, cho phép cài môi trường riêng, tải weights và chạy video ở đây. Quy định cloud của CN2VI cũ không áp dụng cho execution VNLE đã được chỉ định. Tests/builds vẫn dùng CI. Không mặc định credentials cloud cũ còn hiệu lực. Một run khóa config/model/code, không đổi giữa run hay fallback âm thầm.

Trước sửa code dùng codebase-memory-mcp: index nếu thiếu, get_architecture, search_graph, trace_path, đọc đúng file; detect_changes sau sửa. Graph CN2VI cũ không là kiến trúc VNLE.

User cho phép tự sửa lỗi và chạy lại trong giai đoạn prototype Windows này, báo nếu các cách sửa có căn cứ vẫn thất bại. Mỗi lượt vẫn khóa config/code/model và lưu báo cáo; chỉ sửa giữa các lượt, không đổi model/vùng khoanh hoặc fallback âm thầm.

SubAI là công cụ ngoài repo chỉ tham khảo hành vi. Không sửa SubAI, không copy mã dịch ngược, không sửa repo/tài sản ngoài root đã xác nhận.

Giữ private backups/cache/credentials/media ngoài Git. Không xóa tài sản chưa sao lưu xác minh; không reset/clean nhánh cũ. Đọc docs/REPOSITORY_RESET.md trước recovery.

Máy nghiệm thu Windows GTX1650Ti4GB, SSD, cắm điện. SLA VNLE <=600s cho600s MP4 H2648bit720p<=30FPS gồm cold load/decode/OCR/dịch/render/encode/QC, không tính SubAI. Không giảm resolution/bỏ event/giấu quality errors để đạt SLA; số chưa đo ghi chưa đo.

Algorithm-first, native FFmpeg/OpenCV/ORT/libass; AI OCR hoặc semantic uncertainty. Run khóa model/config/input/code, không fallback âm thầm. Custom C++ cần profiler justification.

Mode A render trước SubAI vẫn là yêu cầu; B chỉ so sánh, chưa đổi mặc định. Không claim compatibility trước actual test. Không mang dependency/API/model cũ sang mặc định.

Antigravity Gemini3.8FlashHigh chỉ cho việc hữu hạn cần thiết; primary đối chiếu nguồn và thiết kế. Không tự retry agent empty/failed report. Báo cáo tiếng Việt.

User khoanh subtitle exclusion trước discovery/OCR; không nhận diện trong ROI. Lưu transform/interval/hash; report EXCLUDED_BY_USER, không tự nới ROI.
