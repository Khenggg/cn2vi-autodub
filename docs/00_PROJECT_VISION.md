# 00 — Tầm nhìn VNLE

Ngày nghiên cứu: 09/10/2026. **Đây là thiết kế, chưa có sản phẩm VNLE và chưa xác nhận SLA.** Quy ước: FACT = thấy trong nguồn; JUDGMENT = quyết định kỹ thuật; HYPOTHESIS = cần thử; BUDGET = giới hạn thiết kế, không phải số đo.

## Mục tiêu và phần không làm

Việt hóa chữ cần để hiểu truyện: tên/định danh nhân vật, cấp tu luyện, bảng HP/MP/EXP, nhiệm vụ, kỹ năng, vật phẩm, tin nhắn, thư, địa điểm/thời gian và manh mối. Target gồm phim ngắn Trung Quốc, hoạt hình AI/manhua, hệ thống, xuyên không, tái sinh, tận thế. Chữ có thể ở bất kỳ vị trí, tồn tại 0,3–0,5 giây, nghiêng, chuyển động hoặc thay số trên cùng bảng.

Không làm ASR, dịch lời thoại, TTS, phụ đề hội thoại, tách âm, lip-sync hay thay soundtrack. SubAI tiếp tục xử lý các việc đó và **không được sửa** trong nhiệm vụ mới. Không mang mã dịch ngược của SubAI vào sản phẩm. Không mặc định dùng inpainting sinh ảnh hoặc VLM từng frame.

SubAI là tham khảo về trải nghiệm, timecode và dùng thư viện native, không là bằng chứng VNLE đủ nhanh: OCR toàn khung, nội dung hệ thống và quyết định importance khác extraction subtitle ROI.

## Hai luồng

Mode A giữ là luồng yêu cầu: `Original → user subtitle exclusion → discovery → temporal events → importance → translate → layout → render/verify → SubAI → final`.

Mode B chỉ để so sánh: `Original → analyze/store localization metadata → SubAI → time/geometry mapping → render/verify → final`. Không tự chuyển mặc định sang B. Tương thích cả hai chưa thử với SubAI; B cũng có thể sai nếu công cụ cắt/tăng tốc/crop video.

## Máy nghiệm thu và đồng hồ

Windows x64, GTX 1650 Ti Laptop thường 4 GB, SSD nội bộ, cắm điện. Ghi CPU/RAM/driver/power/thermal thực trong benchmark. Profile bắt buộc: MP4 H.264 8-bit, 1280×720, dài 600 giây, tối đa 30 FPS. 1080p 30 FPS là target khảo sát riêng.

SLA VNLE <=600 giây: từ nhận job đến media đóng/mux xong và QC tự động hoàn tất. Bao gồm startup/load, probe, decode/scan, detection/OCR, tracking/events, semantics, dịch (kể cả mạng), layout, render/encode và kiểm tra. Không tính SubAI. Cài đặt/tải weights báo riêng; không tạo cache phân tích trước run rồi giấu chi phí. Cold và warm đều phải đo.

Không đổi output resolution, bỏ event, thu chữ không đọc được hoặc giảm chất lượng âm thầm để đạt deadline. Metadata codec/resolution không giới hạn mật độ chữ; **không thể bảo đảm mọi video tùy ý** chỉ dựa profile này. Corpus nghiệm thu phải có bảng dày, short events, motion; không âm thầm loại clip khó.

## Thành công

Thời gian, chất lượng, tương thích và license là bốn cổng độc lập. NEEDS_REVIEW có trong báo cáo, không tính là đã Việt hóa thành công. Người dùng duyệt kiến trúc trước xây sản phẩm; nhiệm vụ này dừng ở tài liệu. Nguồn kỹ thuật tại [13](13_RESEARCH_SOURCES.md), quality gate tại [11](11_ACCEPTANCE_CRITERIA.md).

## Người dùng khoanh vùng phụ đề — yêu cầu bổ sung

Trước submit job, người dùng xem video gốc và khoanh một hoặc nhiều vùng phụ đề. VNLE **không phát hiện chữ và không OCR** trong các vùng này; quét chữ quan trọng ở phần còn lại. Không tự đoán bottom band thay người dùng. ROI mặc định toàn video, có thể theo PTS interval nếu vị trí thay đổi. Preview phần bị loại; không tự nới ROI. Lưu rect/polygon, transform display→source (rotation/SAR), active interval, version/hash. Chữ trong ROI là EXCLUDED_BY_USER, không phải omission bị giấu. Đây là scope override của yêu cầu toàn khung trước đó.
