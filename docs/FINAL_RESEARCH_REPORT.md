# Báo cáo kỹ thuật VNLE — nghiên cứu và kiến trúc

Ngày 09/10/2026. **Đã hoàn tất reset có thể khôi phục và thiết kế nghiên cứu. Chưa xây sản phẩm VNLE hoặc xác nhận SLA.** Yêu cầu bổ sung đã được đưa vào kiến trúc: người dùng khoanh vùng text sub trước; vùng đó không được phát hiện chữ/OCR, chỉ phân tích phần hình còn lại.

## A. Kết quả reset

Repo đã xác nhận là D:\Video, remote Khenggg/cn2vi-autodub. Nhánh mới codex/vnle-research-reset; bản CN2VI HEAD7c7ed60 được giữ bằng nhánh lưu, bundle đã verify, bản sao 9.640 tệp đối chiếu SHA và bản gốc archive. Source, frontend, test, dependencies, CI và scripts cũ đã rời active tree. Không xóa SubAI hoặc tài sản ngoài repo. Cache khó truy cập giữ nguyên, giới hạn backup được công bố tại [báo cáo reset](REPOSITORY_RESET.md).

Bộ nghiên cứu gồm đúng 14 tài liệu00–13, thêm báo cáo reset và báo cáo này. Không có mã sản phẩm mới, installer, weights hoặc môi trường runtime đã nghiệm thu.

## B. Kết quả nghiên cứu nguồn

Khảo sát 10 ứng viên chính và 19 repo nguồn, gồm các native engine; sổ bằng chứng lưu 68 file snapshot với revision tại [13](13_RESEARCH_SOURCES.md). Đọc implementation liên quan, không chỉ README; không chạy tests hoặc model upstream.

Đề nghị tái dùng FFmpeg/ffprobe, libass, OpenCV, PyAV, ORT và RapidOCR qua adapter. VSE giúp học grouping/queues; Find-That-Text gần phạm vi VNLE hơn sản phẩm dubbing, có events/evidence/reuse. Không lấy mặc định sampling23frames, gap-only hoặc force CPU của nó làm tối ưu VNLE.

VisualTranslator giữ mask tĩnh giữa các lần OCR, sampling1Hz, force Paddle CPU và thiếu license rõ nên không copy. Manga repos giúp hiểu layout, glossary và quality gate, nhưng xử lý ảnh tĩnh và GPL/AGPL/weights cần audit riêng; hiện LEARN ONLY. CoTracker có license chính CC-BY-NC, không làm dependency thương mại mặc định. TransDETR có custom CUDA legacy và vấn đề release/license cần xác minh; SAM2 không OCR, speed benchmark A100 không nghiệm thu laptop.

Không suy license code cấp quyền cho mọi weights/font/dataset. GPL/AGPL vẫn cho phép thương mại khi tuân thủ nghĩa vụ; policy phân phối VNLE chưa chốt nên chưa nhúng source đó. Model card small ONNX chính thức ghi Apache-2.0 và registry có SHA là bằng chứng sơ bộ; chưa tải và verify artifact bytes. So sánh đầy đủ tại [01](01_OPEN_SOURCE_RESEARCH.md).

## C. Kiến trúc đề nghị

`Probe/preview → user subtitle exclusion → submit → PTS decode/change scan ngoài ROI → detector watchdog/candidates → một worker GPU det/rec crop batch → temporal events/revisions → rules/importance/uncertainty → glossary/TM/translator adapter → layout → native render/NVENC → full QC → SubAI`.

Mode A vẫn là yêu cầu. Mode B lưu metadata rồi dựng sau SubAI chỉ là lựa chọn thay thế, cần mapping thời gian/geometry và phê duyệt trước đổi. Mỗi event là đơn vị OCR, ngữ nghĩa, dịch và layout. Không ASR/TTS hoặc VLM từng frame.

Contracts giữ integer PTS/rational timebase, polygon transforms, shot/appearance/revision, raw observations/evidence và input/code/config/model/ROI hashes. NEEDS_REVIEW hiển thị trong báo cáo, không tính là đã Việt hóa thành công. Manual ROI loại ngay trước model và được dùng làm vùng tránh va chạm cho SubAI; không tự chọn hoặc nới ROI thay người dùng.

Chọn Python điều phối + thư viện native trước. Custom C++/pybind11 hoặc Cython chỉ thêm khi profiler chứng minh nút thắt. CPU libass + NVENC là baseline, không gọi đó là render toàn bộ bằng GPU; compositor GPU cần benchmark format/transfer/color. Chi tiết tại [02](02_ARCHITECTURE_COMPARISON.md), [03](03_ALGORITHM_RESEARCH.md), [04](04_OCR_TRACKING_RESEARCH.md), [05](05_SEMANTIC_DECISION_DESIGN.md), [06](06_RENDERING_ENGINE_RESEARCH.md) và [09](09_TECH_STACK_DECISIONS.md).

## D. Hiệu năng và giới hạn

Budget thiết kế: 520 giây công việc + 80 giây headroom, **không phải dự báo đã đo**. Phân bổ probe5, load15, scan85, det70, rec35, tracking10, semantics10, dịch50, layout5, render/encode200 và QC35 giây. Thời gian exclusive/critical path tách rõ, không cộng inference đã nằm trong OCR tổng lần nữa.

Watchdog4Hz tạo khoảng 2.400 detector calls/600 giây, đòi khoảng29,2ms/call để vừa budget70 giây; chưa có bằng chứng GTX1650Ti làm được. Render18.000frames/200 giây đòi90fps; full-decode QC35 giây đòi khoảng514fps. Đây là **throughput cần có**, không phải năng lực quan sát được. Event density, triggers, tiles, API và nhiệt độ có thể làm FAIL.

ROI loại sub giảm nhu cầu recognition, nhưng mask trên tensor toàn khung không tự giảm CNN latency theo diện tích. SLA600 giây trên1650Ti chưa xác nhận, rủi ro cao ở coverage của sự kiện ngắn và render/QC. 1080p khảo sát riêng, không downscale ngầm. Kế hoạch cold/warm,3/5repeats, sustained30phút, video thật/holdout và provider trace tại [07](07_PERFORMANCE_FEASIBILITY.md). Chất lượng và tốc độ đều bắt buộc.

## E. Lộ trình

Giả định một kỹ sư có kinh nghiệm và corpus/máy sẵn: feasibility3–5ngày; prototype thêm8–12; MVP thêm12–18; stable thêm15–25; perspective nâng cao thêm20–40ngày. Đây là effort estimate, không cam kết calendar; phụ thuộc dữ liệu, annotation, runtime và provider. Không xây web/microservices/installer trước proof.

Exit gates về coverage, không bỏ critical events, số liệu, timing, media, tài nguyên, giấy phép và tương thích nằm tại [10](10_PROJECT_ROADMAP.md) và [11](11_ACCEPTANCE_CRITERIA.md).

## F. Vấn đề chưa chốt

Corpus gold và thuật ngữ series; cấu hình CPU/RAM/driver/power benchmark; translator/provider/model và lựa chọn chi phí/quyền riêng tư; SubAI version/settings và compatibility thật; quyền phân phối từng artifact; policy rút gọn text chỉ tồn tại0,3 giây; proof1080p. Profile codec/resolution không giới hạn mật độ chữ, nên không thể hứa mọi clip720p đều đạt SLA. Scope manual ROI đã chốt, tọa độ cụ thể chờ từng job. Rủi ro tại [12](12_RISK_REGISTER.md).

## G. Việc đầu tiên sau khi duyệt

Prototype nhỏ trên video720p thật: preview/khoanh ROI sub, probe/PTS, decode/change scan ngoài ROI, dựng lớp chữ mẫu bằng engine native và kiểm đầu ra đầy đủ. Đo cold/warm/thermal floor trước; nếu đã gần600 giây thì sửa kiến trúc trước thêm OCR/AI. Sau đó đo detector-only, crop batch và recall0,3–0,5 giây để chọn model/cadence bằng bằng chứng. Không cần thuê GPU mạnh để chứng minh target laptop.

Nhiệm vụ hiện tại dừng ở nghiên cứu. Không gọi API, chạy inference/media benchmark hoặc xây đầy đủ VNLE. Static checks và backup verification không chứng minh runtime pipeline pass.
