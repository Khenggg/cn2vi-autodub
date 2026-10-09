# 08 — Tương thích công cụ ngoài SubAI

**SubAI không được sửa và không nằm trong repo. Chưa thử tương thích VNLE.** ModeA vẫn là yêu cầu, ModeB là alternative để người dùng chọn sau evidence.

| Tiêu chí | A: render trước SubAI | B: analyze trước, render sau |
|---|---|---|
| Time | Overlay gắn nguồn nhưng SubAI có thể đổi timeline | Phải map nguồn→output, không suy duration giống=PTS giống |
| OCR | Chữ Việt có thể bị SubAI xem như subtitle/logo | VNLE OCR bản gốc; không đọc lại chữ Việt để tạo events |
| Layout | SubAI có thể che/crop/đè nhãn | Biết output geometry/subtitles trước placement |
| Chất lượng | VNLE encode rồi SubAI encode: generation loss | Vẫn có2encode nếu SubAI output rồi VNLE render; không claim chỉ1 |
| Operations | Một bước VNLE trước tool | Hai pha + metadata/provenance + mapping/QC |
| SLA VNLE | Tất cả VNLE trước SubAI <=600 | Cộng analysis trước + mapping/render/QC sau, không reset đồng hồ để né SLA; thời gian SubAI loại riêng |
| Failure | Overlay biến dạng phải phát hiện ở final | Mapping thiếu confidence phải review/fail closed |

Không có API/private hooks của SubAI trong thiết kế. Không vượt license/auth công cụ ngoài; input/output media và settings người dùng cung cấp là interface.

## Risk trên ModeA

VNLE nhãn ở bottom band bị che; OCR hybrid tưởng text Việt là cue Trung; logo remover/delogo ăn cả nhãn; output resize/crop/reframe làm out-of-bounds; subtitles Việt chồng bảng; external tốc độ khác1× gây thời lượng đọc khác; reencode làm chữ nhỏ mờ. Chỉ dành safe zones chưa đủ bảo đảm. Không gọi trình độ SubAI nhanh là compatibility đã qua.

## Contract ModeB

Metadata khóa input SHA/PTS/canvas/rotation/SAR và output media SHA. `MediaMapping` có segment time mapping và geometry affine/homography cùng uncertainty/evidence/version. Nếu chỉ resize/letterbox biết rõ, map geometry bằng chính transform được xác minh. Crop mất region báo event occluded/unrenderable; không overlay ở chỗ đoán.

Dùng frame fingerprints/local features+audio alignment tùy chọn để kiểm match nhiều anchors đầu/giữa/cuối/cuts, không chỉ duration. Affine time `t_out=a*t_in+b` chỉ khi no cuts/speed constant đã xác minh. Cuts/reorder/nonlinear speed cần piecewise monotonic mapping; map interval nửa mở và keyframes, split tại cut. Mapping ambiguity/shot thiếu: NEEDS_REVIEW, không dựng bbox sai. Không gọi duration equal là bằng chứng không lệch.

## Ma trận thử, sau khi xây prototype được duyệt

Ghi SubAI version/hash và config frozen; sử dụng cùng clip có nhãn ở top/bottom/center, panel, motion0,3s và text Việt có dấu. A/B cùng input, điều kiện phụ đề on/off, che sub/logo on/off, giữ720p, resize1080p, crop/letterbox, tốc độ1×/khác1× (chỉ profile bổ sung, không cùng SLA). Không đổi SubAI để kết quả đẹp.

Kiểm trước/sau: frame/PTS/media duration/audio, coverage overlay theo expected intervals, glyph / contrast / khả năng đọc, bbox/occlusion và subtitle collision. Output final của SubAI cũng cần QC; runtime QC này được ghi rõ bên ngoài VNLE trước-tool SLA củaA, không giấu trong tuyên bố final quality. ModeB QC final là VNLE stage và tính budget.

Cổng: không mất/che event critical, không serious collisions, mapping error theo[11](11_ACCEPTANCE_CRITERIA.md). Một config PASS không đại diện mọi setting/version. NếuA fails trong config được yêu cầu, báo lỗi và soB; không tự thay luồng. Khả năng đọc event0,3s dài vẫn là vấn đề ngay cả mapping chính xác.

## Manual subtitle ROI

Người dùng khoanh trên nguồn trước analysis; Mode A tránh overlay vào đó. Mode B map ROI theo output geometry thật, kiểm subtitle output có thể ở vị trí mới. ROI nguồn không chứng minh tương thích SubAI. Không nhận diện chữ trong ROI để đối chiếu.
