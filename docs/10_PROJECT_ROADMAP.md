# 10 — Lộ trình có điều kiện

Ước lượng effort, không cam kết lịch: một kỹ sư toàn thời gian có kinh nghiệm Python/CV/media, dùng native libs, người dùng cung cấp corpus có quyền sử dụng và giúp gán nhãn, có máy1650 Ti, không sửa SubAI. Một ngày6–8 giờ tập trung. Phụ thuộc dữ liệu/driver/provider có thể tăng effort; advanced Level3 không nằm MVP.

| Pha | Effort ước lượng | Reuse / phần mới | Nhiệm vụ và exit gate |
|---|---|---|---|
| P0: feasibility microbenchmark | 3–5 ngày | FFmpeg/PyAV/OpenCV/libass; mớiharness+provenance | Probe/PTS, scan và render/QC floor, cold / warm / nhiệt độ. Gate: đo thật+counters+quality; nếu floor gần/vượt600, revise trước thêmAI |
| P1: technical prototype | thêm8–12 ngày | RapidOCR/ORT/CV; mớicandidate/event state | Det/rec crop batch, watchdog sự kiện ngắn, boundary/shot/revision và annotatedreport. Gate: candidate recall+event/numeral/timing đủ, ngân sách có headroom trên corpus |
| P2: MVPLevel1+selected2 | thêm12–18 ngày | libass/FFmpeg/SQLite; mớirules/glossary/translationadapter/layout | VNLE Mode AE2E đầy đủ, collision, sổ lỗi/resume, ma trận tương thích SubAI. Gate: frozen720p10min corpus cold/warm<=600+quality+compatibility |
| P3: stableversion | thêm15–25 ngày | Matureengines; mớirecovery/packaging/QC/regressions | Windowscleanenv, longpaths, noisy/dense/movingcorpus, repeatedthermal, license/artifact manifest. Gate: allacceptance+không fallback ẩn/data loss |
| P4: perspective/occlusionLevel3 | thêm20–40 ngày | OpenCVwarp/tracking, advancedmodelsnếucần; mớimotioncompositor | Planarity/occlusionconfidence, temporalquality, per-eventcost. Gate: hardmotioncases+resourcebudget; nếu neuralneededlicense+modelbenchriêng |

Cộng lũy kế: prototype khoảng11–17 ngày; MVP23–35 ngày; stable38–60 ngày; Level3stable58–100 ngày. Đây là estimate dưới giả định trên, không SLA phát triển. Không tính chờ data/annotation/phê duyệt API trong effort nhưng có trong calendar. 1080p target chỉ thêm sau baseline720p pass; không tự tuyên bố hỗ trợ.

## Thứ tự xây sau khi duyệt

1. Media/provenance/benchmark harness tối thiểu, chưa UI hoàn chỉnh. Đo native render và full verification trước để biết thời gian còn choCV.
2. Gold corpus và temporal event contract. Đo sự kiện ngắns/phase/motion trên proposal, không viết nhiều thuật toán chưa có oracle.
3. OCR adapter/một worker GPU, batch/counters, provider trace, cache crop ổn định; không import toànmanga/copySubAI.
4. Rule templates/glossary/TM và translator được chọn sau quality/costscreening; semantic escalation không trở thànhdependencybắtbuộc.
5. Layout Level1, selectedpanelsLevel2, QC và ModeAcompatibility thật.
6. Packaging/resume/regression; customC++ chỉ sau attributionprofile, không vì cảm giác Pythonchậm.

## Giảm effort

Reuse FFmpeg demux/codec/mux, libass font+raster, OpenCVgeometry, ORTprovider, RapidOCRbatch. Học Find-That-Text event/evidence/report pattern; không dùng default cadence/gap filter. Manga layout/glossarypattern có giá trị nhưng code copyleft chưa chốt reuse. Hoãn web/cloud/multi-user/neuralrestoration/Level3; đủ CLI + report prototype trướcUI.

## Dependencies và cổng quyết định

P0 không cần translatorAI; P1 cần approvedsmallweights / license sau research, corpus+laptop; P2 cần provider/model+glossary+SubAI settings và output để kiểm; P3 cần Windowsfreshenv và licensesmanifest; P4 cần perspectivedataset/oracle chất lượng. Mỗi pha có failcriteria: misscritical, thaystat, CPU/GPUclaim sai, outputcorrupt, >600 hoặc unknownlicenses. Không nhảy gate bằng video dễ/synthetic hay đổi yêu cầu.

Nhiệm vụ hiện tại dừng ở nghiên cứu. Các harness/adapters/UI ở bảng là công việc tương lai, **chưa được viết hoặc chạy**.

## Bổ sung P0/P1

Preview tối thiểu và thao tác khoanh subtitle exclusion, validate geometry/time trước OCR; không cần dựng lại web CN2VI. Gate: pixels trong ROI không vào model, cache theo ROI hash và phạm vi loại trừ có report.
