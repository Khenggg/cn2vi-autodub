# 07 — Tính khả thi SLA600 giây

**Kết luận hiện tại: chưa chứng minh đạt SLA; rủi ro cao ở coverage detector và render/QC.** Không có benchmark VNLE GTX1650 Ti trong nhiệm vụ nghiên cứu này. Không mượn thời gian SubAI ROI, cloud hoặc V100 để nghiệm thu.

## Ngân sách thiết kế, không phải dự báo đã đo

Baseline tuần tự hai pass, không lấy lợi ích overlap trước khi đo. Mỗi hàng là exclusive budget cap để thiết kế phép thử, không latency dự đoán. Detection tách recognition; render bao gồm decode pass2/composite/upload/encode/mux.

| Công việc | BUDGET giây | Phải đo |
|---|---:|---|
| Probe/provenance | 5 | Metadata/hash và disk I/O |
| Process/model initialization | 15 | DLL/context/weights/session load cold |
| Pass1 decode + frame selection/change scan | 85 | 18.000frames, conversion/thumbnail/transfers |
| Text detection | 70 | Watchdog/triggers/tiles/pre-post |
| Crop recognition | 35 | Batch normalize/inference/CTC/reorder |
| Tracking/events/boundary refinement | 10 | Flow/association/redecode bổ sung |
| Importance/semantic decisions | 10 | Rules + selected escalation |
| Translation | 50 | Batch/cache/validation/network nếu có |
| Layout/assets | 5 | Font load/shaping/layout/raster |
| Pass2 render/encode/mux | 200 | Toàn18.000frames, alpha/ASS/NVENC |
| Output verification | 35 | Full decode/media/PTS/audio và overlay audit |
| Tổng cap công việc | **520** | Chưa đo, không claim dưới600 |
| Headroom | **80** | Scheduler/I/O/thermal/API variance |
| Hard maximum | **600** | Đo actual wall bắt buộc |

Ngân sách520 là **mục tiêu phân bổ**, không trung bình hay khoảng tin cậy. Nếu load/QC/tiled OCR vượt cap phải sửa thiết kế/benchmark lại, không bỏ stage. Tải weights cài đặt không nằm warm run; cold process/session/page-cache ghi riêng và startup đều tính.

## Điều kiện số học kiểm khả thi

600 s30FPS có18.000frames. Render200s đòi90fps aggregate (=3× realtime); scan85s đòi~212fps; QC35s full decode đòi~514fps. Các số này là **throughput cần có**, không năng lực đã quan sát. Full output decode QC có thể tự trở thành bottleneck và không được thay sample QC rồi giữ cùng chứng nhận.

Watchdog 4 Hz có~2.400 detector calls chưa thêm triggers/tiles. Budget70s đòi average~29,2ms/call gồm overhead. Nếu2400calls thực tế100ms thì detection240s, riêng detection vượt cap170s và tổngcap thành690s: FAIL nếu các stage khác không giảm đủ. Nếu50ms thì120s, tổng570s còn30s headroom; dense panels hoặc độ trễ API có thể làm FAIL. Đây là **sensitivity scenarios**, không benchmark.

Formula khảo sát: `T = probe + load + scan + Ndet*tdet + Ncrop*tcrop_effective + track/refine + semantic + translation + layout + render/mux + verify + overhead`. Khi crop batching, effective per-crop là tổng batch wall/crops, không inference kernel chia tùy ý. `Ndet,Ncrop,Nevents,Nsemantic,Nrequests` là counters thật trong report, không cố định giả để đẹp số.

PP-OCRv6 upstream200ảnh có smallORTV1000,53s/ảnh E2E. Không dùng con số này như detector-only hay extrapolate1650 Ti; nó nhắc rằng full OCR ở mỗi watchdog có thể quá đắt. Cần tách detector/rec, reuse stable crops và crop proposals. Bảng nguồn ở[13](13_RESEARCH_SOURCES.md).

## Overlap và constraints

Có thể overlap CPU scan với OCR GPU, event aggregation, và translation mạng; render sau đủ metadata. Report exclusive spans+critical path, không cộng kernel nằm trong OCR tổng; cũng không lấy max(stage) khi có dependency khiến overlap không thật. Một GPU worker chống duplicate models/VRAM. Bounded queues, backpressure, no candidate dropping.

VRAM 4 GB bao gồm Windows/display/driver, OCR sessions, tensors và video surfaces. Đề xuất cap tổng process dedicatedVRAM<=3,2GiB và RSS<=8GiB là **acceptance resource budgets**, chưa đo và phải đối chiếu free RAM thực. Không nhận cả video vào GPU. Không VLM resident mặc định.

1080p có2,25× pixel720p (số học), không nhất thiết2,25×wall. Convert/filter/transfer/OCR tiling tăng; benchmark riêng, output giữ1080p. Thermal laptop và CPU/GPU tranh điện có thể giảm sustained speed dù cold test nhanh.

## Benchmark tái lập, chưa thực thi

1. Khóa code/settings/weights/dictionary/font hashes, versions/FFmpeg configure, driver/GPU/CPU/RAM/SSD/power/temperature. Media SHA và PTS index, codec/profile/bitdepth/SAR/color.
2. Corpus thật được phép dùng: ít chữ, nhiều chữ/bảng dày, text0,3–0,5s nhiều phase, moving/perspective, blur/small/art, overlay chữ cùng dialogue, cut/occlusion/reappearance/stat updates. 720p 30 FPS và1080p 30 FPS, VFR có test riêng.
3. Đo từng route decode-to-null/change-scan, detector-only, crop rec, native render20/200/1000events, output full decode. Kiểm quality trước discard candidate/model.
4. Cold start ít nhất3 lần process mới; warm ít nhất5 lần từng profile. Cold page-cache nói rõ cách kiểm, không tự gọi process fresh là disk cold. Chạy sustained back-to-back>=30 phút để xem throttling, cắm điện, cùng settings. Ghi median/min/max/P95 và mọi lầnFAIL, không lấy best-only.
5. ORT profile node/provider; Windows ETW/WDDM hoặc tool tương ứng cho decode/compute/copy/encode, CPU sampling, RSS/VRAM/queue bytes theo timeline. Bắt actual worker PID, stage markers; không đo nhầm TTS/SubAI process.
6. Full E2E từ request đến verified output, không prewarm hidden/cache parse miễn phí. Replay cache là profile riêng, không thay cold acceptance. API network p95/timeouts/usage/cost ghi nếu dùng; chưa có provider thì UNKNOWN.
7. Gold annotation bởi hai người và adjudication, dev/test split theo series. Đo quality gates[11](11_ACCEPTANCE_CRITERIA.md) đồng thời; synth edgecase không thay video thật.

## Nếu không đạt

Đầu tiên đo render/no-OCR floor và discovery quality. Tối ưu copies/batching/bounded caching/ROIproposal/native primitives; customC++ chỉ khi profiler. Nếu vẫn vượt600 hoặc miss critical, báo bottleneck và so lựa chọn hardware/profile/workflow; thay bất kỳ yêu cầu nào phải được người dùng duyệt. Không giảm watchdog / resize / output / chất lượng âm thầm. Mạng API không thể cung cấp hard latency guarantee nếu provider không có SLA; cần corpus và failure policy, hoặc offline translator được benchmark riêng.

## Chi phí và benchmark ROI

Khoanh ROI là input configuration trước submit, như chọn tệp. Từ submit, load/validate/map ROI tính trong probe/SLA. Chỉnh ROI sau submit tạo job/provenance mới, không giấu rework. Mọi đối chứng giữ cùng ROI, report area/time coverage excluded. Không suy mask sub giảm CNN toàn khung latency theo diện tích.
