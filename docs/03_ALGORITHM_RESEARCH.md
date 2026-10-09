# 03 — Algorithm-first discovery và temporal events

Ký hiệu F số frame, P pixel phân tích, K track đang sống, M observation/frame. Complexity dưới đây không phải số đo FPS.

| Phương pháp | Lợi thế | Lỗi thường gặp | Vai trò |
|---|---|---|---|
| OCR cố định 1 Hz | Ít inference | Event 0,3 giây nằm hoàn toàn giữa mẫu | Baseline đối chứng, không production coverage |
| Histogram/scene change | O(FP), native | Chữ đổi trong cùng cảnh không tạo cut | Shot boundary + trigger |
| Tile frame diff | O(FP), bắt thay đổi cục bộ | Pan/ánh sáng/nén; chữ nhỏ mất khi downscale | Scan mỗi frame + motion compensation |
| SSIM vùng | So cấu trúc thay vì pixel thuần | Cost cao hơn; toàn ảnh che thay đổi nhỏ | Patch chưa rõ, không full frame mỗi lúc |
| Edge/component/MSER | Proposal không weights | Texture/font nghệ thuật gây miss/false positive | Chỉ proposal, không cổng loại chữ tuyệt đối |
| Optical flow/template | Theo dõi bbox đã biết | Drift, blur, occlusion, chữ thay trên nền cũ | Forward/backward checks + refresh |
| Coarse-to-fine ML detector | OCR crop thay toàn cảnh | Cổng coarse bỏ chữ nhỏ, tile overhead | Giữ watchdog toàn khung; audit recall từng cổng |

Nguồn: OpenCV LK/homography, PySceneDetect, VSE và Find-That-Text tại [13](13_RESEARCH_SOURCES.md).

## Discovery đề nghị — giả thuyết

1. Decode tuần tự presentation PTS. H.264 cần frame tham chiếu; giảm OCR không xóa decode. Tránh seek theo mỗi mẫu.
2. Mỗi frame tính luma/edge thumbnail và diff tile; bù camera motion nếu đủ điểm nền đáng tin. Lưu change bbox/PTS/quality. Thumbnail chỉ proposal; crop OCR dùng source resolution.
3. Watchdog detector toàn khung khoảng **250 ms tối đa** trong profile thử, thêm cut và tile-change triggers. 4 Hz giúp có một mẫu thời gian trong event >=0,3 giây nếu cadence PTS hợp lệ; không bảo đảm detector/OCR thành công. Burst scan khi chữ typing/fading.
4. Lấy crop rõ theo sharpness/contrast/clipping. Một observation tốt ở event ngắn được giữ, không bắt buộc hai OCR confirmations.
5. Chữ nhỏ/low confidence chạy crop/tile gốc bổ sung; lưu resize/padding transforms và tile overlap. Không giảm resolution output hoặc ngầm chấp nhận mất chữ.
6. Cảnh pan gây overload: backpressure, ledger, re-decode interval nếu hết buffer. Không drop events để giữ SLA. Detector 4 Hz đã ~2.400 calls/600 giây chưa tính triggers/tiles; đây là rủi ro thật.

Watchdog thưa hơn + change scan chỉ thay sau calibration short-event recall. Frame diff không bảo đảm chữ rất nhỏ/low contrast; scene cut cũng không. Ngưỡng là tham số thử, chưa cấu hình production.

## Buffer và boundaries

Ring buffer theo byte/PTS, chỉ giữ thumbnail/crop và ít source frames. RGB720p ~2,76 MB/frame, RGB1080p ~6,22 MB/frame tính số học, chưa gồm RSS/overhead; 30 frame720p ~83 MB. Không buffer hàng nghìn ảnh.

Bracket start/end bằng mẫu trước/sau; refine patch bằng native matching trên frame lân cận theo PTS. Binary search chỉ khi appearance predicate đơn điệu. Nhấp nháy/typing/occlusion phải scan patch đầy đủ; không bisection mù. Boundary uncertainty luôn ghi. Re-decode tốn thời gian được tính vào SLA.

## Lifecycle và association

`CANDIDATE → OBSERVED → CONFIRMED → ACTIVE → OCCLUDED/PENDING_END → CLOSED`, nhánh NEEDS_REVIEW. Event ngắn có một crop tốt + change boundaries vẫn CONFIRMED.

Match bằng IoU sau warp, center distance, crop appearance, text similarity. Chi phí O(KM), Hungarian O(n³) chỉ cho cụm cạnh tranh. Không merge chỉ vì chữ giống: tách vị trí, shot, appearance và revision.

- HP `100 → 90`: typed value và patch change tạo revision, không fuzzy-merge thành text cũ.
- Scene cut đóng track, context xuyên shot tách khỏi bbox tracking.
- Text giống quay lại sau khoảng vắng tạo appearance mới; TM reuse không kéo interval.
- Homography RANSAC chỉ cho mặt phẳng, đủ inlier phân bố, kiểm reprojection/condition/orientation. Parallax/occlusion giảm confidence, đổi annotation an toàn hoặc review.
- Typing/scrolling tách revisions; không kéo lớp che/mask cũ qua disappearance.

## Phép thử thuật toán

Gán nhãn event0,3/0,5s ở nhiều phase so cadence, nhỏ/góc hình, bảng dày, thay một số/ký tự, cut/pan/blur/fade, text lặp. Đo candidate recall → detector recall → OCR correctness → association → localized recall. Mọi drop có reason ledger. Không chỉ đo số cue cuối hoặc model confidence. Dev/test không chung scene/series để tránh threshold overfit.

## Manual exclusion trước discovery

Bước 0: user khoanh subtitle ROI. Skip tiles/crops trong ROI; detector toàn khung blank vùng loại trừ trên bản phân tích, không sửa frame gốc. Chặn proposal ở biên mask giả. Không chỉ vứt kết quả sau OCR. Watchdog/candidate scan phần còn lại. Test rectangles/polygons/interval/rotation/SAR/biên; không nới vùng hoặc bỏ thêm pixel ngoài ROI. ROI thay invalidate cache tương ứng.
