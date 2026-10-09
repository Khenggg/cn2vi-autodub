# 06 — Engine dựng hình và chất lượng thị giác

## Ba cấp dựng

| Cấp | Cách làm | MVP | Giới hạn |
|---|---|---|---|
| 1 Floating annotations | Nhãn Việt gần chữ gốc, nền alpha nhỏ, ASS/libass | Ưu tiên | Không thay sạch chữ gốc; phải tránh nhân vật/dialogue |
| 2 UI-aware overlay | Panel key/value, layout nhóm, nền phẳng/alpha tile cache | Chọn panel ổn định | Không tái dựng cả UI tùy ý; số cập nhật đúng revision |
| 3 Perspective replacement | Rectify, text texture, homography warp, occlusion | Sau MVP | Planarity, tracking drift, alpha/color và occlusion khó; không default SAM2 |

JUDGMENT: Level1 + Level2 chọn lọc là đường nhanh nhất để có giá trị. Không gọi annotation là xóa/thay chữ hoàn hảo. Original vẫn là evidence; không inpaint từng frame chỉ để đẹp.

## Engine, source và license

FFmpeg `vf_subtitles.c` gọi `ass_render_frame` rồi blend mask vào frame trong filter CPU. NVENC ở cuối **không làm ASS thành GPU**. libass `ass_render.c` có font/glyph/bitmap/composite cache; nên compile một ASS track thay nhiều lượt filter chữ. libass ISC; FFmpeg mặc định LGPL-2.1-or-later nhưng build bật thành phần GPL sẽ đổi nghĩa vụ; `--enable-nonfree` không được mặc định phân phối. Audit đúng binary/config/dependencies, không chỉ tên FFmpeg. Nguồn pinned tại [13](13_RESEARCH_SOURCES.md).

ASS dùng PlayResX/Y, `pos`, `move`, `clip`, transforms/fades cho nhãn/đường đơn giản; thời gian ASS centisecond phải round về source PTS với uncertainty. Không giả ASS giải mọi homography/occlusion. Vietnamese shaping/glyph/diacritics cần font thật; glyph tofu/bbox chữ có dấu là QC failure. Font chọn loại có giấy phép rõ, ví dụ Noto, lưu font hash/license riêng.

PIL/FreeType có thể đo font/raster **một lần mỗi layout/revision**, không BGR↔RGB+font load trên18.000 frame. libass hoặc cached RGBA tiles tận dụng raster native. Không reimplement glyph shaping. Level2 group labels giữ field links, wrap theo font metrics; binary search font size chỉ trong min/max đã duyệt, không thu xuống chữ vô nghĩa.

## CPU/GPU routes cần đối chứng

1. Baseline: CPU decode → CPU libass/overlay → upload → NVENC. Ít transfers vòng lại, triển khai đơn giản; nghẽn CPU là rủi ro cần đo.
2. NVDEC → download → CPU ASS → upload → NVENC. Có thể decode nhanh nhưng transfer làm chậm; chỉ chọn nếu E2E thắng route1.
3. NVDEC → GPU compositor → NVENC. Đề nghị cấp cao hơn khi CPU blending là bottleneck. FFmpeg `overlay_cuda` yêu cầu CUDA surfaces và format hạn chế; source snapshot main NV12/YUV420P, overlay thêm YUVA420P. **Không nhận RGBA tile/PIL tùy ý rồi gọi zero-copy**. Phải kiểm alpha/chroma/color và upload assets/driver/build.

NVENC/NVDEC là engine chức năng riêng, không suy tốc độ từ phần trăm GPU tổng. Không có route nào được đo cho VNLE trên GTX 1650 Ti. Đừng chọn full C++/CUDA compositor trước thấy CPU route không đạt.

## Layout và collisions

Planner xếp bbox/polygon text source, dialogue-safe region, panel, khuôn mặt/chi tiết quan trọng được cung cấp hoặc phát hiện tùy chọn. MVP dùng geometry+safe zones; face model không tự được thêm chỉ vì cần tránh va chạm. Giải bằng candidate anchors, weighted overlap penalty, viewport clipping, leader lines và thời gian overlap.

Layout cache theo text+style/font+canvas+panel revision; vị trí theo keyframes. Đừng đổi font/anchor mỗi frame tạo jitter. Chữ motion có smoothing giới hạn error và confidence; thất bại tracking đổi annotation/review có báo cáo, không dán bbox cũ.

Chưa biết subtitle band SubAI tương lai: ModeA dành safe zone dự kiến vẫn phải test. Source Chinese bên dưới và nhãn Việt gần đó có thể collide; che nền nhỏ chỉ khi policy/region đủ rõ. Không che toàn chiều ngang vì bbox chung lớn.

## Bảo toàn media và QC

Giữ nguồn720p/1080p, fps/PTS/SAR/rotation/color, không upscale mặc định. Audio stream-copy khi container hỗ trợ; không EQ/mix. Color route alpha premultiplied/straight, limited/full range, BT.709 và chroma420 kiểm bằng test patterns sau khi được phép implementation.

Level3 dùng homography+texture polygon cho mặt phẳng, clipping/occlusion masks; planar fill/Telea chỉ vùng nhỏ nền đơn giản có quality gate, không dự kiến neural inpainting. Chất lượng temporal và chữ đọc được xét cùng runtime. Nếu nguyên bản đã che khuất chữ quan trọng, không hallucinate nội dung từ tracking.

## Vùng dành cho SubAI

Manual subtitle exclusion ROI là vùng tránh overlay mặc định của Mode A. Không che/raster chữ VNLE trong vùng đó. Không có vị trí đọc được thì collision/review, không dán cưỡng ép.
