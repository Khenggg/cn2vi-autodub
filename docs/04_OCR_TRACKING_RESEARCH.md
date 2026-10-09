# 04 — OCR và tracking

## Lựa chọn có điều kiện

Prototype đề nghị **RapidOCR + ONNX Runtime**, PP-OCRv6 small det/rec là ứng viên đầu. So tiny và PP-OCRv5 mobile trên cùng corpus trước chốt; không medium mặc định. Chưa tải weights, chạy inference hoặc đo VRAM trong nhiệm vụ này.

FACT: RapidOCR recognizer sắp crops theo aspect ratio, padding theo crop rộng nhất trong batch, dùng `rec_batch_num`, trả đúng thứ tự ban đầu. ORT session load một lần, có provider/config/hash validation. Pipeline hoàn chỉnh không chứng minh detector batch nhiều full frames. Xem [13](13_RESEARCH_SOURCES.md).

| Ứng viên | Quyết định | Rủi ro cần kiểm |
|---|---|---|
| RapidOCR/ORT CUDA | REUSE DIRECTLY engine, adapter own | DLL/CUDA/cuDNN, CPU partition, contour/CTC preprocessing |
| PaddleOCR/PaddleX | ADAPT adapter, đối chứng model | Dependencies rộng, 2.x/3.x API, export ops |
| v6 small / tiny / v5 mobile | Benchmark riêng, khóa một cặp trong run | Accuracy chữ nhỏ/art, latency và VRAM máy4GB |
| v6 medium | Chưa chọn; chỉ nếu quality small thiếu và envelope cho phép | Không suy accuracy cao = E2E tốt |
| DirectML | Profile riêng Windows | Gói/runtime/operators khác, không gọi CUDA |
| TransDETR | LEARN ONLY | Custom Linux CUDA legacy, missing release/source/weights audit |
| CoTracker | REJECT production thương mại mặc định | CC-BY-NC, point tracking không OCR |
| SAM2 | LEARN ONLY advanced | State/video memory, A100 benchmark, không OCR |

PP-OCRv6 upstream đo 200 ảnh tính I/O+pre/post+inference; small ORT V100 0,53s/ảnh trong profile đó. Không phải detector-only720p, không quy đổi sang1650 Ti. Bảng quality upstream không nghiệm thu corpus tu luyện. Không có benchmark VNLE trên GTX 1650 Ti hoặc số VRAM thật ở đây.

RapidOCR `MODEL_LICENSES.md` ghi URL/SHA default det/rec small và classifier; official PaddlePaddle small det/rec ONNX model cards đánh Apache-2.0. Artifact ngoài danh mục audit riêng; conversion không xóa license. Model card revision và hash thực phải khóa lúc tải sau này.

## Tổ chức CPU/GPU

Một worker owns det/rec sessions, không session mỗi crop. Batch rec từ nhiều event, bucket width, thử4/8/16 với byte/VRAM cap. Đây là tham số khảo sát. Detector batching phải kiểm dynamic axes/postprocess chứ không mặc định API hỗ trợ. Angle classification chỉ khi hướng chưa rõ; không loại chữ dọc/nghiêng bằng giả định font ngang.

Memory = weights + CUDA context + arenas + tensors + output + decode surfaces. ORT `gpu_mem_limit` không là cap tổng process VRAM. Không ép FP16/INT8 khi export/backend chưa hỗ trợ và chưa qua quality corpus. Chưa xác nhận quantized v6 phù hợp1650 Ti.

Audit ba lớp: available providers, session providers, **ORT per-node profile/trace thật**. Danh sách CUDA/CPU không chứng minh mọi node GPU. Ghi transfers và CPU resize/contours/CTC riêng. GPU3D thấp trong Task Manager không chứng minh CPU inference.

Pin ORT/CUDA/cuDNN theo compatibility matrix; Windows VC runtime/PATH/DLL kiểm trên env sạch. Không dùng runtime hoặc weights thuộc SubAI. CUDA và DirectML env riêng; không fallback model/provider âm thầm trong run. Unsupported backend báo rõ, không claim SLA.

## Crops và cache

Polygon → OpenCV rectify → recognition, giữ transform về source. Dòng dài split có reading order khi cần. Font panel khác subtitle; không loại chữ chỉ vì chiều cao lệch median. Confidence missing=null, không suy cao.

OCR cache key gồm aligned crop appearance/hash, model/preprocess config, shot/track/revision. Translation cache tách riêng. Chỉ cache nền không đủ khi số đổi; value crop invalidate. Fresh watchdog và disappearance đóng event/overlay. OCR gốc không bị sửa thành glossary term thiếu bằng chứng.

Tracking baseline: sparse LK/template/geometry trước, feature trên viền bảng/nền đủ texture; glyph thay số không phải anchor ổn định. Affine/homography theo điều kiện mặt phẳng, kiểm drift/inlier/reprojection; dense/neural tracking chỉ sau failed classical quality gate và đo tài nguyên.

VSE/Find-That-Text gợi ý grouping/cache; VNLE cần thêm shot, revision, PTS, Chinese importance và renderer. Không copy code thiếu license hoặc dùng sampling 23 frames để claim short-event recall.

## ROI trước model

Không đưa pixels chữ trong active subtitle ROI vào detector/recognizer. Mask/skip ngay input, không OCR rồi loại output. Crop/detector cache thêm exclusion hash/version. Mask full-frame vẫn có thể gần cùng latency CNN; lợi ích ít proposals/recognition phải đo, không quy đổi phần trăm diện tích thành speedup.
