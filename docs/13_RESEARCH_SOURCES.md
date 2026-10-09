# 13 — Nguồn và sổ bằng chứng

Ngày truy cập09/10/2026. SHA dưới đây do GitHubAPI trả tại thời điểm nghiên cứu; source snapshot dùng để đối chiếu, không phải dependency version đã cài hoặc đã qua test. File được đọc ở các phần liên quan; không tuyên bố audit toàn bộ repo. Không tải model weights có chủ ý hoặc inference.

## YaoFANGUK/video-subtitle-extractor

[Repo](https://github.com/YaoFANGUK/video-subtitle-extractor) · [revision](https://github.com/YaoFANGUK/video-subtitle-extractor/commit/85746f7df5bf85978fd05f3ca6ce66e321a87a72) · commit `2026-04-09T15:27:37Z`.

Phần kiểm tra: FPS seek, sequential det, cached text/geometry, producer/consumer and ROI selection.

- [backend/main.py](https://github.com/YaoFANGUK/video-subtitle-extractor/blob/85746f7df5bf85978fd05f3ca6ce66e321a87a72/backend/main.py)
- [backend/tools/subtitle_ocr.py](https://github.com/YaoFANGUK/video-subtitle-extractor/blob/85746f7df5bf85978fd05f3ca6ce66e321a87a72/backend/tools/subtitle_ocr.py)
- [backend/tools/subtitle_detect.py](https://github.com/YaoFANGUK/video-subtitle-extractor/blob/85746f7df5bf85978fd05f3ca6ce66e321a87a72/backend/tools/subtitle_detect.py)

## overcrash66/video-translator

[Repo](https://github.com/overcrash66/video-translator) · [revision](https://github.com/overcrash66/video-translator/commit/8e91ebcca2265a7e8a4e1c567da549852ef02638) · commit `2026-07-16T02:21:32Z`.

Phần kiểm tra: Force CPU Paddle, GPU/CPU retries EasyOCR,1Hz samples, TTL500/4h, stale mask,Telea/PIL/mp4v.

- [src/translation/visual_translator.py](https://github.com/overcrash66/video-translator/blob/8e91ebcca2265a7e8a4e1c567da549852ef02638/src/translation/visual_translator.py)
- [src/translation/text_translator.py](https://github.com/overcrash66/video-translator/blob/8e91ebcca2265a7e8a4e1c567da549852ef02638/src/translation/text_translator.py)

## zyddnys/manga-image-translator

[Repo](https://github.com/zyddnys/manga-image-translator) · [revision](https://github.com/zyddnys/manga-image-translator/commit/441d07c59a735c7db3db2e7bb8b07920afd8a9cc) · commit `2026-09-25T02:45:14Z`.

Phần kiểm tra: Line graph grouping/direction, glyph cache, model lifecycle and GPL.

- [manga_translator/manga_translator.py](https://github.com/zyddnys/manga-image-translator/blob/441d07c59a735c7db3db2e7bb8b07920afd8a9cc/manga_translator/manga_translator.py)
- [manga_translator/textline_merge/__init__.py](https://github.com/zyddnys/manga-image-translator/blob/441d07c59a735c7db3db2e7bb8b07920afd8a9cc/manga_translator/textline_merge/__init__.py)
- [manga_translator/rendering/text_render.py](https://github.com/zyddnys/manga-image-translator/blob/441d07c59a735c7db3db2e7bb8b07920afd8a9cc/manga_translator/rendering/text_render.py)
- [manga_translator/utils/inference.py](https://github.com/zyddnys/manga-image-translator/blob/441d07c59a735c7db3db2e7bb8b07920afd8a9cc/manga_translator/utils/inference.py)
- [LICENSE](https://github.com/zyddnys/manga-image-translator/blob/441d07c59a735c7db3db2e7bb8b07920afd8a9cc/LICENSE)

## mrdhnto/libre-manga-translator

[Repo](https://github.com/mrdhnto/libre-manga-translator) · [revision](https://github.com/mrdhnto/libre-manga-translator/commit/19c4cf43fd7d3aad622ea4969f5852c954f21aaa) · commit `2026-10-05T07:22:48Z`.

Phần kiểm tra: AGPL stable license, fill/denoise/Telea ladder, context dictionary, validator pad/truncate.

- [LICENSE](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/LICENSE)
- [src/lib/detections/boxes.ts](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/src/lib/detections/boxes.ts)
- [src/lib/inpaint/ladder.ts](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/src/lib/inpaint/ladder.ts)
- [src/lib/prompts.ts](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/src/lib/prompts.ts)
- [src/lib/server/validator.ts](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/src/lib/server/validator.ts)
- [src/entrypoints/offscreen/main.ts](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/src/entrypoints/offscreen/main.ts)

## weijiawu/TransDETR

[Repo](https://github.com/weijiawu/TransDETR) · [revision](https://github.com/weijiawu/TransDETR/commit/73355e6e81a26f501e974dd04724ec97ca46e7d3) · commit `2024-03-28T13:03:24Z`.

Phần kiểm tra: Transformer track queries/memory, RuntimeTrackerBase, Linux custom ROI CUDA; README release/license ambiguity.

- [README.md](https://github.com/weijiawu/TransDETR/blob/73355e6e81a26f501e974dd04724ec97ca46e7d3/README.md)
- [models/TransDETR.py](https://github.com/weijiawu/TransDETR/blob/73355e6e81a26f501e974dd04724ec97ca46e7d3/models/TransDETR.py)
- [models/Rotated_ROIAlign/setup.py](https://github.com/weijiawu/TransDETR/blob/73355e6e81a26f501e974dd04724ec97ca46e7d3/models/Rotated_ROIAlign/setup.py)

## facebookresearch/co-tracker

[Repo](https://github.com/facebookresearch/co-tracker) · [revision](https://github.com/facebookresearch/co-tracker/commit/82e02e8029753ad4ef13cf06be7f4fc5facdda4d) · commit `2025-01-21T21:30:41Z`.

Phần kiểm tra: Online/offline point tracking and primary CC-BY-NC license.

- [LICENSE.md](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/LICENSE.md)
- [README.md](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/README.md)
- [cotracker/predictor.py](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/predictor.py)
- [cotracker/models/core/cotracker/cotracker3_online.py](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/models/core/cotracker/cotracker3_online.py)

## facebookresearch/sam2

[Repo](https://github.com/facebookresearch/sam2) · [revision](https://github.com/facebookresearch/sam2/commit/2b90b9f5ceec907a1c18123530e92e794ad901a4) · commit `2024-12-16T00:47:17Z`.

Phần kiểm tra: Memory/state/CPU offload/propagation; README benchmark A100, Apache checkpoints.

- [LICENSE](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE)
- [README.md](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md)
- [sam2/sam2_video_predictor.py](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py)
- [sam2/build_sam.py](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/build_sam.py)

## RapidAI/RapidOCR

[Repo](https://github.com/RapidAI/RapidOCR) · [revision](https://github.com/RapidAI/RapidOCR/commit/0700743fc8bfb3943cd8e40e84547f14c7d15dfc) · commit `2026-10-08T15:21:37Z`.

Phần kiểm tra: Aspect-sort crop batching, ORT session/provider validation, model license and hash registry.

- [python/MODEL_LICENSES.md](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/MODEL_LICENSES.md)
- [python/pyproject.toml](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/pyproject.toml)
- [python/rapidocr/main.py](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/rapidocr/main.py)
- [python/rapidocr/ch_ppocr_rec/main.py](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/rapidocr/ch_ppocr_rec/main.py)
- [python/rapidocr/inference_engine/onnxruntime/main.py](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/rapidocr/inference_engine/onnxruntime/main.py)
- [python/rapidocr/inference_engine/onnxruntime/provider_config.py](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/rapidocr/inference_engine/onnxruntime/provider_config.py)

## PaddlePaddle/PaddleOCR

[Repo](https://github.com/PaddlePaddle/PaddleOCR) · [revision](https://github.com/PaddlePaddle/PaddleOCR/commit/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf) · commit `2026-09-16T03:30:53Z`.

Phần kiểm tra: Det/rec orchestration, crop batches, upstream200-image accuracy and latency table.

- [docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)
- [tools/infer/predict_system.py](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/tools/infer/predict_system.py)
- [tools/infer/predict_rec.py](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/tools/infer/predict_rec.py)

## MichaelBrandonFalk/Find-That-Text

[Repo](https://github.com/MichaelBrandonFalk/Find-That-Text) · [revision](https://github.com/MichaelBrandonFalk/Find-That-Text/commit/dde8a147e8b74604afff6278a94388ca1b6d2a0e) · commit `2026-10-01T05:49:09Z`.

Phần kiểm tra: PTS decode, frame23 sampling, geometry/text matcher, tile reuse refresh, CPU OCR, en/es relevance.

- [src/find_that_text/scanner.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/scanner.py)
- [src/find_that_text/video/decoder.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/video/decoder.py)
- [src/find_that_text/video/sampling.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/video/sampling.py)
- [src/find_that_text/tracking/matcher.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/tracking/matcher.py)
- [src/find_that_text/tracking/relevance.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/tracking/relevance.py)
- [src/find_that_text/ocr/reuse.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/ocr/reuse.py)
- [src/find_that_text/ocr/engine.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/ocr/engine.py)

## FFmpeg/FFmpeg

[Repo](https://github.com/FFmpeg/FFmpeg) · [revision](https://github.com/FFmpeg/FFmpeg/commit/c766e0de78b0239a36e49e50573e2297a7b500b4) · commit `2026-10-09T02:43:37Z`.

Phần kiểm tra: CPU ASS mask blend and CUDA overlay format constraints.

- [libavfilter/vf_subtitles.c](https://github.com/FFmpeg/FFmpeg/blob/c766e0de78b0239a36e49e50573e2297a7b500b4/libavfilter/vf_subtitles.c)
- [libavfilter/vf_overlay.c](https://github.com/FFmpeg/FFmpeg/blob/c766e0de78b0239a36e49e50573e2297a7b500b4/libavfilter/vf_overlay.c)
- [libavfilter/vf_overlay_cuda.c](https://github.com/FFmpeg/FFmpeg/blob/c766e0de78b0239a36e49e50573e2297a7b500b4/libavfilter/vf_overlay_cuda.c)
- [LICENSE.md](https://github.com/FFmpeg/FFmpeg/blob/c766e0de78b0239a36e49e50573e2297a7b500b4/LICENSE.md)

## libass/libass

[Repo](https://github.com/libass/libass) · [revision](https://github.com/libass/libass/commit/f61db567e6593df3470e91594bcd4ad2d0473aff) · commit `2026-09-17T00:00:00Z`.

Phần kiểm tra: ASS_Image, glyph/bitmap/composite caches and render_frame.

- [libass/ass_render.c](https://github.com/libass/libass/blob/f61db567e6593df3470e91594bcd4ad2d0473aff/libass/ass_render.c)
- [libass/ass.h](https://github.com/libass/libass/blob/f61db567e6593df3470e91594bcd4ad2d0473aff/libass/ass.h)
- [COPYING](https://github.com/libass/libass/blob/f61db567e6593df3470e91594bcd4ad2d0473aff/COPYING)

## opencv/opencv

[Repo](https://github.com/opencv/opencv) · [revision](https://github.com/opencv/opencv/commit/67ede46c6513d6bca02e92820df34e943ea357d6) · commit `2026-10-08T11:13:57Z`.

Phần kiểm tra: Sparse LK, homography/RANSAC and Python wrapper PyAllowThreads.

- [modules/video/src/lkpyramid.cpp](https://github.com/opencv/opencv/blob/67ede46c6513d6bca02e92820df34e943ea357d6/modules/video/src/lkpyramid.cpp)
- [modules/calib3d/src/fundam.cpp](https://github.com/opencv/opencv/blob/67ede46c6513d6bca02e92820df34e943ea357d6/modules/calib3d/src/fundam.cpp)
- [modules/python/src2/cv2_util.hpp](https://github.com/opencv/opencv/blob/67ede46c6513d6bca02e92820df34e943ea357d6/modules/python/src2/cv2_util.hpp)
- [LICENSE](https://github.com/opencv/opencv/blob/67ede46c6513d6bca02e92820df34e943ea357d6/LICENSE)

## PyAV-Org/PyAV

[Repo](https://github.com/PyAV-Org/PyAV) · [revision](https://github.com/PyAV-Org/PyAV/commit/52e6691c8221a53ac1f1bf111f282cefa7d568e1) · commit `2026-10-03T01:32:24Z`.

Phần kiểm tra: Frame/codec PTS and receive_frame/packet inside cython.nogil.

- [av/video/frame.py](https://github.com/PyAV-Org/PyAV/blob/52e6691c8221a53ac1f1bf111f282cefa7d568e1/av/video/frame.py)
- [av/codec/context.py](https://github.com/PyAV-Org/PyAV/blob/52e6691c8221a53ac1f1bf111f282cefa7d568e1/av/codec/context.py)
- [LICENSE.txt](https://github.com/PyAV-Org/PyAV/blob/52e6691c8221a53ac1f1bf111f282cefa7d568e1/LICENSE.txt)

## microsoft/onnxruntime

[Repo](https://github.com/microsoft/onnxruntime) · [revision](https://github.com/microsoft/onnxruntime/commit/f289d7a87cd1c6c5431748a2a1c14620cd89c03f) · commit `2026-10-09T08:25:47Z`.

Phần kiểm tra: C++ Session/Value/IOBinding API; runtime licensing.

- [include/onnxruntime/core/session/onnxruntime_cxx_api.h](https://github.com/microsoft/onnxruntime/blob/f289d7a87cd1c6c5431748a2a1c14620cd89c03f/include/onnxruntime/core/session/onnxruntime_cxx_api.h)
- [LICENSE](https://github.com/microsoft/onnxruntime/blob/f289d7a87cd1c6c5431748a2a1c14620cd89c03f/LICENSE)

## Breakthrough/PySceneDetect

[Repo](https://github.com/Breakthrough/PySceneDetect) · [revision](https://github.com/Breakthrough/PySceneDetect/commit/81c414cb4b706e58648f98efd381024790b1565f) · commit `2026-09-21T04:43:46Z`.

Phần kiểm tra: Content/adaptive score windows and frame buffer length.

- [scenedetect/detectors/content_detector.py](https://github.com/Breakthrough/PySceneDetect/blob/81c414cb4b706e58648f98efd381024790b1565f/scenedetect/detectors/content_detector.py)
- [scenedetect/detectors/adaptive_detector.py](https://github.com/Breakthrough/PySceneDetect/blob/81c414cb4b706e58648f98efd381024790b1565f/scenedetect/detectors/adaptive_detector.py)
- [LICENSE](https://github.com/Breakthrough/PySceneDetect/blob/81c414cb4b706e58648f98efd381024790b1565f/LICENSE)

## pybind/pybind11

[Repo](https://github.com/pybind/pybind11) · [revision](https://github.com/pybind/pybind11/commit/bb605110db023ff4a64259630ecd1902a9a917de) · commit `2026-10-09T04:10:12Z`.

Phần kiểm tra: GIL ownership and scoped release headers.

- [include/pybind11/gil.h](https://github.com/pybind/pybind11/blob/bb605110db023ff4a64259630ecd1902a9a917de/include/pybind11/gil.h)
- [LICENSE](https://github.com/pybind/pybind11/blob/bb605110db023ff4a64259630ecd1902a9a917de/LICENSE)

## cython/cython

[Repo](https://github.com/cython/cython) · [revision](https://github.com/cython/cython/commit/7648be017e1aa5ec2c31cb5c9cf1530f09ed02d5) · commit `2026-10-08T18:45:34Z`.

Phần kiểm tra: Typed/nogil boundary, not automatic parallel inference.

- [docs/src/userguide/nogil.rst](https://github.com/cython/cython/blob/7648be017e1aa5ec2c31cb5c9cf1530f09ed02d5/docs/src/userguide/nogil.rst)
- [LICENSE.txt](https://github.com/cython/cython/blob/7648be017e1aa5ec2c31cb5c9cf1530f09ed02d5/LICENSE.txt)

## Nuitka/Nuitka

[Repo](https://github.com/Nuitka/Nuitka) · [revision](https://github.com/Nuitka/Nuitka/commit/30b5b77fe236417925b072218ff6f3006c286da3) · commit `2026-10-08T14:15:38Z`.

Phần kiểm tra: Compiler/standalone distribution scope, no external CUDA speed proof.

- [README.rst](https://github.com/Nuitka/Nuitka/blob/30b5b77fe236417925b072218ff6f3006c286da3/README.rst)
- [LICENSE.txt](https://github.com/Nuitka/Nuitka/blob/30b5b77fe236417925b072218ff6f3006c286da3/LICENSE.txt)

## Tài liệu chính thức và giới hạn

- [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html): ASS/timeline/overlay; capabilities cần kiểm binary build.
- [FFmpeg license/build obligations](https://ffmpeg.org/legal.html): LGPL/GPL theo thành phần, nghĩa vụ redistribution.
- [NVENC application note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html): engine encoder riêng graphics/CUDA; benchmark hardware khác không chuyển thẳng1650 Ti.
- [NVDEC guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html) và [GPU codec matrix](https://developer.nvidia.com/video-encode-decode-support-matrix): xác minh exactGPU/profile/driver trước chọn codec route.
- [ORT CUDA matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html): CUDA/cuDNN compatibility, memory limit/runtime libraries.
- [ORT DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html): device selection/Windows/constraints; sustained engineering, WinML direction.
- [OpenCV optical flow](https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html): LK/classical tracking, không benchmark VNLE.
- [PySceneDetect detectors](https://www.scenedetect.com/docs/latest/api/detectors.html): API0.7.1 lúc truy cập, source SHA khóa riêng.
- [PyAV frames](https://pyav.org/docs/develop/api/frame.html): PTS/time_base; site docs hiện trả9.x cũ hơn source snapshot, không dùng label stable đó để chọn wheel mới.
- [pybind11 GIL](https://pybind11.readthedocs.io/en/stable/advanced/misc.html): explicit release/acquire, no implicit speed gain.
- [Nuitka manual](https://nuitka.net/user-documentation/user-manual.html): standalone/data/DLL packaging, không evidence tăng tốc CUDA.
- [PaddlePaddle small det ONNX card](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx) và [small rec ONNX card](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx): metadataApache-2.0 quan sát; card main mutable, model commit/hash phải khóa khi tải sau này.
- [Noto fonts](https://github.com/notofonts/latin-greek-cyrillic): font candidate, chọn chính artifact và license lúc đóng gói, chưa tải font.
- [MMOCR](https://github.com/open-mmlab/mmocr), [IOPaint](https://github.com/Sanster/IOPaint): mở rộng screening, chưa audit chi tiết và không shortlisted dependency.

## Phân biệt bằng chứng

FACT là code/documentation đọc được. Upstream benchmark là upstream, chưa reproduced; hardware/input khác và chưa chứng minh quality genre. BUDGET/SENSITIVITY là số học thiết kế, không latency/VRAM measured. HYPOTHESIS là pipeline đề xuất cần phép thử. Unknown checkpoint/license/cost vẫn UNKNOWN.

Một lượt Antigravity Gemini3.8FlashHigh nghiên cứu2manga repos, run2026-10-09T082520.488761Z-c909ccbb. Artifact thật không rỗng được đọc; primary đối chiếu source và license. Loại claim không có evidence về milliseconds/page, zeroVRAM, accuracy tuyệt đối và diễn giải AGPL quá rộng. Báo cáo agent không là chứng nhận tốc độ/chất lượng hoặc quyền thương mại từng model. Không tự retry agent.

Metadata và source snapshots riêng tại `.research-cache/`, hashes ở `.cache/vnle-source-evidence.json`, đều Gitignored. Không copy implementation upstream vào source sản phẩm. Source text/licensing notices nằm cache chỉ phục vụ nghiên cứu; không đưa weights/binaries/cache vào repository mới.
