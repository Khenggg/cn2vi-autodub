# 01 — Nghiên cứu mã nguồn mở

Nguồn được chốt revision và mở code liên quan, không chỉ README; snapshot nghiên cứu không phải release đã cài/thử. Không chạy tests/inference, tảiweights hoặc benchmarklaptop. Code license, weights, runtime, font và dataset là các đối tượng khác nhau. Nhận định giấy phép là ghi nhận điều kiện kỹ thuật, chưa clearance pháp lý cho phân phối.

Phân loại: REUSE DIRECTLY=library đủ điều kiện sơ bộ, vẫn audit artifacts; ADAPT=thay flow/adapter với tuân thủlicense; LEARN ONLY=họcpattern, chưa copy; REJECT=không chọn integration hiện tại. GPL/AGPL không cấm thương mại khi tuân thủ nghĩa vụ; hiện policy phân phối VNLE chưa chốt nên không nhúng code copyleft. Commercial/NC là rào khác.

## YaoFANGUK/video-subtitle-extractor

Repo: [GitHub](https://github.com/YaoFANGUK/video-subtitle-extractor). Revision `85746f7df5bf85978fd05f3ca6ce66e321a87a72`; commit 2026-04-09T15:27:37Z. License API: Apache-2.0. Có 0 tệp tên giống test trong tree khảo sát; không chạy test hoặc đo coverage. Commit gần đây không tự chứng minh ổn định.

**ADAPT** cho cách tổ chức extraction/grouping, không dùng nguyên GUI/pipeline. Mục tiêu hard-sub→SRT/TXT; source có FPS/detection/VideoSubFinder modes. `extract_frame_by_fps` seek theo CAP_PROP_POS_FRAMES mỗi mẫu; detection mode đọc tuần tự, cache kết quả và so text similarity. Consumer load OCR một lần, producer/consumer queue chuyển nhiệm vụ/frame; ROI selection dùng intersection/threshold. Có watermark filtering, temporal text/coordinate comparisons và subtitle generation.

Đưa sang VNLE: học queue, boundary và duplicate elimination; thay random seek bằng PTS decode, thêm motion/shot/revision và importance toàn khung **ngoài manual subtitle exclusion**. Không suy frame_no/fps đủ đúng VFR. Hardware: tùy Paddle/CPU/CUDA/DirectML và model; windows build workflows có nhưng không nghiệm thu laptop. Chưa thấy benchmark VNLE600 s / GTX 1650 Ti trong file khảo sát. Apache-2.0 code; weights/font/VideoSubFinder phụ thuộc audit riêng. Integration vừa; ROI/dialogue assumptions và detector-only paths không thay narrative semantics.

Evidence code: [backend/main.py](https://github.com/YaoFANGUK/video-subtitle-extractor/blob/85746f7df5bf85978fd05f3ca6ce66e321a87a72/backend/main.py), [backend/tools/subtitle_ocr.py](https://github.com/YaoFANGUK/video-subtitle-extractor/blob/85746f7df5bf85978fd05f3ca6ce66e321a87a72/backend/tools/subtitle_ocr.py), [backend/tools/subtitle_detect.py](https://github.com/YaoFANGUK/video-subtitle-extractor/blob/85746f7df5bf85978fd05f3ca6ce66e321a87a72/backend/tools/subtitle_detect.py).

## overcrash66/video-translator

Repo: [GitHub](https://github.com/overcrash66/video-translator). Revision `8e91ebcca2265a7e8a4e1c567da549852ef02638`; commit 2026-07-16T02:21:32Z. License API: không xác định bằng API. Có 41 tệp tên giống test trong tree khảo sát; không chạy test hoặc đo coverage. Commit gần đây không tự chứng minh ổn định.

**LEARN ONLY**, không copy mã khi chưa có giấy phép rõ. VisualTranslator là module visual-text trong sản phẩm dubbing rộng hơn. Source force PaddleOCR CPU (`use_gpu=False`), EasyOCR có GPU rồi retry CPU; default OCR interval1s, giảm OCR width, giữ last_mask/last_boxes/translations giữa các mẫu. TTL translation cache500 entries/4h và key(text,target_lang); OpenCVTelea+PIL glyph/fit, decode mọi frame bằng VideoCapture, encode mp4v VideoWriter. Cache và GC có giới hạn nhưng gọi chuyển PIL/paint/inpaint trên từng frame không là event-native renderer.

Missing: importance,0,3s recall, moving bbox/disappearance, panel revisions, PTS/audio/provenance/QC đầy đủ; translation cache thiếu series/context. HardwareCPU/GPU tùy engine nhưng không benchmarkGTX 1650 Ti SLA trong file đã đọc. Repo tree/API không có root license xác định; dependencies/model weights tách riêng chưa audit. Integration code cao/blocked licensing; học TTL và adapter, loại1Hz/static-mask/defaultfallback làm baselineproduction.

Evidence code: [src/translation/visual_translator.py](https://github.com/overcrash66/video-translator/blob/8e91ebcca2265a7e8a4e1c567da549852ef02638/src/translation/visual_translator.py), [src/translation/text_translator.py](https://github.com/overcrash66/video-translator/blob/8e91ebcca2265a7e8a4e1c567da549852ef02638/src/translation/text_translator.py).

## zyddnys/manga-image-translator

Repo: [GitHub](https://github.com/zyddnys/manga-image-translator). Revision `441d07c59a735c7db3db2e7bb8b07920afd8a9cc`; commit 2026-09-25T02:45:14Z. License API: GPL-3.0. Có 5 tệp tên giống test trong tree khảo sát; không chạy test hoặc đo coverage. Commit gần đây không tự chứng minh ổn định.

**LEARN ONLY** toàn sản phẩm và code module hiện tại, không port trực tiếp khi VNLE chưa chọn copyleft distribution policy. Mục tiêu ảnh manga tĩnh: detection/OCR adapters→line grouping→translation→mask/inpainting→glyph rendering. Source grouping dùng graph/components, distance/font constraints, tree-based splitting và reading-direction vote; rendererFreeType có font/glyph cache, size/vertical punctuation. Inference base có model mapping/download/hash/lifecycle; orchestrator có dictionaries và translation validation.

Pattern hữu ích: reading order, per-region masks, font metrics, glossary/context, adapter separation. Missing video PTS/shot/tracking/sự kiện ngắns và narrative priority. HardwarePython/PyTorch/CUDA tùy engine, models có thể lớn; chưa có E2E1650 Ti benchmark trong nguồn khảo sát. GPL-3.0 code, checkpoint/third-party models/font license không suy từ root; chưa xác nhận đủ weights dùng thương mại. Integration cao. Không giữ nhận định agent về milliseconds/page, zeroVRAM hoặc accuracy tuyệt đối vì không evidence. Model sinh inpainting không MVP.

Evidence code: [manga_translator/manga_translator.py](https://github.com/zyddnys/manga-image-translator/blob/441d07c59a735c7db3db2e7bb8b07920afd8a9cc/manga_translator/manga_translator.py), [manga_translator/textline_merge/__init__.py](https://github.com/zyddnys/manga-image-translator/blob/441d07c59a735c7db3db2e7bb8b07920afd8a9cc/manga_translator/textline_merge/__init__.py), [manga_translator/rendering/text_render.py](https://github.com/zyddnys/manga-image-translator/blob/441d07c59a735c7db3db2e7bb8b07920afd8a9cc/manga_translator/rendering/text_render.py), [manga_translator/utils/inference.py](https://github.com/zyddnys/manga-image-translator/blob/441d07c59a735c7db3db2e7bb8b07920afd8a9cc/manga_translator/utils/inference.py).

## mrdhnto/libre-manga-translator

Repo: [GitHub](https://github.com/mrdhnto/libre-manga-translator). Revision `19c4cf43fd7d3aad622ea4969f5852c954f21aaa`; commit 2026-10-05T07:22:48Z. License API: NOASSERTION. Có 0 tệp tên giống test trong tree khảo sát; không chạy test hoặc đo coverage. Commit gần đây không tự chứng minh ổn định.

**LEARN ONLY** implementation; algorithm patterns có thể nghiên cứu độc lập nhưng không gắn nhãn ADAPT rồi sao chép AGPLcode không tuân thủ. Browser extensionTypeScript/Svelte, ONNXWeb/WASM/WebGPU/offscreen components. Boxes grouping/order, OCR adapters, prompt series summary/dictionary/history, Canvas text-fit và manual correction. Inpaint ladder có fill/denoise/Telea và quality LaMa path, region scoring/decline; đây là ví dụ không bắt buộc model sinh cho mọi patch.

Validator có tùy chọn strict nhưng mặc định có thể pad/truncate số bản dịch: VNLE không nhận empty field như thành công. Context model tự tạo glossary cần version/approval để không khuếch đại lỗi. Missing video events/motion/PTS/native encode; WebGPU không phải CUDA ORT Windows. License file **AGPL-3.0-or-later từ stable**, betaMIT không là quyền cho HEAD. Code/weights licenses khác nhau, cần xác minh từng artifact/model card, chưa chốt toàn registry. Chưa thấy benchmark E2E1650 Ti trong nguồn khảo sát; không dùng estimate RAM trong docs như đo thực. Integration cao, học component isolation/glossary/quality rollback; không nhúng extension vào videoengine.

Evidence code: [LICENSE](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/LICENSE), [src/lib/detections/boxes.ts](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/src/lib/detections/boxes.ts), [src/lib/inpaint/ladder.ts](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/src/lib/inpaint/ladder.ts), [src/lib/prompts.ts](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/src/lib/prompts.ts), [src/lib/server/validator.ts](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/src/lib/server/validator.ts), [src/entrypoints/offscreen/main.ts](https://github.com/mrdhnto/libre-manga-translator/blob/19c4cf43fd7d3aad622ea4969f5852c954f21aaa/src/entrypoints/offscreen/main.ts).

## weijiawu/TransDETR

Repo: [GitHub](https://github.com/weijiawu/TransDETR). Revision `73355e6e81a26f501e974dd04724ec97ca46e7d3`; commit 2024-03-28T13:03:24Z. License API: không xác định bằng API. Có 3 tệp tên giống test trong tree khảo sát; không chạy test hoặc đo coverage. Commit gần đây không tự chứng minh ổn định.

**LEARN ONLY**, không dependency MVP. Video text spotting/tracking bằng backbone+transformer queries, rotatedROI features/recognition, tracker score/miss tolerance, memory bank và per-frame inference. Source RuntimeTrackerBase, inference_single_image và customRotatedROI CUDA phải xét cùng state/IDs. Ý tưởng identity/temporal memory hữu ích hơn dánbboxstatic, nhưng không cần mạng nặng trước classicalLK/homography.

README có kết quả ICDAR2015 tracking/spotting và training8V100; **traininghardware không là inference requirement hoặc FPS**. README cũ ghi chưa release recognition vì quy định công ty dù laterupdate/source có recognition paths: phải xác minh completeness/checkpoint, không coi turnkey. Linux/CUDA/GCC/PyTorchlegacy+binaryartifacts gây integrationWindows cao. README tuyên bố MIT, root license không xác định API; dependencyRotatedROI có license riêng, weights/datasets chưa đủ rights record. Missing VNLEsemantics/translation/render/QC và benchmark1650 Ti.

Evidence code: [models/TransDETR.py](https://github.com/weijiawu/TransDETR/blob/73355e6e81a26f501e974dd04724ec97ca46e7d3/models/TransDETR.py), [models/Rotated_ROIAlign/setup.py](https://github.com/weijiawu/TransDETR/blob/73355e6e81a26f501e974dd04724ec97ca46e7d3/models/Rotated_ROIAlign/setup.py), [README.md](https://github.com/weijiawu/TransDETR/blob/73355e6e81a26f501e974dd04724ec97ca46e7d3/README.md).

## facebookresearch/co-tracker

Repo: [GitHub](https://github.com/facebookresearch/co-tracker). Revision `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`; commit 2025-01-21T21:30:41Z. License API: NOASSERTION. Có 1 tệp tên giống test trong tree khảo sát; không chạy test hoặc đo coverage. Commit gần đây không tự chứng minh ổn định.

**REJECT** cho dependencyproduction thương mại mặc định; chỉ học phương pháp. Joint point tracking có online/offline predictor và temporal windows, không text detector/OCR/importance. Có visibility và tracks, có thể hỗ trợ motion khó nhưng thêm PyTorch/model state/context và GPU overhead. ClassicalOpenCV LK/template/homography ít integration hơn.

Source license chính CC-BY-NC4.0, README nêu một số third-party portionsMIT/Apache riêng; không suy phần đó cấp quyền toàntracker. Checkpoint/model cards cần audit riêng. Research tracking benchmarks không chứng minh latency/VRAM1650 Ti; không download weights. Integration cao, maintainedsnapshot commit01/2025 chỉ là ngày quan sát, không tuyên bố maintenance hiện mạnh. Commercial license cần nguồn cấp quyền khác trước dùng.

Evidence code: [LICENSE.md](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/LICENSE.md), [cotracker/predictor.py](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/predictor.py), [cotracker/models/core/cotracker/cotracker3_online.py](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/models/core/cotracker/cotracker3_online.py).

## facebookresearch/sam2

Repo: [GitHub](https://github.com/facebookresearch/sam2). Revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`; commit 2024-12-16T00:47:17Z. License API: Apache-2.0. Có 0 tệp tên giống test trong tree khảo sát; không chạy test hoặc đo coverage. Commit gần đây không tự chứng minh ổn định.

**LEARN ONLY**, tùy chọn advancedocclusion, không default. Video predictor quản lý inference state, object IDs/prompts, feature cache, memory/propagation; CPUoffload giảm GPUmemory nhưng có performance tradeoff. Segmentation không đọc/hiểu chữ, vẫn cần OCR/events/layout. WindowsREADME khuyến nghịWSL/customCUDA; khó packaging nativeWindows baseline.

README speed đoA100 vớiTorch2.5.1/CUDA12.4; không áp lên1650 Ti. Code+modelcheckpoints Apache-2.0 theo README/LICENSE, demofonts/third-party postprocess có terms riêng. Chưa cóGTX 1650 Ti E2E/VRAM proof. Integration cao; nếu Level3 cần objectocclusion và classical không đủ, benchmark theo event/shot riêng, không giữ toàn600 svideo/tất cảstates trongGPU.

Evidence code: [sam2/sam2_video_predictor.py](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py), [sam2/build_sam.py](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/build_sam.py), [README.md](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md), [LICENSE](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE).

## RapidAI/RapidOCR

Repo: [GitHub](https://github.com/RapidAI/RapidOCR). Revision `0700743fc8bfb3943cd8e40e84547f14c7d15dfc`; commit 2026-10-08T15:21:37Z. License API: Apache-2.0. Có 20 tệp tên giống test trong tree khảo sát; không chạy test hoặc đo coverage. Commit gần đây không tự chứng minh ổn định.

**REUSE DIRECTLY** engine qua adapter own, khóa version/artifacts sau spike. Det/cls/rec tách component, ORT và backend adapters, modelURL/hash registry. Recognizer sortwidth/batchpad/restoreorder; ORTsession load một lần và provider verification. Dùng crop batch và oneGPUowner thay đọc toànvideo mỗi frame; provider audit phải per-node, không chỉ availableCUDA.

HardwareCPU/CUDA/DirectML và backend khác tùy config; staticAPI hỗ trợ không chứng minh particularGPU. MODEL_LICENSES ghi code và listeddefaultweights Apache-2.0 riêng, SHA/URLs cho v6smalldet/rec+classifier; officialcards smallONNX corroborate, chưa download/verify artifact bytes. Model ngoài list audit riêng. Missing temporal discovery/tracking/importance/TM/render/QC. Chưa thấy VNLEbenchmark600 s trên GTX 1650 Ti trong nguồn đã đọc; actuallatency/VRAM unknown. Integration thấp–vừa, DLL/pre-post/copy/automaticfallback phải kiểm.

Evidence code: [python/rapidocr/main.py](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/rapidocr/main.py), [python/rapidocr/ch_ppocr_rec/main.py](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/rapidocr/ch_ppocr_rec/main.py), [python/rapidocr/inference_engine/onnxruntime/main.py](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/rapidocr/inference_engine/onnxruntime/main.py), [python/rapidocr/inference_engine/onnxruntime/provider_config.py](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/rapidocr/inference_engine/onnxruntime/provider_config.py), [python/MODEL_LICENSES.md](https://github.com/RapidAI/RapidOCR/blob/0700743fc8bfb3943cd8e40e84547f14c7d15dfc/python/MODEL_LICENSES.md).

## PaddlePaddle/PaddleOCR

Repo: [GitHub](https://github.com/PaddlePaddle/PaddleOCR). Revision `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`; commit 2026-09-16T03:30:53Z. License API: Apache-2.0. Có 109 tệp tên giống test trong tree khảo sát; không chạy test hoặc đo coverage. Commit gần đây không tự chứng minh ổn định.

**ADAPT** model/adapter và benchmarkđốichứng, không nhập cả suite. Toolsinfer tách det/rec/cls và crop/widthbatch; currentv6tiny/small/medium là các model/backbone/quality envelope khác. Chinese scene text và multilingual variants có accuracy benchmarks upstream, nhưng genre/fonts/smallvideochars cần corpus riêng. PaddleX/API3.x và ONNXexport kiểm operator/runtime thay assumeequivalence.

Apache-2.0 code; officialv6smallONNXcards Apache-2.0, artifact/dictionary/version/hash/license phải giữ. Docs v6 benchmark200ảnh includeI/O/pre-post/model, hardwareA100/V100/Xeon/M4; small ORT V100 0,53 s/ảnh không là1650 Ti latency hay detector-only. Quantization availability/quality chưa xác minh cho candidate, không tự claimINT8. Missing all videoevents/importance/layout/SLA. Integration vừa, model loading/backendcompatibility/tiles và uncertaintyerrorphải audit. Small là ứng viên, medium không automaticwinner.

Evidence code: [tools/infer/predict_system.py](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/tools/infer/predict_system.py), [tools/infer/predict_rec.py](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/tools/infer/predict_rec.py), [docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md).

## MichaelBrandonFalk/Find-That-Text

Repo: [GitHub](https://github.com/MichaelBrandonFalk/Find-That-Text). Revision `dde8a147e8b74604afff6278a94388ca1b6d2a0e`; commit 2026-10-01T05:49:09Z. License API: MIT. Có 14 tệp tên giống test trong tree khảo sát; không chạy test hoặc đo coverage. Commit gần đây không tự chứng minh ổn định.

**ADAPT**, ứng viên bổ sung gần phạm vi hơn video dubbing. Python modules phân decoder/sampling, OCR, tracking/matcher/relevance, evidence reports. Source decodePyAV tuần tự, frame.time; normalize+IoU/center/textsimilarity gom events; reuseframe theo local32pxchanges và refresh sau2reuses. Rawdetections/report không xóa hết lowrelevance nên hữu ích audit.

**Không dùng default**: frame_step 23 (~0,767s ở30FPS), gap-onlyfilter có thể bỏ plottext trong lời thoại. Relevance vocabularyen/es không hiểuChinese; engine source dùng PaddleTextDetection/TextRecognition và force deviceCPU dù modelv6small. Vì vậy không coi nó là VNLE tối ưu GPU. Missing Chineseimportance/glossary/translation/overlayrender/statrevisions/exactboundaries; không chứng minh recall critical.

MITcode, phụ thuộcruntime / models / fonts táchterms; model card ứng viên nhưPaddle cần audit. Snapshot10/2026 có Windows/Linux/Mac packaging trong repo, không làGTX 1650 Ti SLAproof; benchmarkE2Etarget chưa thấy. Integration vừa: học event/evidence và reuse, tự thiết kế manualexclusion/watchdog/revisions, không sao chép defaultdrop policies.

Evidence code: [src/find_that_text/scanner.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/scanner.py), [src/find_that_text/video/decoder.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/video/decoder.py), [src/find_that_text/video/sampling.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/video/sampling.py), [src/find_that_text/tracking/matcher.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/tracking/matcher.py), [src/find_that_text/ocr/reuse.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/ocr/reuse.py), [src/find_that_text/ocr/engine.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/ocr/engine.py), [src/find_that_text/tracking/relevance.py](https://github.com/MichaelBrandonFalk/Find-That-Text/blob/dde8a147e8b74604afff6278a94388ca1b6d2a0e/src/find_that_text/tracking/relevance.py).

## Native engines và mở rộng screening

FFmpeg/ffprobe,NVENC/NVDEC,libass,OpenCV,PyAV,ORT,PySceneDetect **REUSE DIRECTLY** với license/build/version audit; chi tiết chức năng, source và giới hạn tại[06](06_RENDERING_ENGINE_RESEARCH.md),[09](09_TECH_STACK_DECISIONS.md),[13](13_RESEARCH_SOURCES.md). pybind11/Cython **ADAPT chỉ khi hotspot đã đo**; Nuitka **REUSE DIRECTLY cho packaging**, không claimtăng tốc CUDA.

Tìm thêm RapidVideOCR,MMOCR,IOPaint và Find-That-Text. URLRapidAI/RapidVideOCR trả404 lúc kiểm, không kết luận tính năng/reuse từ snippet. MMOCR là toolboxOCR/understanding, không quyết định tốt hơn native nhẹ nếu chưa auditcheckpoint/Windows stack; chưa shortlisted. IOPaint tập trung inpainting ảnh, không giải temporalSLA nên không MVP. Find-That-Text được đưa shortlist và đọc source như trên; đây là mở rộng có evidence, không bám danh sách ban đầu.

## Kết luận tái dùng

Tốt nhất cho MVP: maturemedia/CV/ORT/libass + RapidOCRadapter; học VSE temporalgrouping và Find-That-Text evidence/events. Manga repos có layout/glossary/quality gate tốt để học, nhưng engine ảnh và copyleft / weights không chuyển nguyên sangvideo. Academictrackers không default 4 GB. Manual subtitle exclusion trước mọidiscovery/OCR là requirement mới và VNLE phải thêm, không giả upstream đã làm đúng yêu cầu đó.

Tests có tên trong tree không chứng minh coverage hoặc pass; số inventory là chỉ báo, có thể chứa utility/trainingtest. Chưa audit issue triage toàn repo hoặc kiểm mỗicheckpoint. Những giới hạn này cần nói rõ, không biến ngàycommit thành lời đảm bảo support.
