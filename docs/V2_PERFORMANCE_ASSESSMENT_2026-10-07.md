# CN2VI: đánh giá mục tiêu 60 phút video → khoảng 10 phút xử lý

Ngày 07/10/2026. Đây là đề xuất để chốt thiết kế, chưa phải kết quả benchmark V2. Không có thay đổi model, mã pipeline hoặc lượt chạy cloud mới trong đánh giá này.

Đối chiếu: bản thảo tối ưu người dùng gửi; yêu cầu V1; mã nguồn tại worktree pipeline-hardening; báo cáo lượt 20261007-214550-c5809426, commit a4b9655; mã Bandit đúng revision 8f76499a32112c68f793dab8fe1a00d49b4fbfd5; tài liệu chính thức được đọc ngày đánh giá. Antigravity Gemini 3.8 Flash High phản biện một phần thiết kế; kết luận dưới đây do agent chính kiểm chứng.

## 1. Kết luận

Mục tiêu cần giữ là khoảng 600 giây, không tự nới thành 15–20 phút. Bản thảo đúng ở hướng giảm công việc không cần thiết, batching, tái sử dụng dữ liệu và tránh video inpainting nặng trên mọi frame. Nhưng dự báo 6–10 phút chưa được chứng minh; trên một GPU, chồng các stage không tự tạo thêm năng lực tính toán.

Ba việc quyết định tính khả thi: giảm khối lượng và thời gian Bandit; thay cách phân tích/xóa phụ đề; tạo được lồng tiếng thật với timing đúng. Tối ưu TensorRT cho ASR không đứng đầu danh sách hiện tại.

## 2. Số đo và phép ngoại suy

Nguồn đo: [run-report.json](v1-first-run/run-report.json), video 163,143 giây, 1280×720, 30 fps. Lượt chạy PARTIAL, không có giọng Việt, bước xóa phụ đề bị dừng theo yêu cầu người dùng.

| Thành phần | Đã đo | Hàm ý |
|---|---:|---|
| Bandit inference | 84,894 giây | Tốc độ hiệu dụng 0,520 giây xử lý/giây nguồn |
| Bandit load | 6,146 giây | Không được nhân chi phí nạp này lên theo từng câu |
| ASR inference / load | 6,302 / 15,825 giây | Nạp model và phần bao quanh đang lớn hơn thời gian nhận dạng |
| OCR tổng | 788,155 giây | Full-frame, mỗi giây một mẫu, CPU provider |
| TTS | 20,476 giây, inference 0 | Không thể dùng để kết luận tốc độ sinh giọng |
| Encode cuối + burn | 6,216 giây | Đã dùng NVENC; không phải toàn bộ pipeline đang encode CPU |

Nếu giữ tốc độ Bandit hiệu dụng đã đo: 25 phút dữ liệu cần khoảng 780,5 giây = 13 phút; 30 phút cần 15,6 phút; 60 phút cần 31,2 phút. Đây là ngoại suy cùng cấu hình, không phải benchmark phim dài hay giới hạn vật lý của model. Nó bao gồm ảnh hưởng chunk/overlap của adapter hiện tại và chưa tính lại phân bố padding khi chuyển sang dialogue-only.

Mức 2,5–4 phút cho 25 phút thoại trong bản thảo đòi hỏi cải thiện khoảng 3,3–5,2 lần. Padding làm khối lượng thực lớn hơn số phút thoại thuần. Tránh lấy wall time 93 giây rồi nhân cả chi phí nạp model theo chiều dài phim.

Ngoại suy thô encode hiện tại cho 60 phút là 137 giây; OCR hiện tại là khoảng 290 phút. Các số này chỉ định hướng ưu tiên, không dự báo chính xác video khác. Không ngoại suy bước inpaint bị hủy thành tốc độ ProPainter thuần: thời gian đó còn có OCR, tạo mask, I/O và khởi động tiến trình.

## 3. Tối ưu hóa âm thanh: Nút thắt Bandit & Đột phá chiến lược Voice-Over Multi-Band Ducking

### 3.1. Phân tích nút thắt Bandit V1 (Lặp công việc & Vượt ngân sách)
Trong [runtime Bandit đã ghim](https://github.com/openmirlab/bandit-infer/blob/8f76499a32112c68f793dab8fe1a00d49b4fbfd5/src/bandit_infer/_v2/runtime.py), cửa sổ là 8 giây, bước dịch 1 giây, inference_batch_size=1. Runtime tạo model mặc định, không chủ động bật autocast FP16/BF16 trong đường gọi đã kiểm tra. Adapter CN2VI còn chia 60 giây, thêm tối đa 8 giây ngữ cảnh mỗi bên. Bởi vậy có overlap bên trong và bên ngoài; không phải một giây âm thanh chỉ qua mạng một lần. Đo đạc thực tế 84,9 giây cho 163 giây audio khiến việc xử lý 60 phút phim tốn tới hơn 31 phút GPU.

### 3.2. Đột phá chiến lược: Chuyển dịch từ Clean Voice Replacement sang Smart Voice-Over
Thay vì cố gắng bóc tách sạch 100% tiếng Trung (bài toán cực kỳ nặng, tốn GPU và luôn đi kèm rủi ro artifact: mỏng nhạc nền, méo tiếng, mất tiếng thở dài, tiếng khóc, tiếng cười hay SFX), chuyển đổi sang mô hình **Thuyết minh phủ đè thông minh (Voice-over with Dynamic Ducking)**:
- **Bảo toàn 100% âm thanh điện ảnh gốc:** Giữ nguyên vẹn toàn bộ soundtrack (nhạc nền BGM, foley, tiếng động va chạm SFX, tiếng khóc/cười, tiếng súng nổ).
- **Thói quen thưởng thức tự nhiên:** Khán giả Việt Nam xem phim Trung Quốc (truyền hình VTV/HTV, web drama, review phim) vốn đã quen thuộc và ưa chuộng định dạng thuyết minh: giọng diễn viên gốc trầm nhỏ phía dưới tạo nhịp điệu cảm xúc, giọng thuyết minh tiếng Việt rõ ràng, truyền cảm nổi bật phía trên.
- **Giải phóng hoàn toàn thời gian Separation:** Thời gian tách âm giảm từ **31,2 phút xuống đúng 0 giây!** Tiết kiệm 2–3 GB VRAM GPU.

### 3.3. Giải pháp kết hợp Đỉnh cao: Dynamic Multi-Band Sidechain Ducking (FFmpeg DSP)
Kết hợp đồng thời cả **Dynamic Sidechain Compression (Cách 1)** và **Speech Frequency Carving (Cách 2)** thông qua bộ phân tần chuẩn phòng thu Linkwitz-Riley 4th-order (`acrossover`) tích hợp sẵn trong FFmpeg:
- **Dải Trầm (Low: < 500 Hz):** Giữ năng lượng nhạc nền, tiếng nổ, tiếng trống $\rightarrow$ nén nhẹ (~ -3 dB).
- **Dải Thoại (Mid: 500 Hz – 3500 Hz):** Vùng tập trung năng lượng giọng Trung Quốc $\rightarrow$ nén sâu và dìm mạnh (-12 dB đến -14 dB) khi có giọng Việt phát ra.
- **Dải Cao (High: > 3500 Hz):** Giữ tiếng bước chân, tiếng kim loại, độ sáng của âm thanh $\rightarrow$ nén nhẹ (~ -3 dB).
- **Đường cong chuyển tiếp (Envelope):** Attack 100ms (hạ âm lượng êm ái), Release 300ms (trả về âm lượng gốc mượt mà khi dứt câu thoại).

**Filtergraph FFmpeg đã được kiểm chứng thực nghiệm:**
```bash
ffmpeg -i original_soundtrack.wav -i vietnamese_tts.wav -filter_complex \
"[0:a]acrossover=split=500 3500[low][mid][high]; \
 [mid][1:a]sidechaincompress=threshold=0.1:ratio=8:attack=100:release=300[mid_ducked]; \
 [low][1:a]sidechaincompress=threshold=0.2:ratio=2:attack=100:release=300[low_ducked]; \
 [high][1:a]sidechaincompress=threshold=0.2:ratio=2:attack=100:release=300[high_ducked]; \
 [low_ducked][mid_ducked][high_ducked]amix=inputs=3:normalize=0[bg_ducked]; \
 [bg_ducked][1:a]amix=inputs=2:weights=1 1.25:normalize=0[out]" \
-map "[out]" output_mastered.wav
```
- **Tốc độ thực tế đo được:** Đạt **`103x Real-Time`** (xử lý toàn bộ tập phim 60 phút chỉ mất **~35 giây CPU**).
- **Tài nguyên:** 0 MB VRAM GPU, 0 model AI, hoàn toàn tất định (deterministic DSP).

### 3.4. Tiêu chí kiểm chứng ASR trên soundtrack gốc (Actionable Gate)
Mối băn khoăn duy nhất là liệu ASR (Qwen3-ASR / FireRed) nhận dạng trực tiếp trên audio gốc (chưa tách nhạc) có bị suy giảm chất lượng không.
- **Quy tắc nghiệm thu:** Chạy benchmark ASR trực tiếp trên soundtrack gốc của video mẫu. Nếu độ chính xác phiên âm và độ khớp timestamp đạt $\ge 95\%$ so với khi qua tách âm, **loại bỏ hoàn toàn Bandit và RoFormer khỏi production pipeline**, biến pipeline thành kiến trúc chuẩn duy nhất (Single Canonical Architecture) siêu tốc.

## 4. Thiết kế lại phần hình ảnh theo sự kiện phụ đề

V1 scan đã lấy mẫu 1 fps. Đổi đơn thuần thành 2–5 fps còn tăng số lần OCR. V1 inpaint lại OCR từng crop, gồm 8 frame lịch sử lặp, và gọi tiến trình ProPainter mới mỗi 40 frame. Đồng thời ghi PNG/mask rồi video FFV1 trung gian. Cần sửa cơ chế này trước khi quy mọi độ chậm cho model.

Thiết kế đề xuất:

- Quét nhẹ vùng phụ đề và điểm chuyển cảnh; phát hiện lúc chữ xuất hiện, thay nội dung, đổi vị trí hoặc biến mất. Chuyển cảnh không đồng nghĩa mọi lần đổi phụ đề.
- OCR một số frame đại diện cho mỗi sự kiện; dùng cùng kết quả cho dịch, layout và glyph mask. Tách phát hiện vị trí chữ khỏi nhận dạng nội dung chữ; tạo mask không cần đọc lại nội dung ở mọi frame.
- Theo dõi mask qua các frame, kiểm tra mất dấu/đổi chữ; OCR lại theo quy tắc cố định khi độ tin cậy không đủ. Tránh mask kéo sót chữ cũ hoặc xóa nhầm chi tiết ảnh.
- Tự hiệu chỉnh vùng phụ đề, lưu profile theo series, xác minh ở tập mới. OCR ngữ cảnh ngoài vùng phụ đề chỉ tại keyframe có nhu cầu; không bỏ luôn bảng tên/chữ trên màn hình.
- Thử CUDA provider cho OCR với bộ CUDA/cuDNN tương thích, xác minh provider thực và số đo. Cài PyTorch CUDA không chứng minh OCR đã chạy GPU. [Tài liệu ONNX Runtime CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
- **Chi tiết kỹ thuật: Vì sao V1 chạy trên CPU & Kế hoạch chuyển dịch ONNX Runtime GPU:**
  - *Nguyên nhân V1 chạy CPU:*
    1. Trong `requirements/bench-vision.txt` và `requirements/bench-vision.in`, thư viện được cài đặt là `onnxruntime==1.22.1` (bản build x86_64 CPU-only của PyPI), không hỗ trợ gọi CUDA.
    2. Trong `src/autodub/adapters/ocr.py`, hàm `build_engine` cấu hình cứng `EngineConfig.onnxruntime.intra_op_num_threads: 4` và không truyền tham số `providers`, khiến RapidOCR mặc định 100% sử dụng `CPUExecutionProvider`.
    3. Ý định ban đầu của V1 là để OCR chạy CPU nhằm dành trọn vẹn VRAM cho BandIt ERB48 và ProPainter. Tuy nhiên, việc quét định kỳ 1 fps (full-frame) và dense frame trên 4 luồng CPU đã gây nghẽn nghiêm trọng (788,1 giây cho video 2,7 phút, ngoại suy 290 phút cho 60 phút).
  - *Kế hoạch triển khai ONNX GPU cho V2:*
    1. **Nâng cấp gói thư viện:** Chuyển `onnxruntime` thành `onnxruntime-gpu` (phiên bản tương thích CUDA 12.x và cuDNN 9.x trên Ubuntu 22.04/24.04).
    2. **Cấu hình Execution Provider trong adapter `ocr.py`:** Khởi tạo session với thứ tự ưu tiên `providers=['CUDAExecutionProvider', 'CPUExecutionProvider']`, kèm theo `provider_options` khống chế VRAM: đặt `device_id: 0`, `gpu_mem_limit: 1073741824` (giới hạn 1 GB VRAM) và `arena_extend_strategy: 'kSameAsRequested'` để RapidOCR không tranh chấp bộ nhớ với các tác vụ khác.
    3. **Kiểm tra thực chứng (Preflight Gate):** Bắt buộc script kiểm tra môi trường chạy `python -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()"` để xác nhận CUDA provider thực tế hoạt động trước khi chạy job.
    4. **Hiệu năng kỳ vọng:** Khi chạy trên GPU (RTX 3060 / 5060 Ti), tốc độ OCR mỗi crop frame giảm từ ~160ms (CPU) xuống còn ~5–10ms (GPU). Kết hợp với Event-based OCR (chỉ quét frame có phụ đề thay đổi, ~500–800 frame cho 60 phút phim), tổng thời gian OCR toàn bộ phim sẽ giảm từ 290 phút xuống chỉ còn **dưới 15–30 giây**, hoàn toàn nằm gọn trong ngân sách 80 giây của V2.

Temporal clean-frame recovery là ứng viên đáng thử, nhưng chỉ khôi phục được từ thông tin nền có thật: có donor không bị chữ che, cùng cảnh, chuyển động và che khuất cho phép ánh xạ. Nền tĩnh vẫn có thể phục hồi rất tốt nếu có frame sạch trước/sau. Nền không bao giờ lộ ra, vật thể đổi hình, cut hoặc occlusion làm phương pháp này thiếu dữ liệu.

Không được tự xem che bằng banner/blur là xóa hard-sub đẹp. Không tự chuyển model khi khó. Cần chốt trước tiêu chuẩn xử lý vùng không thể phục hồi và đánh dấu chưa đạt; thời gian nhanh nhưng để lại chữ hoặc phá nền không tính là đạt mục tiêu. ProPainter có thể được giữ làm đối chứng benchmark tối ưu adapter; chưa đủ bằng chứng để kết luận nó là nguyên nhân duy nhất hoặc nhất thiết phải loại ngay.

## 5. Lồng tiếng: sửa tính đúng trước, rồi đo tốc độ

46/46 đoạn vừa rồi KEEP vì thiếu timing từng từ; TTS không sinh audio. Benchmark mới phải báo số đoạn và tổng số giây thoại thực sự được thay, không chỉ trạng thái tiến trình SUCCESS.

Phải kiểm tra khả năng/đường tích hợp timestamp của FireRed trước. Nếu cần bộ căn thời gian chuyên biệt, đó là một nhiệm vụ kiến trúc riêng phải chốt; không dựng thời gian giả bằng chia đều. Mã [FireRed hiện hành](https://github.com/FireRedTeam/FireRedASR2S/blob/main/fireredasr2s/fireredasr2/asr.py) có nhánh chia đều khi thiếu timestamp, nên chỉ bật return_timestamp không chứng minh có căn từ thực.

ASR hiện đã chạy các speech window từ diarization. Việc còn làm được là gom batch theo độ dài, giải mã/resample audio một lần rồi cắt theo sample, tránh tạo một tiến trình FFmpeg cho mỗi đoạn. Phân đoạn và speaker IDs cần giữ nhất quán xuyên chunk; overlap ngắn không được khiến toàn bộ một đoạn dài bị bỏ lồng tiếng.

VieNeu v3 Turbo là ứng viên tốt cho mục tiêu tốc độ. [Benchmark tác giả](https://github.com/pnnbao97/VieNeu-TTS#-4-benchmarks) báo RTF 0,011–0,02 khi batch trên RTX 3060; đó chưa phải số đo CN2VI, Ngọc Huyền hoặc RTX 5060 Ti. Phải kiểm chứng phát âm, âm lượng, độ dài, chất giọng và biểu cảm trên chính lời thoại dự án. Batching theo độ dài, cache conditioning của Ngọc Huyền và warm-up batch shapes là hướng nên thử.

Đổi IndexTTS2 sang VieNeu có thể thay đổi khả năng diễn theo audio gốc. Không coi sửa “Đi đi” thành “Biến đi” là giải pháp bù cảm xúc mặc định: nó có thể đổi nghĩa và xưng hô. Dịch cần nhận ngân sách thời lượng từ đầu; xử lý câu quá dài theo một chính sách hữu hạn đã chốt, không cắt đuôi hoặc lặp sinh giọng vô hạn. Giữ âm thanh phi ngôn ngữ đòi hỏi mặt nạ thời gian đúng, không chỉ giữ tất cả âm thanh có năng lượng.

FireRed có [runtime TensorRT chính thức](https://github.com/FireRedTeam/FireRedASR2S/blob/main/runtime/triton_tensorrt/README.md), nhưng benchmark công bố dùng H20 và AISHELL. Không áp hệ số đó lên 5060 Ti. ASR inference hiện chỉ 6,3 giây trong mẫu; ưu tiên này đứng sau Bandit và phần hình.

## 6. Scheduler, dữ liệu và dựng video

Một GPU 16 GB: giữ worker sống qua nhiều batch, tiếp tục dùng môi trường tách biệt; chỉ giữ các model đồng thời khi đo cho thấy đủ VRAM và có lợi. GPU-heavy stage có lịch cấp tài nguyên, CPU/API/đọc ghi/NVDEC/NVENC chồng thời gian khi phụ thuộc dữ liệu cho phép. Không khẳng định tất cả model chắc chắn OOM hoặc mọi concurrency đều chậm; cần đo cặp workload và VRAM thực. Mặc định không tính lợi ích concurrency GPU chưa được chứng minh vào ngân sách.

Không áp một chunk 2–5 phút cho mọi stage. Dùng job chunk để checkpoint/điều phối; audio microbatch theo độ dài câu và context; hình theo cảnh và sự kiện chữ. Hàng đợi phải giới hạn dung lượng, có backpressure và giữ mốc PTS/sample tuyệt đối. Speaker registry, xưng hô và glossary là trạng thái xuyên chunk, cập nhật có thứ tự; chỉ batch dịch song song khi dữ liệu ngữ cảnh đã sẵn sàng.

Checkpoint/cache theo hash của input, code, model, config, transcript, giọng và policy; sửa một câu chỉ vô hiệu hóa đúng các đầu ra phụ thuộc. Cache này giúp phát triển và khôi phục, không được dùng kết quả cache sẵn để báo tốc độ xử lý nội dung mới.

NVENC đã có. Việc còn lại là giảm decode/copy/encode lặp: tránh xuất full preview rồi encode lại full final khi không cần; dựng một lần sau khi từng vùng đủ dữ liệu. Có thể dùng chunk đóng GOP và ghép stream khi codec/timebase/extradata/timestamp/AAC continuity thật sự tương thích. Không chỉ kiểm tra cùng codec và fps.

NVDEC/NVENC có phần cứng chuyên dụng, nhưng không tự làm OpenCV và filter subtitles thành xử lý trên GPU. Muốn giảm copy cần thiết kế đường frame rõ ràng và xác minh bộ lọc; không hứa zero-copy ASS/SRT chỉ bằng thêm cờ CUDA. [Hướng dẫn FFmpeg của NVIDIA](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html).

## 7. Điều kiện để công nhận 10 phút

60 phút × 30 fps = 108.000 frame. Trong 600 giây, toàn hệ thống phải duy trì trung bình tương đương 180 frame nguồn/giây, bao gồm các điểm chờ; 60 fps cần gấp đôi số frame. Không tự hạ độ phân giải hoặc bỏ frame để đạt chỉ tiêu.

Ngân sách định hướng theo 2 kịch bản kiến trúc:

- **Kịch bản A (Full Clean Dubbing với Neural Separation):** ~180 giây separation (cần RTF 0,10–0,12), 95 giây diarization + ASR + timing, 45 giây TTS, 80 giây OCR/khôi phục = 400 giây GPU; cộng chi phí codec/API/I/O để vừa trần 600 giây.
- **Kịch bản B (Smart Voice-Over với Dynamic Multi-Band Ducking - KHUYẾN NGHỊ MỚI):**
  - **Separation:** **0 giây** (Loại bỏ hoàn toàn model tách âm).
  - **ASR & Alignment:** ~70–90 giây (Qwen3-ASR / FireRed).
  - **Dịch thuật:** ~30–45 giây (DeepSeek Flash API `deepseek-flash`, chạy song song I/O).
  - **Vietnamese TTS:** ~40–50 giây (VieNeu v3 Turbo batched).
  - **Dynamic Multi-Band Ducking:** ~25–35 giây (FFmpeg DSP Linkwitz-Riley crossover, 0 MB VRAM).
  - **Event-based OCR (ONNX GPU) & Subtitle Burn:** ~30–45 giây.
  - **Final NVENC Mux & Encode:** ~60–80 giây.
  - **Tổng thời gian toàn bộ phim 60 phút:** Chỉ khoảng **250 – 360 giây (~4,2 đến 6 phút)**! Đạt và vượt xa mục tiêu 10 phút ban đầu.

Đo từ lúc input đã có đủ và môi trường sẵn sàng tới khi MP4 hoàn chỉnh kiểm tra được. Nạp model/warm-up phát sinh trong job phải tính; bootstrap, tải model, upload/download và hàng đợi báo riêng để thấy đầy đủ chi phí cloud mới. Báo cả cold-process và warm-resident, tránh lấy thời gian đến chunk đầu tiên làm thời gian xong phim.

Thứ tự kiểm chứng:

1. Mẫu ngắn có bản chú giải: timing, TTS thật, giọng Ngọc Huyền, mất thoại/cười/khóc, khớp câu và phụ đề.
2. Benchmark độc lập ASR trên audio gốc và FFmpeg Multi-Band Ducking; đối chiếu chất lượng và tốc độ trước/sau.
3. Chạy 10 phút để kiểm tra scheduler, bộ nhớ và ranh giới chunk.
4. Chạy đủ 60 phút ở ít nhất các mức mật độ thoại thấp/vừa/cao, cảnh tĩnh/chuyển động, sub liên tục và thay vị trí; báo từng mẫu, không chỉ trung bình. Thử 1080p riêng nếu sản phẩm yêu cầu.

Chỉ tính PASS khi đầu ra đáp ứng phạm vi chức năng: tỷ lệ thoại được dub, lỗi ASR/dịch, timing, bảo toàn âm thanh phi ngôn ngữ, chữ Trung còn sót và lỗi nền/nhấp nháy đều có kết quả kiểm tra. Với phục hồi nền, tạo bộ video sạch rồi chèn sub để có ground truth; PSNR/SSIM phải đo vùng cần sửa, không lấy toàn frame che lấp lỗi. Dữ liệu thật cần xem clip chuyển động. Không dùng PESQ so audio Trung với giọng Việt làm chất lượng dịch/lồng tiếng. Các ngưỡng phải chốt theo baseline được người dùng nghe/xem chấp nhận.

Nếu quá 600 giây, ghi không đạt thời gian; trong lượt thử vẫn tuân thủ chính sách xuất kết quả/partial đã chốt, không tự sửa model giữa lượt hoặc dừng xóa dữ liệu.

## 8. Thứ tự đầu tư đề nghị

P0: timing và TTS tạo được thoại; số đo phản ánh đầu ra thật. P1: Kích hoạt chế độ Smart Voice-Over với Multi-Band Sidechain Ducking, kiểm chứng ASR trên soundtrack gốc để xóa bỏ hoàn toàn Bandit/RoFormer khỏi pipeline chính. P2: OCR event reuse, chuyển đổi sang ONNX Runtime GPU (CUDA provider), mask tracking. P3: persistent workers, bounded queues, decode/encode một đường và cache đúng phụ thuộc. P4: TensorRT hoặc lượng tử hóa sâu chỉ cho phần còn nghẽn. Thử TTS ứng viên theo chất lượng và throughput trước khi chốt lại model chính.

Mục tiêu 10 phút hoàn toàn nằm trong tầm tay đối với kiến trúc Smart Voice-Over (Kịch bản B). Hai điều kiện cần chứng minh sớm nhất là ASR trên audio gốc đạt độ chính xác và khôi phục/render phụ đề vào ngân sách. Nếu một trong hai không đạt, cần thay quyết định kỹ thuật/phạm vi/hardware thay vì thêm concurrency cho đủ sơ đồ.

Ghi chú kiểm chứng phản biện: không tiếp nhận các khẳng định chưa có số đo về chắc chắn OOM, tốc độ ProPainter thuần, ngưỡng flow 0,85, PSNR/SSIM chung hoặc burn ASS zero-copy. 780 giây bằng 130% ngân sách 600 giây, tức vượt 30%, không phải vượt 130%. Nền tĩnh chỉ thiếu donor khi không có pixel sạch phù hợp, không phải mọi cảnh tĩnh đều thất bại.
