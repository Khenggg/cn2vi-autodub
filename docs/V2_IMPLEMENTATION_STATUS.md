# Nghiệm thu mã nguồn CN2VI V2 — 08/10/2026

## Phạm vi đã thực hiện

- V2 là mặc định; DAG và production install plan không gọi/tải Bandit, RoFormer/Kim, ProPainter hay IndexTTS2. Adapter/lock lịch sử giữ riêng để đối chứng, không phải fallback V2.
- Extract soundtrack mono 16 kHz một lần, WebRTC VAD tìm candidate; FireRed AED resident xuyên batch, ghép kết quả theo utterance ID để tránh thứ tự batch làm sai timing. FireRedPunc giữ metadata. Không tạo word timing chia đều; dùng native words khi hợp lệ, nếu thiếu dùng VAD_WINDOW.
- Audio và event OCR chạy song song; OCR PP-OCRv6 Medium yêu cầu CUDA trên cả detector/recognizer/classifier thực, disable provider fallback, giới hạn arena. Gate audio/OCR phân loại dialogue/unsubtitled/OST/nonlexical/ambiguous; whitelist homophone correction, bảo toàn câu đã duyệt và character context.
- DeepSeek Flash batch có giới hạn concurrency, cùng character context và ngân sách thời lượng; key riêng từ env/file. Không gửi key vào artifacts. Câu đã duyệt không bị API viết lại.
- VieNeu v3 Turbo PyTorch BF16, Ngọc Huyền cố định theo hash; load/enroll một lần mỗi job, sort utterances theo độ dài, batch giới hạn. Model process kết thúc sau stage, không resident xuyên nhiều job.
- PCM voice bus dùng memmap để tránh giữ audio cả giờ trong RAM. Câu quá dài không bị cắt đuôi; báo thiếu coverage. Crossover Linkwitz-Riley bậc 4 ở 500/3500 Hz, sidechain nominal −3/−14/−3 dB, attack 100/release 300 ms, tiếng Việt overlay stereo, limiter. Original soundtrack được giữ làm nguồn; filter/duck/limiter thay đổi waveform, không phải bit-exact hay bảo đảm SFX ở dải thoại không giảm.
- Sparse event OCR trong subtitle ROI, signature/cut/heartbeat để tái dùng kết quả. Donor trước/sau cùng cảnh + LK/RANSAC registration; sửa trong glyph mask, giữ pixel khi không chắc. Burn và NVENC một lần, không full preview/inpaint trung gian.
- Scheduler/domain/API/UI hỗ trợ V2_RUNNING, drain/resume frozen snapshot và checkpoint, ngăn xóa/sửa khi đang chạy. Báo cáo coverage thực, PARTIAL khi thiếu giọng/donor, không báo inference hoàn chỉnh chỉ vì endpoint còn sống.
- Fresh-cloud installer khóa ba profile GPU asr/tts/vision và core, 7 assets ~5.64 GiB, SDK/API check, FFmpeg/codec/fonts/espeak, ORT GPU CUDA/cuDNN 9, PyTorch 2.8 CUDA 12.8; không yêu cầu Docker hoặc Drive restore. `setup_new_server.sh` gọi installer chuẩn.

## Kiểm thử

Toàn suite: **317 passed, 0 failed, 0 skipped** trong 18,36 giây. Ruff pass sau chuẩn hóa import. Chạy trên môi trường CPU Ubuntu/WSL Python 3.12; không tải weights, không chạy GPU inference, không gọi DeepSeek thật. Test media chạy FFmpeg/OpenCV thật trên audio/video tổng hợp, đo ducking theo dải và kiểm tra phục hồi/burn một lần; test SDK/provider/ASR/TTS/API dùng mock kiểm tra contract. Xem [pytest log](validation/v2-pytest.log) và [Ruff log](validation/v2-ruff.log). Thời gian test không phải thời gian xử lý video.

Frontend đã pass `tsc --noEmit` và `vite build`. Linux setup scripts được kiểm tra cú pháp và kế hoạch dry-run. Thêm kiểm tra chặn artifact thoát frozen run trước khi probe tồn tại, giới hạn số/thời gian event và lookup bằng bisect, giới hạn filter fitting trước khi chạy FFmpeg. Không xem mock hoặc clip tổng hợp là chứng minh chất lượng phim thật.

## Cần cloud để nghiệm thu sản phẩm

Sẵn sàng **cài và chạy thử V2** trên Ubuntu RTX 3060 12GB sau khi installer thực tế pass. Chưa thể xác nhận 100% inference/chất lượng/SLA: chưa load weights trên GPU mới, chưa đo VRAM concurrent OCR/ASR, chưa chạy video thật 3 phút hoặc phim 60 phút.

Các ngưỡng VAD/OST, OCR role/ROI và ASR confidence chưa được hiệu chuẩn; tên riêng ngoài whitelist không tự thay nếu thiếu bằng chứng. Character registry không tự suy đoán người nói chưa rõ danh tính. Temporal recovery không thể tạo nền chưa bao giờ lộ ra; VFR/offset không hỗ trợ phục hồi. TensorRT và zero-copy video chưa triển khai, không được tính vào tốc độ công bố.

Tài liệu thiết kế có dự phóng và số đo DSP từ cuộc thảo luận trước. Chúng không phải benchmark end-to-end của code V2 này. `sla_verified=false` và `quality_and_sla_verified=false` là trạng thái đúng cho tới lượt GPU được đánh giá.
