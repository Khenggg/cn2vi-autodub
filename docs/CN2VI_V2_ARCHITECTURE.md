# CN2VI AutoDub - Kiến Trúc V2 Tối Ưu (Fast Production Pipeline)

> **Mục tiêu SLA:** Xử lý video nguồn 60 phút $\rightarrow$ Hoàn thành trong khoảng 10–15 phút trên GPU đơn lẻ (NVIDIA RTX 3060 12GB / RTX 5060 Ti 16GB).
> **Quyết định cốt lõi:** Loại bỏ hoàn toàn khâu Stem Separation (BandIt / Mel-RoFormer), giữ nguyên 100% Soundtrack gốc, chuyển sang Voice-over kết hợp Dynamic Multi-Band Ducking và Multimodal Dialogue Detection.

---

## 1. Quyết định chính & Lý do loại bỏ BandIt / Source Separation

**Loại bỏ hoàn toàn BandIt và Kim_Vocal_2 / Source Separation khỏi production pipeline.**

### Lý do:
1. **Thay đổi yêu cầu sản phẩm:** Không bắt buộc phải xóa sạch 100% tiếng Trung như phim chiếu rạp. Khán giả phim truyền hình/web drama/review tại Việt Nam rất quen thuộc với định dạng **Thuyết minh phủ đè (Voice-Over)**: giọng gốc trầm nhỏ bên dưới giữ cảm xúc diễn viên, giọng thuyết minh tiếng Việt rõ ràng, truyền cảm nổi bật phía trên.
2. **Loại bỏ hoàn toàn audio artifacts:** Bất kỳ mô hình AI tách âm nào cũng có rủi ro làm mỏng nhạc nền, méo tiếng, mất tiếng cười/khóc/hét/tiếng thở hoặc để rò rỉ âm thanh (leakage). Giữ nguyên soundtrack gốc đảm bảo **100% foley, ambience, SFX, và nhạc nền nguyên bản**.
3. **Tiết kiệm tối đa tài nguyên:** Khâu tách âm tốn hơn 31 phút GPU trên video 60 phút. Khi bỏ khâu này: **Thời gian separation = 0 giây**, tiết kiệm 2–3 GB VRAM GPU.
4. **Đơn giản hóa hệ thống:** Không còn logic chia chunk 60s, bù context 8s, fader window overlap-add phức tạp.

```text
[KIẾN TRÚC CŨ]                           [KIẾN TRÚC V2 MỚI]
Original Audio                           Original Audio
      │                                        │
   BandIt                                (Giữ nguyên vẹn 100%)
   ├── Dialogue (xóa)                          │
   ├── Music    (tái tạo)                      ▼
   └── SFX      (tái tạo)                Audio Ducking + Overlay TTS
```

---

## 2. Nguyên tắc Multimodal Merge: Không dùng Subtitle làm nguồn duy nhất

Hard-sub rất hữu ích nhưng **tuyệt đối không được dùng làm gate bắt buộc**.
- **Sai lầm cần tránh:** "Có subtitle $\rightarrow$ có dialogue; không có subtitle $\rightarrow$ không có dialogue" (Phim luôn có các cảnh nhân vật nói chuyện ngoài khung hình, tiếng thì thầm, hoặc video không phụ đề).
- **Nguyên tắc vàng:**
  > **Audio đảm bảo Coverage (độ bao phủ). OCR cung cấp Evidence (bằng chứng xác thực).**
  > **Không có hard-sub không đồng nghĩa với việc không có lời thoại.**

```text
                     ORIGINAL VIDEO
                    /              \
                   /                \
              AUDIO PATH          VIDEO PATH
                  │                   │
          Speech detection          PP-OCRv6 (ONNX GPU)
                  │                   │
          Speech candidates       Hard-sub events
                  │                   │
                  └─────────┬─────────┘
                            ↓
                   MULTIMODAL MERGE
                            ↓
                    FireRedASR2-AED
                            ↓
                  Dialogue Decision
```

---

## 3. Vai trò của Audio Speech Detector

Audio path sử dụng bộ phát hiện giọng nói (VAD / Energy-based candidate detector) để tìm các vùng có khả năng chứa giọng người:
- Ví dụ: `00:10–00:13` (voice), `00:18–00:20` (voice), `00:30–00:45` (vocal).
- **Quy tắc phân loại:** `Voice Detected` $\neq$ `Dialogue Confirmed`.
  - Vùng `00:30–00:45` có thể là ca sĩ đang hát nhạc phim (OST) chứ không phải nhân vật đang đối thoại.
  - Cần kết hợp với OCR và ASR để đưa ra quyết định cuối cùng.

---

## 4. Vai trò của OCR: Nguồn bằng chứng phân loại (Evidence Engine)

PP-OCRv6 chạy song song trên nền tảng **ONNX Runtime GPU (`CUDAExecutionProvider`)**:
- Nhận diện các loại văn bản trên màn hình:
  - Chinese dialogue subtitle (Phụ đề thoại chính).
  - Lyric subtitle (Lời bài hát / OST karaoke).
  - Caption / Sign / Bảng tên nhân vật.
  - Logo / Watermark / Chữ quảng cáo.
- **Xây dựng Dialogue Subtitle Profile (sau tập đầu tiên):**
  - Vị trí trục Y chuẩn hóa (thường là bottom-center).
  - Font size, chiều cao dòng, số dòng (thường 1–2 dòng).
  - Thời lượng hiển thị (1–4 giây cho thoại, lyric thường dài hơn và có hiệu ứng chạy chữ).
- **Mục tiêu của OCR:** Không quyết định âm thanh có tồn tại hay không, mà trả lời câu hỏi:
  > **Giọng nói đang nghe là Lời thoại (Dialogue) hay Bài hát (Lyric/OST)?**

---

## 5. Logic xác định Dialogue (Multimodal Decision Matrix)

| Trường hợp | Audio Speech | Hard-sub OCR | Bối cảnh & ASR | Phân loại | Hành động hệ thống |
|---|:---:|:---:|---|:---:|---|
| **A** | CÓ | CÓ (Dialogue Profile) | ASR nhận dạng ra câu thoại rõ ràng | **DIALOGUE** | Translate $\rightarrow$ TTS $\rightarrow$ Ducking & Dub |
| **B** | CÓ | KHÔNG | ASR ra câu thoại (ví dụ: *"你先走，我马上来"*), scene đối thoại | **UNSUBTITLED_DIALOGUE** | Translate $\rightarrow$ TTS $\rightarrow$ Ducking & Dub |
| **C** | CÓ | KHÔNG hoặc Có Lyric | Vocal kéo dài liên tục 30–60s, scene mở đầu/kết thúc/montage | **SINGING_OST** | **KHÔNG DUB**, giữ nguyên âm thanh bài hát gốc |
| **D** | CÓ | KHÔNG | Đoạn cực ngắn (< 0.5s), ASR không ra từ ngữ có nghĩa (tiếng cười, khóc, hét, thở) | **NONLEXICAL** | **KHÔNG DUB**, giữ trọn vẹn tiếng động gốc |
| **E** | CÓ | Mâu thuẫn | Nhạc nền lớn, ASR độ tin cậy thấp, không khớp OCR | **AMBIGUOUS** | Ghi nhãn Needs Review, giữ audio gốc an toàn |

---

## 6. Xử lý Âm nhạc có lời (OST)

- Tuyệt đối không áp dụng suy diễn thô sơ: *"Có giọng người $\rightarrow$ Lồng tiếng"*.
- Nhạc phim (OST) tiếng Trung có ca sĩ hát phải được nhận diện và bảo vệ. Không được tự ý sinh giọng đọc đè lên các đoạn ca sĩ hát nhạc phim trừ khi có phụ đề thoại của nhân vật nói chuyện đè lên nhạc.

---

## 7. FireRed ASR chỉ chạy trên Candidate Windows (Tiết kiệm Compute)

- **Không chạy ASR full 60 phút:**
  - Từ video 60 phút, bộ Speech Detector chỉ lọc ra khoảng **25–28 phút** chứa candidate windows.
  - FireRedASR2-AED chỉ nhận dạng trên ~28 phút này, tiết kiệm hơn 50% thời gian xử lý ASR.
- **Runtime:** Khuyến nghị FireRedASR2-AED hoặc Qwen3-ASR (0.6B) với inference FP16/TensorRT đạt RTF $\le 0.03$.

---

## 8. Hard-sub dùng để sửa lỗi ASR (Multimodal Fusion)

Khi thoại có cả Audio và Subtitle tiếng Trung:
- Giải quyết hiện tượng đồng âm (Homophones): ví dụ âm `/tā/` $\rightarrow$ OCR xác định chữ **她** (cô ấy) thay vì **他** (anh ấy).
- Chuẩn hóa tên riêng, địa danh, thuật ngữ kiếm hiệp/cổ trang, số hiệu.
- Kết hợp Audio + OCR thành **Canonical Chinese Dialogue** chuẩn xác nhất trước khi gửi sang bước Dịch.

---

## 9. Khâu Dịch thuật: DeepSeek Flash API

- Sử dụng **DeepSeek Flash API (`deepseek-flash`)** siêu tốc:
  - Đầu vào: Chinese text, speaker ID, ngữ cảnh nhân vật (Character Registry & Relationship Graph).
  - Đầu ra:
    - `subtitle_vi`: Bản dịch đầy đủ, chuẩn ngữ pháp tiếng Việt để hiển thị phụ đề.
    - `dub_vi`: Bản dịch cô đọng, vừa vặn với thời lượng câu thoại gốc (speech duration constraint).
  - Thời gian gọi API cho cả tập phim chỉ mất **~30–45 giây**, hoàn toàn không tốn VRAM.

---

## 10. Khâu TTS: VieNeu-TTS v3 Turbo

- **Mô hình chính:** `VieNeu-TTS-v3-Turbo` (48 kHz, tự nhiên, siêu nhẹ).
- **Tối ưu hóa:**
  - Nạp model 1 lần duy nhất (Load once, resident memory).
  - Cache speaker conditioning vector (cố định giọng đọc hoặc multi-speaker).
  - Gom batch các câu thoại theo độ dài tương đương (Batched generation) $\rightarrow$ RTF đạt **0.015 – 0.02** trên RTX 3060/5060 (sinh 25 phút thoại chỉ tốn ~30–45 giây GPU).

---

## 11. Tự động giữ nguyên Biểu cảm phi ngôn ngữ (Non-lexical Expressions)

Các âm thanh đặc biệt của diễn viên:
- Tiếng cười, tiếng khóc nức nở, tiếng thét, tiếng thở dài, tiếng ngáp, tiếng va chạm.
- Do chúng ta **không xóa soundtrack gốc**, các âm thanh này tự động được giữ lại 100% nguyên bản mà không cần bất kỳ mô hình AI nào phải phục hồi hay tổng hợp lại.

---

## 12. Hòa âm: Dynamic Multi-Band Sidechain Ducking (FFmpeg DSP)

Áp dụng kỹ thuật phân tần Linkwitz-Riley (`acrossover`) kết hợp nén động (`sidechaincompress`) qua FFmpeg:
- **Dải Trầm (< 500 Hz):** Giữ độ dày của nhạc nền, tiếng trống, tiếng nổ $\rightarrow$ chỉ nén nhẹ (~ -3 dB).
- **Dải Thoại (500 Hz – 3500 Hz):** Dìm sâu giọng Trung Quốc (-12 dB đến -14 dB) khi có giọng Việt phát ra.
- **Dải Cao (> 3500 Hz):** Giữ tiếng động SFX leng keng, bước chân $\rightarrow$ chỉ nén nhẹ (~ -3 dB).
- **Envelope:** Attack 100ms, Release 300ms. Khi dứt câu thoại tiếng Việt, âm thanh tự động trả về 0 dB mượt mà.
- **Hiệu năng:** Tốc độ đo đạc thực tế đạt **`103x Real-Time`** (~35 giây CPU cho 60 phút phim, **0 MB VRAM**).

---

## 13. Khâu Hình ảnh chạy song song: Event-based OCR & Temporal Subtitle Restoration

1. **Video Path chạy song song với Audio Path:** Không đợi xong audio mới xử lý video.
2. **Không OCR 108.000 frame:**
   - Dùng sparse sampling, phát hiện chuyển cảnh (scene change) và thời điểm phụ đề xuất hiện/biến mất.
   - Chỉ OCR các keyframe đại diện trên ONNX Runtime GPU (mỗi frame mất ~5–10ms, tổng cộng vài trăm frame mất <10 giây).
3. **Loại bỏ ProPainter:**
   - ProPainter quá chậm (mất hàng chục phút).
   - Thay bằng **Temporal clean-frame recovery**: Lấy background thật từ các frame lân cận không có chữ (neighboring clean frames) để bù đắp vùng chữ bị che, không dùng AI sinh hình nặng nề.

---

## 14. Toàn bộ Sơ đồ Pipeline V2 Mới

```text
                                VIDEO NGUỒN
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
            AUDIO TRACK                             VIDEO FRAMES
                 │                                       │
        Audio Speech Detection                  Event-based Sampling
                 │                                       │
          Vocal Candidates                      PP-OCRv6 (ONNX GPU)
                 │                                       │
                 │                               Hard-sub Event Evidence
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ↓
                          Multimodal Merge & Gate
                       (Phân loại Dialogue / OST / SFX)
                                     ↓
                              FireRedASR2-AED
                                     ↓
                                FireRedPunc
                                     ↓
                         Canonical Chinese Dialogue
                                     ↓
                             Character Context
                                     ↓
                         DeepSeek Flash API Dịch
                                     ↓
                           subtitle_vi / dub_vi
                                     ↓
                          VieNeu-TTS v3 Turbo
                                     ↓
                            Vietnamese Voice
                                     │
SOUNDTRACK GỐC ──────────────────────┼───────────────────────┐
                                     ↓                       │
                     Multi-Band Sidechain Ducking            │
                                     │                       │
                               Mastered Audio                │
                                     │                       │
OCR GLYPH MASK ──→ Temporal Background Restoration           │
                                     │                       │
                                Clean Frames                 │
                                     │                       │
                            Burn Phụ đề Tiếng Việt           │
                                     │                       │
                                     └───────────┬───────────┘
                                                 ↓
                                            NVENC MUX
                                                 ↓
                                          DUBBED FINAL MP4
```

---

## 15. Bảng Đối Chiếu Danh Mục Thành Phần Hệ Thống

| Hạng mục | Đã Loại Bỏ Khỏi Pipeline (❌) | Giữ Lại & Nâng Cấp (✅) |
|---|---|---|
| **Stem Separation** | ❌ `BandIt v2`, ❌ `Mel-RoFormer`, ❌ `Kim_Vocal_2` | ✅ **Original Soundtrack + FFmpeg Multi-Band Ducking** |
| **ASR & Alignment** | ❌ Chạy ASR mù quáng full-stream | ✅ **Speech-gated FireRedASR2-AED / Qwen3-ASR** |
| **Dịch thuật** | ❌ Chạy model local 4B/9B tốn VRAM | ✅ **DeepSeek Flash API (`deepseek-flash`)** |
| **TTS Engine** | ❌ `IndexTTS2` (nặng, chậm) | ✅ **`VieNeu-TTS-v3-Turbo` batched 48kHz** |
| **Xóa phụ đề** | ❌ `ProPainter` video inpainting | ✅ **Temporal clean-frame background recovery** |
| **OCR Engine** | ❌ CPU full-frame 1 fps | ✅ **PP-OCRv6 trên ONNX Runtime GPU (CUDA Provider)** |
| **Video Encode** | ❌ CPU Libx264 lặp nhiều lần | ✅ **NVIDIA NVENC hardware encoding 1-pass** |

---

## 16. Dự phóng Thời gian Thực tế cho Video 60 phút

| Công đoạn | Thời gian ước tính | Thiết bị thực thi |
|---|---:|---|
| Phân tích âm thanh & Speech Gating | ~15 – 20 s | CPU / DSP |
| ASR & Punctuation (trên ~25 phút candidate) | ~70 – 90 s | GPU (CUDA) |
| Multimodal Merge & Character Context | ~5 s | CPU |
| Dịch thuật (DeepSeek Flash API) | ~30 – 40 s | Network API |
| Lồng tiếng (VieNeu-TTS batched) | ~40 – 50 s | GPU (CUDA) |
| Multi-Band Audio Ducking | ~25 – 35 s | CPU / FFmpeg |
| Event OCR & Subtitle Processing | ~30 – 45 s | GPU / ONNX |
| Subtitle Burn & Final NVENC Encode | ~60 – 80 s | GPU NVENC |
| **TỔNG THỜI GIAN TOÀN BỘ PIPELINE** | **~5 – 7 PHÚT** | **Đạt tuyệt đối mục tiêu SLA!** |
