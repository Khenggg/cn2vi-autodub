Được. Mình viết phần **“nội dung những gì chúng ta đã bàn và đã chốt”** theo dạng có thể đưa thẳng cho Codex/Agent làm tài liệu chuẩn của dự án.

---

# CN2VI AutoDub – Các quyết định kiến trúc đã chốt

## 1. Mục tiêu sản phẩm

CN2VI AutoDub là hệ thống tự động Việt hóa video/phim tiếng Trung.

Đầu vào:

```text
Video tiếng Trung
có thể có hard-sub Trung nằm trực tiếp trên hình
```

Đầu ra:

```text
Video hoàn chỉnh
├── lời thoại tiếng Việt
├── giữ nhạc nền
├── giữ SFX
├── giữ tiếng hét/cười/khóc/thở và âm thanh phi ngôn ngữ
├── xóa hard-sub Trung
├── burn Vietsub
└── xuất MP4 phù hợp xem/upload YouTube
```

Mục tiêu quan trọng:

```text
Tự động tối đa
Nhanh
Chi phí thấp
Chạy trên GPU cloud thuê theo giờ
Ít thao tác thủ công
```

Reference machine hiện tại:

```text
Ubuntu 24.04
RTX 5060 Ti 16 GB VRAM
~28 GB RAM
~1 Gbps network
ephemeral disk
```

---

# 2. Nguyên tắc model

Đã chốt nguyên tắc:

> **Một nhiệm vụ = một model chính.**

Không xây:

```text
Model A lỗi
→ model B
→ model C
```

Không model fallback.

Nếu một model không đủ tốt sau benchmark:

```text
OLD MODEL OUT
NEW MODEL IN
```

thay hoàn toàn model đó.

Được phép:

```text
retry cùng model
chunk lại input
sửa preprocessing
đổi tham số
```

nhưng không được:

```text
FireRed lỗi → Whisper
IndexTTS lỗi → VieNeu
ProPainter lỗi → LaMa
```

Mục tiêu là:

```text
ít model
ít dependency
dễ debug
dễ benchmark
dễ cài cloud
```

---

# 3. Pipeline chính thức

Pipeline mới:

```text
VIDEO
 │
 ▼
FFmpeg / ffprobe
 │
 ├── probe media
 ├── extract audio
 └── quản lý timeline
 │
 ▼
Bandit v2 Cinematic
 │
 ├── Dialogue
 ├── Music
 └── SFX
 │
 ▼
pyannote Community-1
 │
 ├── vùng có speech
 ├── speaker segmentation
 ├── speaker identity
 └── overlapping speakers
 │
 ▼
FireRedASR2-AED
 │
 ├── Chinese transcript
 ├── word timestamps
 └── confidence
 │
 ▼
FireRedPunc
 │
 └── punctuation / sentence segmentation
 │
 ├─────────────────────────────┐
 │                             │
 ▼                             ▼
PP-OCRv6 Medium         Character Context Engine
 │                             │
 │                             ├── character IDs
 │                             ├── names
 │                             ├── aliases
 │                             ├── speaker
 │                             ├── addressee
 │                             ├── relationship
 │                             ├── addressing history
 │                             └── series memory
 │                             │
 └──────────────┬──────────────┘
                ▼
         Translation LLM
                │
         ┌──────┴──────┐
         ▼             ▼
   subtitle_vi       dub_vi
                       │
                       ▼
             IndexTTS2 Vietnamese
                       │
              ┌────────┼─────────┐
              │        │         │
            text     timbre    emotion
              │        │         │
              │        │         └─ audio Trung gốc
              │        └─ fixed female voice
              │
              └─ dub_vi
                       │
                       ▼
               Vietnamese speech
                       │
                       ▼
                  FFmpeg Mixer
                       │
      ┌────────────────┼────────────────┐
      │                │                │
    Music             SFX        Dialogue residual
                                     +
                               Vietnamese speech
                       │
                       ▼
                  PREVIEW VIDEO
                       │
                       ▼
            Auto Subtitle Detection
                       │
                       ▼
                PP-OCRv6 Medium
                       │
                       ▼
                 glyph mask
                       │
                       ▼
                  ProPainter
                       │
                       ▼
               restored frames
                       │
                       ▼
                 burn Vietsub
                       │
                       ▼
              H.264 NVENC + AAC
                       │
                       ▼
                    FINAL
```

---

# 4. Tách âm thanh: Bandit v2 Cinematic

Model cũ:

```text
Kim_Vocal_2
```

được bỏ.

Lý do:

Kim tập trung vào:

```text
Vocals
Instrumental
```

trong khi phim cần:

```text
Dialogue
Music
Effects
```

Model mới:

> **Bandit v2 Cinematic**

Output:

```text
Dialogue stem
Music stem
SFX stem
```

Đây phù hợp trực tiếp với bài toán dubbing phim.

---

# 5. Speaker detection: pyannote Community-1

Không chỉ cần biết:

```text
có lời thoại hay không
```

mà còn cần:

```text
ai đang nói
khi nào đổi người nói
hai người có nói đè nhau không
```

Do đó dùng:

> **pyannote Community-1**

Nó đảm nhiệm:

```text
speech segmentation
speaker diarization
speaker change
overlap detection
```

Vì pyannote đã có speech segmentation nên **không cần FireRedVAD riêng**.

---

# 6. ASR: FireRedASR2-AED

Bỏ:

```text
Faster-Whisper large-v3-turbo
WhisperX
forced alignment model cũ
```

Thay bằng:

> **FireRedASR2-AED**

Dùng cho:

```text
Mandarin / Chinese ASR
word timestamps
confidence
```

Pipeline được đơn giản từ:

```text
Whisper
→ WhisperX
→ alignment
```

thành:

```text
FireRedASR2-AED
```

Không dùng ASR fallback.

---

# 7. Dấu câu: FireRedPunc

ASR text:

```text
你怎么来了我不是叫你别来吗
```

không nên đưa thẳng sang dịch.

Qua:

> **FireRedPunc**

để thành dạng:

```text
你怎么来了？
我不是叫你别来吗？
```

Dịch tốt hơn và chia segment tự nhiên hơn.

---

# 8. OCR hard-sub Trung

Dùng:

> **PP-OCRv6 Medium**

runtime:

```text
RapidOCR
```

OCR có hai nhiệm vụ.

### Nhiệm vụ 1 – hỗ trợ hiểu lời thoại

Ví dụ audio Mandarin phát âm:

```text
tā
```

ASR có thể khó phân biệt:

```text
他
她
```

Nhưng hard-sub trên video ghi:

```text
她
```

OCR cung cấp thông tin chữ viết.

Do đó:

```text
ASR transcript
+
OCR subtitle
+
context
```

được dùng để xây canonical Chinese text.

### Nhiệm vụ 2 – xóa subtitle Trung

OCR tìm:

```text
exact text bounding boxes / polygons
```

để tạo mask cho ProPainter.

---

# 9. Không bắt người dùng vẽ ROI mỗi tập

Thiết kế cũ:

```text
upload
→ preview
→ người dùng vẽ ROI
→ mới xóa sub
```

được bỏ khỏi normal workflow.

Hệ thống phải:

> **tự phát hiện vùng subtitle.**

Thông thường hard-sub nằm ở vùng dưới video.

Ví dụ:

```text
0% ┌──────────────────────┐
   │                      │
   │        VIDEO         │
   │                      │
70%├──────────────────────┤
   │    subtitle zone     │
   │   中文字幕........   │
95%└──────────────────────┘
```

Auto detector dùng:

```text
spatial location
OCR frequency
font size consistency
temporal appearance
center alignment
speech timing
text duration
1–2 line structure
```

để phát hiện vùng subtitle.

---

# 10. Series Subtitle Profile

Nếu một series có:

```text
20 episode
```

không cần detect lại hoàn toàn mỗi episode.

EP01:

```text
auto-calibrate subtitle layout
```

lưu:

```text
subtitle band
alignment
typical line height
font characteristics
line count
```

EP02–EP20:

```text
reuse profile
+
automatic verification
```

Nếu layout khác mới recalibrate.

Manual ROI chỉ còn là:

> **Override Region**

trong trường hợp cực kỳ bất thường.

Không phải bước bắt buộc.

---

# 11. Character Context Engine

Đây là thành phần mới rất quan trọng.

Dịch Trung → Việt không thể chỉ dịch từng câu.

Ví dụ:

```text
你
我
他
她
```

không đủ để biết tiếng Việt nên là:

```text
anh
em
chị
cô ấy
anh ấy
tôi
ta
ngươi
mày
tao
```

Do đó phải có:

> **Character Context Engine**

Nó quản lý:

```text
Character ID
Name
Chinese name
Vietnamese name
Aliases
Speaker ID
Addressee
Relationship
Addressing history
Series glossary
Scene context
```

Ví dụ:

```text
C01 = Trần Ngạn
C02 = Sơ Sơ

C01 → C02
anh / em

C02 → C01
em / anh
```

hoặc phim tu tiên:

```text
C02 → C01
muội / sư huynh
```

---

# 12. Character Registry

Ví dụ:

```json
{
  "C01": {
    "name_zh": "陈彦",
    "name_vi": "Trần Ngạn",
    "aliases": ["陈彦"],
    "role": "main_character"
  },

  "C02": {
    "name_zh": "初初",
    "name_vi": "Sơ Sơ",
    "aliases": ["黎初"]
  }
}
```

Quan hệ:

```json
{
  "source": "C02",
  "target": "C01",
  "relationship": "younger_to_older",
  "self_term": "em",
  "target_term": "anh"
}
```

---

# 13. Translation không chạy từng câu độc lập

Translator phải nhận:

```text
current text
speaker
addressee
previous dialogue
next dialogue nếu có
character registry
relationship graph
addressing history
glossary
scene context
OCR evidence
```

Ví dụ:

Chinese:

```text
在这里等我。
```

Nếu không có context:

```text
"Hãy ở đây đợi tôi."
```

Nếu biết:

```text
speaker = anh
addressee = em
```

thì:

```text
"Em ở đây chờ anh."
```

---

# 14. Output translation

Không cần LLM trả `emotion`.

Emotion lấy từ audio gốc.

Translation output nên tập trung vào:

```json
{
  "segment_id": "seg_001",

  "speaker_id": "C01",
  "addressee_id": "C02",

  "subtitle_vi": "Em ở đây chờ anh.",
  "dub_vi": "Em chờ anh ở đây.",

  "addressing": {
    "self": "anh",
    "target": "em"
  },

  "relationship_updates": []
}
```

Có hai bản:

### `subtitle_vi`

Ưu tiên:

```text
đủ nghĩa
tự nhiên
dễ đọc
```

### `dub_vi`

Ưu tiên:

```text
ngắn hơn nếu cần
khớp thời lượng
tự nhiên khi đọc
```

---

# 15. TTS: IndexTTS2 Vietnamese

Bỏ:

```text
VieNeu-TTS v3 Turbo
```

Model mới:

> **IndexTTS2 Vietnamese**

Không dùng IndexTTS 2.5 official vì hiện chưa có hỗ trợ Vietnamese chính thức đủ tin cậy.

---

# 16. Một giọng nữ duy nhất

Toàn series sử dụng:

```text
fixed female speaker reference
```

Tức timbre luôn cố định.

Ví dụ:

```text
female_reference.wav
```

được dùng cho tất cả câu.

Không automatic voice switching.

---

# 17. Cảm xúc TTS lấy từ audio Trung gốc

Không để LLM đoán:

```text
happy
sad
fear
angry
```

nếu không cần.

Dùng chính audio thoại Trung:

```text
original_chinese_segment.wav
```

làm:

```text
emotion reference
```

IndexTTS2 nhận:

```text
Timbre:
fixed female reference

Emotion:
Chinese original dialogue

Text:
Vietnamese dub

Duration:
target dialogue slot
```

Mục tiêu:

```text
cùng một giọng nữ Việt
+
cách diễn theo diễn viên gốc
```

Ví dụ nhân vật gốc:

```text
sợ hãi
thở gấp
nói nhỏ
giận dữ
khóc
hét
```

thì giọng Việt cố gắng giữ prosody/cảm xúc đó.

---

# 18. Timing

FireRedASR2-AED cung cấp word timestamps.

Mỗi thoại có:

```text
start
end
target_duration
```

IndexTTS2 sinh audio theo slot đó.

Nếu lệch nhẹ:

```text
FFmpeg atempo/time-stretch
```

Nếu câu quá dài đáng kể:

```text
LLM rewrite dub_vi ngắn hơn
```

Không cắt đuôi audio.

---

# 19. Giữ âm thanh phi ngôn ngữ

Một yêu cầu bắt buộc:

```text
scream
cry
laugh
breathing
grunt
gasp
pain sound
```

phải được giữ.

Ví dụ:

```text
快跑啊啊啊啊!!!
```

không được biến thành chỉ:

```text
"Chạy mau!"
```

và mất:

```text
AAAAAA!!!
```

Timeline phải hiểu:

```text
lexical Chinese speech
vs
non-verbal vocal sound
```

Chỉ phần lexical Chinese cần mute/thay thế.

Audio cuối:

```text
Music stem
+
SFX stem
+
non-verbal Dialogue residual
+
Vietnamese TTS
```

---

# 20. Multi-speaker overlap

Nếu hai người nói cùng lúc:

```text
Speaker A
+
Speaker B
```

pyannote phát hiện overlap.

Timeline không được chỉ là:

```text
segment 1
segment 2
segment 3
```

mà phải hỗ trợ:

```text
multi-track events
```

Ví dụ:

```text
Track A: ─── speech ──────
Track B:      ─── speech ─────
```

Nếu processing không thể xử lý chắc chắn:

```text
Needs Review
```

nhưng không chạy một model ASR khác.

---

# 21. Subtitle removal

Hard-sub Trung được xử lý:

```text
Auto subtitle band
↓
PP-OCRv6
↓
text polygons
↓
tight glyph mask
↓
ProPainter
```

Không xóa nguyên rectangle.

Chỉ sửa:

```text
pixel chữ
outline
shadow
```

và vùng context cần thiết.

---

# 22. ProPainter

Bỏ LaMa làm production remover.

Dùng:

> **ProPainter**

vì đây là video inpainting có temporal consistency.

Không chạy full frame nếu không cần.

Ví dụ video:

```text
1920×1080
```

subtitle chỉ nằm:

```text
1920×150
```

thì chỉ crop subtitle area + context.

Giúp:

```text
nhanh hơn
ít VRAM hơn
ít tiền cloud hơn
```

---

# 23. Video engine

Engine dựng video chính:

> **FFmpeg + ffprobe**

FFmpeg dùng cho:

```text
extract audio
cut
mux
concat
delay
time stretch
audio mixing
subtitle burn
encode
normalize
NVENC
```

OpenCV/PyAV dùng cho:

```text
frame processing
OCR masks
crop
tracking
image manipulation
```

---

# 24. Final output

Final:

```text
Container:
MP4

Video:
H.264
NVENC

Audio:
AAC-LC
48 kHz
```

Không giữ Chinese speech track riêng.

---

# 25. Chunking và ghép video

Video dài có thể xử lý theo chunk.

Ví dụ:

```text
2 giờ
↓
24 × 5 phút
```

Nếu output chunks cùng:

```text
codec
fps
resolution
timebase
audio format
```

thì cuối cùng:

```text
FFmpeg concat
-c copy
```

để tránh encode lại 2 giờ video lần nữa.

Nguyên tắc:

> **Một pixel chỉ nên encode lại một lần nếu có thể.**

---

# 26. Error policy

Technical failure:

```text
retry cùng stage 2–3 lần
```

Nếu vẫn lỗi:

```text
Needs Review
```

Queue vẫn tiếp tục episode tiếp theo.

Không:

```text
model fallback
```

Retry nên ưu tiên:

```text
chỉ segment lỗi
```

không rerun toàn episode.

---

# 27. Needs Review vẫn tạo output

Đã chốt:

```text
Needs Review episode
```

vẫn:

```text
xuất video
auto-download
+
issue report
```

không chặn toàn batch.

---

# 28. Series structure

Data hierarchy:

```text
Workspace
  ↓
Series
  ↓
Episode
```

Series lưu:

```text
glossary
characters
relationships
addressing rules
subtitle profile
translation memory
settings
```

---

# 29. Scheduler

Series có priority.

Ví dụ:

```text
Series A priority 1
Series B priority 2
```

Nhưng không phải:

```text
A xong toàn bộ
→ mới B
```

Mà:

```text
A được ưu tiên

resource nào A không dùng
→ B được tận dụng
```

GPU-heavy stages không chạy đồng thời nếu gây OOM.

CPU/API/network có thể overlap.

---

# 30. GPU management

RTX 5060 Ti có 16 GB.

Không load tất cả model một lúc.

Luồng:

```text
Bandit
↓ unload

pyannote
↓ unload nếu cần

FireRed
↓ unload

OCR
↓ unload

IndexTTS2
↓ unload

ProPainter
↓ unload
```

Mỗi model được sử dụng gần toàn bộ VRAM khi tới lượt.

---

# 31. Cloud deployment

Project phải nhẹ.

GitHub chỉ chứa:

```text
source code
Docker/config
installer
model manifest
dependency locks
tests
docs
```

Không chứa:

```text
model weights
video
node_modules
venv
CUDA package
cache
output
```

Cloud tự tải.

---

# 32. Bootstrap

Mục tiêu trên cloud mới:

```bash
git clone ...
cd cn2vi-autodub
sudo ./install.sh
```

Installer tự:

```text
check Ubuntu
check NVIDIA driver
install Docker/tooling nếu thiếu
install NVIDIA Container Toolkit nếu cần
create /data
install dependencies
download pinned models
verify checksum
build frontend
configure runtime
health check
start application
```

Driver NVIDIA:

```text
installer chỉ kiểm tra
```

Không tự nâng cấp nếu `nvidia-smi` đang hoạt động.

---

# 33. Thin repository

Repo ban đầu càng nhỏ càng tốt.

Không nhét:

```text
20–40 GB models
```

vào GitHub.

Models tải về:

```text
/data/models
```

Cache:

```text
/data/cache
```

Runtime:

```text
/data/working
```

Outputs:

```text
/data/outputs
```

---

# 34. Workspace / recovery

Cloud disk ephemeral nên hỗ trợ:

```text
workspace.aidub
```

chứa:

```text
metadata
series
glossary
character context
relationships
subtitle profile
settings
```

không chứa video lớn.

Episode chưa xử lý xong có:

```text
episode.resume.zip
```

để máy cloud mới có thể:

```text
source video
+
resume snapshot
→ resume
```

---

# 35. Drain Worker

Trước khi tắt cloud:

```text
Drain Worker
```

hệ thống:

```text
stop accepting new jobs
↓
finish/checkpoint active stage
↓
flush state
↓
READY TO SHUTDOWN
```

---

# 36. Benchmark mục tiêu

Reference workload:

```text
20 episodes × 5 phút
≈100 phút source
```

và:

```text
120 phút source
```

cần đo:

```text
wall time
GPU time
VRAM peak
RAM peak
API cost
cloud cost
Needs Review
ASR quality
speaker accuracy
subtitle detection
emotion quality
TTS duration
inpainting quality
```

Mục tiêu chi phí ban đầu:

```text
~5.000 VND / 2 giờ source
```

có thể nhỉnh hơn một ít.

Không muốn thường xuyên lên:

```text
~10.000 VND / 2 giờ
```

---

# 37. Những model cũ chính thức bị loại khỏi V1

```text
Kim_Vocal_2                 ❌
Faster-Whisper              ❌
WhisperX                    ❌
Qwen ASR fallback           ❌
VieNeu-TTS                   ❌
LaMa production inpainting  ❌
manual ROI mandatory        ❌
```

---

# 38. Stack V1 chốt

```text
Audio separation
→ Bandit v2 Cinematic

Speaker / overlap
→ pyannote Community-1

ASR
→ FireRedASR2-AED

Punctuation
→ FireRedPunc

OCR
→ PP-OCRv6 Medium + RapidOCR

Translation/context
→ Character Context Engine + one LLM API

TTS
→ IndexTTS2 Vietnamese

Video inpainting
→ ProPainter

Media engine
→ FFmpeg / ffprobe

Frame processing
→ OpenCV / PyAV

Encode
→ NVENC

Storage
→ SQLite
```

Và ba nguyên tắc nền tảng phải được ghi ngay đầu tài liệu mới:

> **1. One task, one model. No model fallback.**  
> **2. Automatic by default; manual interaction only for exceptional review cases.**  
> **3. Preserve information from the original film whenever possible: speaker, emotion, timing, relationships, non-verbal sound and subtitle text are all evidence, không được vứt bỏ sớm trong pipeline.**

Đây là phần **thứ nhất – bản tổng hợp đầy đủ những gì chúng ta đã bàn và đã chốt**. Bạn nói tiếp **thứ hai** bạn cần mình viết gì, mình sẽ làm nó dựa trên chính baseline này để hai tài liệu không mâu thuẫn nhau.