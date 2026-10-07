Đúng, phần thứ 2 này nên được ghi thành **quy tắc vận hành bắt buộc trong giai đoạn thử nghiệm**, vì nếu không Agent rất dễ thấy lỗi rồi tự sửa giữa chừng, khiến benchmark mất giá trị.

Bạn có thể đưa nguyên văn phần dưới cho Agent/Codex:

# TESTING & PIPELINE EXECUTION POLICY

Trong giai đoạn hiện tại, mục tiêu không phải chỉ làm cho pipeline “PASS”, mà là **quan sát chính xác pipeline thực tế hoạt động ra sao, mất bao lâu, lỗi ở đâu và video cuối cùng trông như thế nào**.

Vì vậy phải tuân thủ các quy tắc sau.

## 1. Mọi lần sản xuất video thử nghiệm đều phải đo thời gian

Bất kỳ lần nào chạy pipeline trên video thật đều phải đo:

```text
TOTAL WALL TIME
+
thời gian từng stage
```

Tối thiểu phải ghi:

```text
Input duration

PREPARING
FFmpeg extraction

SEPARATION
Bandit v2

DIARIZATION
pyannote Community-1

ASR
FireRedASR2-AED

PUNCTUATION
FireRedPunc

OCR
PP-OCRv6

CONTEXT
Character Context Engine

TRANSLATION
LLM/API

TTS
IndexTTS2 Vietnamese

AUDIO MIX
FFmpeg

SUBTITLE DETECTION

INPAINT
ProPainter

SUBTITLE RENDER

FINAL ENCODE
NVENC

TOTAL PIPELINE TIME
```

Ví dụ report:

```text
Source duration:          05:00

Preparation:             4.2 s
Bandit:                  31.8 s
pyannote:                 8.1 s
FireRed ASR:             12.6 s
FireRedPunc:              1.3 s
OCR:                      7.8 s
Context:                  0.4 s
Translation API:          3.7 s
IndexTTS2:               19.5 s
Audio mixing:             2.8 s
Subtitle detection:       1.5 s
ProPainter:              42.7 s
Subtitle render/encode:  18.9 s

TOTAL:                  155.3 s
```

Phải tính thêm:

```text
RTF = processing time / source duration
```

Ví dụ:

```text
300 s video
155 s processing

RTF = 0.52
```

Tức pipeline xử lý nhanh hơn realtime khoảng:

```text
1 / 0.52 ≈ 1.92×
```

---

## 2. Phải phân biệt thời gian model load và inference

Không được chỉ ghi:

```text
ASR = 15 s
```

nếu có thể đo chi tiết.

Nên ghi:

```text
FireRedASR2-AED

model load:        3.2 s
preprocess:        0.8 s
inference:         7.1 s
postprocess:       1.0 s

total:            12.1 s
```

Tương tự:

```text
Bandit
pyannote
IndexTTS2
ProPainter
```

Điều này rất quan trọng vì cloud thực tế có thể xử lý nhiều episode liên tục.

Ví dụ:

```text
cold:
IndexTTS load 8 s + inference 10 s

warm:
inference 10 s
```

Nếu không tách ra sẽ không biết optimization nên tập trung vào đâu.

---

## 3. Ghi cả Cold Run và Warm Run

Trong benchmark cần phân biệt:

```text
COLD RUN
```

model chưa load/cache.

với:

```text
WARM RUN
```

model/cache đã có.

Không được trộn hai loại rồi lấy trung bình.

Report:

```text
COLD RUN
Total: 185 s

WARM RUN
Total: 142 s
```

---

# 4. TUYỆT ĐỐI KHÔNG tự sửa pipeline giữa một lần test

Đây là quy tắc quan trọng.

Ví dụ đang chạy:

```text
Bandit
↓
FireRed
↓
Translation
↓
IndexTTS2
```

Agent phát hiện:

```text
FireRed transcript bất thường
```

KHÔNG được:

```text
stop
↓
sửa code
↓
đổi threshold
↓
chạy lại
```

hoặc:

```text
đổi ASR
đổi model
đổi prompt
đổi preprocessing
```

trong cùng benchmark.

Lần test đó phải được coi là:

> **một snapshot của pipeline tại thời điểm bắt đầu chạy.**

---

# 5. Nếu phát hiện lỗi cần sửa code/config/model → PHẢI hỏi người dùng trước

Agent không được tự ý sửa.

Ví dụ phát hiện:

```text
FireRed timestamp sai
```

Agent phải báo:

```text
Detected issue:

Stage:
FireRedASR2-AED

Problem:
word timestamps exceed speech segment boundary.

Evidence:
segment 14
expected: 32.4–34.8
output: 32.4–41.2

Pipeline test will continue without modifying code.

Suggested correction:
...

Do you want me to modify the implementation?
```

Chỉ khi người dùng đồng ý mới được:

```text
edit code
change configuration
change algorithm
change model
change prompt
change thresholds
```

---

# 6. Lỗi không đồng nghĩa với dừng pipeline

Trong giai đoạn thử nghiệm hiện tại:

> **Ưu tiên chạy hết pipeline để xem kết quả cuối cùng.**

Không được gặp một lỗi chất lượng ở giữa rồi dừng toàn bộ quá trình.

Ví dụ:

```text
ASR có 3 câu sai
```

→ ghi lỗi

→ vẫn translation

→ vẫn TTS

→ vẫn mix

→ vẫn subtitle processing

→ vẫn final encode.

Mục tiêu là để người dùng xem:

```text
"ASR sai như thế này thì cuối video thực tế tệ đến mức nào?"
```

chứ không phải chỉ xem log ASR.

---

# 7. Best-effort continuation

Các stage phải hỗ trợ:

```text
SUCCESS
DEGRADED
FAILED_FATAL
```

### SUCCESS

Stage chạy bình thường.

```text
continue
```

### DEGRADED

Output có vấn đề nhưng vẫn dùng được.

Ví dụ:

```text
ASR confidence thấp
một vài từ không nhận được
speaker không chắc
OCR bỏ sót subtitle
TTS duration lệch
inpainting có artifact
```

Phải:

```text
log issue
↓
mark DEGRADED
↓
continue pipeline
```

KHÔNG tự sửa.

### FAILED_FATAL

Chỉ sử dụng khi thực sự không thể tạo input cho stage tiếp theo.

Ví dụ:

```text
video không decode được
audio file hoàn toàn không tồn tại
CUDA process crash không tạo artifact
disk full
final encoder không thể mở source
```

Lúc đó mới được dừng nhánh phụ thuộc.

---

# 8. Nếu một stage fatal, vẫn phải tạo sản phẩm xem được tối đa có thể

Ví dụ:

```text
ProPainter crash
```

không có lý do để vứt toàn bộ video dub đã tạo.

Hệ thống nên xuất:

```text
preview_with_chinese_sub.mp4
```

với:

```text
Vietnamese dubbing ✓
Chinese hard-sub còn ✗
```

và report:

```text
FINAL STATUS: PARTIAL

Audio pipeline: completed
Vietnamese dubbing: completed
Subtitle removal: failed
Final encode: completed using uncleaned source frames
```

Tương tự nếu TTS một số segment lỗi:

```text
các segment thành công → dùng Viet dub
segment lỗi → để silence/source according to existing policy
```

nhưng không tự tạo một phương pháp mới giữa benchmark.

Mục tiêu:

> **Luôn cho người dùng xem output gần cuối nhất có thể.**

---

# 9. Không được giấu lỗi bằng fallback

Đặc biệt cấm:

```text
FireRed lỗi
→ Whisper chạy thay
```

vì người dùng nhìn video cuối sẽ tưởng FireRed hoạt động tốt.

Tương tự:

```text
Bandit lỗi → Kim
IndexTTS2 lỗi → VieNeu
ProPainter lỗi → LaMa
```

đều bị cấm.

Nếu FireRed sai:

> Video cuối phải phản ánh hậu quả thật sự của FireRed sai.

Đây mới là benchmark có ý nghĩa.

---

# 10. Không được tự rerun đoạn lỗi bằng implementation đã sửa

Ví dụ test lần đầu:

```text
segment 17 ASR sai
```

Agent phát hiện bug và nghĩ rằng sửa một dòng code là được.

Không được tự:

```text
edit
rerun segment 17
replace artifact
continue
```

vì kết quả cuối khi đó là hỗn hợp:

```text
pipeline version A
+
pipeline version B
```

không còn giá trị benchmark.

Phải hoàn thành run với version A trước.

Sau khi báo cáo:

```text
Run #001 complete
```

và người dùng cho phép sửa thì mới:

```text
fix
↓
Run #002
```

---

# 11. Mỗi pipeline run phải có ID và version cố định

Ví dụ:

```text
RUN ID:
2026-10-07-001

Git commit:
abc12345

Pipeline version:
V2

Model manifest:
models.lock sha256: ...

Config hash:
...

Input SHA256:
...
```

Như vậy khi so sánh:

```text
Run 001
Run 002
Run 003
```

biết chính xác thay đổi gì.

---

# 12. Trong lúc chạy, lỗi chỉ được ghi nhận chứ không được sửa

Ví dụ issue list:

```text
ISSUE-001
Stage: ASR
Segment: 17
Severity: quality
Description:
Chinese transcript appears repeated.

ISSUE-002
Stage: TTS
Segment: 23
Severity: quality
Description:
Vietnamese speech exceeds slot by 18%.

ISSUE-003
Stage: Inpainting
Time: 04:21
Severity: visual
Description:
Background flicker visible for 5 frames.
```

Pipeline tiếp tục.

---

# 13. Báo cáo cuối phải đi cùng video

Một benchmark không được coi là hoàn thành nếu chỉ có:

```text
logs
JSON
unit tests
```

Phải có ít nhất:

```text
final.mp4
+
run-report.json
+
run-report.md
```

Nếu final đầy đủ không thể tạo:

```text
partial-final.mp4
+
run-report
```

---

# 14. Báo cáo cuối phải nói rõ lỗi có ảnh hưởng gì tới video

Không chỉ:

```text
ASR failed at segment 17
```

mà phải:

```text
Segment 17
00:42.1–00:44.8

ASR:
"xxxxx"

Expected/observed issue:
likely repeated recognition.

Downstream effect:
Vietnamese translation is incorrect.

TTS effect:
incorrect Vietnamese dialogue generated.

Final video:
error remains visible/audible.

Automatic correction:
NONE
```

Đây mới giúp đánh giá pipeline thật.

---

# 15. Báo cáo thời gian phải có bottleneck

Cuối mỗi run:

```text
TIME BREAKDOWN

Bandit          20.5%
pyannote         5.1%
FireRed          8.3%
Translation      2.0%
IndexTTS2       14.7%
ProPainter      32.8%
Encode          13.0%
Other            3.6%
```

và:

```text
Largest bottleneck:
ProPainter
```

Nhưng chỉ **báo**, không tự optimization.

---

# 16. Phải ước tính chi phí từ thời gian thực

Nếu cloud:

```text
6,000 VND / hour
```

và pipeline chạy:

```text
15 min
```

report:

```text
Cloud compute:
15 / 60 × 6,000
≈ 1,500 VND
```

API:

```text
Translation tokens
input
output
estimated API cost
```

Cuối:

```text
Total estimated cost:
compute + API
```

Từ đó quy đổi:

```text
cost / minute source
cost / 2 hours source
```

để kiểm tra mục tiêu:

```text
~5,000 VND / 2h source
```

---

# 17. Không được tối ưu benchmark sau khi nhìn kết quả

Một run phải sử dụng config cố định từ đầu đến cuối.

Không được:

```text
"ProPainter chậm quá nên từ phút thứ 3 giảm resolution"
```

hoặc:

```text
"TTS lâu nên từ episode 4 đổi batch size"
```

Nếu muốn thử config khác:

```text
Run A
→ complete

Run B
→ config mới
```

sau đó so sánh.

---

# 18. Workflow trong giai đoạn hiện tại

Toàn bộ quá trình nên là:

```text
LOCK PIPELINE VERSION
        ↓
START RUN
        ↓
measure every stage
        ↓
detect issue
        ↓
LOG ONLY
        ↓
DO NOT FIX
        ↓
continue
        ↓
another issue
        ↓
LOG ONLY
        ↓
continue
        ↓
FINAL / PARTIAL FINAL VIDEO
        ↓
REPORT
        ↓
USER REVIEWS VIDEO + REPORT
        ↓
Discuss issues with user
        ↓
Ask permission
        ↓
FIX
        ↓
NEW RUN
```

---

# 19. Quy tắc dành riêng cho Agent

Có thể ghi thẳng vào `AGENTS.md`:

> During the current experimental phase, never modify implementation, model selection, prompts, thresholds, preprocessing, postprocessing, or configuration in response to an error discovered during an active pipeline run.
>
> Record the issue and continue the run whenever technically possible.
>
> Quality errors are not fatal errors.
>
> The objective is to observe the real downstream effect of errors on the final produced video.
>
> Always produce the most complete playable video possible and a full run report.
>
> After the run completes, explain discovered problems to the user and request explicit approval before making corrective changes.
>
> Never hide a model failure by invoking another model or fallback implementation.
>
> Every experimental video production run must measure per-stage wall time, model-load time where practical, total wall time, RTF, resource usage and estimated cost.

Và mình sẽ thêm một câu rất quan trọng:

> **Passing the pipeline is not the goal of an experimental run. Observing the pipeline truthfully is the goal.**

Ví dụ FireRed nhận sai 10 câu mà Agent âm thầm sửa đến khi video đẹp thì **benchmark đó gần như vô dụng**. Bạn cần nhìn video sai, report chỉ ra vì sao sai, mất bao lâu, rồi **bạn quyết định có sửa hay không**.

Đó là cách mình nghĩ giai đoạn hiện tại nên vận hành.