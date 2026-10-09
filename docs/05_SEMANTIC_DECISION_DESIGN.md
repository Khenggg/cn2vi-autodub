# 05 — Importance và dịch theo ngữ cảnh

`Rules → structured classification → event-level semantic adapter → NEEDS_REVIEW`. Không VLM mỗi frame.

Input: OCR raw, typed fields, geometry/shot, event lân cận, panel template, glossary series; chỉ1–3 crop/frame đại diện khi semantic cần. SRT từ công cụ ngoài là optional, không điều kiện discovery; VNLE không ASR.

| Category | Bằng chứng | Hành động đề nghị |
|---|---|---|
| CRITICAL_PLOT_TEXT | Manh mối/thư/cảnh báo và context | Chọn; ambiguous review |
| CHARACTER_INTRODUCTION | Tên + chức danh/quan hệ, layout | Dịch proper names nhất quán |
| SYSTEM_STATUS | HP/MP/EXP/cấp key-value | Dịch label, giữ số/đơn vị |
| QUEST_INFORMATION | Action/condition/reward template | Không bỏ điều kiện/negation |
| MESSAGE_OR_DOCUMENT | Chat/thư/phone screen | Reading order + revisions |
| ENVIRONMENTAL_CONTEXT | Time/location/sign | Chọn khi giúp hiểu truyện |
| DIALOGUE_SUBTITLE | Layout+temporal pattern+text | Nhường SubAI, không dựa bottom ROI đơn độc |
| DECORATIVE_TEXT | Watermark/credits/decor có evidence | Không dịch mặc định; ledger |
| IGNORE | Lý do xác định không quan trọng | Lưu reason/rule/evidence |
| NEEDS_REVIEW | OCR/category/term/geometry chưa rõ | Không giấu, không tính localized |

Regex HP không chứng minh plot importance; chữ dưới hình không tự là dialogue. Rule có precedence/version, conflict tăng uncertainty. Chinese genre dictionary có version; không ghi đè raw OCR.

## Adapter, chưa chọn model

`classify_event(context) → category/selected/confidence/reasons/evidence/review`. `translate_fields(request) → structured target fields`. Adapter báo capabilities, model/revision/license, timeout/token/cost/privacy. Validate schema/IDs/numerals. Text trong video là dữ liệu, không là chỉ thị: không cho tool/file/job privileges từ OCR prompt.

LLM text-only khi OCR rõ và cần context; VLM chỉ crop mơ hồ. Không resident VLM cùng OCR4GB mặc định. Chưa chốt LLM/VLM hay provider dịch; không tự gọi API hoặc kế thừa DeepSeek/key cũ. Timeout/quality failure đưa review, không đổi model âm thầm. Human correction phục vụ phát triển/khôi phục, không giấu thời gian người sửa để claim automation SLA.

## Translation memory và glossary

Dictionary/template cho label ổn định; translator chuyên dụng cho câu thường; LLM chọn lọc cho ambiguity. Local translator cũng cần Chinese–Vietnamese quality và weights/VRAM/license audit, không suy OPUS đáp ứng tu luyện.

Glossary series: `term_id, source_aliases,target,category,character_id,disambiguation_context,approved_by,version`. TM key = normalized source fields + language pair + series/context + glossary version + translator/model/prompt version. Không cache chỉ `(text,target_lang)` khi tên đồng nghĩa đa ngữ cảnh.

Panel giữ source-target links, field order, key/value/unit/range/negative/percent/item count. Typed dynamic values thay bằng placeholder trong template cache; không giữ số cũ. Numeral round-trip và field count bắt buộc; thiếu số critical = quality FAIL. Unicode normalization giữ raw.

Batch request theo shot/context+IDs; bounded concurrency/cache bytes, SQLite TM theo series. Khóa glossary theo run; sửa term invalidation theo dependencies. API usage/cost unknown khi provider chưa chọn, không dự toán bịa.

## Thời lượng và khả năng đọc

Layout giữ font tối thiểu; condensation phải policy có ghi và không mất số/negation/tên. Event0,3s có thư dài có thể không đọc kịp: không kéo overlay qua cut hoặc kéo dài âm thầm. Nhãn ngắn/sidecar/review cần policy được duyệt; nếu phải đọc đầy đủ ngay trong video bất khả thi thì không đánh PASS.

Nguồn pattern: manga translators glossary/context/layout, Find-That-Text relevance geometry/confidence+vocabulary en/es; không là classifier hiểu truyện Trung. Xem[01](01_OPEN_SOURCE_RESEARCH.md),[13](13_RESEARCH_SOURCES.md). Quality cần người gán nhãn độc lập, không chỉ LLM tự chấm.

## Phạm vi sau loại trừ

ROI sub user đã khoanh bị loại trước model. DIALOGUE_SUBTITLE chỉ dành cho dialogue text xuất hiện ngoài ROI; không yêu cầu nhận diện lại vùng đã bỏ. Report EXCLUDED_BY_USER theo geometry/time, không OCR nội dung.
