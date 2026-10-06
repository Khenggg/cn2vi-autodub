# DashScope translation provider

`DashScopeTranslationProvider` implements the `TranslationProvider` contract through Alibaba Cloud Model Studio's OpenAI-compatible Chat Completions interface. It uses the standard library only. Configure both `DASHSCOPE_API_KEY` and `QWEN_TRANSLATION_MODEL`; the model name is required from the deployment owner because model availability and access depend on the account and region. The provider deliberately has no model-name fallback. The optional constructor config accepts `base_url`, `timeout`, `max_attempts` (1 through 3), and `force_review` (default false); `force_review: true` marks translated segments for review without changing their actions.

The default base URL is Beijing. Set `DASHSCOPE_BASE_URL` or pass `base_url` to select exactly one of these allowlisted endpoints:

- Beijing: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Singapore: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- US (Virginia): `https://dashscope-us.aliyuncs.com/compatible-mode/v1`

The API key and model must belong to the selected region. The provider posts to `/chat/completions`, sends the configured model and `response_format: {"type":"json_object"}`, and reads `choices[0].message.content` plus optional token counts in `usage`. Its prompt requests JSON explicitly as required by DashScope's JSON-object mode. Token counts are exposed in `provider.last_usage`; no cost is estimated.

Each response must contain exactly one result per input ID and exactly the output fields `id`, `subtitle_vi`, `dub_vi`, `emotion`, and `punctuation`. Unknown, duplicate, or missing IDs, extra/missing fields, invalid JSON, or unsupported punctuation reject the entire batch. The punctuation field is the final mark (`.`, `!`, `?`, ellipsis, or corresponding Chinese mark); it is appended to each Vietnamese output only if that text lacks terminal punctuation. Timings, words, action, confidence, schema version, and review flags are copied from the original segment. The provider does not set the action to `DUB` or change an existing action. A segment already marked `needs_review` stays marked; translation output is not semantic proof and still needs review under the product workflow.

HTTP 429, HTTP 5xx, timeouts, and network transport errors retry with exponential delays, with at most three attempts. Other HTTP errors fail immediately. Exceptions contain only a generic failure and safe status code; API keys and response bodies are never included. Requests use an injectable transport so tests can inspect payloads and simulate failures without making network calls.

`translation_context_hash(series_id, source_text, nearby_context, glossary_version)` creates a SHA-256 key over normalized source/context text and glossary version for a future translation-memory lookup. It does not write to SQLite or retain text itself.

## Documentation verification

Verified against Alibaba Cloud's official docs on 2026-10-06: the [regional base URL table](https://help.aliyun.com/en/model-studio/base-url) lists the Beijing, Singapore, and US compatible endpoints; the [OpenAI-compatible Chat API](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions) documents `POST /chat/completions`, the `choices[].message.content` response, usage token counts, and `response_format: {"type":"json_object"}`; the [structured output guide](https://help.aliyun.com/en/model-studio/qwen-structured-output) requires the prompt to include the JSON keyword. No model inference or API request was made while implementing or testing this adapter.
