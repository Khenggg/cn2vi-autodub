import json

import pytest

from autodub.adapters.translation import (
    ALLOWED_BASE_URLS,
    BEIJING_BASE_URL,
    DashScopeTranslationProvider,
    TranslationProviderError,
    translation_context_hash,
)
from autodub.contracts import Segment, Word


def _segment(identifier="s1", *, action="DUB", needs_review=False):
    return Segment(id=identifier, start_ms=100, end_ms=900, zh_text="你好。",
                   words=[Word(t="你好", s=120, e=500)], subtitle_vi="", dub_vi="",
                   action=action, emotion="neutral", confidence={"asr": 0.91},
                   needs_review=needs_review)


def _completion(items, *, usage=None):
    return json.dumps({"choices": [{"message": {"content": json.dumps({"segments": items}, ensure_ascii=False)}}],
                       "usage": usage}, ensure_ascii=False).encode("utf-8")


def _item(identifier="s1", **overrides):
    value = {"id": identifier, "subtitle_vi": "Xin chào", "dub_vi": "Chào bạn",
             "emotion": "warm", "punctuation": "!"}
    value.update(overrides)
    return value


def test_payload_uses_locked_glossary_explicit_model_json_mode_and_exact_endpoint(monkeypatch):
    captured = {}

    def transport(url, headers, body, timeout):
        captured.update(url=url, headers=headers, body=json.loads(body), timeout=timeout)
        return 200, _completion([_item()], usage={"prompt_tokens": 31, "completion_tokens": 12,
                                                 "total_tokens": 43})

    provider = DashScopeTranslationProvider(api_key="test-secret", model="chosen-model", transport=transport)
    result = provider.translate([_segment()], {"张老师": "thầy Trương"})
    assert captured["url"] == BEIJING_BASE_URL + "/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-secret"
    assert captured["body"]["model"] == "chosen-model"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["stream"] is False
    prompt = " ".join(message["content"] for message in captured["body"]["messages"])
    assert "JSON" in prompt.upper()
    assert "locked_glossary_zh_to_vi" in prompt and "thầy Trương" in prompt
    assert "exactly" in prompt and "punctuation" in prompt
    assert result[0].subtitle_vi == "Xin chào!"
    assert result[0].dub_vi == "Chào bạn!"
    assert result[0].emotion == "warm"
    assert result[0].start_ms == 100 and result[0].end_ms == 900
    assert result[0].words == [Word(t="你好", s=120, e=500)]
    assert result[0].action == "DUB" and result[0].confidence == {"asr": 0.91}
    assert provider.last_usage == {"prompt_tokens": 31, "completion_tokens": 12, "total_tokens": 43}


def test_preserves_keep_needs_review_and_all_nontranslation_fields():
    source = _segment(action="KEEP", needs_review=True)
    provider = DashScopeTranslationProvider(api_key="k", model="m",
        transport=lambda *_: (200, _completion([_item()])))
    translated = provider.translate([source], {})[0]
    assert translated.action == "KEEP"
    assert translated.needs_review is True
    assert translated.start_ms == source.start_ms and translated.end_ms == source.end_ms
    assert translated.words == source.words and translated.confidence == source.confidence


def test_config_force_review_marks_output_without_changing_action():
    source = _segment(action="KEEP")
    provider = DashScopeTranslationProvider({"force_review": True}, api_key="k", model="m",
        transport=lambda *_: (200, _completion([_item()])))
    translated = provider.translate([source], {})[0]
    assert translated.action == "KEEP"
    assert translated.needs_review is True


@pytest.mark.parametrize("items", [
    [_item("unknown")],
    [],
    [_item(), _item()],
    [_item(extra="surprise")],
    [{key: value for key, value in _item().items() if key != "emotion"}],
])
def test_invalid_ids_or_fields_fail_whole_batch(items):
    provider = DashScopeTranslationProvider(api_key="k", model="m",
        transport=lambda *_: (200, _completion(items)))
    with pytest.raises(TranslationProviderError, match="batch rejected|ids do not match"):
        provider.translate([_segment()], {})


def test_invalid_json_is_never_skipped_or_partially_returned():
    body = json.dumps({"choices": [{"message": {"content": "not-json"}}]}).encode()
    provider = DashScopeTranslationProvider(api_key="k", model="m", transport=lambda *_: (200, body))
    with pytest.raises(TranslationProviderError, match="invalid JSON"):
        provider.translate([_segment(), _segment("s2")], {})


def test_retries_429_and_5xx_with_bounded_exponential_backoff():
    outcomes = iter([(429, b"secret provider body"), (503, b"failure"),
                     (200, _completion([_item()]))])
    delays = []
    calls = []

    def transport(*args):
        calls.append(args)
        return next(outcomes)

    provider = DashScopeTranslationProvider(api_key="secret-key", model="m", transport=transport,
                                             sleep=delays.append)
    assert provider.translate([_segment()], {})[0].subtitle_vi == "Xin chào!"
    assert len(calls) == 3
    assert delays == [0.25, 0.5]


def test_timeout_retries_three_times_then_returns_redacted_error():
    delays = []
    calls = []

    def transport(*_args):
        calls.append(1)
        raise TimeoutError("socket error includes secret-key")

    provider = DashScopeTranslationProvider(api_key="secret-key", model="m", transport=transport,
                                             sleep=delays.append)
    with pytest.raises(TranslationProviderError) as error:
        provider.translate([_segment()], {})
    assert len(calls) == 3 and delays == [0.25, 0.5]
    assert "secret-key" not in str(error.value)


def test_non_transient_errors_are_redacted_and_not_retried():
    calls = []

    def transport(*_args):
        calls.append(1)
        raise RuntimeError("failure with secret-key")

    provider = DashScopeTranslationProvider(api_key="secret-key", model="m", transport=transport)
    with pytest.raises(TranslationProviderError, match="transport failed") as error:
        provider.translate([_segment()], {})
    assert calls == [1]
    assert "secret-key" not in str(error.value)


def test_missing_key_or_model_fails_before_transport(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_TRANSLATION_MODEL", raising=False)
    called = []
    provider = DashScopeTranslationProvider(transport=lambda *_: called.append(1))
    with pytest.raises(TranslationProviderError, match="DASHSCOPE_API_KEY"):
        provider.translate([_segment()], {})
    assert called == []
    with pytest.raises(TranslationProviderError, match="QWEN_TRANSLATION_MODEL"):
        DashScopeTranslationProvider(api_key="k", transport=lambda *_: called.append(1)).translate([_segment()], {})
    assert called == []


def test_endpoints_are_exact_allowlist_and_supported_regions_are_documented():
    assert len(ALLOWED_BASE_URLS) == 3
    for invalid in ("https://evil.example/compatible-mode/v1", BEIJING_BASE_URL + "/", "http://dashscope.aliyuncs.com/compatible-mode/v1"):
        with pytest.raises(ValueError, match="allowlisted"):
            DashScopeTranslationProvider(base_url=invalid)


def test_translation_memory_hash_uses_series_source_context_and_glossary_version():
    original = translation_context_hash("series-1", " 你好\n世界 ", "前一句", 3)
    assert original == translation_context_hash("series-1", "你好 世界", "前一句", 3)
    assert original != translation_context_hash("series-2", "你好 世界", "前一句", 3)
    assert original != translation_context_hash("series-1", "你好 世界", "后一句", 3)
    assert original != translation_context_hash("series-1", "你好 世界", "前一句", 4)
