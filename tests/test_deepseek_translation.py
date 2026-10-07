import json
from unittest.mock import MagicMock, patch

from autodub.adapters.local_translation import (
    _messages,
    _parse_completion,
    _translate_with_deepseek,
    find_deepseek_api_key,
    run,
)
from autodub.contracts import Segment


def test_find_deepseek_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    assert find_deepseek_api_key() == "sk-unit-test"


def test_translation_carries_scene_names_and_marks_timing_as_advisory():
    segment = Segment(id="role", start_ms=0, end_ms=100, zh_text="我要当baby")
    messages = _messages([segment], {"baby": "Baby"}, {
        "scene_context": "Children choose roles in a game.",
        "nearby_context": {"role": "昨天就是你"},
    })
    payload = json.loads(messages[1]["content"])
    assert payload["translation_priority"] == "faithful_meaning_before_duration"
    assert payload["scene_context"] == "Children choose roles in a game."
    assert payload["locked_glossary_zh_to_vi"] == {"baby": "Baby"}
    assert payload["segments"][0]["nearby_context"] == "昨天就是你"
    assert "advisory timing estimates" in messages[0]["content"]


def test_faithful_dialogue_is_preserved_even_when_it_exceeds_the_timing_estimate():
    segment = Segment(id="role", start_ms=0, end_ms=100, zh_text="我不能当baby吗")
    content = json.dumps({"segments": [{"id": "s1", "subtitle_vi": "Tớ không được đóng vai Baby sao?",
                                       "dub_vi": "Tớ không được đóng vai Baby sao?", "emotion": "questioning",
                                       "punctuation": "?"}]}, ensure_ascii=False)
    translated = _parse_completion([segment], content, {"baby": "Baby"})[0]
    assert translated.dub_vi == "Tớ không được đóng vai Baby sao?"
    assert translated.start_ms == 0 and translated.end_ms == 100


def test_translate_with_deepseek_mocked(tmp_path):
    seg = Segment(id="seg_001", start_ms=0, end_ms=2000, zh_text="你好。")
    mock_response = json.dumps({
        "choices": [{
            "message": {
                "content": json.dumps({
                    "segments": [{
                        "id": "s1",
                        "subtitle_vi": "Xin chào.",
                        "dub_vi": "Chào bạn.",
                        "emotion": "thân thiện",
                        "punctuation": ".",
                    }]
                })
            }
        }]
    }).encode("utf-8")

    mock_resp_obj = MagicMock()
    mock_resp_obj.read.return_value = mock_response
    mock_resp_obj.__enter__.return_value = mock_resp_obj

    with patch("urllib.request.urlopen", return_value=mock_resp_obj):
        res = _translate_with_deepseek([seg], {}, {}, "sk-dummy")
        assert len(res) == 1
        assert res[0].id == "seg_001"
        assert res[0].subtitle_vi == "Xin chào."
        assert res[0].dub_vi == "Chào bạn."


def test_run_deepseek_integration(tmp_path):
    seg = Segment(id="seg_001", start_ms=0, end_ms=2000, zh_text="你好。")
    mock_response = json.dumps({
        "choices": [{
            "message": {
                "content": json.dumps({
                    "segments": [{
                        "id": "s1",
                        "subtitle_vi": "Xin chào.",
                        "dub_vi": "Chào bạn.",
                        "emotion": "thân thiện",
                        "punctuation": ".",
                    }]
                })
            }
        }]
    }).encode("utf-8")

    mock_resp_obj = MagicMock()
    mock_resp_obj.read.return_value = mock_response
    mock_resp_obj.__enter__.return_value = mock_resp_obj

    out_dir = tmp_path / "out"
    dummy_source = tmp_path / "source.mp4"
    dummy_source.touch()

    with patch("urllib.request.urlopen", return_value=mock_resp_obj):
        result = run(dummy_source, {
            "output_dir": str(out_dir),
            "segments": [seg.model_dump()],
            "deepseek_api_key": "sk-dummy",
        })
        assert result["quality_metrics"]["translated_segments"] == 1
        trans_file = out_dir / "translation.json"
        assert trans_file.exists()
        saved = json.loads(trans_file.read_text(encoding="utf-8"))
        assert saved["model_id"] == "deepseek-chat"
        assert saved["segments"][0]["subtitle_vi"] == "Xin chào."
