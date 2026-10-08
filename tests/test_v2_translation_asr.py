import json
import sys
import wave
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodub.adapters import v1_translation, v2_speech, v2_translation
from autodub.credentials import deepseek_key
from autodub.storage import atomic_json, sha256_file


def test_parallel_flash_batches_have_separate_artifacts_and_preserve_ost(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-never-serialize")
    monkeypatch.setenv("AUTODUB_WORKER_OUTPUT", str(tmp_path / "output"))
    requests = []

    def respond(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        content = {"segments": [{"id": "s1", "subtitle_vi": "Xin chào", "dub_vi": "Xin chào",
                                 "emotion": "neutral", "punctuation": "."}], "character_context": []}
        return BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(content)}}],
                                   "usage": {"total_tokens": 12}, "model": "deepseek-flash"}).encode())

    monkeypatch.setattr(v1_translation.urllib.request, "urlopen", respond)
    segments = [{"id": f"s{i}", "start_ms": i * 1000, "end_ms": (i + 1) * 1000,
                 "zh_text": "你好", "action": "DUB"} for i in range(3)]
    segments.append({"id": "ost", "start_ms": 4000, "end_ms": 5000, "zh_text": "歌曲", "action": "KEEP",
                     "dialogue_kind": "SINGING_OST"})
    result = v2_translation.run(tmp_path, {"segments": segments, "translation_batch_size": 1})
    assert len(requests) == 3 and all(r["model"] == "deepseek-flash" for r in requests)
    assert len(list((tmp_path / "output").glob("batch-*/translation.json"))) == 3
    assert result["segments"][-1] == segments[-1]
    assert all(s["dub_vi"] == "Xin chào." for s in result["segments"][:3])
    assert "private-never-serialize" not in json.dumps(result)


def test_approved_translation_is_not_overwritten_by_api(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODUB_WORKER_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(v1_translation.urllib.request, "urlopen", lambda *a, **k: pytest.fail("No API request for an approved line"))
    approved = {"id": "s", "start_ms": 0, "end_ms": 1000, "zh_text": "你好", "action": "DUB",
                "dub_vi": "Lời đã duyệt", "context_provenance": {"human_reviewed": True}}
    assert v2_translation.run(tmp_path, {"segments": [approved]})["segments"] == [approved]


def test_key_reader_accepts_private_file_and_environment_without_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    key_file = tmp_path / "API.txt.txt"
    key_file.write_text('DEEPSEEK_API_KEY="sk-private_fixture_123"\n')
    assert deepseek_key(key_file) == "sk-private_fixture_123"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-priority")
    assert deepseek_key(key_file) == "environment-priority"
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    key_file.write_text("sk-private_fixture_123\nsk-different_fixture_456")
    with pytest.raises(ValueError, match="exactly one"):
        deepseek_key(key_file)


def test_asr_batches_original_audio_without_inventing_word_times(tmp_path, monkeypatch):
    source = tmp_path / "original.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "none"))
        audio.writeframes(b"\0" * 16000 * 2 * 3)
    candidates = tmp_path / "candidates.json"
    atomic_json(candidates, {"source_sha256": sha256_file(source), "windows": [
        {"start_ms": i * 1000, "end_ms": (i + 1) * 1000, "vocal_run_ms": 3000} for i in range(3)]})
    calls = []

    class Model:
        @classmethod
        def from_pretrained(cls, *args):
            calls.append("load")
            return cls()

        def _get_and_fix_timestamp(self, *args):
            pytest.fail("No synthetic equal-spaced timestamps are allowed")

        def transcribe(self, ids, paths):
            calls.append(len(paths))
            return [{"uttid": identifier, "text": f"你好{identifier}", "confidence": 0.99,
                     "timestamp": self._get_and_fix_timestamp({"timestamp": None}, [1, 2], 1.0)} for identifier in reversed(ids)]

    module = SimpleNamespace(FireRedAsr2=Model, FireRedAsr2Config=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, 'torchaudio', SimpleNamespace(
        functional=SimpleNamespace(forced_align=lambda *args, **kwargs: None)))
    monkeypatch.setitem(sys.modules, "fireredasr2s.fireredasr2.asr", module)
    monkeypatch.setattr(v2_speech, "asset", lambda config, identifier: (tmp_path, {
        "id": identifier, "model_revision": "pinned", "weights_sha256": "hash"}))
    monkeypatch.setattr(v2_speech, "configure_offline", lambda config: None)
    monkeypatch.setenv("AUTODUB_WORKER_OUTPUT", str(tmp_path / "asr"))
    result = v2_speech.recognize(source, {"source_sha256": sha256_file(source), "candidates_path": str(candidates),
                                        "asr_batch_size": 2})
    report = json.loads(Path(result["artifacts"][0]).read_text())
    assert calls == ["load", 2, 1]
    assert len(report["segments"]) == 3
    assert all(s["zh_text"] == f"你好{s['id']}" for s in report["segments"])
    assert all(s["timing_source"] == "VAD_WINDOW" and not s["words"] for s in report["segments"])
    assert report["recognition_input"] == "ORIGINAL_SOUNDTRACK" and not report["word_times_fabricated"]


def test_reviewed_character_context_is_preserved():
    from autodub.character_context import build_context
    approved = {"id": "s", "zh_text": "你好", "speaker_id": "speaker-1", "character_id": "hero",
                "addressee_id": "mother", "addressing": {"self": "con", "other": "mẹ"},
                "context_provenance": {"human_reviewed": True}}
    context = build_context([approved], {}, {"speaker_character_map": {"speaker-1": "wrong"}}, {})
    assert approved["character_id"] == context["segments"]["s"]["character_id"] == "hero"
