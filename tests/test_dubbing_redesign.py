import json
import shutil
import sys
import wave
from types import SimpleNamespace

import pytest

from autodub import audio_mix
from autodub.adapters import local_translation, qwen
from autodub.contracts import Segment
from autodub.storage import sha256_file


def _tone(path, ms, rate=24000):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x10\x20" * (rate * ms // 1000))


def _frames_ms(path):
    with wave.open(str(path), "rb") as stream:
        return round(stream.getnframes() * 1000 / stream.getframerate())


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


@needs_ffmpeg
@pytest.mark.parametrize(("clip_ms", "target_ms", "review", "action"), [
    (3000, 1000, True, "REWRITE"),  # do not cut speech to force it into the slot
    (1300, 1000, True, "REWRITE"),
    (400, 1000, True, "REWRITE"),
])
def test_elastic_fit_never_discards_speech(tmp_path, clip_ms, target_ms, review, action):
    clip, out = tmp_path / "clip.wav", tmp_path / "out.wav"
    _tone(clip, clip_ms)
    info = audio_mix.fit_voice(clip, out, target_ms, elastic=True)
    assert info["review_required"] is review
    assert info["action"] == action
    if action == "REWRITE":
        assert not out.exists()
    else:
        assert abs(_frames_ms(out) - target_ms) <= 20


@needs_ffmpeg
def test_strict_fit_still_rejects_large_drift(tmp_path):
    clip = tmp_path / "clip.wav"
    _tone(clip, 3000)
    info = audio_mix.fit_voice(clip, tmp_path / "out.wav", 1000)
    assert info["action"] == "REWRITE" and not (tmp_path / "out.wav").exists()


def test_translation_prompt_carries_a_vietnamese_word_budget():
    segment = Segment(id="s", start_ms=0, end_ms=2000, zh_text="你好")
    assert local_translation.word_budget(2000) == 11
    assert local_translation.word_budget(100) == 2
    messages = local_translation._messages([segment], {}, {})
    assert json.loads(messages[1]["content"])["segments"][0]["max_vietnamese_words"] == 11
    assert "max_vietnamese_words" in messages[0]["content"]


def test_run_alignment_emits_one_segment_per_utterance_and_trims_chunk_overlap(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"fake source")
    transcript = tmp_path / "transcript.json"
    chunks = [
        {"id": "seg_000000000000", "start_ms": 0, "end_ms": 10000, "zh_text": "你好吗。再见。"},
        {"id": "seg_000000009000", "start_ms": 9000, "end_ms": 20000, "zh_text": "再见。谢谢你。"},
        {"id": "seg_000000019000", "start_ms": 19000, "end_ms": 30000, "zh_text": "   "},
    ]
    transcript.write_text(json.dumps({"schema_version": 1, "source_sha256": sha256_file(source),
                                      "segments": chunks}), encoding="utf-8")

    def item(text, start, end):
        return SimpleNamespace(text=text, start_time=start, end_time=end)

    # Times are relative to each chunk.  "再见" is spoken at absolute 9.5-10.5 s,
    # i.e. inside the overlap, and must be emitted exactly once.
    results = iter([
        [[item("你好吗", "1.0", "2.0"), item("再见", "9.5", "10.0")]],
        [[item("再见", "0.5", "1.5"), item("谢谢你", "5.0", "6.0")]],
    ])

    class FakeAligner:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def align(self, **_kwargs):
            return next(results)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ForcedAligner=FakeAligner))
    monkeypatch.setattr(qwen, "configure_offline", lambda *_: None)
    monkeypatch.setattr(qwen, "asset", lambda *_: (tmp_path, {"id": "aligner"}))
    monkeypatch.setattr(qwen, "output_folder", lambda _config: tmp_path)
    monkeypatch.setattr(qwen, "torch_device", lambda *_: ("cpu", None))
    monkeypatch.setattr(qwen, "synchronize", lambda *_: None)
    monkeypatch.setattr(qwen, "allocator_metrics", lambda *_: {})
    monkeypatch.setattr(qwen, "identity", lambda *_: {})
    monkeypatch.setattr(qwen, "word_boundary_error", lambda *_: {})
    monkeypatch.setattr(qwen.media, "probe_media", lambda *_: {"duration_ms": 30000})
    monkeypatch.setattr(qwen, "extract_audio", lambda *_a, **_k: tmp_path / "chunk.wav")

    qwen.run_alignment(source, {"transcript_path": str(transcript), "device": "cpu"})

    aligned = json.loads((tmp_path / "alignment.json").read_text(encoding="utf-8"))
    segments = aligned["segments"]
    assert [s["zh_text"] for s in segments] == ["你好吗。", "再见。", "谢谢你。"]
    assert [(s["start_ms"], s["end_ms"]) for s in segments] == [(1000, 2000), (9500, 10500), (14000, 15000)]
    assert len({s["id"] for s in segments}) == 3
    assert all(a["end_ms"] <= b["start_ms"] for a, b in zip(segments, segments[1:], strict=False))
    assert all(s["words"] for s in segments)
    assert aligned["segmentation"] == "utterance"
