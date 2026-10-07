import json
import shutil
import subprocess
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autodub.adapters.v2_mix import RATE
from autodub.adapters.v2_mix import run as mix
from autodub.adapters.v2_speech import candidate_windows
from autodub.adapters.v2_vision import recover_crop
from autodub.config import DEFAULT_VOICE_ID
from autodub.multimodal import canonical_text, decide
from autodub.storage import sha256_file


def segment(text="你先走我马上来", **kwargs):
    return {"id": "speech", "start_ms": 1000, "end_ms": 3000, "zh_text": text,
            "action": "KEEP", "timing_source": "VAD_WINDOW", **kwargs}


def test_candidates_keep_short_speech_and_do_not_cap_dense_dialogue():
    windows = candidate_windows([False, True, False], 90, padding_ms=0, gap_ms=0)
    assert windows == [{"start_ms": 30, "end_ms": 60, "vocal_run_ms": 30}]
    dense = candidate_windows([True] * 2000, 60000)
    assert sum(w["end_ms"] - w["start_ms"] for w in dense) == 60000
    assert all(w["end_ms"] - w["start_ms"] <= 15000 for w in dense)
    assert candidate_windows([False] * 100, 3000) == []


@pytest.mark.parametrize("payload,events,kind,action", [
    (segment(), [], "UNSUBTITLED_DIALOGUE", "DUB"),
    (segment("哈啊啊", end_ms=1200), [], "NONLEXICAL", "KEEP"),
    (segment("走", end_ms=1200), [], "UNSUBTITLED_DIALOGUE", "DUB"),
    (segment(confidence={"asr": 0.2}), [], "AMBIGUOUS", "KEEP"),
    (segment(dialogue_evidence={"vocal_run_ms": 40000, "pitched_fraction": 0.95, "pitch_span_semitones": 8}),
     [{"id": "lyric", "start_ms": 0, "end_ms": 5000, "kind": "LYRIC", "text": "♫爱情", "score": 0.9}],
     "SINGING_OST", "KEEP"),
    (segment(dialogue_evidence={"vocal_run_ms": 40000, "pitched_fraction": 0.95, "pitch_span_semitones": 8}),
     [{"id": "bottom-lyric", "start_ms": 0, "end_ms": 5000, "kind": "DIALOGUE", "text": "你先走我马上来", "score": 0.9}],
     "AMBIGUOUS", "KEEP"),
    (segment(), [{"id": "caption", "start_ms": 0, "end_ms": 5000, "kind": "DIALOGUE",
                  "text": "完全不同的名字", "score": 0.9}], "AMBIGUOUS", "KEEP"),
])
def test_multimodal_matrix(payload, events, kind, action):
    result = decide(payload, events)
    assert (result["dialogue_kind"], result["action"]) == (kind, action)
    assert result["timing_source"] == "VAD_WINDOW"
    assert result["words"] == []  # Candidate times must not become fabricated word alignment.


def test_homophones_use_hanzi_evidence_without_overwriting_conflicting_text():
    assert canonical_text("他来了。", "她来了")[0] == "她来了。"
    assert canonical_text("他来了。", "你走开")[0] == "他来了。"
    event = {"id": "e1", "start_ms": 900, "end_ms": 3100, "kind": "DIALOGUE", "text": "她来了", "score": 0.95}
    result = decide(segment("他来了。"), [event])
    assert result["zh_text"] == "她来了。" and result["action"] == "DUB"
    assert result["dialogue_evidence"]["homophone_corrections"]


def test_recovery_changes_only_observed_mask_pixels_and_rejects_wrong_background():
    clean = np.full((80, 160, 3), (80, 40, 20), dtype="uint8")
    frame = clean.copy()
    mask = np.zeros((80, 160), dtype="uint8")
    mask[40:55, 50:95] = 255
    frame[mask > 0] = 255
    recovered, used = recover_crop(frame, clean, mask)
    assert used and np.array_equal(recovered, clean)
    assert np.array_equal(recovered[mask == 0], frame[mask == 0])
    wrong = np.full_like(clean, 220)
    kept, used = recover_crop(frame, wrong, mask)
    assert not used and np.array_equal(kept, frame)


def write_wav(path, values, rate=RATE):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1 if values.ndim == 1 else values.shape[1], 2, rate, 0, "NONE", "none"))
        audio.writeframes((np.clip(values, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as audio:
        values = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype("float32") / 32768
        return values.reshape(-1, audio.getnchannels())


def amplitude(audio, frequency, start, end):
    values = audio[round(start * RATE):round(end * RATE), 0]
    carrier = np.exp(-2j * np.pi * frequency * np.arange(len(values)) / RATE)
    return float(abs(np.dot(values, carrier)) * 2 / len(values))


def test_real_ffmpeg_ducking_has_band_specific_depth_and_recovers_after_voice(tmp_path, monkeypatch):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg media checks run on Ubuntu/CI")
    source, clip, output = tmp_path / "source.wav", tmp_path / "clip.wav", tmp_path / "mix"
    time = np.arange(RATE * 5) / RATE
    soundtrack = sum(0.03 * np.sin(2 * np.pi * frequency * time) for frequency in (200, 1000, 6000))
    write_wav(source, np.column_stack((soundtrack, soundtrack)))
    voice = 0.1 * np.sin(2 * np.pi * 700 * np.arange(RATE * 2) / RATE)
    write_wav(clip, voice)
    monkeypatch.setenv("AUTODUB_WORKER_OUTPUT", str(output))
    monkeypatch.setenv("AUTODUB_WORKER_RUN_ROOT", str(tmp_path))
    result = mix(source, {"duration_ms": 5000, "segments": [segment(action="DUB", dub_vi="Xin chào")],
                          "clips": {"speech": str(clip)}, "ffmpeg_bin": ffmpeg})
    mastered = read_wav(result["audio"])
    assert mastered.shape == (RATE * 5, 2)
    mid = 20 * np.log10(amplitude(mastered, 1000, 1.6, 2.7) / amplitude(mastered, 1000, 0.3, 0.8))
    low = 20 * np.log10(amplitude(mastered, 200, 1.6, 2.7) / amplitude(mastered, 200, 0.3, 0.8))
    high = 20 * np.log10(amplitude(mastered, 6000, 1.6, 2.7) / amplitude(mastered, 6000, 0.3, 0.8))
    assert -15.5 <= mid <= -12.5
    assert -4 <= low <= -2 and -4 <= high <= -2
    assert amplitude(mastered, 700, 1.6, 2.7) > 0.1
    restored = amplitude(mastered, 1000, 4.3, 4.8) / amplitude(mastered, 1000, 0.3, 0.8)
    assert 0.9 <= restored <= 1.1
    report = json.loads(Path(result["artifacts"][0]).read_text())
    assert report["separation_used"] is False and report["voice_added_count"] == 1


def test_long_tts_is_reported_and_never_truncated(tmp_path, monkeypatch):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg media checks run on Ubuntu/CI")
    source, clip = tmp_path / "source.wav", tmp_path / "long.wav"
    write_wav(source, np.zeros(RATE * 5))
    write_wav(clip, np.ones(RATE * 4) * 0.03)
    monkeypatch.setenv("AUTODUB_WORKER_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("AUTODUB_WORKER_RUN_ROOT", str(tmp_path))
    result = mix(source, {"duration_ms": 5000, "segments": [segment(action="DUB", dub_vi="Dài")],
                          "clips": {"speech": str(clip)}, "ffmpeg_bin": ffmpeg})
    report = json.loads(Path(result["artifacts"][0]).read_text())
    assert report["voice_added_count"] == 0
    assert report["issues"][0]["code"] == "TTS_TOO_LONG_NO_TRUNCATION"
    assert result["stage_status"] == "DEGRADED"


def test_v2_running_prevents_delete_and_preserves_api_secret(client, app, uploaded, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-PRIVATE_NEVER_RETURN")
    app.state.db.execute("UPDATE episode SET status='V2_RUNNING' WHERE id=?", (uploaded["id"],))
    assert client.delete(f"/api/episodes/{uploaded['id']}").status_code == 409
    assert "PRIVATE_NEVER_RETURN" not in client.get("/api/system/status").text


def test_review_accepts_v2_candidate_timing_without_forging_word_times(client, app, uploaded):
    app.state.db.execute("UPDATE episode SET status='NEEDS_REVIEW',duration_ms=5000 WHERE id=?", (uploaded["id"],))
    from autodub.contracts import Segment
    value = Segment.model_validate(segment(action="KEEP", needs_review=True))
    app.state.db.execute("INSERT INTO segment VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
        "db:speech", uploaded["id"], 1000, 3000, value.zh_text, "", "", None,
        "KEEP", "PENDING", value.model_dump_json()))
    app.state.service.settings = replace(app.state.service.settings, pipeline_generation="v2")
    payload = value.model_copy(update={"action": "DUB", "dub_vi": "Anh đi trước", "needs_review": False}).model_dump()
    result = client.put(f"/api/episodes/{uploaded['id']}/segments", json=[payload])
    assert result.status_code == 200
    assert result.json()[0]["words"] == [] and result.json()[0]["timing_source"] == "VAD_WINDOW"
    from autodub.pipeline import Pipeline
    Pipeline(app.state.db, app.state.service.settings)._save_segments_to_db(
        uploaded["id"], [Segment.model_validate(result.json()[0])])
    assert app.state.db.one("SELECT status FROM segment WHERE episode_id=?", (uploaded["id"],))["status"] == "APPROVED"


def test_v2_plan_selects_only_requested_production_assets():
    import sys
    repo = Path(__file__).resolve().parents[1]
    plan = json.loads(subprocess.check_output([sys.executable, str(repo / "scripts/cloud_plan.py")], text=True))
    assert plan["profiles"] == ["asr", "tts", "vision"]
    assert not any("bandit" in name or "roformer" in name or "propainter" in name or "indextts" in name for name in plan["assets"])
    tts = subprocess.check_output([sys.executable, str(repo / "scripts/cloud_plan.py"), "--profile", "tts", "--field", "assets"], text=True)
    assert set(tts.splitlines()) == {"vieneu-turbo", "moss-torch", "ngoc-huyen-reference"}


def test_voice_reference_is_pinned_to_user_selected_audio():
    repo = Path(__file__).resolve().parents[1]
    lock = json.loads((repo / "benchmarks/models.lock.json").read_text())
    reference = next(m for m in lock["models"] if m["id"] == "ngoc-huyen-reference")
    assert reference["files"][0]["sha256"] == "194625b4844e250683b442640abd4ae60d794540b9e915c879b9e98e2d11a734"
    assert DEFAULT_VOICE_ID == "Ngọc Huyền"


def test_batched_tts_loads_and_enrolls_fixed_voice_once(tmp_path, monkeypatch):
    import sys

    from autodub.adapters import v2_tts
    reference = tmp_path / "ngoc.wav"
    reference.write_bytes(b"fixed reference")
    output = tmp_path / "output"
    calls = []

    class Model:
        sample_rate = RATE

        def __init__(self, **kwargs):
            calls.append(("load", kwargs))

        def add_voice(self, name, path, **kwargs):
            calls.append(("reference", name, str(path)))

        def infer_batch(self, texts, **kwargs):
            calls.append(("batch", len(texts), kwargs["voice"]))
            return [np.ones(1000, dtype="float32") * 0.05 for _ in texts]

        def close(self):
            calls.append(("close",))

    monkeypatch.setitem(sys.modules, "vieneu", SimpleNamespace(Vieneu=Model))
    monkeypatch.setattr(v2_tts, "asset", lambda config, identifier: (tmp_path, {"id": identifier, "model_revision": "pinned", "weights_sha256": "hash"}))
    monkeypatch.setattr(v2_tts, "configure_offline", lambda config: None)
    from contextlib import nullcontext
    monkeypatch.setattr(v2_tts, "local_hub_files", lambda config: nullcontext())
    monkeypatch.setenv("AUTODUB_WORKER_OUTPUT", str(output))
    result = v2_tts.run(reference, {"voice_reference": str(reference), "voice_reference_sha256": sha256_file(reference),
                                  "tts_batch_size": 2, "segments": [segment(id=f"s{i}", action="DUB", dub_vi="Chào") for i in range(5)]})
    assert len(result["clips"]) == 5
    assert sum(c[0] == "load" for c in calls) == sum(c[0] == "reference" for c in calls) == 1
    assert [c[1] for c in calls if c[0] == "batch"] == [2, 2, 1]
    assert all(c[2] == DEFAULT_VOICE_ID for c in calls if c[0] == "batch")
    assert calls[-1] == ("close",)
