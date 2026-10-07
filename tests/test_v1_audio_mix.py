import json
import shutil
import wave

import numpy as np
import pytest

from autodub.adapters.v1_mix import run


def pcm(path, value, seconds=1):
    with wave.open(str(path), "wb") as writer:
        writer.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        writer.writeframes(np.full((round(seconds * 48000), 2), value, dtype="<i2").tobytes())
    return path


def setup_mix(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("Media boundary test requires FFmpeg on CI")
    stems = {name: str(pcm(tmp_path / f"{name}.wav", level))
             for name, level in (("speech", 1000), ("music", 100), ("effects", 200))}
    segment = {"id": "s", "start_ms": 200, "end_ms": 700, "action": "DUB", "speech_kind": "lexical",
               "words": [{"s": 200, "e": 600, "t": "你好"}]}
    clip = pcm(tmp_path / "voice.wav", 400, 0.4)
    return {"stems": stems, "segments": [segment], "clips": {"s": str(clip)},
            "output_dir": str(tmp_path / "out")}


def test_only_lexical_span_replaced_background_and_nonverbal_retained(tmp_path):
    config = setup_mix(tmp_path)
    result = run(tmp_path / "unused", config)
    with wave.open(result["audio"], "rb") as reader:
        assert reader.getnframes() == 48000
        mixed = np.frombuffer(reader.readframes(48000), dtype="<i2").reshape(-1, 2)
    assert mixed[0, 0] == 1300
    assert mixed[24000, 0] == 700
    assert mixed[40000, 0] == 1300
    assert result["stage_status"] == "SUCCESS"


def test_overlong_voice_is_not_cut_or_mutes_original(tmp_path):
    config = setup_mix(tmp_path)
    pcm(tmp_path / "voice.wav", 400, 2)
    result = run(tmp_path / "unused", config)
    with wave.open(result["audio"], "rb") as reader:
        mixed = np.frombuffer(reader.readframes(48000), dtype="<i2")
    assert np.all(mixed == 1300)
    report = json.loads(open(result["artifacts"][0], encoding="utf-8").read())
    assert report["issues"][0]["code"] == "VOICE_EXCEEDS_FROZEN_DURATION_POLICY"


def test_keep_event_on_other_track_protects_nonverbal_audio(tmp_path):
    config = setup_mix(tmp_path)
    config["segments"].append({"id": "scream", "start_ms": 300, "end_ms": 500, "action": "KEEP"})
    result = run(tmp_path / "unused", config)
    assert result["stage_status"] == "DEGRADED"
    assert result["quality_evidence"]["issues"][0]["code"] == "PROTECTED_EVENT_COLLISION_KEEP_ORIGINAL"
