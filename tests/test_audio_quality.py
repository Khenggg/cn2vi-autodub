import array
import hashlib
import json
import shutil
import wave

import pytest

from autodub.audio_mix import fit_voice, mix_preview

RATE = 48_000


def _write_stereo(path, duration_ms, left, right):
    frames = RATE * duration_ms // 1000
    data = array.array("h", (sample for _ in range(frames) for sample in (left, right)))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(data.tobytes())


def _read_stereo(path):
    with wave.open(str(path), "rb") as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (2, 2, RATE)
        return array.array("h", stream.readframes(stream.getnframes()))


def _segment(segment_id="s1", action="DUB", start=500, end=2500):
    return {"id": segment_id, "action": action, "start_ms": start, "end_ms": end,
            "words": [{"t": "你好", "s": 800, "e": 1000},
                      {"t": "再见", "s": 1800, "e": 2000}] if action == "DUB" else []}


def _separation(tmp_path, dialogue):
    manifest = tmp_path / "separation.json"
    manifest.write_text(json.dumps({"schema_version": 1, "windows": [
        {"start_ms": 0, "end_ms": 3000, "dialogue": dialogue.name}]}), encoding="utf-8")
    return manifest


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


@needs_ffmpeg
def test_word_removal_preserves_source_gaps_and_adds_full_voice_slot(tmp_path):
    source = tmp_path / "source.wav"
    dialogue = tmp_path / "dialogue.wav"
    voice = tmp_path / "voice.wav"
    output = tmp_path / "preview.wav"
    _write_stereo(source, 3000, 1000, 2000)
    # The separator estimate is nonzero only under the two aligned Chinese words.
    dialogue_values = array.array("h")
    for frame in range(RATE * 3):
        ms = frame * 1000 // RATE
        value = 600 if 800 <= ms < 1000 or 1800 <= ms < 2000 else 0
        dialogue_values.extend((value, value))
    with wave.open(str(dialogue), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(dialogue_values.tobytes())
    _write_stereo(voice, 2000, 100, 300)
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    result = mix_preview(source, output, [_segment()], _separation(tmp_path, dialogue), {"s1": voice})
    samples = _read_stereo(output)

    # Source dialogue is removed under an aligned word; original ambience stays
    # under the word and across the inter-word gap while Vietnamese plays there.
    word = 900 * RATE // 1000 * 2
    gap = 1400 * RATE // 1000 * 2
    outside = 100 * RATE // 1000 * 2
    assert samples[word:word + 2] == array.array("h", [500, 1700])
    assert samples[gap:gap + 2] == array.array("h", [1100, 2300])
    assert samples[outside:outside + 2] == array.array("h", [1000, 2000])
    assert result["modified_ranges_ms"] == [[800, 1000], [1800, 2000]]
    assert result["voice_added_ranges_ms"] == [[500, 2500]]
    assert result["source_component_preserved_outside_removal_mask"] is True
    assert result["output_differs_outside_removal_mask_due_to_voice_addition"] is True
    assert result["human_listening_required"] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before == result["source_sha256"]


@needs_ffmpeg
def test_all_keep_is_bit_exact_stereo_passthrough_without_separation_windows(tmp_path):
    source, output = tmp_path / "source.wav", tmp_path / "preview.wav"
    _write_stereo(source, 1000, 1234, -2345)

    result = mix_preview(source, output, [_segment(action="KEEP", start=0, end=1000)],
                         tmp_path / "no-separation.json", {})

    assert _read_stereo(output) == _read_stereo(source)
    assert result["qc"]["channels"] == 2
    assert result["qc"]["sample_rate_hz"] == RATE
    assert result["qc"]["technical_quality_pass"] is True
    assert result["modified_ranges_ms"] == []


@needs_ffmpeg
def test_empty_separation_windows_return_decoded_source_without_failure(tmp_path):
    source, output = tmp_path / "source.wav", tmp_path / "preview.wav"
    manifest = tmp_path / "separation.json"
    _write_stereo(source, 3000, 321, -654)
    manifest.write_text(json.dumps({"schema_version": 1, "windows": []}), encoding="utf-8")

    result = mix_preview(source, output, [_segment()], manifest, {"s1": tmp_path / "not-produced.wav"})

    assert _read_stereo(output) == _read_stereo(source)
    assert result["modified_ranges_ms"] == []
    assert result["voice_added_ranges_ms"] == []
    assert result["technical_quality_pass"] is True


@needs_ffmpeg
def test_clipping_is_reported_as_technical_quality_failure(tmp_path):
    source, dialogue, voice = (tmp_path / name for name in ("source.wav", "dialogue.wav", "voice.wav"))
    _write_stereo(source, 3000, 30000, 30000)
    _write_stereo(dialogue, 3000, 0, 0)
    _write_stereo(voice, 2000, 10000, 10000)
    output = tmp_path / "preview.wav"

    result = mix_preview(source, output, [_segment()], _separation(tmp_path, dialogue), {"s1": voice})

    assert result["qc"]["clipped_sample_ratio"] > 0
    assert result["qc"]["clipping_pass"] is False
    assert result["technical_quality_pass"] is False
    assert result["human_listening_required"] is True
    assert result["automatic_quality_pass"] is False


def test_duplicate_dub_ids_and_orphan_clips_are_rejected(tmp_path):
    segment_a, segment_b = _segment(), _segment(start=2500, end=3000)
    segment_b["words"] = [{"t": "再见", "s": 2600, "e": 2800}]
    with pytest.raises(ValueError, match="ids must be unique"):
        mix_preview(tmp_path / "unused", tmp_path / "out", [segment_a, segment_b],
                    tmp_path / "unused.json", {"s1": tmp_path / "clip.wav"})
    with pytest.raises(ValueError, match="match DUB segment ids"):
        mix_preview(tmp_path / "unused", tmp_path / "out", [_segment(action="KEEP")],
                    tmp_path / "unused.json", {"orphan": tmp_path / "clip.wav"})


@needs_ffmpeg
def test_fit_overrun_requests_rewrite_without_cutting_audio(tmp_path):
    source, output = tmp_path / "long.wav", tmp_path / "fitted.wav"
    _write_stereo(source, 1500, 1000, 2000)

    result = fit_voice(source, output, 1000)

    assert result["action"] == "REWRITE"
    assert result["failure_code"] == "DURATION_DRIFT_OVER_20_PERCENT"
    assert result["raw_ms"] == 1500
    assert result["target_ms"] == 1000
    assert result["review_required"] is True
    assert not output.exists()
