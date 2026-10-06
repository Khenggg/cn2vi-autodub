import json
from unittest.mock import patch

from autodub.adapters import roformer


def test_is_roformer_available():
    # Should safely return a boolean without crashing
    res = roformer.is_roformer_available()
    assert isinstance(res, bool)


def test_roformer_fallback_to_bandit(tmp_path):
    dummy_source = tmp_path / "source.mp4"
    dummy_source.touch()
    out_dir = tmp_path / "out"

    # When audio-separator is not installed, it should fallback to bandit
    with patch("autodub.adapters.roformer.is_roformer_available", return_value=False), \
         patch("autodub.adapters.bandit.run", return_value={"artifacts": ["dummy.json"]}) as mock_bandit:
        res = roformer.run(dummy_source, {
            "output_dir": str(out_dir),
            "ffprobe_bin": "ffprobe",
            "windows": [{"start_ms": 0, "end_ms": 1000}],
        })
        mock_bandit.assert_called_once()
        assert res["artifacts"] == ["dummy.json"]


def test_roformer_run_mocked(tmp_path):
    dummy_source = tmp_path / "source.mp4"
    dummy_source.touch()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    vocals = out_dir / "vocals.wav"
    inst = out_dir / "instrumental.wav"
    vocals.touch()
    inst.touch()

    with patch("autodub.adapters.roformer.is_roformer_available", return_value=True), \
         patch("autodub.adapters.roformer.probe_media", return_value={"duration_ms": 60000}), \
         patch("autodub.adapters.roformer.extract_audio"), \
         patch("autodub.adapters.roformer._run_with_audio_separator", return_value=(vocals, inst)):

        res = roformer.run(dummy_source, {
            "output_dir": str(out_dir),
            "ffprobe_bin": "ffprobe",
            "ffmpeg_bin": "ffmpeg",
        })

        assert res["processed_media_ms"] == 60000
        sep_json = out_dir / "separation.json"
        assert sep_json.exists()
        data = json.loads(sep_json.read_text(encoding="utf-8"))
        assert len(data["windows"]) == 1
        assert data["windows"][0]["dialogue"] == "vocals.wav"
        assert data["windows"][0]["start_ms"] == 0
        assert data["windows"][0]["end_ms"] == 60000
