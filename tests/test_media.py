import os
import shutil
import subprocess
from pathlib import Path

import pytest

from autodub.media import MediaError, probe_audio, probe_media


def test_real_media_probe_and_preparation(tmp_path, monkeypatch):
    ffmpeg = os.getenv("FFMPEG_BIN", "ffmpeg")
    ffprobe = os.getenv("FFPROBE_BIN", "ffprobe")
    if not shutil.which(ffmpeg) or not shutil.which(ffprobe):
        pytest.skip("FFmpeg/FFprobe not installed")
    source = tmp_path / "smoke.mp4"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=24:duration=1", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000:duration=1", "-c:v", "libx264",
                    "-c:a", "aac", "-shortest", str(source)], check=True, timeout=30)
    metadata = probe_media(source, ffprobe)
    assert metadata["duration_ms"] == 1000
    assert metadata["video"]["width"] == 320 and metadata["audio"]["codec_name"] == "aac"
    from fastapi.testclient import TestClient

    from autodub.config import Settings
    from autodub.main import create_app
    app = create_app(Settings(data_dir=tmp_path / "data", admin_token="test", ffprobe_bin=ffprobe,
                              frontend_dir=tmp_path / "none"), start_worker=False)
    with TestClient(app) as client:
        client.headers.update({"X-Autodub-Request": "1", "Authorization": "Bearer test"})
        series = client.post("/api/series", json={"title": "Real smoke"}).json()
        episode = client.post("/api/episodes", json={"series_id": series["id"], "ordinal": 1,
                                                   "filename": source.name, "total_bytes": source.stat().st_size}).json()
        assert client.put(f"/api/uploads/{episode['id']}/chunks", content=source.read_bytes(),
                          headers={"Upload-Offset": "0"}).status_code == 200
        client.post(f"/api/episodes/{episode['id']}/start")
        assert app.state.scheduler.tick()
        assert client.get(f"/api/episodes/{episode['id']}").json()["status"] == "CHECKPOINTED"


def test_invalid_input_and_missing_ffprobe_are_controlled_errors(tmp_path):
    source = tmp_path / "invalid.mp4"
    source.write_bytes(b"not a video")
    with pytest.raises(MediaError, match="unavailable"):
        probe_media(source, str(tmp_path / "missing-ffprobe"))


def test_audio_only_input_rejected(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    payload = {"format": {"duration": "1"}, "streams": [{"codec_type": "audio"}]}
    monkeypatch.setattr("autodub.media.subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(payload)))
    with pytest.raises(MediaError, match="video and audio"):
        probe_media(Path("fixture.mp4"))
    assert probe_audio(Path("vocals.wav"))["duration_ms"] == 1000


@pytest.mark.parametrize("duration", ["0", "-1", "NaN", "Infinity"])
def test_recognition_audio_rejects_invalid_duration(monkeypatch, duration):
    import json
    from types import SimpleNamespace
    payload = {"format": {"duration": duration}, "streams": [{"codec_type": "audio"}]}
    monkeypatch.setattr("autodub.media.subprocess.run", lambda *_a, **_k:
                        SimpleNamespace(returncode=0, stdout=json.dumps(payload)))
    with pytest.raises(MediaError, match="valid audio"):
        probe_audio(Path("vocals.wav"))
