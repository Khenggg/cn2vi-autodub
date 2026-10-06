import json
import threading

from autodub.media import MediaError

MEDIA = {"schema_version": 1, "duration_ms": 5000, "video": {"width": 1280, "height": 720}, "audio": {"sample_rate": "48000"}}


def test_preparation_checkpoint_stops_before_asr(client, app, uploaded, monkeypatch):
    monkeypatch.setattr("autodub.scheduler.probe_media", lambda *_: MEDIA)
    client.post(f"/api/episodes/{uploaded['id']}/start")
    assert app.state.scheduler.tick()
    episode = client.get(f"/api/episodes/{uploaded['id']}").json()
    assert episode["status"] == "CHECKPOINTED" and episode["next_stage"] == "ASR"
    assert episode["duration_ms"] == 5000 and episode["progress"] < 1
    artifact = episode["artifacts"][0]
    checkpoint = client.get(f"/api/download/{artifact['id']}").json()
    assert checkpoint["source_sha256"] == uploaded["source_sha256"] and checkpoint["schema_version"] == 1
    assert not app.state.scheduler.tick()
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 409


def test_priority_and_episode_ordinal(client, app):
    for title, priority in [("B", 10), ("A", 0)]:
        series = client.post("/api/series", json={"title": title, "priority": priority}).json()
        for ordinal in [2, 1]:
            episode = client.post("/api/episodes", json={"series_id": series["id"], "ordinal": ordinal, "filename": "s.mp4", "total_bytes": 1}).json()
            client.put(f"/api/uploads/{episode['id']}/chunks", content=b"x", headers={"Upload-Offset": "0"})
            client.post(f"/api/episodes/{episode['id']}/start")
    episode = app.state.scheduler.next_episode()
    assert episode["ordinal"] == 1
    assert app.state.service.require("series", episode["series_id"])["title"] == "A"


def test_failed_probe_does_not_block_next_and_retry(client, app, uploaded, monkeypatch):
    def fail(*_):
        raise MediaError("Invalid media")
    monkeypatch.setattr("autodub.scheduler.probe_media", fail)
    client.post(f"/api/episodes/{uploaded['id']}/start")
    app.state.scheduler.tick()
    episode = client.get(f"/api/episodes/{uploaded['id']}").json()
    assert episode["status"] == "FAILED" and episode["issues"][0]["code"] == "MEDIA_INVALID"
    monkeypatch.setattr("autodub.scheduler.probe_media", lambda *_: MEDIA)
    assert client.post(f"/api/episodes/{uploaded['id']}/retry").status_code == 200
    app.state.scheduler.tick()
    assert client.get(f"/api/episodes/{uploaded['id']}").json()["status"] == "CHECKPOINTED"


def test_drain_waits_for_active_checkpoint_and_blocks_delete(client, app, uploaded, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def probe(*_):
        entered.set()
        assert release.wait(5)
        return MEDIA
    monkeypatch.setattr("autodub.scheduler.probe_media", probe)
    client.post(f"/api/episodes/{uploaded['id']}/start")
    worker = threading.Thread(target=app.state.scheduler.tick)
    worker.start()
    assert entered.wait(5)
    try:
        assert client.post("/api/worker/drain").json()["worker_state"] == "DRAINING"
        assert client.delete(f"/api/episodes/{uploaded['id']}").status_code == 409
    finally:
        release.set()
        worker.join(5)
    assert app.state.scheduler.state() == "READY_TO_SHUTDOWN"
    assert client.get(f"/api/episodes/{uploaded['id']}").json()["status"] == "CHECKPOINTED"


def test_source_corruption_is_rejected(client, app, uploaded, monkeypatch):
    monkeypatch.setattr("autodub.scheduler.probe_media", lambda *_: MEDIA)
    episode = app.state.service.require("episode", uploaded["id"])
    (app.state.settings.data_dir / episode["source_path"]).write_bytes(b"changed")
    client.post(f"/api/episodes/{uploaded['id']}/start")
    app.state.scheduler.tick()
    assert client.get(f"/api/episodes/{uploaded['id']}").json()["status"] == "FAILED"


def test_restart_recovers_interrupted_preparation(client, app, uploaded, monkeypatch):
    monkeypatch.setattr("autodub.scheduler.probe_media", lambda *_: MEDIA)
    app.state.db.transition(uploaded["id"], "PREPARING", "Simulated crash", queue_requested=1)
    checkpoint = app.state.settings.data_dir / "checkpoints" / uploaded["id"] / "state.json"
    # Drain state is preserved across process restarts; recovery remains queued until resume.
    app.state.scheduler.drain()
    app.state.scheduler.start()
    try:
        assert client.get(f"/api/episodes/{uploaded['id']}").json()["status"] == "QUEUED"
        episode = app.state.db.one("SELECT next_stage,queue_requested FROM episode WHERE id=?",
                                   (uploaded["id"],))
        assert episode["next_stage"] == "PREPARING" and episode["queue_requested"] == 1
        assert not checkpoint.exists()
        assert app.state.scheduler.state() == "READY_TO_SHUTDOWN"
    finally:
        app.state.scheduler.close()
    events = app.state.db.rows("SELECT * FROM job_event WHERE job_id=?", (uploaded["id"],))
    assert any("Recovered" in event["message"] for event in events)
    assert all(json.loads(event["metrics_json"]) is not None for event in events)
