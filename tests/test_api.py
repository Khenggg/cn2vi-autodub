import hashlib
import io
import json
import zipfile

from fastapi.testclient import TestClient


def test_auth_cookie_csrf_and_rate_limit(app):
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/api/series").status_code == 401
        assert client.post("/api/auth/login", json={"token": "test-token"}).status_code == 403
        response = client.post("/api/auth/login", json={"token": "test-token"}, headers={"X-Autodub-Request": "1"})
        assert "HttpOnly" in response.headers["set-cookie"] and "SameSite=strict" in response.headers["set-cookie"]
        assert client.get("/api/series").status_code == 200
        assert client.post("/api/series", json={"title": "Bad"}, headers={"X-Autodub-Request": "1", "Origin": "https://evil.invalid"}).status_code == 403
        client.post("/api/auth/logout", headers={"X-Autodub-Request": "1"})
        assert client.get("/api/series").status_code == 401
        for _ in range(10):
            assert client.post("/api/auth/login", json={"token": "wrong"}, headers={"X-Autodub-Request": "1"}).status_code == 401
        assert client.post("/api/auth/login", json={"token": "wrong"}, headers={"X-Autodub-Request": "1"}).status_code == 429


def test_upload_resume_offsets_checksum_and_duplicate_ordinal(client, app):
    series = client.post("/api/series", json={"title": "Film"}).json()
    body = {"series_id": series["id"], "ordinal": 1, "filename": "source.mp4", "total_bytes": 6}
    slot = client.post("/api/episodes", json=body).json()
    assert client.post("/api/episodes", json=body).status_code == 409
    url = f"/api/uploads/{slot['id']}/chunks"
    assert client.post(f"/api/episodes/{slot['id']}/start").status_code == 409
    assert client.put(url, content=b"abc", headers={"Upload-Offset": "0"}).status_code == 200
    assert client.put(url, content=b"abc", headers={"Upload-Offset": "0"}).status_code == 409
    assert client.get(f"/api/uploads/{slot['id']}").json()["uploaded_bytes"] == 3
    # Simulate a write that reached disk but did not commit its DB offset.
    part = app.state.settings.data_dir / "uploads" / series["id"] / slot["id"] / "source.part"
    with part.open("ab") as stream:
        stream.write(b"uncommitted")
    done = client.put(url, content=b"def", headers={"Upload-Offset": "3"}).json()
    assert done["status"] == "QUEUED" and done["queue_requested"] == 0
    assert done["source_sha256"] == hashlib.sha256(b"abcdef").hexdigest()
    assert client.get(f"/api/episodes/{slot['id']}/source").content == b"abcdef"
    assert client.put(url, content=b"x", headers={"Upload-Offset": "6"}).status_code == 409


def test_upload_recovers_rename_before_db_commit(client, app):
    series = client.post("/api/series", json={"title": "Film"}).json()
    slot = client.post("/api/episodes", json={"series_id": series["id"], "ordinal": 1, "filename": "s.mp4", "total_bytes": 3}).json()
    folder = app.state.settings.data_dir / "uploads" / series["id"] / slot["id"]
    (folder / "source.part").write_bytes(b"abc")
    (folder / "source.part").rename(folder / "source.mp4")
    result = client.put(f"/api/uploads/{slot['id']}/chunks", content=b"abc", headers={"Upload-Offset": "0"})
    assert result.status_code == 200
    assert result.json()["source_sha256"] == hashlib.sha256(b"abc").hexdigest()


def test_bad_filename_size_and_roi_before_preview(client, uploaded):
    series = client.get("/api/series").json()[0]
    for filename in ["../x.mp4", "C:\\x.mp4", "x.exe", "folder/x.mp4"]:
        assert client.post("/api/episodes", json={"series_id": series["id"], "ordinal": 2,
                                                 "filename": filename, "total_bytes": 10}).status_code == 400
    assert client.put(f"/api/episodes/{uploaded['id']}/roi", json={"x": 0, "y": 0.5, "w": 1, "h": 0.5}).status_code == 409
    assert client.put(f"/api/episodes/{uploaded['id']}/roi", json={"x": 0.9, "y": 0.5, "w": 1, "h": 0.5}).status_code == 422


def test_glossary_lock_and_export_excludes_media_and_secrets(client, uploaded, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "SECRET_DO_NOT_EXPORT")
    series_id = uploaded["series_id"]
    url = f"/api/series/{series_id}/glossary"
    assert client.put(url, json=[{"zh": "夜采", "vi": "Dạ Thái", "locked_by_user": True}]).status_code == 200
    assert client.put(url, json=[{"zh": "夜采", "vi": "Other", "locked_by_user": False}]).status_code == 409
    assert client.get(f"/api/series/{series_id}").json()["glossary"][0]["vi"] == "Dạ Thái"
    exported = client.post("/api/workspace/export").json()
    response = client.get(exported["download_url"])
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"manifest.json", "workspace.sqlite.export.json", "checksums.json"}
        payload = archive.read("workspace.sqlite.export.json")
        assert b"SECRET_DO_NOT_EXPORT" not in payload and b"test-content" not in payload and b"source_path" not in payload
        assert json.loads(archive.read("checksums.json"))["workspace.sqlite.export.json"] == hashlib.sha256(payload).hexdigest()
    assert "SECRET_DO_NOT_EXPORT" not in client.get("/api/system/status").text


def test_drain_blocks_start_until_resume(client, uploaded):
    assert client.post("/api/worker/drain").json()["worker_state"] == "READY_TO_SHUTDOWN"
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 409
    assert client.post("/api/worker/resume").json()["worker_state"] == "ACCEPTING"
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 200


def test_delete_cascades_and_removes_source(client, uploaded, app):
    series_id = uploaded["series_id"]
    folder = app.state.settings.data_dir / "uploads" / series_id / uploaded["id"]
    assert folder.is_dir()
    assert client.delete(f"/api/series/{series_id}").status_code == 200
    assert client.get(f"/api/episodes/{uploaded['id']}").status_code == 404
    assert not folder.exists()
