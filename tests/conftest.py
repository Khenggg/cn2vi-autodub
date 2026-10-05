import pytest
from fastapi.testclient import TestClient

from autodub.config import Settings
from autodub.main import create_app


@pytest.fixture
def app(tmp_path):
    return create_app(Settings(data_dir=tmp_path, admin_token="test-token", frontend_dir=tmp_path / "no-ui"), start_worker=False)


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        client.headers.update({"X-Autodub-Request": "1"})
        response = client.post("/api/auth/login", json={"token": "test-token"})
        assert response.status_code == 200
        yield client


@pytest.fixture
def uploaded(client):
    series = client.post("/api/series", json={"title": "Test Series", "priority": 0}).json()
    episode = client.post("/api/episodes", json={"series_id": series["id"], "ordinal": 1,
                                               "filename": "episode.mp4", "total_bytes": 12}).json()
    result = client.put(f"/api/uploads/{episode['id']}/chunks", content=b"test-content",
                        headers={"Upload-Offset": "0"})
    assert result.status_code == 200
    return result.json()
