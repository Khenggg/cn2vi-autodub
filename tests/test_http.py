import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from vnle.pipeline import DEFAULT_CONFIG
from vnle.server import Handler, Store


@pytest.fixture
def endpoint(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.store = Store(tmp_path, "ffprobe", None, DEFAULT_CONFIG, False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", server.store
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_ui_starts_without_model_dependencies(endpoint):
    url, _ = endpoint
    with urllib.request.urlopen(url + "/api/status") as response:
        status = json.load(response)
    assert status["analysis_enabled"] is False
    assert status["models_configured"] is False
    with urllib.request.urlopen(url) as response:
        assert 'lang="vi"' in response.read().decode()


def test_ui_only_server_cannot_start_inference(endpoint):
    url, _ = endpoint
    request = urllib.request.Request(
        url + "/api/runs", b'{"video_id":"ignored"}', headers={"Content-Type": "application/json"}
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 400
    assert "disabled" in json.load(error.value)["error"]


def test_cross_origin_cannot_upload_or_start(endpoint):
    url, store = endpoint
    request = urllib.request.Request(
        url + "/api/videos?name=test.mp4",
        b"anything",
        headers={"Origin": "https://outside.example"},
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 400
    assert not list((store.root / "videos").iterdir())


def test_media_suffix_ranges_and_path_traversal(endpoint):
    url, store = endpoint
    identifier = "a" * 32
    folder = store.root / "videos" / identifier
    folder.mkdir()
    (folder / "input.mp4").write_bytes(b"0123456789")
    (folder / "metadata.json").write_text('{"id":"' + identifier + '"}')
    request = urllib.request.Request(url + "/media/" + identifier, headers={"Range": "bytes=-3"})
    with urllib.request.urlopen(request) as response:
        assert response.status == 206
        assert response.headers["Content-Range"] == "bytes 7-9/10"
        assert response.read() == b"789"
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(url + "/artifacts/../../AGENTS.md")
    assert error.value.code == 404


def test_orphaned_run_is_interrupted_after_restart(endpoint):
    _, store = endpoint
    identifier = "b" * 32
    folder = store.root / "runs" / identifier
    folder.mkdir()
    (folder / "run-report.json").write_text('{"status":"RUNNING","error":null}')
    assert store.job(identifier)["status"] == "INTERRUPTED"
