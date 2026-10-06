import hashlib
import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodub.model_assets import (
    _tracked_git_files,
    _verify_download,
    fetch_asset,
    load_manifest,
    main,
    manifest_digest,
)


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, *, status: int = 200, headers: dict | None = None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}

    def getcode(self):
        return self.status


def _write_manifest(folder: Path, entries: list[tuple[str, bytes]], *, digest: str | None = None):
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for relative, content in entries:
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files.append({"path": relative, "bytes": len(content),
                      "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"schema_version": 1, "id": folder.name, "model_revision": "a" * 40,
                "weights_sha256": digest or manifest_digest(files), "files": files}
    (folder / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_manifest_digest_is_canonical_and_load_verifies_real_files(tmp_path):
    folder = tmp_path / "fixture-model"
    manifest = _write_manifest(folder, [("weights.bin", b"fake weights"), ("config.json", b"{}")])
    reverse = list(reversed(manifest["files"]))
    assert manifest_digest(reverse) == manifest["weights_sha256"]
    loaded = load_manifest(folder)
    assert loaded["id"] == "fixture-model"
    assert loaded["model_revision"] == "a" * 40


def test_manifest_seal_and_file_hash_or_size_mismatch_are_rejected(tmp_path):
    folder = tmp_path / "model"
    manifest = _write_manifest(folder, [("weights.bin", b"original")])
    manifest["weights_sha256"] = "0" * 64
    (folder / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest checksum"):
        load_manifest(folder)

    manifest["weights_sha256"] = manifest_digest(manifest["files"])
    (folder / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset checksum"):
        load_manifest(folder)


@pytest.mark.parametrize("bad_id", ["..", "../escape", "model/child", "UpperCase", "with space"])
def test_asset_id_must_be_safe_before_any_filesystem_or_network_work(tmp_path, monkeypatch, bad_id):
    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen",
                        lambda *_args, **_kwargs: pytest.fail("unsafe ID must not download"))
    spec = {"id": bad_id, "revision": "revision", "kind": "http", "files": []}
    with pytest.raises(ValueError, match="Unsafe asset ID"):
        fetch_asset(spec, tmp_path)


def test_traversal_file_path_rejected_before_network_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen",
                        lambda *_args, **_kwargs: calls.append(1))
    spec = {"id": "safe-model", "revision": "source", "kind": "http", "files": [
        {"path": "../outside.bin", "url": "https://models.example/file", "bytes": 4,
         "sha256": hashlib.sha256(b"data").hexdigest()}]}
    with pytest.raises(ValueError, match="Path outside"):
        fetch_asset(spec, tmp_path)
    assert calls == []


def test_integrity_verification_refuses_missing_upstream_hash(tmp_path):
    downloaded = tmp_path / "download.bin"
    downloaded.write_bytes(b"data")
    with pytest.raises(ValueError, match="no upstream integrity checksum"):
        _verify_download(downloaded, {"bytes": 4})


def test_partial_download_never_becomes_an_asset_or_manifest(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    target_bytes = b"complete-model-file"

    def fake_urlopen(_request, timeout):
        assert timeout == 120
        return FakeResponse(b"partial", headers={"Content-Length": "7"})

    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen", fake_urlopen)
    spec = {"id": "tiny-model", "revision": "http-source", "kind": "http", "files": [
        {"path": "weights.bin", "url": "https://models.example/weights.bin",
         "bytes": len(target_bytes), "sha256": hashlib.sha256(target_bytes).hexdigest()}]}
    with pytest.raises(ValueError, match="length mismatch"):
        fetch_asset(spec, root)
    assert not (root / "tiny-model" / "weights.bin").exists()
    assert not (root / "tiny-model" / "model-manifest.json").exists()


def _http_spec(content: bytes, *, url="https://models.example/weights.bin"):
    return {"id": "tiny-model", "revision": "http-source", "kind": "http", "files": [
        {"path": "weights.bin", "url": url, "bytes": len(content),
         "sha256": hashlib.sha256(content).hexdigest()}]}


def test_http_download_resumes_partial_with_verified_content_range(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    content = b"verified-data"
    temporary = root / "tiny-model" / "weights.bin.part"
    temporary.parent.mkdir()
    temporary.write_bytes(content[:4])
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        assert request.get_header("Range") == f"bytes={len(content[:4])}-"
        return FakeResponse(content[4:], status=206, headers={
            "Content-Range": f"bytes 4-{len(content) - 1}/{len(content)}",
            "Content-Length": str(len(content) - 4),
        })

    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen", fake_urlopen)
    manifest = fetch_asset(_http_spec(content), root)
    target = root / "tiny-model" / "weights.bin"
    assert target.read_bytes() == content
    assert not temporary.exists()
    assert len(calls) == 1 and calls[0][1] == 120
    assert manifest["files"][0]["sha256"] == hashlib.sha256(content).hexdigest()


def test_http_server_ignoring_range_overwrites_partial_cleanly(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    content = b"complete-file"
    temporary = root / "tiny-model" / "weights.bin.part"
    temporary.parent.mkdir()
    temporary.write_bytes(b"bad-prefix")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.get_header("Range"))
        return FakeResponse(content, status=200, headers={"Content-Length": str(len(content))})

    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen", fake_urlopen)
    fetch_asset(_http_spec(content), root)
    assert calls == [f"bytes={len(b'bad-prefix')}-"]
    assert (root / "tiny-model" / "weights.bin").read_bytes() == content


def test_complete_verified_partial_is_promoted_without_a_request(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    content = b"already-downloaded"
    temporary = root / "tiny-model" / "weights.bin.part"
    temporary.parent.mkdir()
    temporary.write_bytes(content)
    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen",
                        lambda *_args, **_kwargs: pytest.fail("verified partial should be promoted locally"))
    fetch_asset(_http_spec(content), root)
    assert (root / "tiny-model" / "weights.bin").read_bytes() == content
    assert not temporary.exists()


def test_existing_verified_asset_is_reused_without_a_request(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    content = b"already-verified"
    target = root / "tiny-model" / "weights.bin"
    target.parent.mkdir()
    target.write_bytes(content)
    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen",
                        lambda *_args, **_kwargs: pytest.fail("verified asset should be reused"))
    manifest = fetch_asset(_http_spec(content), root)
    assert target.read_bytes() == content
    assert manifest["files"][0]["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("status", [429, 503])
def test_http_transient_status_retries_without_logging_upstream_url(tmp_path, monkeypatch, status):
    root = tmp_path / "models"
    root.mkdir()
    content = b"retry-me"
    spec = _http_spec(content, url="https://models.example/file?token=SECRET")
    attempts, sleeps = [], []

    def fake_urlopen(request, timeout):
        attempts.append(request.full_url)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(request.full_url, status, "upstream details", {}, io.BytesIO())
        return FakeResponse(content, headers={"Content-Length": str(len(content))})

    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("autodub.model_assets.time.sleep", sleeps.append)
    fetch_asset(spec, root)
    assert len(attempts) == 2
    assert sleeps == [0.1]
    assert (root / "tiny-model" / "weights.bin").read_bytes() == content


def test_http_network_failure_retries_and_resumes(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    content = b"network-retry"
    attempts, sleeps = [], []

    def fake_urlopen(request, timeout):
        attempts.append(request.get_header("Range"))
        if len(attempts) == 1:
            raise urllib.error.URLError("private connection detail")
        return FakeResponse(content, headers={"Content-Length": str(len(content))})

    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("autodub.model_assets.time.sleep", sleeps.append)
    fetch_asset(_http_spec(content), root)
    assert attempts == [None, None]
    assert sleeps == [0.1]
    assert (root / "tiny-model" / "weights.bin").read_bytes() == content


def test_http_retries_are_bounded_and_errors_do_not_expose_url(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    spec = _http_spec(b"body", url="https://models.example/file?token=SECRET")
    attempts, sleeps = [], []

    def always_unavailable(request, timeout):
        attempts.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 503, "secret details", {}, io.BytesIO())

    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen", always_unavailable)
    monkeypatch.setattr("autodub.model_assets.time.sleep", sleeps.append)
    with pytest.raises(RuntimeError) as error:
        fetch_asset(spec, root)
    assert len(attempts) == 3
    assert len(sleeps) == 2
    assert "SECRET" not in str(error.value)
    assert not (root / "tiny-model" / "model-manifest.json").exists()


def test_nontransient_http_error_is_sanitized_and_not_retried(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    spec = _http_spec(b"body", url="https://models.example/file?token=SECRET")
    calls = []

    def not_found(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 404, "private upstream details", {}, io.BytesIO())

    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen", not_found)
    with pytest.raises(ValueError) as error:
        fetch_asset(spec, root)
    assert len(calls) == 1
    assert "SECRET" not in str(error.value)
    assert "private upstream details" not in str(error.value)
    assert not (root / "tiny-model" / "model-manifest.json").exists()


def test_http_resume_rejects_wrong_content_range_before_append(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    content = b"expected"
    temporary = root / "tiny-model" / "weights.bin.part"
    temporary.parent.mkdir()
    temporary.write_bytes(content[:3])

    def wrong_range(_request, timeout):
        return FakeResponse(content[3:], status=206, headers={
            "Content-Range": f"bytes 2-{len(content) - 1}/{len(content)}",
            "Content-Length": str(len(content) - 2),
        })

    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen", wrong_range)
    with pytest.raises(ValueError, match="range does not match"):
        fetch_asset(_http_spec(content), root)
    assert temporary.read_bytes() == content[:3]
    assert not (root / "tiny-model" / "weights.bin").exists()
    assert not (root / "tiny-model" / "model-manifest.json").exists()


def test_interrupted_range_body_resumes_at_actual_partial_offset(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    content = b"resume-the-tail"
    splits = (5, 10)
    calls = []

    def fake_urlopen(request, timeout):
        start = int(request.get_header("Range").removeprefix("bytes=").removesuffix("-")) \
            if request.get_header("Range") else 0
        calls.append(start)
        if len(calls) == 1:
            # A transport EOF before the declared Content-Range is retried from actual bytes received.
            return FakeResponse(content[:splits[0]], status=206, headers={
                "Content-Range": f"bytes 0-{len(content) - 1}/{len(content)}",
                "Content-Length": str(len(content)),
            })
        return FakeResponse(content[start:], status=206, headers={
            "Content-Range": f"bytes {start}-{len(content) - 1}/{len(content)}",
            "Content-Length": str(len(content) - start),
        })

    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("autodub.model_assets.time.sleep", lambda *_: None)
    fetch_asset(_http_spec(content), root)
    assert calls == [0, splits[0]]
    assert (root / "tiny-model" / "weights.bin").read_bytes() == content


def test_http_integrity_failure_never_promotes_or_writes_manifest(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    spec = _http_spec(b"expected")
    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen", lambda *_args, **_kwargs:
                        FakeResponse(b"tampered", headers={"Content-Length": "8"}))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_asset(spec, root)
    assert not (root / "tiny-model" / "weights.bin").exists()
    assert not (root / "tiny-model" / "model-manifest.json").exists()


def test_http_asset_requires_integrity_before_request(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    calls = []
    monkeypatch.setattr("autodub.model_assets.urllib.request.urlopen",
                        lambda *_args, **_kwargs: calls.append(1))
    spec = {"id": "tiny-model", "revision": "http", "kind": "http", "files": [
        {"path": "weights.bin", "url": "https://models.example/file", "bytes": 4}]}
    with pytest.raises(ValueError, match="no upstream integrity checksum"):
        fetch_asset(spec, root)
    assert calls == []


def test_bad_existing_asset_is_not_accepted_as_verified(tmp_path):
    target = tmp_path / "existing.bin"
    target.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="length mismatch"):
        _verify_download(target, {"bytes": 4, "sha256": hashlib.sha256(b"right").hexdigest()})


def test_git_asset_refuses_dirty_existing_checkout_before_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "bandit-code" / "source"
    checkout.mkdir(parents=True)
    (checkout / ".git").mkdir()
    calls = []

    def fake_git(command, **_kwargs):
        calls.append(command)
        if command[-2:] == ["rev-parse", "--show-toplevel"]:
            return SimpleNamespace(returncode=0, stdout=str(checkout), stderr="")
        if command[-4:] == ["status", "--porcelain", "--untracked-files=all", "--ignored=matching"]:
            return SimpleNamespace(returncode=0, stdout="?? scratch.py\n", stderr="")
        pytest.fail("dirty checkout must be rejected before further Git operations")

    monkeypatch.setattr("autodub.model_assets.subprocess.run", fake_git)
    spec = {"id": "bandit-code", "kind": "git", "repo": "https://example.invalid/bandit.git",
            "revision": "a" * 40}
    with pytest.raises(ValueError, match="local changes or untracked"):
        fetch_asset(spec, tmp_path)
    assert not any("checkout" in command for command in calls)


def test_git_asset_refuses_ignored_source_files_before_pinning(tmp_path, monkeypatch):
    checkout = tmp_path / "bandit-code" / "source"
    checkout.mkdir(parents=True)
    (checkout / ".git").mkdir()
    (checkout / "main.py").write_text("pinned = True\n", encoding="utf-8")
    (checkout / "stale_ignored.py").write_text("not pinned\n", encoding="utf-8")
    revision = "a" * 40

    def fake_git(command, **_kwargs):
        if command[-2:] == ["rev-parse", "--show-toplevel"]:
            return SimpleNamespace(returncode=0, stdout=str(checkout), stderr="")
        if command[-4:] == ["status", "--porcelain", "--untracked-files=all", "--ignored=matching"]:
            return SimpleNamespace(returncode=0, stdout="!! stale_ignored.py\n", stderr="")
        pytest.fail(f"unexpected git invocation: {command}")

    monkeypatch.setattr("autodub.model_assets.subprocess.run", fake_git)
    spec = {"id": "bandit-code", "kind": "git", "repo": "https://example.invalid/bandit.git",
            "revision": revision}
    with pytest.raises(ValueError, match="local changes or untracked"):
        fetch_asset(spec, tmp_path)


def test_git_manifest_hashes_only_committed_regular_tree_files(tmp_path, monkeypatch):
    folder = tmp_path / "bandit-code"
    checkout = folder / "source"
    checkout.mkdir(parents=True)
    tracked_file = checkout / "main.py"
    tracked_file.write_text("pinned = True\n", encoding="utf-8")
    (checkout / "stale_ignored.py").write_text("not pinned\n", encoding="utf-8")
    revision = "a" * 40
    records = (
        b"100644 blob 1111111111111111111111111111111111111111\tmain.py\0"
        b"120000 blob 2222222222222222222222222222222222222222\tlink.py\0"
        b"160000 commit 3333333333333333333333333333333333333333\tsubmodule\0"
    )

    def fake_git(command, **_kwargs):
        assert command[-4:] == ["ls-tree", "-rz", "--full-tree", revision]
        return SimpleNamespace(returncode=0, stdout=records, stderr=b"")

    monkeypatch.setattr("autodub.model_assets.subprocess.run", fake_git)
    files = _tracked_git_files(checkout, folder, revision)
    tracked_bytes = tracked_file.read_bytes()
    assert files == [{"path": "source/main.py", "bytes": len(tracked_bytes),
                      "sha256": hashlib.sha256(tracked_bytes).hexdigest()}]


def test_git_manifest_verification_rechecks_head_and_cleanliness(tmp_path, monkeypatch):
    folder = tmp_path / "bandit-code"
    _write_manifest(folder, [("source/main.py", b"pinned")])
    (folder / "source" / ".git").mkdir()
    manifest_path = folder / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["upstream"] = "https://example.invalid/bandit.git"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    calls = []

    def fake_output(checkout, *arguments):
        calls.append(arguments)
        if arguments == ("rev-parse", "HEAD"):
            return "b" * 40
        if arguments == ("status", "--porcelain", "--untracked-files=all", "--ignored=matching"):
            return ""
        pytest.fail(f"unexpected Git verification call: {arguments}")

    monkeypatch.setattr("autodub.model_assets._git_output", fake_output)
    with pytest.raises(ValueError, match="manifest revision"):
        load_manifest(folder)
    assert calls == [("rev-parse", "HEAD")]


@pytest.mark.parametrize("status", ["?? untracked.py\n", "!! ignored.py\n"])
def test_git_manifest_verification_refuses_untracked_or_ignored_code(tmp_path, monkeypatch, status):
    folder = tmp_path / "bandit-code"
    _write_manifest(folder, [("source/main.py", b"pinned")])
    (folder / "source" / ".git").mkdir()
    manifest_path = folder / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["upstream"] = "https://example.invalid/bandit.git"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_output(_checkout, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return manifest["model_revision"]
        if arguments == ("status", "--porcelain", "--untracked-files=all", "--ignored=matching"):
            return status
        pytest.fail(f"unexpected Git verification call: {arguments}")

    monkeypatch.setattr("autodub.model_assets._git_output", fake_output)
    with pytest.raises(ValueError, match="local changes or untracked"):
        load_manifest(folder)


def test_model_assets_verify_rejects_unknown_only_ids(tmp_path, monkeypatch, capsys):
    lock = tmp_path / "models.lock.json"
    lock.write_text(json.dumps({"schema_version": 1, "models": []}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["model_assets", "verify", "--lock", str(lock),
                                       "--root", str(tmp_path), "--only", "missing-model"])
    assert main() == 1
    assert capsys.readouterr().out == "Asset operation failed: ValueError\n"


def test_model_assets_verify_compares_manifest_revision_to_lock(tmp_path, monkeypatch, capsys):
    lock = tmp_path / "models.lock.json"
    lock.write_text(json.dumps({"schema_version": 1, "models": [{
        "id": "fixture-model", "revision": "a" * 40, "kind": "http", "files": []}]}),
        encoding="utf-8")
    folder = tmp_path / "fixture-model"
    manifest = _write_manifest(folder, [("weights.bin", b"weights")])
    manifest["model_revision"] = "b" * 40
    (folder / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["model_assets", "verify", "--lock", str(lock),
                                       "--root", str(tmp_path)])
    assert main() == 1
    assert capsys.readouterr().out == "Asset operation failed: ValueError\n"
