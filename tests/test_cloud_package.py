import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from autodub.cloud_package import build_package
from autodub.cloud_unpack import digest, extract_checked_package, install_checkout, validate_checkout

REPO = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.run(list(args), check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def installer(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    command("git", "-c", "init.defaultBranch=main", "init", str(project))
    for key, value in (("user.name", "Setup test"), ("user.email", "test@example.invalid"),
                       ("core.autocrlf", "false")):
        command("git", "-C", str(project), "config", key, value)
    (project / ".gitignore").write_text("frontend/dist/\n", encoding="utf-8")
    (project / "README.md").write_text("Committed fixture\n", encoding="utf-8")
    for relative in ("scripts/cloud_install.sh", "src/autodub/cloud_unpack.py"):
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    (project / "frontend/dist/assets").mkdir(parents=True)
    (project / "frontend/dist/index.html").write_text("<html>Built fixture</html>", encoding="utf-8")
    (project / "frontend/dist/assets/app.js").write_text("console.log(1)", encoding="utf-8")
    command("git", "-C", str(project), "add", ".")
    command("git", "-C", str(project), "commit", "-m", "Fixture")
    commit = command("git", "-C", str(project), "rev-parse", "HEAD")
    bundle = tmp_path / "code.bundle"
    command("git", "-C", str(project), "bundle", "create", str(bundle), "main")
    output = tmp_path / "setup.run"
    result = build_package(project, bundle, commit, output)
    data = output.read_bytes()
    marker = b"# AUTODUB_EMBEDDED_PAYLOAD\n"
    offset = data.index(marker) + len(marker)
    payload = tmp_path / "payload.tar.gz"
    payload.write_bytes(data[offset:])
    return project, commit, output, payload, result


def test_standalone_package_offset_checksum_and_real_offline_checkout(installer, tmp_path):
    _, commit, output, payload, result = installer
    header = output.read_bytes().split(b"# AUTODUB_EMBEDDED_PAYLOAD\n")[0]
    assigned_line = next(line for line in header.splitlines() if line.startswith(b"PAYLOAD_LINE="))
    assert int(assigned_line.split(b"=")[1]) == header.count(b"\n") + 2
    assert result["sha256"] == digest(output)
    assert output.with_suffix(".run.sha256").read_text().startswith(result["sha256"])
    staging = tmp_path / "staging"
    metadata = extract_checked_package(payload, digest(payload), staging, commit)
    assert metadata["source_commit"] == commit
    target = tmp_path / "installed"
    validate_checkout(staging, target, commit)
    assert not target.exists()
    install_checkout(staging, target, commit)
    assert command("git", "-C", str(target), "rev-parse", "HEAD") == commit
    assert (target / "frontend/dist/index.html").read_text() == "<html>Built fixture</html>"
    sentinel = tmp_path / "userdata.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    install_checkout(staging, target, commit)
    assert sentinel.read_text() == "preserve"


def test_corrupt_payload_is_rejected_before_extraction(installer, tmp_path):
    _, commit, _, payload, _ = installer
    with pytest.raises(ValueError, match="checksum"):
        extract_checked_package(payload, "0" * 64, tmp_path / "staging", commit)
    assert not (tmp_path / "staging").exists()


@pytest.mark.parametrize("destination", ["bundle", "frontend", "source"])
def test_package_output_cannot_destroy_its_inputs(installer, destination):
    project, commit, _, _, _ = installer
    bundle = project.parent / "code.bundle"
    output = {"bundle": bundle, "frontend": project / "frontend/dist/index.html",
              "source": project / "README.md"}[destination]
    before = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite project inputs"):
        build_package(project, bundle, commit, output)
    assert output.read_bytes() == before


@pytest.mark.parametrize("name,kind", [
    ("../escape", "file"), ("/escape", "file"), ("frontend-dist/../escape", "file"),
    ("frontend-dist\\escape", "file"), ("frontend-dist/link", "symlink"),
    ("frontend-dist/hard", "hardlink"), ("frontend-dist/fifo", "fifo"),
    ("frontend-dist/pax", "pax"),
])
def test_archive_rejects_unsafe_entries_without_writing(name, kind, tmp_path):
    payload = tmp_path / "unsafe.tar.gz"
    with tarfile.open(payload, "w:gz") as archive:
        item = tarfile.TarInfo(name)
        item.size = 1
        if kind == "symlink":
            item.type, item.linkname, item.size = tarfile.SYMTYPE, "../../escape", 0
        elif kind == "hardlink":
            item.type, item.linkname, item.size = tarfile.LNKTYPE, "../../escape", 0
        elif kind == "fifo":
            item.type, item.size = tarfile.FIFOTYPE, 0
        elif kind == "pax":
            item.pax_headers = {"comment": "unsupported"}
        archive.addfile(item, io.BytesIO(b"x") if item.size else None)
    staging = tmp_path / "staging"
    with pytest.raises(ValueError, match="Unsafe"):
        extract_checked_package(payload, digest(payload), staging, "a" * 40)
    assert list(staging.iterdir()) == []
    assert not (tmp_path / "escape").exists()


def test_archive_rejects_duplicate_paths_before_any_write(tmp_path):
    payload = tmp_path / "duplicate.tar.gz"
    with tarfile.open(payload, "w:gz") as archive:
        for _ in range(2):
            item = tarfile.TarInfo("code.bundle")
            item.size = 1
            archive.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        extract_checked_package(payload, digest(payload), tmp_path / "staging", "a" * 40)
    assert not (tmp_path / "staging/code.bundle").exists()


@pytest.mark.parametrize("change", ["tracked", "untracked", "frontend", "wrong_commit"])
def test_rerun_preserves_changes_and_refuses_conflicts(installer, tmp_path, change):
    _, commit, _, payload, _ = installer
    staging, target = tmp_path / "staging", tmp_path / "installed"
    extract_checked_package(payload, digest(payload), staging, commit)
    install_checkout(staging, target, commit)
    if change == "tracked":
        sentinel = target / "README.md"
    elif change == "untracked":
        sentinel = target / "notes.txt"
    elif change == "frontend":
        sentinel = target / "frontend/dist/index.html"
    else:
        with pytest.raises(ValueError, match="pinned source"):
            install_checkout(staging, target, "a" * 40)
        return
    sentinel.write_text("local edit", encoding="utf-8")
    with pytest.raises(ValueError, match="local changes|local edits"):
        install_checkout(staging, target, commit)
    assert sentinel.read_text() == "local edit"


def test_package_file_manifest_detects_changed_file(installer, tmp_path):
    _, commit, _, payload, _ = installer
    content = {}
    with tarfile.open(payload, "r:gz") as archive:
        for item in archive:
            content[item.name] = archive.extractfile(item).read()
    metadata = json.loads(content["setup-package.json"])
    metadata["files"][0]["sha256"] = "0" * 64
    content["setup-package.json"] = json.dumps(metadata).encode()
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(tampered, "w:gz") as archive:
        for name, data in content.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    with pytest.raises(ValueError, match="file checksum"):
        extract_checked_package(tampered, digest(tampered), tmp_path / "staging", commit)


def test_generated_installer_dry_run_on_bash_writes_nothing(installer, tmp_path):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if shutil.which("git") and Path("C:/Program Files/Git/bin/bash.exe").exists() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable")
    _, _, output, _, _ = installer
    before = set(tmp_path.rglob("*"))
    result = subprocess.run([str(bash), str(output), "--dry-run"], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "No changes made" in result.stdout
    assert before == set(tmp_path.rglob("*"))
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest(output)
