import io
import tarfile
from pathlib import Path

import pytest

from scripts.cloud_restore_archive import extract_selected


def _archive(path: Path, entries: list[tuple[str, str, str | bytes | None]]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, kind, value in entries:
            member = tarfile.TarInfo(name)
            if kind == "dir":
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            elif kind == "file":
                data = value if isinstance(value, bytes) else str(value or "").encode()
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            elif kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = str(value)
                archive.addfile(member)
            elif kind == "hardlink":
                member.type = tarfile.LNKTYPE
                member.linkname = str(value)
                archive.addfile(member)
            else:
                raise AssertionError(kind)


def _base() -> list[tuple[str, str, str | bytes | None]]:
    return [("data/models", "dir", None),
            ("opt/autodub/venvs", "dir", None)]


def test_extracts_only_selected_roots_and_repairs_known_python_link(tmp_path):
    archive = tmp_path / "backup.tar"
    entries = _base() + [
        ("data/models/qwen/model.bin", "file", b"model"),
        ("opt/autodub/venvs/core/bin", "dir", None),
        ("opt/autodub/venvs/core/bin/python3.12", "symlink", "/usr/bin/python3.12"),
        ("opt/autodub/venvs/core/bin/python", "symlink", "python3.12"),
        ("data/uploads/user-source.mp4", "file", b"must be ignored"),
        ("etc/ssh/sshd_config", "file", b"must be ignored"),
    ]
    _archive(archive, entries)
    trusted = Path("/bin/true").resolve()
    stage = tmp_path / "stage"

    assert extract_selected(archive, stage, trusted) == 6
    assert (stage / "data/models/qwen/model.bin").read_bytes() == b"model"
    assert not (stage / "data/uploads").exists()
    assert not (stage / "etc").exists()
    assert (stage / "opt/autodub/venvs/core/bin/python3.12").resolve() == trusted
    assert (stage / "opt/autodub/venvs/core/bin/python").resolve() == trusted


@pytest.mark.parametrize("path", [
    "data/models/../../etc/cron.d/escape",
    "/data/models/absolute",
    "data/models/..\\..\\etc\\passwd",
])
def test_rejects_traversal_and_absolute_member_paths(tmp_path, path):
    archive = tmp_path / "unsafe.tar"
    _archive(archive, _base() + [(path, "file", b"bad")])

    with pytest.raises(ValueError):
        extract_selected(archive, tmp_path / "stage", Path("/bin/true").resolve())


@pytest.mark.parametrize("target", ["/tmp/escape", "../../../../etc/passwd", "../outside"])
def test_rejects_unsafe_symlink_targets(tmp_path, target):
    archive = tmp_path / "unsafe-link.tar"
    _archive(archive, _base() + [("data/models/qwen", "dir", None),
                                 ("data/models/qwen/link", "symlink", target)])

    with pytest.raises(ValueError):
        extract_selected(archive, tmp_path / "stage", Path("/bin/true").resolve())


def test_rejects_unsafe_hardlink_target(tmp_path):
    archive = tmp_path / "unsafe-hardlink.tar"
    _archive(archive, _base() + [("data/models/qwen", "dir", None),
                                 ("data/models/qwen/link", "hardlink", "../../etc/passwd")])

    with pytest.raises(ValueError):
        extract_selected(archive, tmp_path / "stage", Path("/bin/true").resolve())


def test_rejects_members_nested_under_a_symlink(tmp_path):
    archive = tmp_path / "symlink-parent.tar"
    _archive(archive, _base() + [("data/models/link", "symlink", "qwen"),
                                 ("data/models/link/payload", "file", b"bad")])

    with pytest.raises(ValueError):
        extract_selected(archive, tmp_path / "stage", Path("/bin/true").resolve())
