#!/usr/bin/env python3
"""Safely stage only the model assets and virtual environments in a restore TAR."""
from __future__ import annotations

import argparse
import os
import posixpath
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ALLOWED_ROOTS = ("data/models", "opt/autodub/venvs")
LEGACY_PYTHON_LINK = "/usr/bin/python3.12"


def _archive_path(raw: str) -> str:
    if not raw or "\\" in raw or raw.startswith("/"):
        raise ValueError("Unsafe archive path")
    parts = [part for part in PurePosixPath(raw).parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError("Unsafe archive path")
    return "/".join(parts)


def _in_allowed_root(path: str) -> bool:
    return any(path == root or path.startswith(root + "/") for root in ALLOWED_ROOTS)


def _link_target(member_path: str, linkname: str, *, hardlink: bool) -> str:
    if not linkname or "\\" in linkname or linkname.startswith("/"):
        raise ValueError("Unsafe archive link target")
    if hardlink:
        target = _archive_path(linkname)
    else:
        target = posixpath.normpath(posixpath.join(posixpath.dirname(member_path), linkname))
        if target in ("", ".", "..") or target.startswith("../"):
            raise ValueError("Unsafe archive link target")
    if not _in_allowed_root(target):
        raise ValueError("Archive link escapes the selected restore roots")
    return target


def _validated_members(archive: tarfile.TarFile, python312: Path) -> list[tuple[tarfile.TarInfo, str, str | None]]:
    members: list[tuple[tarfile.TarInfo, str, str | None]] = []
    by_path: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        path = _archive_path(member.name)
        if not _in_allowed_root(path):
            continue
        if path in by_path:
            raise ValueError("Archive contains duplicate selected paths")
        if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
            raise ValueError("Archive contains an unsupported selected member type")
        link_target = None
        if member.issym() or member.islnk():
            if (member.issym() and member.linkname == LEGACY_PYTHON_LINK
                    and path.startswith("opt/autodub/venvs/")
                    and PurePosixPath(path).name == "python3.12"
                    and "/bin/" in path):
                if not python312.is_file() or not os.access(python312, os.X_OK):
                    raise ValueError("Trusted Python 3.12 executable is unavailable")
                link_target = str(python312.resolve(strict=True))
            else:
                link_target = _link_target(path, member.linkname, hardlink=member.islnk())
        by_path[path] = member
        members.append((member, path, link_target))

    if not any((path == "data/models" or path.startswith("data/models/")) and member.isdir()
               for member, path, _ in members):
        raise ValueError("Archive is missing data/models")
    if not any((path == "opt/autodub/venvs" or path.startswith("opt/autodub/venvs/")) and member.isdir()
               for member, path, _ in members):
        raise ValueError("Archive is missing opt/autodub/venvs")

    for member, _path, target in members:
        if member.issym() and member.linkname != LEGACY_PYTHON_LINK and target not in by_path:
            raise ValueError("Symlink target is not present in the selected archive")
        if target is None or not member.islnk():
            continue
        linked = by_path.get(target)
        if linked is None or not linked.isfile():
            raise ValueError("Hardlink target is missing or is not a regular archived file")

    link_paths = {path for member, path, _ in members if member.issym() or member.islnk()}
    for _, path, _ in members:
        parent = posixpath.dirname(path)
        while parent:
            if parent in link_paths:
                raise ValueError("Archive member is nested beneath an archive link")
            parent = posixpath.dirname(parent)
    return members


def extract_selected(archive_path: Path, destination: Path, python312: Path) -> int:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:*") as archive:
        members = _validated_members(archive, python312)
        for member, relative, _ in members:
            if member.isdir():
                (destination / relative).mkdir(parents=True, exist_ok=True)
        for member, relative, _ in members:
            if not member.isfile():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Archive regular file has no payload")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            os.chmod(target, (member.mode & 0o777) & ~0o022)
        for member, relative, link_target in members:
            if not member.islnk():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(destination / str(link_target), target)
        for member, relative, link_target in members:
            if not member.issym():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.linkname == LEGACY_PYTHON_LINK:
                os.symlink(str(link_target), target)
            else:
                os.symlink(member.linkname, target)
    return len(members)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--python312", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = extract_selected(args.archive, args.destination, args.python312)
    except (OSError, tarfile.TarError, ValueError) as error:
        print(f"cloud_restore_archive: rejected archive ({type(error).__name__})", file=__import__("sys").stderr)
        return 1
    print(f"Validated and staged {count} selected archive members.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
