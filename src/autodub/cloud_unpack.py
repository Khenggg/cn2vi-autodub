"""Stdlib-only unpacker embedded in the standalone cloud installer."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

MAX_FILES = 4096
MAX_UNPACKED_BYTES = 100 * 1024**2


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def extract_checked_package(archive: Path, expected_sha256: str, staging: Path, commit: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or digest(archive) != expected_sha256:
        raise ValueError("Installer payload checksum mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid pinned source commit")
    staging = staging.resolve()
    staging.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as package:
        entries = package.getmembers()
        if len(entries) > MAX_FILES or sum(item.size for item in entries) > MAX_UNPACKED_BYTES:
            raise ValueError("Installer payload exceeds extraction limits")
        seen = set()
        for item in entries:
            relative = PurePosixPath(item.name)
            if (not item.isfile() or item.size < 0 or relative.is_absolute()
                    or ".." in relative.parts or "\\" in item.name or item.name in seen
                    or item.pax_headers or item.sparse is not None
                    or not item.name or item.name != relative.as_posix()
                    or relative.parts[0] not in {"code.bundle", "setup-package.json", "frontend-dist"}):
                raise ValueError("Unsafe installer archive entry")
            target = (staging / item.name).resolve()
            if not target.is_relative_to(staging) or target == staging or target.exists():
                raise ValueError("Installer extraction must use an empty staging directory")
            seen.add(item.name)
        for item in entries:
            target = staging / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.extractfile(item) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    metadata = json.loads((staging / "setup-package.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("source_commit") != commit:
        raise ValueError("Installer source provenance mismatch")
    records = metadata.get("files")
    if not isinstance(records, list) or {item.get("path") for item in records} != seen - {"setup-package.json"}:
        raise ValueError("Installer file manifest mismatch")
    for record in records:
        path = staging / record["path"]
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Installer file checksum mismatch")
    if not (staging / "code.bundle").is_file() or not (staging / "frontend-dist/index.html").is_file():
        raise ValueError("Installer code or frontend is missing")
    return metadata


def git(*arguments: str) -> str:
    result = subprocess.run(["git", *arguments], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError("Installer Git operation failed")
    return result.stdout.strip()


def validate_checkout(staging: Path, target: Path, commit: str) -> Path:
    target = target.expanduser().absolute()
    if target.is_symlink() or target.resolve() != target:
        raise ValueError("Project destination must not traverse symlinks")
    bundle = staging / "code.bundle"
    verification_repo = staging / "verify-repo"
    git("init", str(verification_repo))
    git("-C", str(verification_repo), "bundle", "verify", str(bundle))
    heads = git("bundle", "list-heads", str(bundle))
    if not any(line.split()[0] == commit for line in heads.splitlines()):
        raise ValueError("Bundle does not advertise the pinned source commit")
    if target.exists():
        if not (target / ".git").is_dir() or Path(git("-C", str(target), "rev-parse", "--show-toplevel")).resolve() != target:
            raise ValueError("Existing project destination is not the expected Git checkout")
        if git("-C", str(target), "rev-parse", "HEAD") != commit:
            raise ValueError("Existing project has a different commit; choose another project destination")
        if git("-C", str(target), "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("Existing project has local edits; setup preserves them and stops")
    frontend = target / "frontend" / "dist"
    if frontend.is_symlink() or frontend.resolve() != frontend:
        raise ValueError("Frontend destination must not traverse symlinks")
    expected_files = {path.relative_to(staging / "frontend-dist") for path in (staging / "frontend-dist").rglob("*")
                      if path.is_file()}
    if frontend.exists():
        existing_files = {path.relative_to(frontend) for path in frontend.rglob("*") if path.is_file()}
        if existing_files != expected_files or any(path.is_symlink() for path in frontend.rglob("*")):
            raise ValueError("Existing frontend has local changes; setup preserves them and stops")
        if any(digest(frontend / relative) != digest(staging / "frontend-dist" / relative)
               for relative in expected_files):
            raise ValueError("Existing frontend has local changes; setup preserves them and stops")
    return target


def install_checkout(staging: Path, target: Path, commit: str) -> None:
    target = validate_checkout(staging, target, commit)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--no-checkout", str(staging / "code.bundle"), str(target))
        git("-C", str(target), "config", "core.autocrlf", "false")
        git("-C", str(target), "checkout", "--detach", commit)
        if git("-C", str(target), "rev-parse", "HEAD") != commit:
            raise ValueError("Cloned project commit mismatch")
    frontend = target / "frontend" / "dist"
    if frontend.exists():
        return
    for source in (staging / "frontend-dist").rglob("*"):
        if source.is_file():
            destination = frontend / source.relative_to(staging / "frontend-dist")
            if destination.is_symlink() or destination.resolve() != destination:
                raise ValueError("Frontend file destination must not traverse symlinks")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        extract_checked_package(args.archive, args.sha256, args.staging_dir, args.source_commit)
        if args.validate_only:
            validate_checkout(args.staging_dir, args.project_root, args.source_commit)
        else:
            install_checkout(args.staging_dir, args.project_root, args.source_commit)
    except Exception as error:
        print(f"Cloud source installation failed ({type(error).__name__})")
        return 1
    print(f"Source package {'validated' if args.validate_only else 'installed'}: {args.source_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
