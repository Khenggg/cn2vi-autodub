"""Build a single self-extracting cloud installer from committed code and built frontend."""
import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

from autodub.cloud_unpack import digest


def build_package(project: Path, bundle: Path, commit: str, output: Path) -> dict:
    project, bundle, output = project.resolve(), bundle.resolve(), output.absolute()
    if output.is_symlink() or output.resolve() != output:
        raise ValueError("Installer output must not traverse symlinks")
    if output == bundle or (output.is_relative_to(project) and not output.is_relative_to(project / ".cache")):
        raise ValueError("Installer output must not overwrite project inputs")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("An exact source commit is required")
    frontend = project / "frontend" / "dist"
    if not bundle.is_file() or not (frontend / "index.html").is_file():
        raise ValueError("Git bundle and built frontend are required")
    sources = [("code.bundle", bundle)]
    for path in sorted(frontend.rglob("*")):
        if path.is_symlink():
            raise ValueError("Frontend build must not contain symlinks")
        if path.is_file():
            sources.append(("frontend-dist/" + path.relative_to(frontend).as_posix(), path))
    if output.with_suffix(output.suffix + ".part") == bundle or output.with_suffix(output.suffix + ".sha256") == bundle:
        raise ValueError("Installer output must not overwrite project inputs")
    metadata = {"schema_version": 1, "source_commit": commit, "files": [
        {"path": name, "bytes": path.stat().st_size, "sha256": digest(path)} for name, path in sources]}
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, path in sources:
            record = tarfile.TarInfo(name)
            record.size = path.stat().st_size
            record.mode = 0o644
            with path.open("rb") as source:
                archive.addfile(record, source)
        data = json.dumps(metadata, sort_keys=True).encode("utf-8")
        record = tarfile.TarInfo("setup-package.json")
        record.size, record.mode = len(data), 0o644
        archive.addfile(record, io.BytesIO(data))
    payload_bytes = payload.getvalue()
    template = (project / "scripts/cloud_install.sh").read_text(encoding="utf-8")
    unpacker = (project / "src/autodub/cloud_unpack.py").read_text(encoding="utf-8")
    header = (template.replace("__SOURCE_COMMIT__", commit)
              .replace("__PAYLOAD_SHA256__", hashlib.sha256(payload_bytes).hexdigest())
              .replace("__INSTALLER_PYTHON__", unpacker)).rstrip() + "\n"
    header = header.replace("__PAYLOAD_LINE__", str(header.count("\n") + 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("wb") as stream:
        stream.write(header.encode("utf-8"))
        stream.write(payload_bytes)
    temporary.replace(output)
    checksum = digest(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    return {"source_commit": commit, "installer": str(output), "bytes": output.stat().st_size,
            "sha256": checksum}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_package(args.project_root, args.bundle, args.commit, args.output)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
