"""Download immutable model assets explicitly; inference never downloads from floating branches."""
import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from autodub.storage import atomic_json, safe_path, sha256_file

HTTP_DOWNLOAD_TIMEOUT = 120
HTTP_DOWNLOAD_ATTEMPTS = 3
HTTP_RETRY_BACKOFF_SECONDS = (0.1, 0.2)
HTTP_COPY_CHUNK_BYTES = 1024 * 1024
UNKNOWN_ASSET_SIZE_ESTIMATE = 256 * 1024**2


def manifest_digest(files: list[dict]) -> str:
    canonical = [{"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]}
                 for item in sorted(files, key=lambda value: value["path"])]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_manifest(folder: Path, *, verify: bool = True) -> dict:
    manifest = json.loads((folder / "model-manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or not manifest.get("files") or
            manifest.get("id") != folder.name or not isinstance(manifest.get("model_revision"), str)):
        raise ValueError("Invalid model manifest")
    if manifest_digest(manifest["files"]) != manifest["weights_sha256"]:
        raise ValueError("Model manifest checksum mismatch")
    if verify:
        for item in manifest["files"]:
            path = safe_path(folder, item["path"])
            if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise ValueError("Model asset checksum mismatch")
        git_checkout = folder / "source"
        is_git_asset = str(manifest.get("upstream", "")).endswith(".git")
        if is_git_asset or (git_checkout / ".git").exists():
            if not (git_checkout / ".git").exists():
                raise ValueError("Git model manifest is missing its source checkout")
            head = _git_output(git_checkout, "rev-parse", "HEAD").strip()
            if head != manifest["model_revision"]:
                raise ValueError("Git model asset does not match its manifest revision")
            _require_clean_checkout(git_checkout)
    return manifest


def _verify_download(path: Path, specification: dict) -> None:
    if specification.get("bytes") is not None and path.stat().st_size != specification["bytes"]:
        raise ValueError("Downloaded asset length mismatch")
    if specification.get("sha256"):
        if sha256_file(path) != specification["sha256"]:
            raise ValueError("Downloaded asset SHA-256 mismatch")
    elif specification.get("git_blob_sha1"):
        digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != specification["git_blob_sha1"]:
            raise ValueError("Downloaded asset Git blob mismatch")
    elif specification.get("md5"):
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != specification["md5"]:
            raise ValueError("Downloaded asset upstream MD5 mismatch")
    else:
        raise ValueError("Asset has no upstream integrity checksum")


def _http_integrity_requirements(specification: dict) -> int | None:
    if not any(specification.get(key) for key in ("sha256", "git_blob_sha1", "md5")):
        raise ValueError("Asset has no upstream integrity checksum")
    expected_bytes = specification.get("bytes")
    if expected_bytes is not None and (isinstance(expected_bytes, bool)
                                       or not isinstance(expected_bytes, int) or expected_bytes < 0):
        raise ValueError("Invalid expected HTTP asset length")
    return expected_bytes


def _content_length(headers) -> int | None:
    value = headers.get("Content-Length")
    if value is None:
        return None
    try:
        length = int(value)
    except (TypeError, ValueError):
        raise ValueError("Invalid HTTP asset response length") from None
    if length < 0:
        raise ValueError("Invalid HTTP asset response length")
    return length


def _content_range(headers) -> tuple[int, int, int]:
    value = headers.get("Content-Range")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value or "", flags=re.IGNORECASE)
    if not match:
        raise ValueError("Invalid HTTP asset Content-Range")
    start, end, total = (int(part) for part in match.groups())
    if start > end or end >= total:
        raise ValueError("Invalid HTTP asset Content-Range")
    return start, end, total


def _http_status(response) -> int | None:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    return status


def _download_http_asset(url: str, temporary: Path, root: Path, specification: dict) -> None:
    """Resume an integrity-checked HTTPS transfer without exposing upstream URL errors."""
    expected_bytes = _http_integrity_requirements(specification)
    if temporary.is_symlink():
        raise ValueError("Partial asset path must not be a symlink")
    if temporary.exists() and not temporary.is_file():
        raise ValueError("Partial asset path is not a regular file")

    expected_total = expected_bytes
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset:
        try:
            _verify_download(temporary, specification)
        except ValueError:
            if expected_bytes is not None and offset >= expected_bytes:
                temporary.unlink()
                offset = 0
            else:
                offset = temporary.stat().st_size
        else:
            # A crash after completing the body but before promotion needs no network request.
            return

    minimum_free = 256 * 1024**2
    estimated_size = expected_bytes if expected_bytes is not None else UNKNOWN_ASSET_SIZE_ESTIMATE
    required_remaining = max(0, estimated_size - offset)
    if shutil.disk_usage(root).free < required_remaining + minimum_free:
        raise ValueError("Insufficient disk space for model asset")

    for attempt in range(HTTP_DOWNLOAD_ATTEMPTS):
        request_headers = {"User-Agent": "CN2VI-AutoDub/0.1"}
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=request_headers)
        except (TypeError, ValueError):
            raise ValueError("Invalid HTTP model asset URL") from None
        try:
            response = urllib.request.urlopen(request, timeout=HTTP_DOWNLOAD_TIMEOUT)
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            if status == 416 and offset:
                # A stale or corrupt complete partial may be beyond the upstream object.
                temporary.unlink(missing_ok=True)
                offset = 0
                expected_total = expected_bytes
                continue
            if status == 429 or 500 <= status <= 599:
                if attempt + 1 < HTTP_DOWNLOAD_ATTEMPTS:
                    time.sleep(HTTP_RETRY_BACKOFF_SECONDS[min(attempt, len(HTTP_RETRY_BACKOFF_SECONDS) - 1)])
                    continue
                raise RuntimeError("HTTP model asset download failed after retries") from None
            raise ValueError("HTTP model asset request was rejected") from None
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
            if attempt + 1 < HTTP_DOWNLOAD_ATTEMPTS:
                time.sleep(HTTP_RETRY_BACKOFF_SECONDS[min(attempt, len(HTTP_RETRY_BACKOFF_SECONDS) - 1)])
                continue
            raise RuntimeError("HTTP model asset download failed after retries") from None

        try:
            with response:
                status = _http_status(response)
                headers = response.headers
                response_length = _content_length(headers)
                if status == 429 or (status is not None and 500 <= status <= 599):
                    if attempt + 1 < HTTP_DOWNLOAD_ATTEMPTS:
                        time.sleep(HTTP_RETRY_BACKOFF_SECONDS[min(attempt, len(HTTP_RETRY_BACKOFF_SECONDS) - 1)])
                        continue
                    raise RuntimeError("HTTP model asset download failed after retries") from None
                if status not in {200, 206}:
                    raise ValueError("Unexpected HTTP model asset response")

                if status == 206:
                    start, end, total = _content_range(headers)
                    if start != offset or (expected_bytes is not None and total != expected_bytes):
                        raise ValueError("HTTP model asset range does not match the request")
                    if expected_total is not None and total != expected_total:
                        raise ValueError("HTTP model asset total length changed during resume")
                    expected_total = total
                    range_length = end - start + 1
                    if response_length is not None and response_length != range_length:
                        raise ValueError("HTTP model asset range length mismatch")
                    mode = "ab" if offset else "wb"
                    announced_length = range_length
                else:
                    # A 200 response to Range means the server ignored it; replace the partial.
                    mode = "wb"
                    announced_length = response_length
                    if response_length is not None:
                        if expected_bytes is not None and response_length != expected_bytes:
                            raise ValueError("HTTP model asset length mismatch")
                        if expected_total is not None and response_length != expected_total:
                            raise ValueError("HTTP model asset total length changed during resume")
                        expected_total = response_length

                received = 0
                with temporary.open(mode) as stream:
                    while True:
                        chunk = response.read(HTTP_COPY_CHUNK_BYTES)
                        if not chunk:
                            break
                        stream.write(chunk)
                        received += len(chunk)
                offset = temporary.stat().st_size
                if announced_length is not None and received > announced_length:
                    raise ValueError("HTTP model asset response exceeded its announced length")
                if announced_length is not None and received < announced_length:
                    # The transport ended early; preserve the verified prefix for Range retry.
                    if attempt + 1 < HTTP_DOWNLOAD_ATTEMPTS:
                        time.sleep(HTTP_RETRY_BACKOFF_SECONDS[min(attempt, len(HTTP_RETRY_BACKOFF_SECONDS) - 1)])
                    continue
                if expected_total is not None and offset > expected_total:
                    raise ValueError("Downloaded asset length mismatch")
                if expected_total is not None and offset < expected_total:
                    if attempt + 1 < HTTP_DOWNLOAD_ATTEMPTS:
                        time.sleep(HTTP_RETRY_BACKOFF_SECONDS[min(attempt, len(HTTP_RETRY_BACKOFF_SECONDS) - 1)])
                    continue
                _verify_download(temporary, specification)
                return
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
            if attempt + 1 < HTTP_DOWNLOAD_ATTEMPTS:
                offset = temporary.stat().st_size if temporary.exists() else 0
                time.sleep(HTTP_RETRY_BACKOFF_SECONDS[min(attempt, len(HTTP_RETRY_BACKOFF_SECONDS) - 1)])
                continue
            raise RuntimeError("HTTP model asset download failed after retries") from None

    raise ValueError("Downloaded asset length or checksum mismatch")


def _git_output(checkout: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(checkout), *arguments], check=False,
                                capture_output=True, text=True, encoding="utf-8")
    except OSError:
        raise RuntimeError("Could not verify pinned Git model asset") from None
    if result.returncode != 0:
        raise ValueError("Pinned Git model asset is not a valid checkout")
    return result.stdout


def _require_clean_checkout(checkout: Path) -> None:
    status = _git_output(checkout, "status", "--porcelain", "--untracked-files=all", "--ignored=matching")
    if status.strip():
        raise ValueError("Git model asset checkout has local changes or untracked files")


def _tracked_git_files(checkout: Path, folder: Path, revision: str) -> list[dict]:
    try:
        result = subprocess.run(["git", "-C", str(checkout), "ls-tree", "-rz", "--full-tree", revision],
                                check=False, capture_output=True)
    except OSError:
        raise RuntimeError("Could not enumerate pinned Git model files") from None
    if result.returncode != 0:
        raise ValueError("Could not enumerate files from pinned Git revision")

    files = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_relative = record.split(b"\t", maxsplit=1)
            mode, entry_type, _object_id = header.decode("ascii").split(" ")
        except (ValueError, UnicodeDecodeError):
            raise ValueError("Pinned Git revision contains an invalid tree entry") from None
        # Git symlinks and submodules are not model source files and must not
        # resolve or import content outside the committed regular-file tree.
        if entry_type != "blob" or mode not in {"100644", "100755"}:
            continue
        relative = os.fsdecode(raw_relative)
        path = safe_path(checkout, relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Pinned Git source file is missing or is not a regular file")
        files.append({"path": (Path("source") / relative).as_posix(), "bytes": path.stat().st_size,
                      "sha256": sha256_file(path)})
    if not files:
        raise ValueError("Pinned Git revision contains no regular source files")
    return files


def fetch_asset(specification: dict, root: Path) -> dict:
    identifier = specification["id"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", identifier):
        raise ValueError("Unsafe asset ID")
    folder = safe_path(root, identifier)
    folder.mkdir(parents=True, exist_ok=True)
    revision = specification["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision) and specification["kind"] != "http":
        raise ValueError("Asset must use an immutable commit revision")
    if specification["kind"] == "git":
        checkout = folder / "source"
        environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        if checkout.is_symlink():
            raise ValueError("Git model asset source path must not be a symlink")
        if not checkout.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", specification["repo"],
                            str(checkout)], check=True, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif not checkout.is_dir():
            raise ValueError("Git model asset source path is not a directory")
        else:
            if not (checkout / ".git").exists():
                raise ValueError("Existing Git model asset path is not a Git checkout")
            top_level = Path(_git_output(checkout, "rev-parse", "--show-toplevel").strip()).resolve()
            if top_level != checkout.resolve():
                raise ValueError("Existing Git model asset path is not the expected checkout root")
            _require_clean_checkout(checkout)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", revision], check=True,
                       env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        head = _git_output(checkout, "rev-parse", "HEAD").strip()
        if head != revision:
            raise ValueError("Git model asset checkout does not match the pinned revision")
        _require_clean_checkout(checkout)
        files = _tracked_git_files(checkout, folder, revision)
    else:
        files = []
        for item in specification["files"]:
            path = safe_path(folder, item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            valid = False
            if path.exists():
                try:
                    _verify_download(path, item)
                    valid = True
                except ValueError:
                    pass
            if not valid:
                url = item.get("url")
                if not isinstance(url, str) or not url.startswith("https://"):
                    raise ValueError("Only HTTPS asset downloads are allowed")
                temporary = path.with_suffix(path.suffix + ".part")
                _download_http_asset(url, temporary, root, item)
                _verify_download(temporary, item)
                temporary.replace(path)
            files.append({"path": item["path"], "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"schema_version": 1, "id": identifier, "model_revision": revision,
                "weights_sha256": manifest_digest(files), "files": files,
                "upstream": specification.get("repo", specification.get("source_url"))}
    atomic_json(folder / "model-manifest.json", manifest)
    return manifest


def fetch_lock(lock_path: Path, root: Path, identifiers: list[str] | None = None, *, progress=None) -> list[dict]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported models lock schema")
    root.mkdir(parents=True, exist_ok=True)
    selected = [model for model in lock["models"] if not identifiers or model["id"] in identifiers]
    if identifiers and {model["id"] for model in selected} != set(identifiers):
        raise ValueError("Unknown model asset ID")
    manifests = []
    for model in selected:
        if progress:
            progress(model["id"], "FETCHING")
        manifests.append(fetch_asset(model, root))
        if progress:
            progress(model["id"], "VERIFIED")
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "verify", "list"])
    parser.add_argument("--lock", type=Path, default=Path("benchmarks/models.lock.json"))
    parser.add_argument("--root", type=Path, default=Path("/data/models"))
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            manifests = fetch_lock(args.lock, args.root, args.only,
                                   progress=lambda identifier, status: print(
                                       json.dumps({"asset": identifier, "status": status}), file=sys.stderr, flush=True))
            print(json.dumps({"status": "FETCHED", "assets": [item["id"] for item in manifests]}))
        else:
            lock = json.loads(args.lock.read_text(encoding="utf-8"))
            if lock.get("schema_version") != 1:
                raise ValueError("Unsupported models lock schema")
            known_ids = {model["id"] for model in lock["models"]}
            if args.only and not set(args.only) <= known_ids:
                raise ValueError("Unknown model asset ID")
            models = [m for m in lock["models"] if not args.only or m["id"] in args.only]
            for model in models:
                if args.command == "verify":
                    manifest = load_manifest(safe_path(args.root, model["id"]))
                    if manifest.get("id") != model["id"] or manifest.get("model_revision") != model["revision"]:
                        raise ValueError("Model manifest does not match the pinned lock entry")
                print(f"{model['id']}: {model['revision']}")
    except Exception as error:
        print(f"Asset operation failed: {type(error).__name__}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
