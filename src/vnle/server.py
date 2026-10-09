"""Loopback prototype UI. Cloud access uses an SSH tunnel, not a public port."""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .domain import Request
from .media import probe
from .pipeline import DEFAULT_CONFIG, analyze, validate_config, write_json


def byte_range(header: str | None, size: int) -> tuple[int, int, bool]:
    if not header:
        return 0, max(0, size - 1), False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()) or size == 0:
        raise ValueError("Invalid range")
    a, b = match.groups()
    if not a:
        length = int(b)
        if length <= 0:
            raise ValueError("Invalid suffix range")
        start, end = max(0, size - length), size - 1
    else:
        start, end = int(a), min(int(b), size - 1) if b else size - 1
    if start >= size or start > end:
        raise ValueError("Unsatisfiable range")
    return start, end, True


class Store:
    def __init__(
        self, root: Path, ffprobe: str, models: Path | None, config: dict, enable_analysis: bool
    ):
        self.root = root.resolve()
        (self.root / "videos").mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        self.ffprobe = ffprobe
        self.models = models
        self.config = validate_config(config)
        self.enable_analysis = enable_analysis
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.cancellations: dict[str, threading.Event] = {}

    def video(self, identifier: str):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise FileNotFoundError("Unknown video")
        folder = self.root / "videos" / identifier
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        return folder / "input.mp4", metadata

    def start(self, identifier: str, data: dict):
        if not self.enable_analysis:
            raise ValueError(
                "Analysis is disabled on this UI-only server; enable it on the execution machine"
            )
        if not self.models or not self.models.is_file():
            raise ValueError("Configure the model manifest before starting OCR")
        video, metadata = self.video(identifier)
        request = Request.parse(data, metadata["media"]["duration_s"])
        with self.lock:
            if any(j["status"] == "RUNNING" for j in self.jobs.values()):
                raise ValueError("One OCR worker is already running; wait or cancel it")
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "video_id": identifier, "status": "RUNNING", "progress": {}}
            self.jobs[job_id] = job
            cancel = threading.Event()
            self.cancellations[job_id] = cancel
        frozen_config = dict(self.config)

        def progress(value):
            with self.lock:
                job["progress"] = value

        def work():
            try:
                report = analyze(
                    video,
                    metadata["media"],
                    request,
                    self.root / "runs" / job_id,
                    self.models,
                    frozen_config,
                    progress,
                    cancel,
                )
                with self.lock:
                    job["status"] = report["status"]
                    job["error"] = report["error"]
            except Exception as exc:
                with self.lock:
                    job["status"] = "FAILED"
                    job["error"] = str(exc)

        threading.Thread(target=work, name="vnle-ocr-owner", daemon=True).start()
        return dict(job)

    def job(self, identifier: str):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise FileNotFoundError("Unknown run")
        with self.lock:
            if identifier in self.jobs:
                return json.loads(json.dumps(self.jobs[identifier]))
        path = self.root / "runs" / identifier / "run-report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        # A killed server must not present an orphaned RUNNING run as still active.
        status = "INTERRUPTED" if report["status"] == "RUNNING" else report["status"]
        return {"id": identifier, "status": status, "progress": {}, "error": report.get("error")}


class Handler(BaseHTTPRequestHandler):
    server_version = "VNLE/0.1"

    @property
    def store(self) -> Store:
        return self.server.store

    def json_response(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= 64 * 1024:
            raise ValueError("Invalid JSON payload length")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("Incomplete request")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("JSON object expected")
        return value

    def same_origin(self):
        host = self.headers.get("Host", "")
        name = host.partition(":")[0]
        if name not in ("127.0.0.1", "localhost"):
            raise ValueError("Use the loopback address or an SSH tunnel")
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{host}":
            raise ValueError("Cross-origin writes are disabled")
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked request bodies are not supported")

    def file_response(self, path: Path, head=False):
        size = path.stat().st_size
        try:
            start, end, partial = byte_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        count = end - start + 1 if size else 0
        self.send_response(206 if partial else 200)
        self.send_header(
            "Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(count))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not head:
            with path.open("rb") as file:
                file.seek(start)
                while count:
                    block = file.read(min(count, 1024 * 1024))
                    if not block:
                        break
                    self.wfile.write(block)
                    count -= len(block)

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        try:
            self.same_origin()
            parts = urlsplit(self.path).path.strip("/").split("/")
            if parts == ["api", "status"]:
                self.json_response(
                    {
                        "analysis_enabled": self.store.enable_analysis,
                        "models_configured": bool(
                            self.store.models and self.store.models.is_file()
                        ),
                        "provider": self.store.config["provider"],
                        "version": "0.1.0",
                    }
                )
            elif len(parts) == 3 and parts[:2] == ["api", "videos"]:
                self.json_response(self.store.video(parts[2])[1])
            elif len(parts) == 2 and parts[0] == "media":
                self.file_response(self.store.video(parts[1])[0], head)
            elif len(parts) == 3 and parts[:2] == ["api", "runs"]:
                self.json_response(self.store.job(parts[2]))
            elif len(parts) >= 3 and parts[0] == "artifacts":
                self.store.job(parts[1])
                relative = "/".join(parts[2:])
                if relative not in (
                    "events.json",
                    "run-report.json",
                    "run-report.md",
                    "observations.jsonl",
                ):
                    if not re.fullmatch(r"evidence/e\d{6}\.png", relative):
                        raise FileNotFoundError("Unknown artifact")
                self.file_response(self.store.root / "runs" / parts[1] / relative, head)
            elif len(parts) == 1 and parts[0] in ("", "app.js", "style.css"):
                self.file_response(Path(__file__).parent / "web" / (parts[0] or "index.html"), head)
            else:
                self.json_response({"error": "Not found"}, 404)
        except (FileNotFoundError, KeyError):
            self.json_response({"error": "Not found"}, 404)
        except (ValueError, TypeError) as exc:
            self.json_response({"error": str(exc)}, 400)
        except (ConnectionError, BrokenPipeError):
            pass

    def do_POST(self):
        try:
            self.same_origin()
            path = urlsplit(self.path)
            parts = path.path.strip("/").split("/")
            if parts == ["api", "videos"]:
                name = unquote(parse_qs(path.query).get("name", ["video.mp4"])[0])
                if not name.lower().endswith(".mp4"):
                    raise ValueError("Prototype accepts MP4 input")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 8 * 1024**3:
                    raise ValueError("Upload must contain at most 8 GiB")
                identifier = uuid.uuid4().hex
                folder = self.store.root / "videos" / identifier
                folder.mkdir()
                temp = folder / "upload.part"
                try:
                    with temp.open("wb") as file:
                        remaining = length
                        while remaining:
                            block = self.rfile.read(min(1024 * 1024, remaining))
                            if not block:
                                raise ValueError("Upload was interrupted")
                            file.write(block)
                            remaining -= len(block)
                    metadata = {
                        "id": identifier,
                        "name": name,
                        "bytes": length,
                        "media": probe(temp, self.store.ffprobe),
                    }
                    temp.replace(folder / "input.mp4")
                    write_json(folder / "metadata.json", metadata)
                except Exception:
                    temp.unlink(missing_ok=True)
                    raise
                self.json_response(metadata, 201)
            elif parts == ["api", "runs"]:
                data = self.json_body()
                self.json_response(self.store.start(data["video_id"], data), 202)
            elif len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "cancel":
                self.store.job(parts[2])
                cancellation = self.store.cancellations.get(parts[2])
                if cancellation:
                    cancellation.set()
                self.json_response({"cancel_requested": bool(cancellation)})
            else:
                self.json_response({"error": "Not found"}, 404)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.json_response({"error": str(exc)}, 400)
        except FileNotFoundError:
            self.json_response({"error": "Not found"}, 404)
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as exc:
            self.json_response({"error": f"{type(exc).__name__}: {exc}"}, 500)


def serve(
    root: Path, port=18083, ffprobe="ffprobe", models=None, config=None, enable_analysis=False
):
    store = Store(root, ffprobe, models, config or DEFAULT_CONFIG, enable_analysis)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.store = store
    print(
        f"VNLE: http://127.0.0.1:{port} (analysis {'enabled' if enable_analysis else 'disabled'})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for cancellation in store.cancellations.values():
            cancellation.set()
        server.server_close()
