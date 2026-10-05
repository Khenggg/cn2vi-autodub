import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from autodub.config import Settings
from autodub.contracts import EpisodeCreate, GlossaryEntry, Roi, SeriesCreate, SeriesPatch, StrictModel
from autodub.scheduler import Scheduler
from autodub.service import Conflict, Service
from autodub.storage import Database, safe_path
from autodub.system import status

logger = logging.getLogger("uvicorn.error")


class Login(StrictModel):
    token: str = Field(min_length=1, max_length=512)


def create_app(settings: Settings | None = None, *, start_worker: bool = True) -> FastAPI:
    settings = settings or Settings()
    settings.prepare()
    db = Database(settings.data_dir / "app-state" / "autodub.sqlite")
    service = Service(db, settings)
    scheduler = Scheduler(db, settings)
    token = settings.admin_token or secrets.token_urlsafe(32)
    sessions: dict[str, float] = {}
    attempts: dict[str, tuple[int, float]] = {}
    started = time.monotonic()

    @asynccontextmanager
    async def lifespan(app):
        if not settings.admin_token:
            logger.warning("CN2VI admin token (enter in browser): %s", token)
        if start_worker:
            scheduler.start()
        yield
        await run_in_threadpool(scheduler.close)

    app = FastAPI(title="CN2VI AutoDub", version="0.1.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.db, app.state.service, app.state.scheduler = db, service, scheduler
    app.state.settings = settings

    @app.middleware("http")
    async def protect(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/"):
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                if request.headers.get("x-autodub-request") != "1":
                    return JSONResponse({"detail": "Missing request protection header"}, status_code=403)
                origin = request.headers.get("origin")
                if origin and urlsplit(origin).netloc != request.headers.get("host"):
                    return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
            if path != "/api/auth/login":
                session = request.cookies.get("autodub_session", "")
                expiry = sessions.get(session, 0)
                bearer = request.headers.get("authorization", "")
                valid_bearer = bearer.startswith("Bearer ") and secrets.compare_digest(bearer[7:], token)
                if expiry < time.monotonic() and not valid_bearer:
                    sessions.pop(session, None)
                    return JSONResponse({"detail": "Admin token required"}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(LookupError)
    async def not_found(request, error):
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.get("/healthz")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/api/auth/login")
    def login(payload: Login, request: Request):
        address = request.client.host if request.client else "local"
        current = time.monotonic()
        # Bound unauthenticated per-address accounting. Single-user sessions are process-local.
        for key, (_, until) in list(attempts.items()):
            if until < current:
                attempts.pop(key)
        count, until = attempts.get(address, (0, current + 60))
        if count >= 10:
            raise HTTPException(429, "Too many login attempts; wait one minute")
        if not secrets.compare_digest(payload.token, token):
            if len(attempts) >= 1024 and address not in attempts:
                raise HTTPException(429, "Too many login attempts")
            attempts[address] = (count + 1, until)
            raise HTTPException(401, "Invalid admin token")
        attempts.pop(address, None)
        for key, expiry in list(sessions.items()):
            if expiry < current:
                sessions.pop(key)
        if len(sessions) >= 32:
            sessions.pop(next(iter(sessions)))
        session = secrets.token_urlsafe(32)
        sessions[session] = current + 86400
        response = JSONResponse({"authenticated": True})
        response.set_cookie("autodub_session", session, httponly=True, secure=settings.cookie_secure,
                            samesite="strict", max_age=86400)
        return response

    @app.post("/api/auth/logout")
    def logout(request: Request):
        sessions.pop(request.cookies.get("autodub_session", ""), None)
        response = JSONResponse({"authenticated": False})
        response.delete_cookie("autodub_session")
        return response

    @app.get("/api/system/status")
    def system_status():
        return status(settings, started, scheduler.state(), scheduler.active)

    @app.get("/api/series")
    def list_series():
        return [service.series_detail(row["id"]) for row in db.rows("SELECT id FROM series ORDER BY priority,created_at,id")]

    @app.post("/api/series", status_code=201)
    def create_series(payload: SeriesCreate):
        return service.create_series(payload)

    @app.get("/api/series/{identifier}")
    def get_series(identifier: str):
        return service.series_detail(identifier)

    @app.patch("/api/series/{identifier}")
    def patch_series(identifier: str, payload: SeriesPatch):
        return service.patch_series(identifier, payload)

    @app.delete("/api/series/{identifier}")
    def delete_series(identifier: str):
        service.delete_series(identifier)
        return {"deleted": identifier}

    @app.put("/api/series/{identifier}/glossary")
    def update_glossary(identifier: str, payload: list[GlossaryEntry]):
        return service.glossary(identifier, payload)

    @app.post("/api/episodes", status_code=201)
    def create_episode(payload: EpisodeCreate):
        return service.create_episode(payload)

    @app.get("/api/episodes/{identifier}")
    @app.get("/api/uploads/{identifier}")
    def get_episode(identifier: str):
        return service.episode_detail(identifier)

    @app.put("/api/uploads/{identifier}/chunks")
    async def upload(identifier: str, request: Request, upload_offset: int = Header(ge=0)):
        length = request.headers.get("content-length")
        if length and int(length) > settings.max_chunk_bytes:
            raise HTTPException(413, "Chunk exceeds 8 MiB")
        chunks = bytearray()
        async for block in request.stream():
            chunks.extend(block)
            if len(chunks) > settings.max_chunk_bytes:
                raise HTTPException(413, "Chunk exceeds 8 MiB")
        return await run_in_threadpool(service.append_chunk, identifier, upload_offset, bytes(chunks))

    @app.post("/api/episodes/{identifier}/start")
    def start_episode(identifier: str):
        with db.lock:
            result = service.start_episode(identifier, scheduler.state())
        scheduler.wake_event.set()
        return result

    @app.post("/api/episodes/{identifier}/retry")
    def retry_episode(identifier: str):
        with db.lock:
            result = service.retry_episode(identifier, scheduler.state())
        scheduler.wake_event.set()
        return result

    @app.put("/api/episodes/{identifier}/roi")
    def roi(identifier: str, payload: Roi):
        return service.set_roi(identifier, payload)

    @app.delete("/api/episodes/{identifier}")
    def delete_episode(identifier: str):
        service.delete_episode(identifier)
        return {"deleted": identifier}

    @app.get("/api/episodes/{identifier}/source")
    def source(identifier: str):
        episode = service.require("episode", identifier)
        if not episode["source_path"]:
            raise Conflict("Upload not complete")
        path = safe_path(settings.data_dir, episode["source_path"])
        if not path.is_file():
            raise LookupError("Source not available")
        return FileResponse(path)

    @app.post("/api/worker/drain")
    def drain():
        return {"worker_state": scheduler.drain()}

    @app.post("/api/worker/resume")
    def resume():
        scheduler.resume()
        return {"worker_state": scheduler.state()}

    @app.post("/api/workspace/export")
    def export():
        return service.export_workspace()

    @app.get("/api/download/{identifier}")
    def download(identifier: str):
        artifact = service.require("artifact", identifier)
        path = safe_path(settings.data_dir, artifact["path"])
        if not path.is_file():
            raise LookupError("Artifact not available")
        return FileResponse(path, filename=path.name)

    @app.get("/api/events")
    async def events(request: Request, after: int = 0):
        try:
            cursor = max(0, int(request.headers.get("last-event-id", after)))
        except ValueError:
            raise HTTPException(400, "Invalid event cursor") from None

        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                batch = await run_in_threadpool(db.rows, "SELECT * FROM job_event WHERE id>? ORDER BY id LIMIT 100", (cursor,))
                for event in batch:
                    cursor = event["id"]
                    yield f"id: {cursor}\nevent: job\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                if not batch:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    if settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    else:
        @app.get("/")
        def no_frontend():
            return JSONResponse({"message": "Run npm ci && npm run build in frontend; restart server"})
    return app


# Use uvicorn autodub.main:create_app --factory --workers 1 to retain single SQLite ownership.
