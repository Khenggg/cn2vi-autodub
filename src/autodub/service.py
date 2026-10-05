import hashlib
import json
import os
import shutil
import sqlite3
import uuid
import zipfile
from pathlib import PurePosixPath

from autodub.config import Settings
from autodub.contracts import EpisodeCreate, GlossaryEntry, Roi, SeriesCreate, SeriesPatch
from autodub.storage import Database, now_ms, safe_path, sha256_file


class Conflict(ValueError):
    pass


class Service:
    def __init__(self, db: Database, settings: Settings):
        self.db, self.settings = db, settings

    def require(self, table: str, identifier: str) -> dict:
        if table not in {"series", "episode", "artifact"}:
            raise ValueError("Invalid table")
        row = self.db.one(f"SELECT * FROM {table} WHERE id=?", (identifier,))
        if row is None:
            raise LookupError(f"{table.capitalize()} not found")
        return row

    def series_detail(self, identifier: str) -> dict:
        series = self.require("series", identifier)
        series["roi"] = json.loads(series.pop("roi_json") or "null")
        series["glossary"] = self.db.rows("SELECT zh,vi,confidence,locked_by_user FROM glossary_entry WHERE series_id=?", (identifier,))
        series["episodes"] = [self.episode_detail(row["id"]) for row in self.db.rows(
            "SELECT id FROM episode WHERE series_id=? ORDER BY ordinal", (identifier,))]
        return series

    def episode_detail(self, identifier: str) -> dict:
        episode = self.require("episode", identifier)
        episode.pop("source_path", None)
        episode["roi"] = json.loads(episode.pop("roi_json") or "null")
        episode["flags"] = json.loads(episode.pop("flags_json"))
        episode["issues"] = self.db.rows("SELECT id,code,severity,details_json,resolved FROM issue WHERE episode_id=?", (identifier,))
        for issue in episode["issues"]:
            issue["details"] = json.loads(issue.pop("details_json"))
        episode["artifacts"] = self.db.rows("SELECT id,kind,bytes FROM artifact WHERE episode_id=?", (identifier,))
        return episode

    def create_series(self, payload: SeriesCreate) -> dict:
        title = payload.title.strip()
        if not title:
            raise ValueError("Title cannot be blank")
        identifier = uuid.uuid4().hex
        self.db.execute("INSERT INTO series(id,title,priority,created_at) VALUES(?,?,?,?)",
                        (identifier, title, payload.priority, now_ms()))
        self.db.event(identifier, "CREATED", "Series created")
        return self.series_detail(identifier)

    def patch_series(self, identifier: str, payload: SeriesPatch) -> dict:
        self.require("series", identifier)
        values = payload.model_dump(exclude_unset=True)
        for key, value in values.items():
            if value is None or (key == "title" and not value.strip()):
                raise ValueError("Title and priority must have a value")
            self.db.execute(f"UPDATE series SET {key}=? WHERE id=?", (value.strip() if key == "title" else value, identifier))
        return self.series_detail(identifier)

    def create_episode(self, payload: EpisodeCreate) -> dict:
        with self.db.lock:
            self.require("series", payload.series_id)
            name = payload.filename
            if any(character in name for character in ("/", "\\", ":", "\x00")) or any(ord(c) < 32 for c in name):
                raise ValueError("Filename must be a basename")
            extension = PurePosixPath(name).suffix.lower()
            if extension not in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}:
                raise ValueError("Unsupported video extension")
            if payload.total_bytes > self.settings.max_upload_bytes:
                raise ValueError("Video exceeds upload limit")
            reserved = self.db.one("SELECT COALESCE(SUM(total_bytes),0) AS bytes FROM episode")["bytes"]
            if reserved + payload.total_bytes > self.settings.workspace_quota_bytes:
                raise Conflict("Workspace source quota reached; delete an episode first")
            if shutil.disk_usage(self.settings.data_dir).free < payload.total_bytes + 512 * 1024**2:
                raise Conflict("Not enough free disk space")
            identifier = uuid.uuid4().hex
            folder = self.settings.data_dir / "uploads" / payload.series_id / identifier
            try:
                self.db.execute("""INSERT INTO episode(id,series_id,ordinal,filename,total_bytes,status,created_at)
                                  VALUES(?,?,?,?,?,'UPLOADING',?)""",
                                (identifier, payload.series_id, payload.ordinal, name, payload.total_bytes, now_ms()))
            except sqlite3.IntegrityError as error:
                raise Conflict("Episode ordinal already exists in this Series") from error
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "source.part").touch()
            self.db.event(identifier, "UPLOADING", "Upload slot created")
            return self.episode_detail(identifier)

    def append_chunk(self, identifier: str, offset: int, chunk: bytes) -> dict:
        if not chunk or len(chunk) > self.settings.max_chunk_bytes:
            raise ValueError("Chunk must contain 1 to 8 MiB")
        with self.db.lock:
            episode = self.require("episode", identifier)
            if episode["status"] != "UPLOADING":
                raise Conflict("Upload already finalized")
            if offset != episode["uploaded_bytes"]:
                raise Conflict(f"Offset mismatch; resume at {episode['uploaded_bytes']}")
            if offset + len(chunk) > episode["total_bytes"]:
                raise ValueError("Chunk exceeds declared file length")
            folder = self.settings.data_dir / "uploads" / episode["series_id"] / identifier
            part = folder / "source.part"
            final = folder / ("source" + PurePosixPath(episode["filename"]).suffix.lower())
            # Recovery from a crash after rename but before the DB commit.
            if not part.exists() and final.exists():
                final.replace(part)
            with part.open("r+b") as stream:
                if part.stat().st_size < offset:
                    raise Conflict("Upload data missing; delete slot and upload again")
                stream.truncate(offset)  # Roll back uncommitted tail after an interrupted write.
                stream.seek(offset)
                stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            uploaded = offset + len(chunk)
            if uploaded == episode["total_bytes"]:
                digest = sha256_file(part)
                part.replace(final)
                with self.db.connect() as connection:
                    connection.execute("UPDATE episode SET uploaded_bytes=?,source_path=?,source_sha256=? WHERE id=?",
                                       (uploaded, final.relative_to(self.settings.data_dir).as_posix(), digest, identifier))
                    connection.execute("UPDATE episode SET status='QUEUED' WHERE id=?", (identifier,))
                    connection.execute("INSERT INTO job_event(job_id,ts,state,message,metrics_json) VALUES(?,?,?,?,?)",
                                       (identifier, now_ms(), "QUEUED", "Upload complete; ready to start", "{}"))
            else:
                self.db.execute("UPDATE episode SET uploaded_bytes=? WHERE id=?", (uploaded, identifier))
            return self.episode_detail(identifier)

    def start_episode(self, identifier: str, worker_state: str) -> dict:
        with self.db.lock:
            episode = self.require("episode", identifier)
            if worker_state != "ACCEPTING":
                raise Conflict("Worker is drained; resume it first")
            if episode["status"] == "CHECKPOINTED" and episode["next_stage"] == "ASR":
                raise Conflict("ASR provider is not configured yet; media checkpoint is saved")
            if episode["status"] != "QUEUED" or not episode["source_sha256"]:
                raise Conflict("Episode must finish uploading before it can start")
            if not episode["queue_requested"]:
                self.db.execute("UPDATE episode SET queue_requested=1 WHERE id=?", (identifier,))
                self.db.event(identifier, "QUEUED", "Processing requested")
            return self.episode_detail(identifier)

    def retry_episode(self, identifier: str, worker_state: str) -> dict:
        with self.db.lock:
            episode = self.require("episode", identifier)
            if episode["status"] != "FAILED":
                raise Conflict("Only failed preparation can be retried in phase 1")
            if worker_state != "ACCEPTING":
                raise Conflict("Resume worker before retry")
            self.db.transition(identifier, "RETRYING", "Retry requested")
            self.db.execute("UPDATE issue SET resolved=1 WHERE episode_id=?", (identifier,))
            self.db.transition(identifier, "QUEUED", "Retry queued", queue_requested=1)
            return self.episode_detail(identifier)

    def set_roi(self, identifier: str, roi: Roi) -> dict:
        episode = self.require("episode", identifier)
        if episode["status"] not in {"PREVIEW_READY", "AWAITING_ROI"}:
            raise Conflict("Review the Phase A preview before selecting a subtitle ROI")
        payload = roi.model_dump_json()
        if roi.scope == "series":
            self.db.execute("UPDATE series SET roi_json=? WHERE id=?", (payload, episode["series_id"]))
        self.db.execute("UPDATE episode SET roi_json=? WHERE id=?", (payload, identifier))
        if episode["status"] == "PREVIEW_READY":
            self.db.transition(identifier, "AWAITING_ROI", "ROI saved; final processing awaits confirmation")
        return self.episode_detail(identifier)

    def glossary(self, identifier: str, entries: list[GlossaryEntry]) -> list[dict]:
        self.require("series", identifier)
        if len(entries) > 5000:
            raise ValueError("Too many glossary entries")
        if len({entry.zh for entry in entries}) != len(entries):
            raise ValueError("Duplicate glossary term")
        with self.db.connect() as connection:
            for entry in entries:
                old = connection.execute("SELECT locked_by_user FROM glossary_entry WHERE series_id=? AND zh=?", (identifier, entry.zh)).fetchone()
                if old and old["locked_by_user"] and not entry.locked_by_user:
                    raise Conflict("A model proposal cannot override a user-locked term")
                connection.execute("INSERT OR REPLACE INTO glossary_entry VALUES(?,?,?,?,?,?)",
                                   (identifier, entry.zh, entry.vi, entry.confidence, int(entry.locked_by_user), now_ms()))
            connection.execute("UPDATE series SET glossary_version=glossary_version+1 WHERE id=?", (identifier,))
        return self.series_detail(identifier)["glossary"]

    def delete_episode(self, identifier: str):
        with self.db.lock:
            episode = self.require("episode", identifier)
            if episode["status"] == "PREPARING":
                raise Conflict("Drain the active worker before deleting this episode")
            folders = [self.settings.data_dir / "uploads" / episode["series_id"] / identifier,
                       self.settings.data_dir / "work" / identifier, self.settings.data_dir / "checkpoints" / identifier]
            for artifact in self.db.rows("SELECT path FROM artifact WHERE episode_id=?", (identifier,)):
                safe_path(self.settings.data_dir, artifact["path"]).unlink(missing_ok=True)
            for folder in folders:
                checked = safe_path(self.settings.data_dir, folder.relative_to(self.settings.data_dir).as_posix())
                if checked.exists():
                    shutil.rmtree(checked)
            self.db.execute("DELETE FROM episode WHERE id=?", (identifier,))
            self.db.event(identifier, "DELETED", "Episode deleted by user")

    def delete_series(self, identifier: str):
        with self.db.lock:
            self.require("series", identifier)
            episodes = self.db.rows("SELECT id,status FROM episode WHERE series_id=?", (identifier,))
            if any(episode["status"] == "PREPARING" for episode in episodes):
                raise Conflict("Drain the active worker before deleting this Series")
            for episode in episodes:
                self.delete_episode(episode["id"])
            self.db.execute("DELETE FROM series WHERE id=?", (identifier,))

    def export_workspace(self) -> dict:
        identifier = uuid.uuid4().hex
        folder = self.settings.data_dir / "app-state" / "workspace"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{identifier}.aidub"
        with self.db.lock:
            # Whitelist metadata only: never serialize env, paths, media, or credentials.
            tables = {
                "workspace": self.db.rows("SELECT * FROM workspace"),
                "series": self.db.rows("SELECT * FROM series"),
                "episode": self.db.rows("""SELECT id,series_id,ordinal,filename,total_bytes,source_sha256,
                    duration_ms,status,next_stage,roi_json,flags_json FROM episode"""),
                "glossary_entry": self.db.rows("SELECT * FROM glossary_entry"),
                "segment": self.db.rows("SELECT * FROM segment"),
                "issue": self.db.rows("SELECT * FROM issue"),
            }
            payload = json.dumps(tables, ensure_ascii=False, indent=2).encode()
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps({"schema_version": 1, "created_at": now_ms(),
                                                             "kind": "workspace_metadata", "contains_media": False,
                                                             "import_supported": False}))
                archive.writestr("workspace.sqlite.export.json", payload)
                archive.writestr("checksums.json", json.dumps({"workspace.sqlite.export.json": hashlib.sha256(payload).hexdigest()}))
            self.db.execute("INSERT INTO artifact VALUES(?,?,?,?,?,?,?)",
                            (identifier, None, "workspace", path.relative_to(self.settings.data_dir).as_posix(),
                             sha256_file(path), path.stat().st_size, now_ms()))
        return {"artifact_id": identifier, "download_url": f"/api/download/{identifier}"}
