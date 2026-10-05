import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from autodub.domain import check_transition

SCHEMA = """
CREATE TABLE IF NOT EXISTS workspace(id TEXT PRIMARY KEY, schema_version INTEGER, created_at INTEGER);
INSERT OR IGNORE INTO workspace VALUES('default', 1, strftime('%s','now') * 1000);
CREATE TABLE IF NOT EXISTS runtime_state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO runtime_state VALUES('worker_state', 'ACCEPTING');
CREATE TABLE IF NOT EXISTS series(
 id TEXT PRIMARY KEY, title TEXT NOT NULL, priority INTEGER NOT NULL, status TEXT DEFAULT 'ACTIVE',
 glossary_version INTEGER DEFAULT 0, roi_json TEXT, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS episode(
 id TEXT PRIMARY KEY, series_id TEXT NOT NULL REFERENCES series(id) ON DELETE CASCADE,
 ordinal INTEGER NOT NULL, filename TEXT NOT NULL, total_bytes INTEGER NOT NULL,
 uploaded_bytes INTEGER DEFAULT 0, source_path TEXT, source_sha256 TEXT, duration_ms INTEGER,
 status TEXT NOT NULL, progress REAL DEFAULT 0, flags_json TEXT DEFAULT '[]',
 roi_json TEXT, queue_requested INTEGER DEFAULT 0, next_stage TEXT, created_at INTEGER,
 UNIQUE(series_id, ordinal));
CREATE TABLE IF NOT EXISTS segment(
 id TEXT PRIMARY KEY, episode_id TEXT REFERENCES episode(id) ON DELETE CASCADE,
 start_ms INTEGER, end_ms INTEGER, zh_text TEXT, subtitle_vi TEXT, dub_vi TEXT,
 confidence REAL, action TEXT, status TEXT, contract_json TEXT);
CREATE TABLE IF NOT EXISTS artifact(
 id TEXT PRIMARY KEY, episode_id TEXT REFERENCES episode(id) ON DELETE CASCADE,
 kind TEXT, path TEXT NOT NULL, sha256 TEXT, bytes INTEGER, created_at INTEGER);
CREATE TABLE IF NOT EXISTS issue(
 id TEXT PRIMARY KEY, episode_id TEXT REFERENCES episode(id) ON DELETE CASCADE,
 segment_id TEXT, code TEXT, severity TEXT, details_json TEXT, resolved INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS job_event(
 id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, ts INTEGER, state TEXT, message TEXT, metrics_json TEXT);
CREATE TABLE IF NOT EXISTS translation_memory(
 series_id TEXT REFERENCES series(id) ON DELETE CASCADE, source_hash TEXT, context_hash TEXT,
 result_json TEXT, PRIMARY KEY(series_id, source_hash, context_hash));
CREATE TABLE IF NOT EXISTS glossary_entry(
 series_id TEXT REFERENCES series(id) ON DELETE CASCADE, zh TEXT, vi TEXT,
 confidence REAL, locked_by_user INTEGER, updated_at INTEGER, PRIMARY KEY(series_id, zh));
CREATE INDEX IF NOT EXISTS episode_queue ON episode(status, queue_requested);
CREATE INDEX IF NOT EXISTS issue_episode ON issue(episode_id);
"""


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Path outside data directory")
    return path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        with self.lock:
            connection = sqlite3.connect(self.path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def rows(self, sql: str, values: tuple = ()) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(sql, values)]

    def one(self, sql: str, values: tuple = ()) -> dict | None:
        rows = self.rows(sql, values)
        return rows[0] if rows else None

    def execute(self, sql: str, values: tuple = ()) -> None:
        with self.connect() as connection:
            connection.execute(sql, values)

    def event(self, job_id: str, state: str, message: str, metrics: dict | None = None) -> None:
        self.execute("INSERT INTO job_event(job_id,ts,state,message,metrics_json) VALUES(?,?,?,?,?)",
                     (job_id, now_ms(), state, message, json.dumps(metrics or {})))

    def transition(self, episode_id: str, target: str, message: str, **fields) -> None:
        with self.connect() as connection:
            row = connection.execute("SELECT status FROM episode WHERE id=?", (episode_id,)).fetchone()
            if row is None:
                raise LookupError("Episode not found")
            check_transition(row["status"], target)
            allowed = {"progress", "duration_ms", "queue_requested", "next_stage", "flags_json"}
            if not fields.keys() <= allowed:
                raise ValueError("Invalid transition fields")
            assignments = ["status=?"] + [f"{key}=?" for key in fields]
            connection.execute(f"UPDATE episode SET {','.join(assignments)} WHERE id=?",
                               (target, *fields.values(), episode_id))
            connection.execute("INSERT INTO job_event(job_id,ts,state,message,metrics_json) VALUES(?,?,?,?,?)",
                               (episode_id, now_ms(), target, message, "{}"))
