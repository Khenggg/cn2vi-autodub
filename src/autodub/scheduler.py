import json
import threading
import uuid

from autodub.config import Settings
from autodub.media import MediaError, probe_media
from autodub.storage import Database, atomic_json, now_ms, safe_path, sha256_file


class Scheduler:
    """Phase 1 CPU preparation worker; heavy stages remain disabled behind benchmark gates.

    Only one core process owns SQLite and this thread. Phase 2 adds separate model workers.
    Drain waits for the active bounded probe and its checkpoint before reporting shutdown readiness.
    """

    def __init__(self, db: Database, settings: Settings):
        self.db, self.settings = db, settings
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.active: str | None = None

    def start(self):
        with self.db.lock:
            # A probe is idempotent; recover a process killed before its atomic checkpoint.
            for row in self.db.rows("SELECT id FROM episode WHERE status='PREPARING'"):
                self.db.transition(row["id"], "CHECKPOINTED", "Preparation interrupted; safe to retry")
                self.db.transition(row["id"], "QUEUED", "Recovered preparation", queue_requested=1)
            self.thread = threading.Thread(target=self.run, name="preparation-worker", daemon=True)
            self.thread.start()

    def close(self):
        self.stop_event.set()
        self.wake_event.set()
        if self.thread:
            self.thread.join(timeout=65)

    def state(self) -> str:
        return self.db.one("SELECT value FROM runtime_state WHERE key='worker_state'")["value"]

    def drain(self):
        with self.db.lock:
            state = "DRAINING" if self.active else "READY_TO_SHUTDOWN"
            self.db.execute("UPDATE runtime_state SET value=? WHERE key='worker_state'", (state,))
            self.db.event("worker", state, "Worker drain requested")
        self.wake_event.set()
        return state

    def resume(self):
        with self.db.lock:
            if self.active and self.state() == "DRAINING":
                raise ValueError("Wait for the active stage to checkpoint")
            self.db.execute("UPDATE runtime_state SET value='ACCEPTING' WHERE key='worker_state'")
            self.db.event("worker", "ACCEPTING", "Worker resumed")
        self.wake_event.set()

    def next_episode(self) -> dict | None:
        # Lower priority number wins. Within a Series, prepare episodes in ordinal order.
        return self.db.one("""SELECT e.* FROM episode e JOIN series s ON s.id=e.series_id
          WHERE e.status='QUEUED' AND e.queue_requested=1
          ORDER BY s.priority, s.created_at, s.id, e.ordinal LIMIT 1""")

    def tick(self) -> bool:
        with self.db.lock:
            if self.state() != "ACCEPTING":
                if not self.active:
                    self.db.execute("UPDATE runtime_state SET value='READY_TO_SHUTDOWN' WHERE key='worker_state'")
                return False
            episode = self.next_episode()
            if not episode:
                return False
            self.active = episode["id"]
            self.db.transition(self.active, "PREPARING", "Inspecting media", progress=0.02)
        try:
            self.prepare(episode)
        except Exception as error:
            # Keep the queue moving; never include model/API secrets or arbitrary exception repr in logs.
            code = "MEDIA_INVALID" if isinstance(error, MediaError) else "PREPARATION_FAILED"
            message = str(error) if isinstance(error, MediaError) else "Preparation failed; retry this episode"
            with self.db.lock:
                self.db.execute("INSERT INTO issue VALUES(?,?,?,?,?,?,?)",
                                (uuid.uuid4().hex, episode["id"], None, code, "error",
                                 json.dumps({"message": message}), 0))
                self.db.transition(episode["id"], "FAILED", message, queue_requested=0)
        finally:
            with self.db.lock:
                self.active = None
                if self.state() == "DRAINING":
                    self.db.execute("UPDATE runtime_state SET value='READY_TO_SHUTDOWN' WHERE key='worker_state'")
                    self.db.event("worker", "READY_TO_SHUTDOWN", "Active stage checkpointed")
        return True

    def prepare(self, episode: dict):
        started = now_ms()
        path = safe_path(self.settings.data_dir, episode["source_path"])
        metadata = probe_media(path, self.settings.ffprobe_bin)
        if sha256_file(path) != episode["source_sha256"]:
            raise MediaError("Source checksum mismatch; re-upload the original video")
        checkpoint = self.settings.data_dir / "checkpoints" / episode["id"] / "state.json"
        atomic_json(checkpoint, {"schema_version": 1, "episode_id": episode["id"],
                                "source_sha256": episode["source_sha256"], "completed_stages": ["PREPARING"],
                                "next_stage": "ASR", "media": metadata, "created_at": now_ms()})
        with self.db.lock:
            self.db.execute("DELETE FROM artifact WHERE episode_id=? AND kind='checkpoint'", (episode["id"],))
            self.db.execute("INSERT INTO artifact VALUES(?,?,?,?,?,?,?)",
                            (uuid.uuid4().hex, episode["id"], "checkpoint",
                             checkpoint.relative_to(self.settings.data_dir).as_posix(), sha256_file(checkpoint),
                             checkpoint.stat().st_size, now_ms()))
            self.db.transition(episode["id"], "CHECKPOINTED", "Media ready; ASR benchmark/provider required",
                               duration_ms=metadata["duration_ms"], progress=0.05, next_stage="ASR", queue_requested=0)
            self.db.event(episode["id"], "CHECKPOINTED", "Preparation checkpoint saved",
                          {"stage_wall_ms": now_ms() - started, "provider_ready": False})

    def run(self):
        while not self.stop_event.is_set():
            if not self.tick():
                self.wake_event.wait(timeout=1)
                self.wake_event.clear()
