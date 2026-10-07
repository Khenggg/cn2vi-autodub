import json
import threading
import uuid

from autodub.checkpoint import CheckpointError, fingerprint, read_checkpoint
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
        self.active_pipeline = None

    def start(self):
        with self.db.lock:
            active_states = ("V1_RUNNING", "V2_RUNNING", "PREPARING", "ASR", "ALIGNING", "TRANSLATING", "SEPARATING",
                             "TTS", "TIMING", "AUDIO_MIX", "QC", "PREVIEW_READY",
                             "AWAITING_ROI", "OCR_VERIFY", "TEXT_REMOVAL", "SUBTITLE_RENDER", "ENCODING")
            marks = ",".join("?" for _ in active_states)
            for row in self.db.rows(f"SELECT * FROM episode WHERE status IN ({marks})", active_states):
                next_stage = "ASR"
                try:
                    source = safe_path(self.settings.data_dir, row["source_path"])
                    if sha256_file(source) != row["source_sha256"]:
                        raise CheckpointError("Source checksum mismatch")
                    if row["status"] in {"V1_RUNNING", "V2_RUNNING"}:
                        self.db.transition(row["id"], "CHECKPOINTED", "Interrupted V1 run recovered",
                                           next_stage=row["status"], queue_requested=row["queue_requested"])
                        if row["queue_requested"]:
                            self.db.transition(row["id"], "QUEUED", "Resume frozen V1 checkpoint", queue_requested=1)
                        continue
                    checkpoint_path = self.settings.data_dir / "checkpoints" / row["id"] / "state.json"
                    if row["status"] == "PREPARING" and not checkpoint_path.is_file():
                        # Media probing had not produced a durable checkpoint. Requeue
                        # preparation itself, retaining the request even if the worker
                        # was drained before it restarted.
                        next_stage = "PREPARING"
                        self.db.transition(row["id"], "CHECKPOINTED", "Interrupted preparation recovered",
                                           next_stage=next_stage,
                                           queue_requested=row["queue_requested"])
                        if row["queue_requested"]:
                            self.db.transition(row["id"], "QUEUED", "Recovered interrupted preparation",
                                               queue_requested=1, next_stage=next_stage)
                        continue
                    from autodub.pipeline import Pipeline
                    pipeline = Pipeline(self.db, self.settings)
                    config_fp = fingerprint({"runner": pipeline._runner_config(),
                                             "duration_ms": row.get("duration_ms") or 1000,
                                             "subtitle_mode": pipeline._subtitle_mode_for_episode(row)})
                    checkpoint = read_checkpoint(self.settings.data_dir, row["id"],
                                                 row["source_sha256"], config_fp)
                    next_stage = checkpoint.get("next_stage") or "SEPARATING"
                except (CheckpointError, OSError, ValueError):
                    from autodub.pipeline import Pipeline
                    Pipeline(self.db, self.settings)._fail(
                        row["id"], "CHECKPOINT_INVALID", "Saved pipeline checkpoint is invalid; retry this episode")
                    continue
                self.db.transition(row["id"], "CHECKPOINTED", "Interrupted pipeline recovered",
                                   next_stage=next_stage)
                if row["queue_requested"] and self.state() == "ACCEPTING":
                    self.db.transition(row["id"], "QUEUED", "Recovered queued pipeline",
                                       queue_requested=1, next_stage=next_stage)
            self.thread = threading.Thread(target=self.run, name="preparation-worker", daemon=True)
            self.thread.start()

    def close(self):
        self.stop_event.set()
        self.wake_event.set()
        if self.active_pipeline is not None:
            self.active_pipeline.cancel()
        if self.thread:
            self.thread.join()

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
            for row in self.db.rows("SELECT id FROM episode WHERE status='CHECKPOINTED' AND queue_requested=1"):
                self.db.transition(row["id"], "QUEUED", "Worker resumed queued checkpoint",
                                   queue_requested=1)
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
        existing_checkpoint = False
        if self.settings.enable_pipeline and checkpoint.is_file():
            try:
                existing = json.loads(checkpoint.read_text(encoding="utf-8"))
                existing_checkpoint = (isinstance(existing, dict)
                                       and existing.get("episode_id") == episode["id"]
                                       and existing.get("source_sha256") == episode["source_sha256"])
            except (OSError, json.JSONDecodeError):
                existing_checkpoint = False
        first_stage = "V2_RUNNING" if self.settings.pipeline_generation == "v2" else "SEPARATING"
        if not existing_checkpoint:
            atomic_json(checkpoint, {"schema_version": 1, "episode_id": episode["id"],
                                    "source_sha256": episode["source_sha256"], "completed_stages": ["PREPARING"],
                                    "next_stage": first_stage, "media": metadata, "created_at": now_ms()})
        with self.db.lock:
            self.db.execute("DELETE FROM artifact WHERE episode_id=? AND kind='checkpoint'", (episode["id"],))
            self.db.execute("INSERT INTO artifact VALUES(?,?,?,?,?,?,?)",
                            (uuid.uuid4().hex, episode["id"], "checkpoint",
                             checkpoint.relative_to(self.settings.data_dir).as_posix(), sha256_file(checkpoint),
                             checkpoint.stat().st_size, now_ms()))
            self.db.transition(episode["id"], "CHECKPOINTED", "Media ready; recognition and voice-over providers required",
                               duration_ms=metadata["duration_ms"], progress=0.05,
                               next_stage=existing.get("next_stage", first_stage) if existing_checkpoint else first_stage,
                               queue_requested=1 if self.settings.enable_pipeline else 0)
            self.db.event(episode["id"], "CHECKPOINTED", "Preparation checkpoint saved",
                          {"stage_wall_ms": now_ms() - started, "provider_ready": False})
        if self.settings.enable_pipeline:
            from autodub.pipeline import Pipeline
            self.active_pipeline = Pipeline(self.db, self.settings)
            self.active_pipeline.run(episode, metadata,
                                     should_stop=lambda: self.stop_event.is_set() or self.state() != "ACCEPTING")
            self.active_pipeline = None

    def run(self):
        while not self.stop_event.is_set():
            if not self.tick():
                self.wake_event.wait(timeout=1)
                self.wake_event.clear()
