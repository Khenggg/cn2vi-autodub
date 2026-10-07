"""End-to-end automated dubbing pipeline orchestrator."""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from autodub.audio_mix import fit_voice, mix_preview
from autodub.checkpoint import (
    STAGE_ORDER,
    CheckpointError,
    artifact_record,
    fingerprint,
    invalidate_from_stage,
    read_checkpoint,
    save_checkpoint,
)
from autodub.config import Settings
from autodub.contracts import Roi, Segment
from autodub.media import probe_audio
from autodub.model_runner import ModelRunner
from autodub.storage import Database, now_ms, safe_path, sha256_file

logger = logging.getLogger("autodub.pipeline")


class _PipelineStopped(Exception):
    """Raised at a stage boundary when the worker is draining or stopping."""


class Pipeline:
    """Execute all automated dubbing stages for an episode."""

    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self.runner: ModelRunner | None = None
        self._active_process: subprocess.Popen | None = None
        self.stop_requested = threading.Event()
        self._should_stop = lambda: self.stop_requested.is_set()
        self._checkpoint: dict = {}
        self._config_fingerprint = ""
        self._episode_id = ""
        self._source_sha256 = ""
        self._active_stage: str | None = None
        self._subtitle_mode = self.settings.subtitle_mode

    def cancel(self) -> None:
        self.stop_requested.set()
        if self.runner is not None:
            self.runner.cancel()
        if self._active_process is not None:
            ModelRunner._terminate(self._active_process)

    def _command(self, command: list[str], timeout: int, *, text: bool = False,
                 check: bool = True) -> subprocess.CompletedProcess:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=text, creationflags=flags, start_new_session=os.name != "nt")
        self._active_process = process
        try:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                ModelRunner._terminate(process)
                raise RuntimeError("Bounded media command timed out") from error
        finally:
            self._active_process = None
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check and process.returncode:
            raise RuntimeError("Bounded media command failed")
        return result

    def _fail(self, episode_id: str, code: str, message: str) -> bool:
        with self.db.lock:
            self.db.execute("INSERT INTO issue VALUES(?,?,?,?,?,?,?)",
                            (uuid.uuid4().hex, episode_id, None, code, "error",
                             json.dumps({"message": message}), 0))
            current = self.db.one("SELECT status FROM episode WHERE id=?", (episode_id,))
            if current and current["status"] not in {"FAILED", "COMPLETED", "NEEDS_REVIEW"}:
                self.db.transition(episode_id, "FAILED", message, queue_requested=0)
        return False

    def _checkpoint_stage(self, stage: str, result: dict) -> None:
        """Persist a stage result only after every referenced artifact is durable."""
        artifacts = self._checkpoint.setdefault("artifacts", {})
        for path_value in [*result.get("artifacts", []), *result.get("clips", {}).values()]:
            path = Path(path_value)
            record = artifact_record(self.settings.data_dir, path)
            artifacts[f"{stage}:{path.name}"] = record
        completed = self._checkpoint.setdefault("completed_stages", [])
        needs_review = bool(result.get("review_required") or result.get("action") == "NEEDS_REVIEW")
        if stage not in completed and not needs_review:
            completed.append(stage)
        results = self._checkpoint.setdefault("results", {})
        results[stage] = result
        next_stage = stage if needs_review else {"SEPARATING": "ASR", "ASR": "ALIGNING",
                      "ALIGNING": "TRANSLATING", "TRANSLATING": "TTS",
                      "TTS": "TIMING", "VISION_RENDER": "ENCODING"}.get(stage, "TIMING")
        path = save_checkpoint(self.settings.data_dir, self._episode_id,
                               self._source_sha256, self._config_fingerprint,
                               completed_stages=completed, next_stage=next_stage,
                               artifacts=artifacts, results=results,
                               media=self._checkpoint.get("media"),
                               stage_inputs=self._checkpoint.get("stage_inputs", {}))
        self._checkpoint.update({"schema_version": 2, "episode_id": self._episode_id,
                                 "source_sha256": self._source_sha256,
                                 "config_fingerprint": self._config_fingerprint,
                                 "next_stage": next_stage})
        self._register_artifact(self._episode_id, "checkpoint", path)

    def _run_stage(self, stage: str, source: Path, config: dict) -> dict:
        results = self._checkpoint.setdefault("results", {})
        inputs = self._checkpoint.setdefault("stage_inputs", {})
        input_hash = fingerprint(config)
        if (stage in self._checkpoint.get("completed_stages", [])
                and inputs.get(stage) == input_hash):
            result = results.get(stage)
            if not isinstance(result, dict):
                raise CheckpointError("Checkpoint stage result is missing")
            return result
        # User edits to text/action after a review invalidate this stage and its
        # dependants, while retaining earlier source-derived work.
        order = STAGE_ORDER
        if stage in order:
            start = order.index(stage)
            invalidated = set(order[start:])
            self._checkpoint["completed_stages"] = [s for s in self._checkpoint.get("completed_stages", [])
                                                       if s not in invalidated]
            for item in invalidated:
                results.pop(item, None)
                inputs.pop(item, None)
            self._checkpoint["artifacts"] = {
                key: value for key, value in self._checkpoint.get("artifacts", {}).items()
                if key.split(":", 1)[0] not in invalidated
            }
        if self._should_stop():
            raise _PipelineStopped()
        self._active_stage = stage
        result = self.runner.run(stage, source, config)
        if stage == "TRANSLATING":
            translation_path = Path(result["artifacts"][0])
            translation_data = json.loads(translation_path.read_text(encoding="utf-8"))
            translated_segments = [Segment.model_validate(item)
                                  for item in translation_data["segments"]]
            self._save_segments_to_db(self._episode_id, translated_segments)
        inputs[stage] = input_hash
        self._checkpoint_stage(stage, result)
        self._active_stage = None
        return result

    def _save_boundary(self, completed_stage: str, next_stage: str | None,
                       paths: list[Path] | None = None, details: dict | None = None) -> None:
        artifacts = self._checkpoint.setdefault("artifacts", {})
        for path in paths or []:
            artifacts[f"{completed_stage}:{path.name}"] = artifact_record(self.settings.data_dir, path)
        stages = self._checkpoint.setdefault("completed_stages", [])
        if completed_stage not in stages:
            stages.append(completed_stage)
        if details:
            self._checkpoint.setdefault("results", {})[completed_stage] = details
        path = save_checkpoint(self.settings.data_dir, self._episode_id, self._source_sha256,
                               self._config_fingerprint, completed_stages=stages,
                               next_stage=next_stage, artifacts=artifacts,
                               results=self._checkpoint.setdefault("results", {}),
                               media=self._checkpoint.get("media"),
                               stage_inputs=self._checkpoint.get("stage_inputs", {}))
        self._checkpoint["next_stage"] = next_stage
        self._register_artifact(self._episode_id, "checkpoint", path)

    def _passthrough_audio(self, ep_id: str, episode: dict, source: Path,
                           work_dir: Path, duration_ms: int) -> bool:
        """Keep the original audio when no segment is marked for dubbing."""
        if self._should_stop():
            raise _PipelineStopped()
        preview = work_dir / "preview.wav"
        self.db.transition(ep_id, "AUDIO_MIX", "Không có câu cần lồng tiếng; giữ nguyên âm thanh gốc",
                           progress=0.82, next_stage="QC")
        self._command([self.settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error",
                       "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "48000",
                       "-c:a", "pcm_s16le", "-y", str(preview)], 180)
        self._register_artifact(ep_id, "preview_audio", preview)
        passthrough_report = {"passthrough": True, "source_sha256": sha256_file(source),
                              "technical_quality_pass": True, "coverage_complete": True}
        self._save_boundary("AUDIO_MIX", "QC", [preview], passthrough_report)
        return self._qc_and_encode(ep_id, source, preview, duration_ms, passthrough_report)

    def _qc_and_encode(self, ep_id: str, source: Path, preview_wav: Path,
                       duration_ms: int, mix_report: dict | None = None) -> bool:
        if self._should_stop():
            raise _PipelineStopped()
        self.db.transition(ep_id, "QC", "Kiểm tra kỹ thuật âm thanh trước khi xuất video",
                           progress=0.86, next_stage="ENCODING")
        probe = self._command([self.settings.ffprobe_bin, "-v", "error", "-show_streams",
                               "-show_format", "-of", "json", str(preview_wav)], 60, text=True)
        report = json.loads(probe.stdout)
        audio = next((s for s in report.get("streams", []) if s.get("codec_type") == "audio"), None)
        fmt = report.get("format", {}).get("format_name", "")
        actual_duration = float((audio or {}).get("duration") or report.get("format", {}).get("duration") or 0)
        technical = bool(audio and "wav" in fmt and str(audio.get("codec_name", "")).startswith("pcm")
                         and int(audio.get("sample_rate", 0)) == 48000
                         and int(audio.get("channels", 0)) == 2
                         and abs(actual_duration * 1000 - duration_ms) <= 250)
        loudness = self._command([self.settings.ffmpeg_bin, "-hide_banner", "-i", str(preview_wav),
                                  "-af", "volumedetect", "-f", "null", "-"],
                                 120, text=True, check=False)
        max_volume = None
        for line in loudness.stderr.splitlines():
            if "max_volume:" in line:
                try:
                    max_volume = float(line.split("max_volume:", 1)[1].strip().split()[0])
                except ValueError:
                    pass
        loudness_valid = max_volume is not None and (max_volume == float("-inf")
                                                       or math.isfinite(max_volume) and max_volume < 0.0)
        report_peak = max_volume if max_volume is not None and math.isfinite(max_volume) else None
        technical = technical and loudness_valid
        if mix_report and not mix_report.get("passthrough"):
            source_hash = sha256_file(source)
            rows = self.db.rows("SELECT contract_json FROM segment WHERE episode_id=? AND action='DUB' ORDER BY start_ms, id",
                                (ep_id,))
            expected_words = sorted((word["s"], word["e"]) for row in rows
                                    for word in json.loads(row["contract_json"]).get("words", []))
            expected_slots = [[int(json.loads(row["contract_json"])["start_ms"]),
                               int(json.loads(row["contract_json"])["end_ms"])] for row in rows]
            mix_quality = mix_report.get("qc", {})
            coverage_complete = (mix_report.get("source_sha256") == source_hash
                                 and mix_report.get("technical_quality_pass") is True
                                 and mix_report.get("modified_ranges_ms") == [list(span) for span in expected_words]
                                 and mix_report.get("voice_added_ranges_ms") == expected_slots
                                 and mix_report.get("outside_mask_preserved") is True
                                 and mix_quality.get("technical_quality_pass") is True)
            technical = technical and coverage_complete
        else:
            coverage_complete = bool(mix_report and mix_report.get("coverage_complete"))
            technical = technical and coverage_complete
        qc = {"passed": technical, "sample_rate": int((audio or {}).get("sample_rate", 0) or 0),
              "channels": int((audio or {}).get("channels", 0) or 0),
              "duration_ms": round(actual_duration * 1000), "max_volume_dbfs": report_peak,
              "mix_report": mix_report or {}, "coverage_complete": coverage_complete,
              "technical_quality_pass": technical, "semantic_listening_verified": False}
        qc_path = preview_wav.with_name("audio_qc.json")
        qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
        self._register_artifact(ep_id, "audio_qc", qc_path)
        render_mode = self._subtitle_mode
        render_next = "VISION_RENDER" if render_mode in {"burn", "replace"} else "ENCODING"
        self._save_boundary("QC", render_next, [qc_path], qc)
        if not technical:
            return self._fail(ep_id, "AUDIO_QC_FAILED", "Mixed audio failed technical quality checks")
        if self._should_stop():
            raise _PipelineStopped()
        output_dir = self.settings.data_dir / "outputs" / ep_id
        output_dir.mkdir(parents=True, exist_ok=True)
        final_video = output_dir / "dubbed_final.mp4"
        mode = self._subtitle_mode
        if mode in {"burn", "replace"}:
            episode = self.db.one("SELECT * FROM episode WHERE id=?", (ep_id,))
            series = self.db.one("SELECT roi_json FROM series WHERE id=?", (episode["series_id"],))
            roi = None
            if mode == "replace":
                for value in (episode.get("roi_json"), series.get("roi_json") if series else None):
                    if value:
                        try:
                            roi = Roi.model_validate(json.loads(value))
                            break
                        except (ValueError, TypeError, json.JSONDecodeError):
                            continue
                if roi is None:
                    with self.db.lock:
                        self.db.execute("INSERT INTO issue VALUES(?,?,?,?,?,?,?)",
                                        (uuid.uuid4().hex, ep_id, None, "ROI_REQUIRED", "warning",
                                         json.dumps({"message": "Choose the subtitle region before replacing on-screen text"}), 0))
                        self.db.transition(ep_id, "NEEDS_REVIEW",
                                           "Subtitle replacement needs an approved ROI",
                                           next_stage="VISION_RENDER", queue_requested=0)
                    return False
            renderer_dir = self.settings.data_dir / "work" / ep_id / "vision_render"
            target = renderer_dir / "rendered.mp4"
            segments = [json.loads(row["contract_json"]) for row in self.db.rows(
                "SELECT contract_json FROM segment WHERE episode_id=? ORDER BY start_ms, id", (ep_id,))]
            transition = "TEXT_REMOVAL" if mode == "replace" else "SUBTITLE_RENDER"
            self.db.transition(ep_id, transition,
                               "Đang xử lý phụ đề và hình ảnh" if mode == "replace"
                               else "Đang ghi phụ đề lên video",
                               progress=0.89, next_stage="ENCODING")
            render_config = {"output_dir": str(renderer_dir), "preview_audio": str(preview_wav),
                             "target_output": str(target), "subtitle_mode": mode,
                             "segments": segments, "roi": roi.model_dump() if roi else None,
                             "ffmpeg_bin": self.settings.ffmpeg_bin,
                             "ffprobe_bin": self.settings.ffprobe_bin,
                             "timeout_seconds": 900}
            render_result = self._run_stage("VISION_RENDER", source, render_config)
            artifacts = render_result.get("artifacts", [])
            video_artifact = next((Path(path) for path in artifacts
                                   if Path(path).suffix.lower() == ".mp4" and Path(path).is_file()), None)
            if render_result.get("review_required") or render_result.get("action") == "NEEDS_REVIEW" or video_artifact is None:
                with self.db.lock:
                    self.db.execute("INSERT INTO issue VALUES(?,?,?,?,?,?,?)",
                                    (uuid.uuid4().hex, ep_id, None, "VISION_RENDER_REQUIRED", "warning",
                                     json.dumps({"message": "Subtitle rendering needs review",
                                                 "reason": render_result.get("failure_code")}), 0))
                    self.db.transition(ep_id, "NEEDS_REVIEW", "Subtitle rendering needs review",
                                       next_stage="VISION_RENDER", queue_requested=0)
                return False
            temp_final = final_video.with_suffix(".partial.mp4")
            shutil.copy2(video_artifact, temp_final)
            temp_final.replace(final_video)
            self.db.transition(ep_id, "ENCODING", "Video rendering complete; saving final output",
                               progress=0.96, next_stage="COMPLETED")
        else:
            if self._should_stop():
                raise _PipelineStopped()
            self.db.transition(ep_id, "ENCODING", "Đang xuất video hoàn chỉnh (FFmpeg)",
                               progress=0.90, next_stage="COMPLETED")
            temp_video = output_dir / "dubbed_final.partial.mp4"
            cmd = [self.settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error",
                   "-i", str(source), "-i", str(preview_wav), "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                   "-movflags", "+faststart", "-y", str(temp_video)]
            self._command(cmd, 300)
            temp_video.replace(final_video)
        self._register_artifact(ep_id, "final_video", final_video)
        self._save_boundary("ENCODING", None, [final_video], {"completed": True})
        with self.db.lock:
            self.db.transition(ep_id, "COMPLETED", "Hoàn tất lồng tiếng video!",
                               progress=1.0, next_stage=None, queue_requested=0)
            self.db.event(ep_id, "COMPLETED", "Pipeline completed; technical audio QC passed")
        return True

    def _runner_config(self) -> dict:
        v = self.settings.venvs_dir
        model_manifests = {}
        if self.settings.models_dir.is_dir():
            for folder in sorted(self.settings.models_dir.iterdir(), key=lambda item: item.name):
                manifest = folder / "model-manifest.json"
                if folder.is_dir() and manifest.is_file():
                    # Model manifests are small provenance files; never hash the large weights here.
                    model_manifests[folder.name] = sha256_file(manifest)
        return {
            "models_root": str(self.settings.models_dir),
            "model_manifest_sha256": model_manifests,
            "cache_root": str(self.settings.data_dir / "cache"),
            "interpreters": {
                "asr": str(v / "asr" / "bin" / "python"),
                "translation": str(v / "translation" / "bin" / "python") if (v / "translation" / "bin" / "python").exists() else None,
                "bandit": str(v / "separation" / "bin" / "python"),
                "separation": str(v / "separation" / "bin" / "python"),
                "tts": str(v / "tts" / "bin" / "python"),
                "vision": str(v / "vision" / "bin" / "python"),
            },
            "voice_id": self.settings.voice_id,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "ffmpeg_bin": self.settings.ffmpeg_bin,
            "ffprobe_bin": self.settings.ffprobe_bin,
            "timeout_seconds": 1800,
        }

    def _subtitle_mode_for_episode(self, episode: dict) -> str:
        try:
            flags = json.loads(episode.get("flags_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            flags = []
        if isinstance(flags, list):
            for flag in flags:
                if isinstance(flag, str) and flag.startswith("SUBTITLE_MODE="):
                    mode = flag.partition("=")[2]
                    if mode in {"off", "burn", "replace"}:
                        return mode
        return self.settings.subtitle_mode

    def _register_artifact(self, episode_id: str, kind: str, path: Path) -> None:
        path = path.resolve()
        rel_path = path.relative_to(self.settings.data_dir.resolve()).as_posix()
        with self.db.lock:
            self.db.execute("DELETE FROM artifact WHERE episode_id=? AND kind=?", (episode_id, kind))
            self.db.execute(
                "INSERT INTO artifact VALUES(?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, episode_id, kind, rel_path, sha256_file(path), path.stat().st_size, now_ms()),
            )

    def _save_segments_to_db(self, episode_id: str, segments: list[Segment]) -> None:
        with self.db.lock:
            self.db.execute("DELETE FROM segment WHERE episode_id=?", (episode_id,))
            for s in segments:
                self.db.execute(
                    "INSERT INTO segment VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        f"{episode_id}:{s.id}", episode_id, s.start_ms, s.end_ms, s.zh_text,
                        s.subtitle_vi, s.dub_vi, None, s.action,
                        "PENDING",
                        json.dumps(s.model_dump(), ensure_ascii=False),
                    ),
                )

    def _recognition_audio(self, separation_path: Path, duration_ms: int) -> Path:
        """Require an uncropped, zero-offset vocal stem on the source timeline."""
        metadata = json.loads(separation_path.read_text(encoding="utf-8"))
        windows = metadata.get("windows", [])
        if (metadata.get("schema_version") != 1 or len(windows) != 1
                or windows[0].get("start_ms") != 0
                or windows[0].get("end_ms") != duration_ms):
            raise ValueError("Recognition requires full-length separated vocals")
        window = windows[0]
        dialogue = window.get("dialogue")
        if not isinstance(dialogue, str) or dialogue not in window.get("stems", []):
            raise ValueError("Separated vocal stem is missing")
        root = separation_path.parent.resolve()
        vocals = (root / dialogue).resolve(strict=True)
        if not vocals.is_relative_to(root) or not vocals.is_file():
            raise ValueError("Separated vocal stem path is invalid")
        audio = probe_audio(vocals, self.settings.ffprobe_bin)
        if abs(audio["duration_ms"] - duration_ms) > 250:
            raise ValueError("Separated vocal stem changed the source timeline")
        return vocals

    def run(self, episode: dict, metadata: dict, should_stop=None) -> bool:
        ep_id = episode["id"]
        source = safe_path(self.settings.data_dir, episode["source_path"])
        if sha256_file(source) != episode.get("source_sha256"):
            return self._fail(ep_id, "SOURCE_CHECKSUM_MISMATCH", "Source checksum mismatch; re-upload the original video")
        duration_ms = metadata.get("duration_ms", episode.get("duration_ms", 1000))
        work_dir = self.settings.data_dir / "work" / ep_id
        work_dir.mkdir(parents=True, exist_ok=True)
        runner_config = self._runner_config()
        self.runner = ModelRunner(runner_config)
        self._should_stop = should_stop or self.stop_requested.is_set
        self._episode_id = ep_id
        self._source_sha256 = episode.get("source_sha256") or sha256_file(source)
        self._subtitle_mode = self._subtitle_mode_for_episode(episode)
        self._config_fingerprint = fingerprint({"runner": runner_config, "duration_ms": duration_ms,
                                                 "subtitle_mode": self._subtitle_mode})

        try:
            self._checkpoint = read_checkpoint(self.settings.data_dir, ep_id, self._source_sha256,
                                               self._config_fingerprint)
            self._checkpoint.setdefault("artifacts", {})
            self._checkpoint.setdefault("results", {})
            self._checkpoint.setdefault("completed_stages", [])
            self._checkpoint["media"] = metadata
            # ---------------------------------------------------------
            # Stage 1: SEPARATING (5% -> 15%). Both stems retain the full timeline.
            # Pin the provider; never silently run ASR on the original mixture.
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "SEPARATING", "Đang tách giọng khỏi nhạc nền trước ASR (Kim_Vocal_2)",
                progress=0.10, next_stage="ASR",
            )
            sep_res = self._run_stage("SEPARATING", source, {
                "output_dir": str(work_dir / "separating"),
                "separation_provider": "roformer",
                "separation_policy_revision": 2,
                "windows": [{"start_ms": 0, "end_ms": duration_ms}],
            })
            sep_path = Path(sep_res["artifacts"][0])
            self._register_artifact(ep_id, "separation", sep_path)
            recognition_audio = self._recognition_audio(sep_path, duration_ms)
            recognition_hash = sha256_file(recognition_audio)

            # ---------------------------------------------------------
            # Stage 2: ASR (15% -> 25%)
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "ASR", "Đang nhận diện giọng nói và phát hiện ngôn ngữ (Faster-Whisper)",
                progress=0.20, next_stage="ALIGNING",
            )
            asr_res = self._run_stage("ASR", recognition_audio, {
                "output_dir": str(work_dir / "asr"),
                "recognition_audio_sha256": recognition_hash,
                "original_source_sha256": self._source_sha256,
                "audio_input_kind": "separated_vocals",
            })
            transcript_path = Path(asr_res["artifacts"][0])
            self._register_artifact(ep_id, "transcript", transcript_path)
            transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
            segments = [Segment.model_validate(s) for s in transcript_data["segments"]]
            if not segments:
                with self.db.lock:
                    self.db.transition(ep_id, "NEEDS_REVIEW", "Không nhận diện được lời thoại; cần kiểm tra nguồn âm thanh",
                                       next_stage="ASR", queue_requested=0)
                    self.db.event(ep_id, "NEEDS_REVIEW", "ASR returned no transcript segments",
                                  {"language": transcript_data.get("language")})
                return False

            # ---------------------------------------------------------
            # Stage 3: ALIGNING (25% -> 35%)
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "ALIGNING", "Đang căn chỉnh thời gian từng từ theo ngôn ngữ (WhisperX)",
                progress=0.30, next_stage="TRANSLATING",
            )
            from autodub.adapters.whisper import ALIGNMENT_POLICY_REVISION
            align_config = {
                "output_dir": str(work_dir / "align"),
                "transcript_path": str(transcript_path),
                "alignment_policy_revision": ALIGNMENT_POLICY_REVISION,
                "recognition_audio_sha256": recognition_hash,
                "transcript_sha256": sha256_file(transcript_path),
            }
            had_alignment_checkpoint = (
                "ALIGNING" in self._checkpoint.get("completed_stages", [])
                and self._checkpoint.get("stage_inputs", {}).get("ALIGNING") == fingerprint(align_config)
            )
            align_res = self._run_stage("ALIGNING", recognition_audio, align_config)
            align_path = Path(align_res["artifacts"][0])
            self._register_artifact(ep_id, "alignment", align_path)
            align_data = json.loads(align_path.read_text(encoding="utf-8"))
            # AutoDub default: mark segments as DUB so pipeline runs end-to-end
            segments = [
                Segment.model_validate({**s, "action": s.get("action", "DUB"),
                                        "needs_review": s.get("needs_review", False)})
                for s in align_data["segments"]
            ]
            if not had_alignment_checkpoint:
                self._save_segments_to_db(ep_id, segments)
            if align_data.get("alignment_status") == "UNSUPPORTED_LANGUAGE_NO_WORD_TIMES":
                required_asset = align_data.get("required_asset")
                with self.db.lock:
                    self.db.transition(ep_id, "NEEDS_REVIEW",
                                       f"Thiếu model căn chỉnh thời gian cho ngôn ngữ {align_data.get('language') or 'không xác định'}",
                                       next_stage="ALIGNING", queue_requested=0)
                    self.db.event(ep_id, "NEEDS_REVIEW", "Pinned language aligner is unavailable",
                                  {"language": align_data.get("language"), "required_asset": required_asset})
                return False

            # ---------------------------------------------------------
            # Stage 4: TRANSLATING (35% -> 65%)
            # ---------------------------------------------------------
            from autodub.adapters.local_translation import TRANSLATION_POLICY_REVISION, find_deepseek_api_key
            trans_label = "DeepSeek-V3 API" if find_deepseek_api_key() else "Qwen3.5-4B Offline"
            self.db.transition(
                ep_id, "TRANSLATING", f"Đang dịch tiếng Trung sang tiếng Việt ({trans_label})",
                progress=0.50, next_stage="TTS",
            )
            glossary = {
                row["zh"]: row["vi"]
                for row in self.db.rows("SELECT zh,vi FROM glossary_entry WHERE series_id=?", (episode["series_id"],))
            }
            context = {
                s.id: " / ".join(n.zh_text for n in (segments[max(0, i - 3):i] + segments[i + 1:i + 4]))
                for i, s in enumerate(segments)
            }
            had_translation_checkpoint = "TRANSLATING" in self._checkpoint.get("completed_stages", [])
            trans_res = self._run_stage("TRANSLATING", source, {
                "output_dir": str(work_dir / "translation"),
                "segments": [s.model_dump() for s in segments],
                "glossary": glossary,
                "nearby_context": context,
                "translation_policy_revision": TRANSLATION_POLICY_REVISION,
            })
            trans_path = Path(trans_res["artifacts"][0])
            self._register_artifact(ep_id, "translation", trans_path)
            trans_data = json.loads(trans_path.read_text(encoding="utf-8"))
            segments = [Segment.model_validate(s) for s in trans_data["segments"]]
            # Segment rows are the user-editable review source of truth after a
            # translation checkpoint has been saved.
            loaded_review_rows = False
            if had_translation_checkpoint:
                rows = self.db.rows("SELECT contract_json FROM segment WHERE episode_id=? ORDER BY start_ms, id",
                                    (ep_id,))
                if rows:
                    try:
                        segments = [Segment.model_validate(json.loads(row["contract_json"])) for row in rows]
                        loaded_review_rows = True
                    except (ValueError, TypeError, json.JSONDecodeError):
                        raise CheckpointError("Saved reviewed segments are invalid") from None
            if not loaded_review_rows:
                self._save_segments_to_db(ep_id, segments)

            unresolved = [s for s in segments if s.needs_review or s.action == "NEEDS_REVIEW"
                          or (s.action == "DUB" and (not s.zh_text.strip() or not s.dub_vi.strip()))]
            if unresolved:
                with self.db.lock:
                    self.db.transition(ep_id, "NEEDS_REVIEW", "Review unresolved transcript or translation lines",
                                       next_stage="TTS", queue_requested=0)
                    self.db.event(ep_id, "NEEDS_REVIEW", "Automatic review gate held unresolved lines",
                                  {"segment_count": len(unresolved)})
                return False

            dub_segments = [s for s in segments if s.action == "DUB" and s.dub_vi.strip()]
            if not dub_segments:
                return self._passthrough_audio(ep_id, episode, source, work_dir, duration_ms)

            # ---------------------------------------------------------
            # Stage 5: TTS (65% -> 75%)
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "TTS", "Đang sinh giọng lồng tiếng Việt (VieNeu-TTS-v3)",
                progress=0.70, next_stage="TIMING",
            )
            tts_res = self._run_stage("TTS", source, {
                "output_dir": str(work_dir / "tts"),
                "segments": [s.model_dump() for s in dub_segments],
                "voice_id": self.settings.voice_id,
            })
            clips = tts_res.get("clips", {})
            self._register_artifact(ep_id, "tts_clips", Path(tts_res["artifacts"][0]))

            # ---------------------------------------------------------
            # Stage 6: TIMING & AUDIO_MIX (75% -> 85%)
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "TIMING", "Đang khớp thời lượng câu thoại (Pitch-Preserved Fitting)",
                progress=0.78, next_stage="AUDIO_MIX",
            )
            fitted_clips: dict[str, str] = {}
            fit_report: dict[str, dict] = {}
            timing_dir = work_dir / "timing"
            timing_dir.mkdir(parents=True, exist_ok=True)
            for seg in dub_segments:
                if self._should_stop():
                    raise _PipelineStopped()
                if seg.id in clips and Path(clips[seg.id]).exists():
                    clip_file = Path(clips[seg.id])
                    target_ms = seg.end_ms - seg.start_ms
                    fitted_file = timing_dir / f"{seg.id.replace(':', '_')}_fitted.wav"
                    fitted_info = fit_voice(clip_file, fitted_file, target_ms, self.settings.ffmpeg_bin,
                                            elastic=False)
                    fit_report[seg.id] = {key: fitted_info[key] for key in
                                          ("action", "review_required", "target_ms", "speed_factor",
                                           "failure_code") if key in fitted_info}
                    if not fitted_info["review_required"]:
                        fitted_clips[seg.id] = str(fitted_info["wav"])
            (timing_dir / "fit_report.json").write_text(
                json.dumps(fit_report, ensure_ascii=False, indent=1), encoding="utf-8")
            fit_report_path = timing_dir / "fit_report.json"
            flagged = [sid for sid, info in fit_report.items() if info["review_required"]]
            if flagged:
                self._save_boundary("TIMING", "TTS",
                                    [fit_report_path, *[Path(p) for p in fitted_clips.values()]],
                                    {"fit_report": fit_report})
                with self.db.lock:
                    for sid in flagged:
                        self.db.execute("INSERT INTO issue VALUES(?,?,?,?,?,?,?)",
                                        (uuid.uuid4().hex, ep_id, f"{ep_id}:{sid}",
                                         "DURATION_REWRITE_REQUIRED", "warning",
                                         json.dumps({"segment_id": sid,
                                                     "fit": fit_report[sid]}), 0))
                    self.db.transition(ep_id, "NEEDS_REVIEW",
                                       "Voice duration needs review before audio mixing",
                                       next_stage="TTS", queue_requested=0)
                    self.db.event(ep_id, "NEEDS_REVIEW", "Strict timing gate requires review",
                                  {"segment_ids": flagged})
                return False
            missing = [s.id for s in dub_segments if s.id not in fitted_clips]
            if missing:
                raise RuntimeError("A dubbed voice clip is missing; retry the TTS stage")
            self._save_boundary("TIMING", "AUDIO_MIX",
                                [fit_report_path, *[Path(p) for p in fitted_clips.values()]],
                                {"fit_report": fit_report})
            mix_segments = [s for s in dub_segments if s.id in fitted_clips]
            if not mix_segments:
                raise RuntimeError("No dubbed clips were produced")

            if self._should_stop():
                raise _PipelineStopped()
            self.db.transition(
                ep_id, "AUDIO_MIX", "Đang hòa trộn giọng lồng tiếng với BGM và SFX gốc",
                progress=0.82, next_stage="ENCODING",
            )
            preview_wav = work_dir / "preview.wav"
            # BandIt manifest lists stems; the mixer needs the speech stem as "dialogue".
            sep_meta = json.loads(sep_path.read_text(encoding="utf-8"))
            for win in sep_meta["windows"]:
                if "dialogue" not in win:
                    win["dialogue"] = next(p for p in win["stems"] if p.endswith("speech.wav"))
            mix_sep_path = sep_path.with_name("separation_mix.json")
            mix_sep_path.write_text(json.dumps(sep_meta), encoding="utf-8")
            # Keep aligned word boundaries so the mixer masks only source dialogue.
            mix_report = mix_preview(
                source, preview_wav,
                [s.model_dump() for s in mix_segments],
                mix_sep_path, {k: Path(v) for k, v in fitted_clips.items()},
                ffmpeg_bin=self.settings.ffmpeg_bin,
            )
            mix_report = json.loads(json.dumps(mix_report, default=str)) if isinstance(mix_report, dict) else {}
            self._register_artifact(ep_id, "preview_audio", preview_wav)
            self._save_boundary("AUDIO_MIX", "QC", [preview_wav],
                                {"mix_report": mix_report or {}, "semantic_listening_verified": False})

            # ---------------------------------------------------------
            # Stage 7: FINAL MUX & ENCODING (85% -> 100%)
            # ---------------------------------------------------------
            return self._qc_and_encode(ep_id, source, preview_wav, duration_ms,
                                       mix_report if isinstance(mix_report, dict) else {})

        except _PipelineStopped:
            stage = self._active_stage or self._checkpoint.get("next_stage") or "SEPARATING"
            with self.db.lock:
                current = self.db.one("SELECT status FROM episode WHERE id=?", (ep_id,))
                if current and current["status"] != "CHECKPOINTED":
                    self.db.transition(ep_id, "CHECKPOINTED", "Worker stopped at a durable stage boundary",
                                       next_stage=stage, queue_requested=1)
            return False
        except CheckpointError as error:
            logger.error("Pipeline checkpoint validation failed for episode %s", ep_id)
            if error.stage:
                try:
                    path = invalidate_from_stage(self.settings.data_dir, ep_id, self._source_sha256,
                                                 self._config_fingerprint, error.stage)
                    self._register_artifact(ep_id, "checkpoint", path)
                except (CheckpointError, OSError, ValueError):
                    pass
            return self._fail(ep_id, "CHECKPOINT_INVALID", "Saved pipeline checkpoint is invalid; retry from the last valid stage")
        except Exception:
            if self._should_stop():
                stage = self._active_stage or self._checkpoint.get("next_stage") or "SEPARATING"
                with self.db.lock:
                    current = self.db.one("SELECT status FROM episode WHERE id=?", (ep_id,))
                    if current and current["status"] != "CHECKPOINTED":
                        self.db.transition(ep_id, "CHECKPOINTED", "Model stage cancelled at shutdown",
                                           next_stage=stage, queue_requested=1)
                return False
            logger.error("Pipeline stage failed for episode %s", ep_id)
            return self._fail(ep_id, "PIPELINE_STAGE_FAILED", "Pipeline stage failed; retry this episode")
