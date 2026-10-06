"""End-to-end automated dubbing pipeline orchestrator."""
from __future__ import annotations

import json
import logging
import subprocess
import uuid
from pathlib import Path

from autodub.audio_mix import fit_voice, mix_preview
from autodub.config import Settings
from autodub.contracts import Segment
from autodub.domain import merge_windows
from autodub.model_runner import ModelRunner
from autodub.storage import Database, now_ms, safe_path, sha256_file

logger = logging.getLogger("autodub.pipeline")


class Pipeline:
    """Execute all automated dubbing stages for an episode."""

    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings

    def _runner_config(self) -> dict:
        v = self.settings.venvs_dir
        return {
            "models_root": str(self.settings.models_dir),
            "cache_root": str(self.settings.data_dir / "cache"),
            "interpreters": {
                "asr": str(v / "asr" / "bin" / "python") if (v / "asr" / "bin" / "python").exists() else None,
                "translation": str(v / "translation" / "bin" / "python") if (v / "translation" / "bin" / "python").exists() else None,
                "bandit": str(v / "bandit" / "bin" / "python") if (v / "bandit" / "bin" / "python").exists() else None,
                "tts": str(v / "tts" / "bin" / "python") if (v / "tts" / "bin" / "python").exists() else None,
                "vision": str(v / "vision" / "bin" / "python") if (v / "vision" / "bin" / "python").exists() else None,
            },
            "voice_id": "Trúc Ly",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "ffmpeg_bin": self.settings.ffmpeg_bin,
            "ffprobe_bin": self.settings.ffprobe_bin,
            "timeout_seconds": 1800,
        }

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
                        s.id, episode_id, s.start_ms, s.end_ms, s.zh_text,
                        s.subtitle_vi, s.dub_vi, 1.0, s.action, "APPROVED" if s.action == "DUB" else "PENDING",
                        json.dumps(s.model_dump(), ensure_ascii=False),
                    ),
                )

    def run(self, episode: dict, metadata: dict) -> bool:
        ep_id = episode["id"]
        source = safe_path(self.settings.data_dir, episode["source_path"])
        duration_ms = metadata.get("duration_ms", episode.get("duration_ms", 1000))
        work_dir = self.settings.data_dir / "work" / ep_id
        work_dir.mkdir(parents=True, exist_ok=True)
        runner = ModelRunner(self._runner_config())

        try:
            # ---------------------------------------------------------
            # Stage 1: ASR (5% -> 15%)
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "ASR", "Đang nhận diện giọng nói tiếng Trung (Qwen3-ASR)",
                progress=0.10, next_stage="ALIGNING",
            )
            asr_res = runner.run("ASR", source, {"output_dir": str(work_dir / "asr")})
            transcript_path = Path(asr_res["artifacts"][0])
            self._register_artifact(ep_id, "transcript", transcript_path)
            transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
            segments = [Segment.model_validate(s) for s in transcript_data["segments"]]

            # ---------------------------------------------------------
            # Stage 2: ALIGNING (15% -> 25%)
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "ALIGNING", "Đang căn chỉnh thời gian từng từ (Qwen3-ForcedAligner)",
                progress=0.20, next_stage="TRANSLATING",
            )
            align_res = runner.run("ALIGNING", source, {
                "output_dir": str(work_dir / "align"),
                "transcript_path": str(transcript_path),
            })
            align_path = Path(align_res["artifacts"][0])
            self._register_artifact(ep_id, "alignment", align_path)
            align_data = json.loads(align_path.read_text(encoding="utf-8"))
            # AutoDub default: mark segments as DUB so pipeline runs end-to-end
            segments = [
                Segment.model_validate({**s, "action": "DUB", "needs_review": False})
                for s in align_data["segments"]
            ]
            self._save_segments_to_db(ep_id, segments)

            # ---------------------------------------------------------
            # Stage 3: TRANSLATING (25% -> 45%)
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "TRANSLATING", "Đang dịch tiếng Trung sang tiếng Việt (Qwen3.5-4B Offline)",
                progress=0.35, next_stage="SEPARATING",
            )
            glossary = {
                row["zh"]: row["vi"]
                for row in self.db.rows("SELECT zh,vi FROM glossary_entry WHERE series_id=?", (episode["series_id"],))
            }
            trans_res = runner.run("TRANSLATING", source, {
                "output_dir": str(work_dir / "translation"),
                "segments": [s.model_dump() for s in segments],
                "glossary": glossary,
            })
            trans_path = Path(trans_res["artifacts"][0])
            self._register_artifact(ep_id, "translation", trans_path)
            trans_data = json.loads(trans_path.read_text(encoding="utf-8"))
            segments = [Segment.model_validate(s) for s in trans_data["segments"]]
            self._save_segments_to_db(ep_id, segments)

            # ---------------------------------------------------------
            # Stage 4: SEPARATING (45% -> 65%)
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "SEPARATING", "Đang tách 3 track âm thanh: Thoại, BGM, SFX (BandIt ERB48)",
                progress=0.55, next_stage="TTS",
            )
            dub_segments = [s for s in segments if s.action == "DUB" and s.dub_vi.strip()]
            if not dub_segments:
                dub_segments = segments  # fallback if all text was blank

            spans = []
            for s in dub_segments:
                if s.words:
                    spans.extend((w.s, w.e) for w in s.words)
                else:
                    spans.append((s.start_ms, s.end_ms))

            if spans:
                raw_windows = merge_windows(spans, duration_ms)
            else:
                raw_windows = [(0, min(duration_ms, 30000))]

            # Pad all windows to at least 30s to satisfy BandIt fader requirement
            windows = []
            for w_start, w_end in raw_windows:
                if w_end - w_start < 30000:
                    pad = 30000 - (w_end - w_start)
                    w_start = max(0, w_start - pad // 2)
                    w_end = min(duration_ms, w_start + 30000)
                    if w_end - w_start < 30000 and w_start > 0:
                        w_start = max(0, w_end - 30000)
                windows.append({"start_ms": w_start, "end_ms": w_end})

            sep_res = runner.run("SEPARATING", source, {
                "output_dir": str(work_dir / "separating"),
                "windows": windows,
            })
            sep_path = Path(sep_res["artifacts"][0])
            self._register_artifact(ep_id, "separation", sep_path)

            # ---------------------------------------------------------
            # Stage 5: TTS (65% -> 75%)
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "TTS", "Đang sinh giọng lồng tiếng tiếng Việt (VieNeu-TTS-v3)",
                progress=0.70, next_stage="TIMING",
            )
            tts_res = runner.run("TTS", source, {
                "output_dir": str(work_dir / "tts"),
                "segments": [s.model_dump() for s in dub_segments],
            })
            clips = tts_res.get("clips", {})
            self._register_artifact(ep_id, "tts_clips", Path(tts_res["artifacts"][0]))

            # ---------------------------------------------------------
            # Stage 6: TIMING & AUDIO_MIX (75% -> 85%)
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "TIMING", "Đang khớp thời lượng khẩu hình (Pitch-Preserved Fitting)",
                progress=0.78, next_stage="AUDIO_MIX",
            )
            fitted_clips: dict[str, str] = {}
            timing_dir = work_dir / "timing"
            timing_dir.mkdir(parents=True, exist_ok=True)
            for seg in dub_segments:
                if seg.id in clips and Path(clips[seg.id]).exists():
                    clip_file = Path(clips[seg.id])
                    target_ms = seg.end_ms - seg.start_ms
                    fitted_file = timing_dir / f"{seg.id.replace(':', '_')}_fitted.wav"
                    fitted_info = fit_voice(clip_file, fitted_file, target_ms, self.settings.ffmpeg_bin)
                    fitted_clips[seg.id] = str(fitted_info["wav"])

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
            mix_preview(
                source, preview_wav, [s.model_dump() for s in dub_segments],
                mix_sep_path, {k: Path(v) for k, v in fitted_clips.items()},
                ffmpeg_bin=self.settings.ffmpeg_bin,
            )
            self._register_artifact(ep_id, "preview_audio", preview_wav)

            # ---------------------------------------------------------
            # Stage 7: FINAL MUX & ENCODING (85% -> 100%)
            # ---------------------------------------------------------
            self.db.transition(
                ep_id, "ENCODING", "Đang xuất video hoàn chỉnh (FFmpeg NVENC Mux)",
                progress=0.90, next_stage="COMPLETED",
            )
            output_dir = self.settings.data_dir / "outputs" / ep_id
            output_dir.mkdir(parents=True, exist_ok=True)
            final_video = output_dir / "dubbed_final.mp4"
            temp_video = output_dir / "dubbed_final.partial.mp4"

            # FFmpeg Mux: Combine video with mixed Vietnamese audio
            cmd = [
                self.settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error",
                "-i", str(source), "-i", str(preview_wav),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", "-y", str(temp_video),
            ]
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            temp_video.replace(final_video)
            self._register_artifact(ep_id, "final_video", final_video)

            # Complete!
            with self.db.lock:
                self.db.transition(
                    ep_id, "COMPLETED", "Hoàn tất lồng tiếng video!",
                    progress=1.0, next_stage=None, queue_requested=0,
                )
                self.db.event(ep_id, "COMPLETED", "Full AutoDub pipeline completed successfully")
            return True

        except Exception as error:
            logger.exception("Pipeline failed for episode %s: %s", ep_id, error)
            with self.db.lock:
                self.db.execute(
                    "INSERT INTO issue VALUES(?,?,?,?,?,?,?)",
                    (uuid.uuid4().hex, ep_id, None, "PIPELINE_ERROR", "error",
                     json.dumps({"message": str(error)}), 0),
                )
                self.db.transition(ep_id, "FAILED", f"Lỗi pipeline: {error}", queue_requested=0)
            return False
