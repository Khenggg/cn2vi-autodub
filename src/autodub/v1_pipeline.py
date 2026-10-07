"""Frozen V1 execution with honest partial exports and durable stage boundaries."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from autodub.character_context import build_context, update_memory
from autodub.checkpoint import fingerprint
from autodub.experimental_run import ExperimentalRun
from autodub.model_runner import ModelRunner
from autodub.storage import atomic_json, sha256_file


class RunDrained(Exception):
    """The current stage is durable; an explicit resume may continue this snapshot."""


def read_result(result: dict | None) -> dict:
    if not result:
        return {}
    for path in result.get("artifacts", []):
        if Path(path).suffix == ".json":
            return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


def _srt_time(ms: int) -> str:
    seconds, milli = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milli:03d}"


def write_subtitles(path: Path, segments: list[dict]) -> None:
    lines = []
    for segment in segments:
        text = segment.get("subtitle_vi", "").strip()
        if text:
            lines.extend([str(len(lines) // 4 + 1),
                          f"{_srt_time(segment['start_ms'])} --> {_srt_time(segment['end_ms'])}",
                          text.replace("\r", " ").replace("\n", " "), ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class V1Pipeline:
    def __init__(self, source: Path, root: Path, config: dict, repo: Path,
                 *, notify=lambda stage, result: None, should_stop=lambda: False):
        self.source = source.resolve(strict=True)
        self.config = json.loads(json.dumps(config))
        self.repo = repo
        self.notify, self.should_stop = notify, should_stop
        self.pointer = root / "active-run.json"
        restored = None
        if self.pointer.is_file():
            run_id = json.loads(self.pointer.read_text())["run_id"]
            if Path(run_id).name != run_id:
                raise ValueError("Invalid run checkpoint identity")
            previous = root / run_id
            state = json.loads((previous / "stage-checkpoint.json").read_text())
            if state["state"] != "FINISHED":
                restored = state
                self.run = ExperimentalRun.resume(previous, source, self.config, repo=repo)
        if restored is None:
            self.run = ExperimentalRun(root, source, self.config, repo=repo,
                                       source_duration_ms=config["duration_ms"], cloud_rate=config.get("cloud_rate", 6000))
        self.runner = ModelRunner({**self.config, "package_root": str(self.run.package_root)})
        self.checkpoint = self.run.root / "stage-checkpoint.json"
        self.completed = restored["completed"] if restored else {}
        self.best_video = Path(restored["best_video"]) if restored else source
        self.frozen_reference = self.run.root / "frozen-inputs" / "ngoc-huyen.wav"
        reference_value = self.config.get("voice_reference")
        if reference_value and Path(reference_value).is_file() and not self.frozen_reference.exists():
            self.frozen_reference.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(reference_value, self.frozen_reference)
        if self.frozen_reference.is_file() and sha256_file(self.frozen_reference) != self.config.get("voice_reference_sha256"):
            raise ValueError("Frozen voice reference differs from the requested fixed voice")
        atomic_json(self.checkpoint, {"schema_version": 1, "run_id": self.run.id,
            "completed": self.completed, "state": "RUNNING", "best_video": str(self.best_video)})
        atomic_json(self.pointer, {"schema_version": 1, "run_id": self.run.id})

    def stage(self, name: str, worker: str, source: Path, **kwargs) -> dict | None:
        if self.should_stop():
            atomic_json(self.checkpoint, {"schema_version": 1, "run_id": self.run.id,
                "config_sha256": fingerprint(self.config), "completed": self.completed,
                "state": "CHECKPOINTED", "best_video": str(self.best_video)})
            raise RunDrained()
        options = {**self.config, **kwargs, "output_dir": str(self.run.root / name.lower())}
        options["model_lock_path"] = str(self.run.root / "snapshot/benchmarks/models.lock.json")
        options["run_root"] = str(self.run.root)
        if self.frozen_reference.is_file():
            options["voice_reference"] = str(self.frozen_reference)
        saved = self.completed.get(name)
        if saved:
            if (saved["input_sha256"] != sha256_file(source)
                    or saved["config_sha256"] != fingerprint(options)
                    or any(sha256_file(Path(path)) != digest for path, digest in saved["artifact_hashes"].items())):
                raise ValueError("Frozen stage checkpoint changed; do not silently rerun a different version")
            return saved["result"]
        self.notify(name, None)
        result = self.run.execute(name, lambda: self.runner.run(worker, source, options))
        if result:
            self.completed[name] = {"result": result, "input_sha256": sha256_file(source),
                                    "config_sha256": fingerprint(options),
                                    "artifact_hashes": {p: sha256_file(Path(p)) for p in result["artifacts"]}}
        atomic_json(self.checkpoint, {"schema_version": 1, "run_id": self.run.id,
            "config_sha256": fingerprint(self.config), "completed": self.completed, "state": "RUNNING",
            "best_video": str(self.best_video)})
        self.notify(name, result)
        return result

    def encode(self, video: Path, audio: Path, target: Path, srt: Path | None = None) -> dict:
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [self.config.get("ffmpeg_bin", "ffmpeg"), "-hide_banner", "-loglevel", "error",
                   "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0?",
                   "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-ar", "48000", "-movflags", "+faststart"]
        if srt and srt.stat().st_size > 1:
            command.extend(["-vf", "subtitles=subtitles.srt"])
        command.extend(["-y", str(target)])
        subprocess.run(command, cwd=srt.parent if srt else target.parent,
                       check=True, capture_output=True, timeout=self.config.get("media_timeout_seconds", 7200))
        from autodub.media import probe_media
        info = probe_media(target, self.config.get("ffprobe_bin", "ffprobe"))
        drift = abs(info["duration_ms"] - self.config["duration_ms"])
        return {"video": str(target), "artifacts": [str(target)],
                "stage_status": "DEGRADED" if drift > 500 else "SUCCESS",
                "failure_code": "OUTPUT_DURATION_DRIFT", "quality_evidence": {"duration_drift_ms": drift}}

    def execute(self) -> dict:
        try:
            separation = self.stage("SEPARATION", "V1_SEPARATION", self.source)
            stems = read_result(separation).get("stems", {})
            speech = Path(stems["speech"]) if stems else None
            segments = []
            if speech:
                diarization = self.stage("DIARIZATION", "V1_DIARIZATION", speech)
                if diarization:
                    asr = self.stage("ASR", "V1_ASR", speech, diarization_path=diarization["artifacts"][0])
                    if asr:
                        punctuation = self.stage("PUNCTUATION", "V1_PUNCTUATION", speech,
                                                 transcript_path=asr["artifacts"][0])
                        if not punctuation:
                            self.run.issue("PUNCTUATION", "RAW_ASR_RETAINED_UNCHANGED",
                                           downstream_effect="Translation receives the original ASR text without punctuation")
                        segments = read_result(punctuation or asr).get("segments", [])
                    else:
                        self.run.skip("PUNCTUATION", "ASR artifact unavailable")
                else:
                    self.run.skip("ASR", "No speech/speaker windows")
                    self.run.skip("PUNCTUATION", "ASR skipped")
            else:
                for stage in ("DIARIZATION", "ASR", "PUNCTUATION"):
                    self.run.skip(stage, "Separated speech unavailable; no recognition model fallback")
            ocr = self.stage("OCR", "V1_OCR", self.source)
            ocr_data = read_result(ocr)
            context_path = self.run.root / "character-context.json"
            context = build_context(segments, ocr_data, self.config.get("series_memory", {}),
                                    self.config.get("glossary", {}))
            atomic_json(context_path, context)
            self.run.execute("CONTEXT", lambda: {"artifacts": [str(context_path)],
                "stage_status": "DEGRADED" if any(s["uncertain"] for s in context["segments"].values()) else "SUCCESS",
                "failure_code": "CHARACTER_OR_ADDRESSEE_UNCONFIRMED"})
            translation = None
            if segments:
                translation = self.stage("TRANSLATION", "V1_TRANSLATION", self.source,
                                         segments=segments, character_context=context)
                if translation:
                    segments = translation["segments"]
            else:
                self.run.skip("TRANSLATION", "No recognized dialogue")
            clips = {}
            if speech and translation:
                tts = self.stage("TTS", "V1_TTS", speech, segments=segments)
                if tts:
                    clips = tts.get("clips", {})
            else:
                self.run.skip("TTS", "No translated dialogue or separated speech")
            audio = self.source
            if stems and clips:
                mixed = self.stage("AUDIO_MIX", "V1_MIX", self.source, stems=stems, segments=segments, clips=clips)
                if mixed:
                    audio = Path(mixed["audio"])
            else:
                self.run.skip("AUDIO_MIX", "Retain original audio; no usable dubbed clips")
            preview = self.run.execute("PREVIEW", lambda: self.encode(
                self.source, audio, self.run.root / "preview.mp4"))
            if preview:
                self.best_video = Path(preview["video"])
            video = self.source
            mode = self.config.get("subtitle_mode", "replace")
            layout = ocr_data.get("layout")
            if mode == "replace" and layout:
                self.run.execute("SUBTITLE_DETECTION", lambda: {"artifacts": ocr["artifacts"], "layout": layout})
                clean = self.stage("INPAINT", "V1_INPAINT", self.source, subtitle_layout=layout)
                if clean:
                    video = Path(clean["video"])
            else:
                self.run.skip("SUBTITLE_DETECTION", "Subtitle mode or layout does not permit removal")
                self.run.skip("INPAINT", "Retain original picture")
            subtitles = self.run.root / "subtitles.srt"
            write_subtitles(subtitles, segments if mode != "off" else [])
            self.run.execute("SUBTITLE_RENDER", lambda: {"artifacts": [str(subtitles)]})
            encoded = self.run.execute("FINAL_ENCODE", lambda: self.encode(
                video, audio, self.run.root / "final.mp4", subtitles))
            if encoded:
                self.best_video = Path(encoded["video"])
            if self.best_video == self.source:
                # A fatal encoder cannot erase the already playable input. This
                # is a partial original artifact, explicitly labelled in the report.
                self.run.issue("FINAL_ENCODE", "ORIGINAL_SOURCE_ONLY", downstream_effect="No rendered MP4 available")
            atomic_json(self.run.root / "series-memory.json", update_memory(
                self.config.get("series_memory", {}), segments, self.run.id))
            self.run.report["api_calls"] = translation.get("api_calls", []) if translation else []
            self.run.report["selected_models"] = json.loads((self.repo / "config/cloud-runtime.json").read_text())["assets"]
            report = self.run.finish(self.best_video)
            atomic_json(self.checkpoint, {"schema_version": 1, "run_id": self.run.id,
                "completed": self.completed, "state": "FINISHED", "best_video": str(self.best_video)})
            return report
        except RunDrained:
            self.run.report["status"] = "CHECKPOINTED"
            self.run._save()
            raise
