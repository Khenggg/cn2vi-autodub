"""Canonical V2 voice-over DAG: original soundtrack, event OCR, one final render."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from autodub.character_context import build_context, update_memory
from autodub.checkpoint import fingerprint
from autodub.model_runner import ModelRunner
from autodub.multimodal import merge_dialogue
from autodub.storage import atomic_json, sha256_file
from autodub.v1_pipeline import RunDrained, V1Pipeline, read_result, write_subtitles


class WorkerPool:
    """Serialize large GPU models; CUDA OCR has its own bounded arena lane."""
    GPU_STAGES = frozenset({"V2_ASR", "V2_TTS"})

    def __init__(self, config: dict):
        self.config = config
        self.gpu = threading.Lock()
        self.lock = threading.RLock()
        self.active = set()
        self.cancelled = threading.Event()

    def run(self, stage: str, source: Path, config: dict) -> dict:
        with self.gpu if stage in self.GPU_STAGES else nullcontext():
            if self.cancelled.is_set():
                raise RunDrained()
            runner = ModelRunner(self.config)
            with self.lock:
                self.active.add(runner)
            try:
                return runner.run(stage, source, config)
            finally:
                with self.lock:
                    self.active.discard(runner)

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            for runner in tuple(self.active):
                runner.cancel()


class V2Pipeline(V1Pipeline):
    """Reuse immutable snapshot/checkpoint machinery, never the V1 execution graph."""

    def __init__(self, source, root, config, repo, **kwargs):
        super().__init__(source, root, config, repo, **kwargs)
        self.runner = WorkerPool({**self.config, "package_root": str(self.run.package_root)})
        self._checkpoint_lock = threading.RLock()

    def stage(self, name: str, worker: str, source: Path, **kwargs):
        if self.should_stop():
            raise RunDrained()
        options = {**self.config, **kwargs, "output_dir": str(self.run.root / name.lower()),
                   "model_lock_path": str(self.run.root / "snapshot/benchmarks/models.lock.json"),
                   "run_root": str(self.run.root)}
        if self.frozen_reference.is_file():
            options["voice_reference"] = str(self.frozen_reference)
        saved = self.completed.get(name)
        if saved:
            if (saved["input_sha256"] != sha256_file(source) or saved["config_sha256"] != fingerprint(options)
                    or any(sha256_file(Path(path)) != digest for path, digest in saved["artifact_hashes"].items())):
                raise ValueError("V2 checkpoint changed; resume requires the frozen version")
            return saved["result"]
        self.notify(name, None)
        result = self.run.execute(name, lambda: self.runner.run(worker, source, options))
        with self._checkpoint_lock:
            if result:
                self.completed[name] = {"result": result, "input_sha256": sha256_file(source),
                    "config_sha256": fingerprint(options),
                    "artifact_hashes": {p: sha256_file(Path(p)) for p in result["artifacts"]}}
            self._checkpoint("RUNNING")
        self.notify(name, result)
        return result

    def _checkpoint(self, state):
        atomic_json(self.checkpoint, {"schema_version": 1, "run_id": self.run.id,
            "config_sha256": fingerprint(self.config), "completed": self.completed,
            "state": state, "best_video": str(self.best_video)})

    def _audio_branch(self):
        detected = self.stage("SPEECH_DETECTION", "V2_DETECT", self.source)
        if not detected:
            self.run.skip("ASR", "No vocal candidates artifact")
            return None
        audio = Path(detected["audio"])
        recognized = self.stage("ASR", "V2_ASR", audio, candidates_path=detected["artifacts"][0])
        if not recognized or not read_result(recognized).get("segments"):
            return recognized
        punctuated = self.stage("PUNCTUATION", "V2_PUNCTUATION", audio, transcript_path=recognized["artifacts"][0])
        return punctuated or recognized

    def _video_branch(self):
        ocr = self.stage("OCR", "V2_OCR", self.source)
        plan = None
        if ocr and self.config.get("subtitle_mode", "replace") == "replace":
            plan = self.stage("TEMPORAL_RESTORATION", "V2_RESTORE", self.source, ocr_path=ocr["artifacts"][0])
        else:
            self.run.skip("TEMPORAL_RESTORATION", "Subtitle mode or OCR evidence does not permit restoration")
        return ocr, plan

    def execute(self):
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                audio_task = executor.submit(self._audio_branch)
                video_task = executor.submit(self._video_branch)
                recognition = audio_task.result()
                ocr, restoration = video_task.result()
            recognized = read_result(recognition).get("segments", [])
            ocr_data = read_result(ocr)
            gate = merge_dialogue(recognized, ocr_data.get("events", []))
            approved = {s["id"]: s for s in self.config.get("approved_segments", [])}
            for index, segment in enumerate(gate["segments"]):
                if segment["id"] in approved:
                    gate["segments"][index] = dict(approved[segment["id"]])
                    gate["segments"][index]["context_provenance"] = {**gate["segments"][index].get("context_provenance", {}),
                                                                      "human_reviewed": True}
            gate["issues"] = [issue for issue in gate["issues"] if issue.get("segment_id") not in approved]
            gate_path = self.run.root / "dialogue-gate.json"
            atomic_json(gate_path, gate)
            self.run.execute("MULTIMODAL_GATE", lambda: {"artifacts": [str(gate_path)],
                "stage_status": "DEGRADED" if gate["issues"] else "SUCCESS", "quality_evidence": {"issues": gate["issues"]}})
            segments = gate["segments"]
            context = build_context(segments, ocr_data, self.config.get("series_memory", {}), self.config.get("glossary", {}))
            context_path = self.run.root / "character-context.json"
            atomic_json(context_path, context)
            self.run.execute("CONTEXT", lambda: {"artifacts": [str(context_path)]})
            translation = self.stage("TRANSLATION", "V2_TRANSLATION", self.source,
                                     segments=segments, character_context=context)
            if translation:
                segments = translation["segments"]
            tts = self.stage("TTS", "V2_TTS", self.source, segments=segments)
            clips = tts.get("clips", {}) if tts else {}
            audio = self.source
            mixed = None
            if clips:
                mixed = self.stage("AUDIO_MIX", "V2_MIX", self.source, segments=segments, clips=clips)
                if mixed:
                    audio = Path(mixed["audio"])
            else:
                self.run.skip("AUDIO_MIX", "No generated voice; original soundtrack retained")
            subtitles = self.run.root / "subtitles.srt"
            write_subtitles(subtitles, segments if self.config.get("subtitle_mode") != "off" else [])
            self.run.execute("SUBTITLE_RENDER", lambda: {"artifacts": [str(subtitles)]})
            final = self.stage("FINAL_ENCODE", "V2_ENCODE", self.source, audio_path=str(audio),
                               subtitles_path=str(subtitles), restoration_plan=restoration.get("plan") if restoration else None)
            if final:
                self.best_video = Path(final["video"])
            candidate = read_result(self.completed.get("SPEECH_DETECTION", {}).get("result"))
            expected = sum(s["action"] == "DUB" for s in gate["segments"])
            voiced = mixed.get("voice_added_count", 0) if mixed else 0
            if voiced < expected or not recognized and candidate.get("candidate_duration_ms", 0):
                self.run.issue("TTS", "VOICEOVER_COVERAGE_INCOMPLETE",
                    evidence={"expected_dialogue_count": expected, "voiced_count": voiced},
                    downstream_effect="Original soundtrack remains under missing voice-over")
            if self.best_video == self.source:
                self.run.issue("FINAL_ENCODE", "ORIGINAL_SOURCE_ONLY", downstream_effect="No rendered final artifact")
            atomic_json(self.run.root / "series-memory.json", {**update_memory(self.config.get("series_memory", {}), segments, self.run.id),
                        "subtitle_profile": ocr_data.get("subtitle_profile", {})})
            self.run.report.update(api_calls=translation.get("api_calls", []) if translation else [],
                selected_models=json.loads((self.run.root / "snapshot/config/cloud-runtime.json").read_text())["assets"],
                soundtrack_policy="MULTIBAND_VOICEOVER", separation_used=False, generative_video_model=None,
                candidate_duration_ms=candidate.get("candidate_duration_ms"),
                voiceover_coverage={"dialogue_count": expected, "generated_clips": len(clips), "voiced_count": voiced},
                sla_target_seconds=600, quality_and_sla_verified=False)
            report = self.run.finish(self.best_video)
            self._checkpoint("FINISHED")
            return report
        except RunDrained:
            self.run.report["status"] = "CHECKPOINTED"
            self.run._save()
            self._checkpoint("CHECKPOINTED")
            raise
