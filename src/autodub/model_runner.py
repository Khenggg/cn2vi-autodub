"""Run model stages in short-lived, isolated Python processes."""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

_INTERPRETER_STAGE = {
    "V1_SEPARATION": "separation", "V1_DIARIZATION": "diarization", "V1_ASR": "asr",
    "V1_PUNCTUATION": "punctuation", "V1_OCR": "vision", "V1_TRANSLATION": "core",
    "V1_TTS": "indextts", "V1_MIX": "separation", "V1_INPAINT": "vision",
    "ASR": "asr",
    "ALIGNING": "asr",
    "TRANSLATING": "translation",
    "SEPARATING": "bandit",
    "TTS": "tts",
    "PROPAINTER": "bandit",
    "VISION_RENDER": "vision",
}
_SECRET_KEY = re.compile(r"(?:secret|token|password|credential|api[_-]?key|authorization)", re.I)
_SAFE_ENV = (
    "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "DEEPSEEK_API_KEY",
    "FFMPEG_BIN", "FFPROBE_BIN",
)


class ModelRunnerError(RuntimeError):
    """A bounded, sanitized failure from a model subprocess."""

    def __init__(self, message: str, *, error_code: str = "ModelRunnerError"):
        super().__init__(message)
        self.error_code = error_code


def _public_config(value: Any) -> Any:
    """Copy JSON data while removing credential-like fields at every depth."""
    if isinstance(value, dict):
        return {key: _public_config(item) for key, item in value.items()
                if not _SECRET_KEY.search(str(key))}
    if isinstance(value, (list, tuple)):
        return [_public_config(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Model config must contain JSON-compatible values")


class ModelRunner:
    """Invoke ASR, alignment, separation, and TTS in an isolated interpreter."""

    def __init__(self, config: dict):
        self.config = dict(config)
        self._lock = threading.RLock()
        self._active: subprocess.Popen | None = None

    def run(self, stage: str, source: Path, config: dict) -> dict:
        if stage not in _INTERPRETER_STAGE:
            raise ValueError(f"Unsupported model stage: {stage}")
        interpreter_key = _INTERPRETER_STAGE[stage]
        interpreters = self.config.get("interpreters", {})
        interpreter = interpreters.get(interpreter_key)
        if interpreter is None:
            interpreter = interpreters.get("default")
        if interpreter is not None and not Path(str(interpreter)).exists():
            raise ModelRunnerError("Configured model interpreter is unavailable")
        if interpreter is None:
            interpreter = sys.executable

        source = Path(source).resolve(strict=True)
        job_config = _public_config(config)
        if not isinstance(job_config, dict):
            raise ValueError("Model config must be an object")
        output_value = job_config.get("output_dir")
        if not output_value:
            raise ValueError("Model output_dir is required")
        output_root = Path(output_value).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        job_config.update({key: self.config[key] for key in
                           ("models_root", "cache_root", "ffmpeg_bin", "ffprobe_bin", "device", "dtype")
                           if key in self.config and key not in job_config})
        job_config["output_dir"] = str(output_root)
        job_config["production"] = True

        with tempfile.TemporaryDirectory(prefix="autodub-model-", dir=output_root) as temp_dir:
            request_path = Path(temp_dir) / "job-request.json"
            result_path = Path(temp_dir) / "job-result.json"
            request_path.write_text(json.dumps({"schema_version": 1, "stage": stage,
                                               "source": str(source), "config": job_config},
                                              ensure_ascii=False), encoding="utf-8")
            env = {key: os.environ[key] for key in _SAFE_ENV if key in os.environ}
            package_root = str(Path(self.config.get("package_root", Path(__file__).resolve().parent.parent)).resolve(strict=True))
            env["PYTHONPATH"] = package_root + (os.pathsep + os.environ["PYTHONPATH"]
                                                  if os.environ.get("PYTHONPATH") else "")
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            # Keep checksum-pinned upstream checkouts clean across model stages.
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["AUTODUB_WORKER_OUTPUT"] = str(output_root)
            env["AUTODUB_WORKER_RUN_ROOT"] = str(Path(job_config.get("run_root", output_root)).resolve())
            env["AUTODUB_WORKER_MODELS"] = str(Path(self.config.get("models_root", "/data/models")).resolve())
            command = [str(interpreter), "-m", "autodub.model_worker", "--request", str(request_path),
                       "--result", str(result_path)]
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            timeout = self._timeout(config)
            if stage == "VISION_RENDER" and timeout > 900:
                raise ValueError("VISION_RENDER timeout_seconds must be at most 900")
            try:
                process = subprocess.Popen(command, cwd=package_root, env=env,
                                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, creationflags=creationflags,
                                           start_new_session=(os.name != "nt"))
                with self._lock:
                    self._active = process
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired as error:
                    self._terminate(process)
                    raise ModelRunnerError("Model worker timed out") from error
                finally:
                    with self._lock:
                        if self._active is process:
                            self._active = None
            except OSError as error:
                raise ModelRunnerError(f"Model worker could not be started: {error}") from error

            try:
                envelope = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                if process.returncode != 0:
                    raise ModelRunnerError(f"Model worker failed during {stage}") from error
                raise ModelRunnerError("Model worker returned an invalid result") from error

            if process.returncode != 0:
                error_code = envelope.get("error_code") if isinstance(envelope, dict) else None
                if (isinstance(envelope, dict) and envelope.get("schema_version") == 1
                        and envelope.get("stage") == stage
                        and isinstance(error_code, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", error_code)):
                    raise ModelRunnerError(f"Model worker failed during {stage} ({error_code})", error_code=error_code)
                raise ModelRunnerError(f"Model worker failed during {stage}")

            if (not isinstance(envelope, dict) or envelope.get("schema_version") != 1
                    or envelope.get("stage") != stage or not isinstance(envelope.get("result"), dict)):
                raise ModelRunnerError("Model worker returned an invalid result")

            result = envelope["result"]
            self._validate_artifacts(result, output_root)
            return result

    def cancel(self) -> None:
        """Terminate active child process."""
        with self._lock:
            process = self._active
        if process is not None and process.poll() is None:
            self._terminate(process)

    def _timeout(self, stage_config: dict) -> float:
        value = stage_config.get("timeout_seconds", self.config.get("timeout_seconds", 900))
        try:
            timeout = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("timeout_seconds must be positive") from error
        if isinstance(value, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        return timeout

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   check=False, timeout=5)
            else:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    @staticmethod
    def _validate_artifacts(result: dict, root: Path) -> None:
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ModelRunnerError("Model worker returned an invalid result")
        resolved_root = root.resolve()
        paths = list(artifacts)
        clips = result.get("clips", {})
        if clips:
            if not isinstance(clips, dict) or any(not isinstance(path, str) for path in clips.values()):
                raise ModelRunnerError("Model worker returned an invalid result")
            paths.extend(clips.values())
        for item in paths:
            if not isinstance(item, str):
                raise ModelRunnerError("Model worker returned an invalid result")
            path = Path(item).resolve()
            if not path.is_relative_to(resolved_root) or not path.is_file():
                raise ModelRunnerError(f"Model worker returned an invalid artifact: {item}")
