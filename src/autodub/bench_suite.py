"""Run a benchmark plan sequentially, launching a fresh Python process for every stage."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from autodub.benchmark import DEFAULT_ADAPTERS, STAGES
from autodub.storage import atomic_json

PLAN_SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 3600
MAX_TIMEOUT_SECONDS = 86400
JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
REPO_ROOT = Path(__file__).resolve().parents[2]


class PlanError(ValueError):
    pass


def _resolve_from_plan(value: str | Path, plan_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = plan_path.parent / path
    return path.resolve()


def _python_executable(value: str | None, plan_path: Path) -> str:
    if value is None:
        return sys.executable
    candidate = Path(value).expanduser()
    has_path = candidate.is_absolute() or "/" in value or "\\" in value
    if has_path:
        target = candidate if candidate.is_absolute() else (plan_path.parent / candidate)
        if not target.is_file():
            raise PlanError("Configured Python executable was not found")
        return str(target)
    found = shutil.which(value)
    if found is None:
        raise PlanError("Configured Python executable was not found")
    return found


def load_plan(plan_path: Path) -> tuple[dict, list[dict]]:
    plan_path = Path(plan_path).resolve()
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"Plan could not be read ({type(error).__name__})") from None
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise PlanError("Plan schema_version must be 1")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise PlanError("Plan must contain a non-empty jobs list")
    rate = plan.get("hourly_rate_vnd")
    if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float))
                             or not math.isfinite(rate) or rate < 0):
        raise PlanError("hourly_rate_vnd must be a non-negative number")

    resolved_jobs = []
    seen_ids = set()
    for index, raw in enumerate(jobs):
        if not isinstance(raw, dict):
            raise PlanError(f"Job {index + 1} must be an object")
        job_id = raw.get("id")
        stage = raw.get("stage")
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id) or job_id in seen_ids:
            raise PlanError(f"Job {index + 1} has an invalid or duplicate id")
        seen_ids.add(job_id)
        if stage not in STAGES:
            raise PlanError(f"Job {job_id} has an unsupported stage")
        input_value = raw.get("input")
        if not isinstance(input_value, str) or not input_value:
            raise PlanError(f"Job {job_id} must name an input file")
        source = _resolve_from_plan(input_value, plan_path)
        if not source.is_file():
            raise PlanError(f"Job {job_id} input file was not found")
        config_value = raw.get("config")
        config_path = None
        if config_value is not None:
            if not isinstance(config_value, str) or not config_value:
                raise PlanError(f"Job {job_id} config must be a JSON path")
            config_path = _resolve_from_plan(config_value, plan_path)
            if not config_path.is_file():
                raise PlanError(f"Job {job_id} config file was not found")
            try:
                config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise PlanError(f"Job {job_id} config is invalid ({type(error).__name__})") from None
            if not isinstance(config_payload, dict):
                raise PlanError(f"Job {job_id} config must contain a JSON object")
            output_dir = config_payload.get("output_dir")
            if output_dir is not None and not isinstance(output_dir, str):
                raise PlanError(f"Job {job_id} output_dir must be a path string")
        elif stage != "probe":
            raise PlanError(f"Job {job_id} model stage requires a config JSON path")

        adapter = raw.get("adapter")
        if adapter is not None and (not isinstance(adapter, str) or ":" not in adapter):
            raise PlanError(f"Job {job_id} adapter must use module:function syntax")
        timeout = raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout)
                or timeout <= 0 or timeout > MAX_TIMEOUT_SECONDS):
            raise PlanError(f"Job {job_id} timeout_seconds must be in (0, 86400]")
        python_value = raw.get("python")
        if python_value is not None and not isinstance(python_value, str):
            raise PlanError(f"Job {job_id} python must be a path or executable name")
        python = _python_executable(python_value, plan_path)
        output_value = raw.get("output")
        if output_value is not None and (not isinstance(output_value, str) or not output_value):
            raise PlanError(f"Job {job_id} output must be a JSON report path")
        report_path = (_resolve_from_plan(output_value, plan_path) if output_value
                       else (plan_path.parent / "benchmark-results" / f"{job_id}.json").resolve())
        resolved_jobs.append({
            "id": job_id,
            "stage": stage,
            "input": source,
            "config": config_path,
            "output": report_path,
            "adapter": adapter or DEFAULT_ADAPTERS.get(stage),
            "python": python,
            "timeout_seconds": float(timeout),
        })
    return plan, resolved_jobs


def _terminate_tree(process: subprocess.Popen) -> None:
    try:
        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
    except psutil.Error:
        children = []
        root = None
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.Error:
            pass
    if root is not None:
        try:
            root.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(([root] if root is not None else []) + children, timeout=3)
    for process_info in alive:
        try:
            process_info.kill()
        except psutil.Error:
            pass
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    # Model logs and Chinese/Vietnamese text must survive Windows console code pages.
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    src_path = str(REPO_ROOT / "src")
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not old_pythonpath else src_path + os.pathsep + old_pythonpath
    return env


def _read_report(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _summary_payload(plan_path: Path, plan: dict, jobs: list[dict], status: str = "RUNNING") -> dict:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan": str(plan_path.resolve()),
        "status": status,
        "validation": "QUALITY_REVIEW_REQUIRED",
        "corpus_kind": plan.get("corpus_kind"),
        "representative_quality_pass": False,
        "hourly_rate_vnd": plan.get("hourly_rate_vnd"),
        "jobs": jobs,
        "elapsed_ms": sum(job.get("elapsed_ms", 0) for job in jobs),
        "cost_vnd": None,
    }


def _cost(summary: dict) -> None:
    rate = summary.get("hourly_rate_vnd")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool):
        summary["cost_vnd"] = round(summary["elapsed_ms"] * rate / 3_600_000, 2)


def dry_run(plan_path: Path, summary_path: Path | None = None) -> dict:
    plan, jobs = load_plan(plan_path)
    preflight = []
    for job in jobs:
        try:
            check = subprocess.run([job["python"], "--version"], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise PlanError(f"Job {job['id']} Python preflight failed") from None
        if check.returncode:
            raise PlanError(f"Job {job['id']} Python preflight failed")
        preflight.append({"id": job["id"], "stage": job["stage"], "status": "READY",
                          "input": str(job["input"]), "config": str(job["config"]) if job["config"] else None,
                          "output": str(job["output"]), "python": job["python"],
                          "timeout_seconds": job["timeout_seconds"]})
    summary = _summary_payload(Path(plan_path), plan, preflight, "DRY_RUN")
    if summary_path:
        atomic_json(summary_path, summary)
    return summary


def _command(job: dict) -> list[str]:
    command = [job["python"], "-m", "autodub.benchmark", "--stage", job["stage"],
               "--input", str(job["input"]), "--output", str(job["output"])]
    if job["adapter"]:
        command.extend(["--adapter", job["adapter"]])
    if job["config"]:
        command.extend(["--config", str(job["config"])])
    return command


def run_suite(plan_path: Path, summary_path: Path) -> dict:
    plan, resolved = load_plan(plan_path)
    jobs = [{"id": job["id"], "stage": job["stage"], "status": "PENDING",
             "input": str(job["input"]), "output": str(job["output"]), "elapsed_ms": 0,
             "report": str(job["output"]), "error_type": None} for job in resolved]
    summary = _summary_payload(Path(plan_path), plan, jobs)
    atomic_json(summary_path, summary)

    for index, job in enumerate(resolved):
        item = jobs[index]
        item["status"] = "RUNNING"
        atomic_json(summary_path, summary)
        started = time.perf_counter()
        process = None
        try:
            job["output"].parent.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(_command(job), stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       cwd=REPO_ROOT, env=_child_env())
            try:
                return_code = process.wait(timeout=job["timeout_seconds"])
            except subprocess.TimeoutExpired:
                _terminate_tree(process)
                item["status"] = "TIMED_OUT"
                item["error_type"] = "TimeoutExpired"
            else:
                child_report = _read_report(job["output"])
                if child_report is None:
                    item["status"] = "FAILED"
                    item["error_type"] = "ReportMissing"
                else:
                    item["quality_evidence_missing"] = child_report.get("quality_evidence_missing", [])
                    item["quality_evidence"] = child_report.get("quality_evidence", {})
                    item["corpus_kind"] = child_report.get("corpus_kind")
                    item["representative_quality_pass"] = False
                    if child_report.get("status") == "MEASURED" and return_code == 0:
                        item["status"] = "MEASURED"
                        item["model_revision"] = child_report.get("model_revision")
                        item["weights_sha256"] = child_report.get("weights_sha256")
                    else:
                        item["status"] = "FAILED"
                        item["error_type"] = child_report.get("error_type") or "BenchmarkFailed"
        except KeyboardInterrupt:
            if process is not None:
                _terminate_tree(process)
            item["status"] = "INTERRUPTED"
            item["error_type"] = "KeyboardInterrupt"
            item["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            summary["status"] = "INCOMPLETE"
            summary["interrupted"] = True
            summary["elapsed_ms"] = sum(row.get("elapsed_ms", 0) for row in jobs)
            _cost(summary)
            atomic_json(summary_path, summary)
            return summary
        except Exception as error:
            item["status"] = "FAILED"
            item["error_type"] = type(error).__name__
        finally:
            item["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            summary["elapsed_ms"] = sum(row.get("elapsed_ms", 0) for row in jobs)
            _cost(summary)
            atomic_json(summary_path, summary)

    statuses = {item["status"] for item in jobs}
    if statuses & {"PENDING", "RUNNING", "TIMED_OUT", "INTERRUPTED"}:
        summary["status"] = "INCOMPLETE"
    elif "FAILED" in statuses:
        summary["status"] = "FAILED"
    else:
        summary["status"] = "MEASURED"
    atomic_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate files and Python environments without loading a model")
    args = parser.parse_args()
    summary_path = args.summary or (args.plan.resolve().parent / "benchmark-results" / "suite-summary.json")
    try:
        if args.dry_run:
            summary = dry_run(args.plan, summary_path)
        else:
            summary = run_suite(args.plan, summary_path)
    except PlanError as error:
        print(f"Benchmark suite not started: {error}", file=sys.stderr)
        return 2
    print(f"suite: {summary['status']}; summary: {summary_path}")
    return 0 if summary["status"] in {"MEASURED", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
