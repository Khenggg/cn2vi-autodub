import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodub import bench_plan, bench_suite


def _inputs(tmp_path, monkeypatch, *, kind="representative", include_full_case=True):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"verified fixture source")
    full_case = {
        "id": "speech/clear",
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "start_ms": 0,
        "end_ms": 600_000,
        "category": "speech_clear",
        "zh_reference": "你好",
        "words": [{"t": "你好", "s": 0, "e": 500}],
        "tts_text": "Xin chào",
        "target_ms": 1800,
        "roi": {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.2},
    }
    incomplete_case = {
        "id": "motion case",
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "start_ms": 0,
        "end_ms": 10_000,
        "category": "subtitle_motion",
    }
    cases = [full_case, incomplete_case] if include_full_case else [incomplete_case]
    manifest = {"schema_version": 1, "kind": kind, "env": {"TOKEN": "DO_NOT_COPY"}, "cases": cases}
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(manifest), encoding="utf-8")
    statuses = {"representative": "READY_FOR_REPRESENTATIVE_QUALITY_EVALUATION",
                "synthetic_smoke": "SYNTHETIC_SMOKE_ONLY"}
    categories = sorted({case["category"] for case in cases})
    report_cases = [{"id": case["id"], "source": case["source"],
                     "source_sha256": case["source_sha256"], "source_duration_ms": 600_000,
                     "start_ms": case["start_ms"], "end_ms": case["end_ms"], "category": case["category"]}
                    for case in cases]
    report = {"schema_version": 1, "kind": kind, "status": statuses[kind],
              "representative_quality_pass": False, "total_case_duration_ms": sum(
                  case["end_ms"] - case["start_ms"] for case in cases),
              "covered_categories": categories, "missing_categories": [], "cases": report_cases}
    monkeypatch.setattr(bench_plan, "validate_manifest", lambda *_args: report)
    venv_root = tmp_path / "venvs"
    for profile in {"asr", "tts", "bandit", "vision"}:
        executable = (venv_root / profile / "Scripts" / "python.exe" if os.name == "nt"
                      else venv_root / profile / "bin" / "python")
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(b"python stub")
    version_calls = []

    def version_only(command, **_kwargs):
        version_calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bench_suite.subprocess, "run", version_only)
    return corpus_path, source, manifest, venv_root, version_calls


def test_generate_suite_links_stage_configs_artifacts_and_provenance(tmp_path, monkeypatch):
    corpus, source, original, venv_root, version_calls = _inputs(tmp_path, monkeypatch)
    output_dir = tmp_path / "generated"
    result = bench_plan.generate_plan(
        corpus_path=corpus, output_dir=output_dir, models_root=tmp_path / "models",
        cache_root=tmp_path / "cache", venv_root=venv_root, hourly_rate_vnd=72_000,
    )
    plan_path = Path(result["plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    jobs = {job["id"].rsplit("_", 1)[-1]: job for job in plan["jobs"] if "speech" in job["id"]}
    assert set(jobs) == {"asr", "align", "tts", "bandit", "ocr", "inpaint"}
    assert all(Path(job["input"]).is_absolute() and Path(job["input"]) == source.resolve() for job in plan["jobs"])
    assert plan["hourly_rate_vnd"] == 72_000
    assert plan["validation"] == "QUALITY_REVIEW_REQUIRED"
    assert plan["representative_quality_pass"] is False
    assert result["status"] == "DRY_RUN"
    assert len(version_calls) == len(plan["jobs"])
    assert all(call[1:] == ["--version"] for call in version_calls)
    profiles = {"asr": "asr", "align": "asr", "tts": "tts", "bandit": "bandit",
                "ocr": "vision", "inpaint": "vision"}
    for stage, job in jobs.items():
        expected_python = (venv_root / profiles[stage] / "Scripts" / "python.exe" if os.name == "nt"
                           else venv_root / profiles[stage] / "bin" / "python")
        assert Path(job["python"]) == expected_python.resolve()

    def config_for(job):
        config_path = (plan_path.parent / job["config"]).resolve()
        assert config_path.is_relative_to(output_dir.resolve())
        config = json.loads(config_path.read_text(encoding="utf-8"))
        artifact_dir = (plan_path.parent / job["output"]).with_suffix(".artifacts").resolve()
        assert Path(config["output_dir"]) == artifact_dir
        assert Path(config["output_dir"]).is_relative_to(output_dir.resolve())
        assert config["models_root"] == str((tmp_path / "models").resolve())
        assert config["cache_root"] == str((tmp_path / "cache").resolve())
        assert config["corpus_kind"] == "representative"
        assert config["representative_quality_pass"] is False
        return config

    asr = config_for(jobs["asr"])
    assert (asr["start_ms"], asr["end_ms"], asr["chunk_ms"], asr["overlap_ms"]) == (0, 600_000, 240_000, 1000)
    assert asr["reference_text"] == "你好"
    align = config_for(jobs["align"])
    assert Path(align["transcript_path"]) == Path(config_for(jobs["asr"])["output_dir"]) / "transcript.zh.json"
    assert align["reference_words"] == original["cases"][0]["words"]
    tts = config_for(jobs["tts"])
    assert (tts["text"], tts["target_ms"], tts["voice_id"], tts["device"]) == (
        "Xin chào", 1800, "Mai Anh", "cuda:0")
    assert config_for(jobs["bandit"])["windows"] == [{"start_ms": 0, "end_ms": 600_000}]
    ocr = config_for(jobs["ocr"])
    assert (ocr["roi"], ocr["start_ms"], ocr["end_ms"], ocr["sample_fps"]) == (
        original["cases"][0]["roi"], 0, 600_000, 5)
    assert config_for(jobs["inpaint"])["ocr_manifest_path"] == str(
        Path(config_for(jobs["ocr"])["output_dir"]) / "ocr.json")

    generation_path = Path(result["generation_manifest"])
    generation_text = generation_path.read_text(encoding="utf-8")
    generation = json.loads(generation_text)
    assert "DO_NOT_COPY" not in generation_text and '"env"' not in generation_text
    assert result["generation_manifest_sha256"] == hashlib.sha256(generation_path.read_bytes()).hexdigest()
    assert generation["representative_quality_pass"] is False
    clean_corpus = {key: value for key, value in original.items() if key != "env"}
    clean_digest = hashlib.sha256(json.dumps(clean_corpus, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")).encode("utf-8")).hexdigest()
    assert generation["source_corpus_sha256"] == clean_digest
    assert all(Path(case["source_absolute"]).is_absolute() for case in generation["cases"])
    assert {item["stage"] for item in generation["skipped"] if item["case_id"] == "motion case"} == {
        "tts", "ocr", "inpaint"}
    assert all(Path(path).resolve().is_relative_to(output_dir.resolve())
               for path in [generation_path, plan_path, *[(output_dir / item["path"]) for item in generation["configs"]]])
    original["env"]["TOKEN"] = "CHANGED_SECRET"
    corpus.write_text(json.dumps(original), encoding="utf-8")
    repeated = bench_plan.generate_plan(
        corpus_path=corpus, output_dir=output_dir, models_root=tmp_path / "models",
        cache_root=tmp_path / "cache", venv_root=venv_root, hourly_rate_vnd=72_000,
    )
    assert repeated["generation_manifest_sha256"] == result["generation_manifest_sha256"]


def test_synthetic_requires_flag_and_never_marks_quality_pass(tmp_path, monkeypatch):
    corpus, _source, _manifest, venv_root, _calls = _inputs(
        tmp_path, monkeypatch, kind="synthetic_smoke", include_full_case=False)
    output_dir = tmp_path / "not-allowed"
    with pytest.raises(bench_plan.PlanGenerationError, match="allow-synthetic"):
        bench_plan.generate_plan(corpus_path=corpus, output_dir=output_dir, venv_root=venv_root)
    assert not output_dir.exists()

    result = bench_plan.generate_plan(corpus_path=corpus, output_dir=tmp_path / "allowed",
                                      venv_root=venv_root, allow_synthetic=True)
    plan = json.loads(Path(result["plan"]).read_text(encoding="utf-8"))
    generation = json.loads(Path(result["generation_manifest"]).read_text(encoding="utf-8"))
    assert plan["corpus_kind"] == "synthetic_smoke"
    assert plan["representative_quality_pass"] is False
    assert result["representative_quality_pass"] is False
    assert all("representative_ground_truth" in job["quality_evidence_missing"]
               for job in generation["jobs"])
    for job in plan["jobs"]:
        config = json.loads((Path(result["plan"]).parent / job["config"]).read_text(encoding="utf-8"))
        assert config["corpus_kind"] == "synthetic_smoke"
        assert config["representative_quality_pass"] is False


def test_allow_synthetic_does_not_bypass_invalid_representative_gate(tmp_path, monkeypatch):
    corpus, _source, _manifest, venv_root, _calls = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(bench_plan, "validate_manifest", lambda *_args: {
        "kind": "representative", "status": "INVALID", "representative_quality_pass": False,
        "cases": [], "errors": ["short corpus"]})
    with pytest.raises(bench_plan.PlanGenerationError, match="INVALID"):
        bench_plan.generate_plan(corpus_path=corpus, output_dir=tmp_path / "generated",
                                 venv_root=venv_root, allow_synthetic=True)


def test_missing_environment_blocks_cpu_preflight_but_keeps_generated_plan(tmp_path, monkeypatch):
    corpus, _source, _manifest, venv_root, _calls = _inputs(tmp_path, monkeypatch)
    for profile in {"asr", "tts", "bandit", "vision"}:
        shutil_path = (venv_root / profile / "Scripts" / "python.exe" if os.name == "nt"
                       else venv_root / profile / "bin" / "python")
        shutil_path.unlink()
    result = bench_plan.generate_plan(corpus_path=corpus, output_dir=tmp_path / "generated",
                                      venv_root=venv_root)
    assert result["status"] == "PREFLIGHT_BLOCKED"
    assert Path(result["plan"]).is_file()
    assert result["preflight"]["error_type"] == "PlanError"
