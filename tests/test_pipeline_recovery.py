import json
from pathlib import Path

import pytest

from autodub.checkpoint import CheckpointError, fingerprint, read_checkpoint, save_checkpoint
from autodub.pipeline import Pipeline
from autodub.storage import sha256_file

MEDIA = {"schema_version": 1, "duration_ms": 5000,
         "video": {"width": 320, "height": 240},
         "audio": {"sample_rate": "48000"}}


def _prepare_episode(client, app, uploaded, monkeypatch):
    monkeypatch.setattr("autodub.scheduler.probe_media", lambda *_: MEDIA)
    assert client.post(f"/api/episodes/{uploaded['id']}/start").status_code == 200
    assert app.state.scheduler.tick()
    return app.state.service.require("episode", uploaded["id"])


class FakeRunner:
    calls = []
    configs = []

    def __init__(self, _config):
        pass

    def run(self, stage, _source, config):
        self.calls.append(stage)
        self.configs.append((stage, config))
        output = Path(config["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / f"{stage.lower()}.json"
        if stage == "ASR":
            value = {"segments": [{"id": "seg1", "start_ms": 0, "end_ms": 1000,
                                   "zh_text": "你好", "words": [{"t": "你好", "s": 0, "e": 1000}]},
                                  {"id": "seg2", "start_ms": 1500, "end_ms": 2500,
                                   "zh_text": "再见", "words": [{"t": "再见", "s": 1500, "e": 2500}]}]}
        elif stage == "ALIGNING":
            value = {"segments": [{"id": "seg1", "start_ms": 0, "end_ms": 1000,
                                   "zh_text": "你好", "words": [{"t": "你好", "s": 0, "e": 1000}],
                                   "action": "DUB"},
                                  {"id": "seg2", "start_ms": 1500, "end_ms": 2500,
                                   "zh_text": "再见", "words": [{"t": "再见", "s": 1500, "e": 2500}],
                                   "action": "DUB"}]}
        elif stage == "TRANSLATING":
            value = {"segments": [{**item, "dub_vi": "Xin chào", "subtitle_vi": "Xin chào",
                                   "action": "DUB", "needs_review": False}
                                   for item in config["segments"]]}
        else:
            value = {"windows": []} if stage == "SEPARATING" else {"ok": True}
        artifact.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        result = {"artifacts": [str(artifact)]}
        if stage == "TTS":
            result["clips"] = {}
            for item in config["segments"]:
                clip = output / f"{item['id']}.wav"
                clip.write_bytes(b"synthetic-tts-clip")
                result["clips"][item["id"]] = str(clip)
        return result


def test_review_after_tts_resumes_from_checkpoint_without_repeating_models(
        client, app, uploaded, monkeypatch):
    episode = _prepare_episode(client, app, uploaded, monkeypatch)
    FakeRunner.calls = []
    FakeRunner.configs = []
    monkeypatch.setattr("autodub.pipeline.ModelRunner", FakeRunner)
    fit_calls = 0

    def strict_fit(_clip, output, target_ms, *_args, **_kwargs):
        nonlocal fit_calls
        fit_calls += 1
        if fit_calls == 1:
            return {"action": "REWRITE", "review_required": True,
                    "target_ms": target_ms, "speed_factor": 3.0,
                    "failure_code": "DURATION_DRIFT_OVER_20_PERCENT"}
        output.write_bytes(b"fitted-clip")
        return {"action": "SPEED", "review_required": False, "wav": output,
                "target_ms": target_ms, "speed_factor": 1.1}

    def fake_mix(_source, output, *_args, **_kwargs):
        output.write_bytes(b"preview-audio")
        return {"passed": True}

    monkeypatch.setattr("autodub.pipeline.fit_voice", strict_fit)
    monkeypatch.setattr("autodub.pipeline.mix_preview", fake_mix)
    monkeypatch.setattr(Pipeline, "_qc_and_encode", lambda *_args, **_kwargs: True)
    pipeline = Pipeline(app.state.db, app.state.settings)

    assert not pipeline.run(episode, MEDIA)
    assert app.state.service.require("episode", uploaded["id"])["status"] == "NEEDS_REVIEW"
    checkpoint = json.loads((app.state.settings.data_dir / "checkpoints" / uploaded["id"] / "state.json")
                            .read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == 2
    assert checkpoint["next_stage"] == "TTS"
    assert checkpoint["completed_stages"][0] == "PREPARING"
    model_stages = [stage for stage in checkpoint["completed_stages"] if stage != "PREPARING"]
    assert model_stages[:5] == ["ASR", "ALIGNING", "TRANSLATING", "SEPARATING", "TTS"]

    # Review the timing issue: edit the translation, keep the second line, and
    # choose a speaker voice before retrying from the saved TTS boundary.
    rows = app.state.db.rows("SELECT id,contract_json FROM segment WHERE episode_id=? ORDER BY start_ms",
                             (uploaded["id"],))
    first, second = (json.loads(row["contract_json"]) for row in rows)
    first.update({"dub_vi": "Xin chào bạn", "voice_id": "voice-alt", "action": "DUB"})
    second.update({"action": "KEEP", "dub_vi": ""})
    for row, contract in zip(rows, (first, second), strict=True):
        app.state.db.execute("UPDATE segment SET action=?,dub_vi=?,contract_json=? WHERE id=?",
                             (contract["action"], contract["dub_vi"],
                              json.dumps(contract, ensure_ascii=False), row["id"]))

    app.state.db.transition(uploaded["id"], "RETRYING", "Retry reviewed duration")
    app.state.db.transition(uploaded["id"], "QUEUED", "Queue retry", queue_requested=1)
    app.state.db.transition(uploaded["id"], "PREPARING", "Resume")
    app.state.db.transition(uploaded["id"], "CHECKPOINTED", "Prepared", queue_requested=1)
    resumed = app.state.service.require("episode", uploaded["id"])
    assert pipeline.run(resumed, MEDIA)
    assert FakeRunner.calls == ["ASR", "ALIGNING", "TRANSLATING", "SEPARATING", "TTS", "TTS"]
    revised_tts = [config for stage, config in FakeRunner.configs if stage == "TTS"][-1]
    assert [(item["id"], item["dub_vi"], item["action"], item["voice_id"])
            for item in revised_tts["segments"]] == [("seg1", "Xin chào bạn", "DUB", "voice-alt")]


def test_drain_stops_before_the_next_model_call(client, app, uploaded, monkeypatch):
    episode = _prepare_episode(client, app, uploaded, monkeypatch)
    FakeRunner.calls = []
    monkeypatch.setattr("autodub.pipeline.ModelRunner", FakeRunner)
    pipeline = Pipeline(app.state.db, app.state.settings)

    assert not pipeline.run(episode, MEDIA, should_stop=lambda: True)
    recovered = app.state.service.require("episode", uploaded["id"])
    assert recovered["status"] == "CHECKPOINTED"
    assert recovered["next_stage"] == "ASR"
    assert FakeRunner.calls == []


def test_interrupted_tts_keeps_last_good_checkpoint_for_retry(client, app, uploaded, monkeypatch):
    episode = _prepare_episode(client, app, uploaded, monkeypatch)
    FakeRunner.calls = []
    original_run = FakeRunner.run
    failed = False

    def fail_once(self, stage, source, config):
        nonlocal failed
        if stage == "TTS" and not failed:
            failed = True
            FakeRunner.calls.append(stage)
            raise RuntimeError("simulated worker interruption")
        return original_run(self, stage, source, config)

    monkeypatch.setattr("autodub.pipeline.ModelRunner", FakeRunner)
    monkeypatch.setattr(FakeRunner, "run", fail_once)
    monkeypatch.setattr("autodub.pipeline.fit_voice",
                        lambda _clip, _output, target_ms, *_a, **_k:
                        {"action": "REWRITE", "review_required": True,
                         "target_ms": target_ms, "speed_factor": 3.0,
                         "failure_code": "DURATION_DRIFT_OVER_20_PERCENT"})
    pipeline = Pipeline(app.state.db, app.state.settings)

    assert not pipeline.run(episode, MEDIA)
    failed_episode = app.state.service.require("episode", uploaded["id"])
    checkpoint = json.loads((app.state.settings.data_dir / "checkpoints" / uploaded["id"] / "state.json")
                            .read_text(encoding="utf-8"))
    assert failed_episode["status"] == "FAILED"
    assert checkpoint["next_stage"] == "TTS"
    assert checkpoint["completed_stages"][-1] == "SEPARATING"

    app.state.db.transition(uploaded["id"], "RETRYING", "Retry interrupted TTS")
    app.state.db.transition(uploaded["id"], "QUEUED", "Queue retry", queue_requested=1)
    app.state.db.transition(uploaded["id"], "PREPARING", "Resume")
    app.state.db.transition(uploaded["id"], "CHECKPOINTED", "Prepared", queue_requested=1)
    resumed = app.state.service.require("episode", uploaded["id"])
    assert not pipeline.run(resumed, MEDIA)
    assert FakeRunner.calls == ["ASR", "ALIGNING", "TRANSLATING", "SEPARATING", "TTS", "TTS"]
    assert app.state.service.require("episode", uploaded["id"])["status"] == "NEEDS_REVIEW"


def test_checkpoint_rejects_source_and_artifact_tampering(tmp_path):
    source = tmp_path / "source.mp4"
    artifact = tmp_path / "work" / "alignment.json"
    source.write_bytes(b"source")
    artifact.parent.mkdir()
    artifact.write_bytes(b"alignment")
    source_hash = sha256_file(source)
    config_hash = fingerprint({"model": "fixed"})
    record = {"path": artifact.relative_to(tmp_path).as_posix(), "sha256": sha256_file(artifact)}
    save_checkpoint(tmp_path, "episode", source_hash, config_hash,
                    completed_stages=["ALIGNING"], next_stage="TRANSLATING",
                    artifacts={"ALIGNING:alignment.json": record}, results={"ALIGNING": {}})

    read_checkpoint(tmp_path, "episode", source_hash, config_hash)
    artifact.write_bytes(b"tampered")
    with pytest.raises(CheckpointError, match="artifact checksum"):
        read_checkpoint(tmp_path, "episode", source_hash, config_hash)
    with pytest.raises(CheckpointError, match="source identity"):
        read_checkpoint(tmp_path, "episode", "different", config_hash)


def test_tampered_pipeline_artifact_fails_episode_without_leaking_details(
        client, app, uploaded, monkeypatch):
    episode = _prepare_episode(client, app, uploaded, monkeypatch)
    pipeline = Pipeline(app.state.db, app.state.settings)
    work_artifact = app.state.settings.data_dir / "work" / uploaded["id"] / "alignment.json"
    work_artifact.parent.mkdir(parents=True, exist_ok=True)
    work_artifact.write_bytes(b"valid alignment")
    config_hash = fingerprint({"runner": pipeline._runner_config(), "duration_ms": 5000,
                               "subtitle_mode": pipeline._subtitle_mode_for_episode(episode)})
    record = {"path": work_artifact.relative_to(app.state.settings.data_dir).as_posix(),
              "sha256": sha256_file(work_artifact)}
    save_checkpoint(app.state.settings.data_dir, uploaded["id"], episode["source_sha256"], config_hash,
                    completed_stages=["ASR"], next_stage="ALIGNING",
                    artifacts={"ASR:alignment.json": record},
                    results={"ASR": {"artifacts": [str(work_artifact)]}})
    work_artifact.write_bytes(b"tampered alignment")

    assert not pipeline.run(episode, MEDIA)
    failed = app.state.service.require("episode", uploaded["id"])
    assert failed["status"] == "FAILED"
    repaired_checkpoint = json.loads((app.state.settings.data_dir / "checkpoints" / uploaded["id"] / "state.json")
                                    .read_text(encoding="utf-8"))
    assert repaired_checkpoint["next_stage"] == "ASR"
    assert "ASR" not in repaired_checkpoint["completed_stages"]
    issue = app.state.db.one("SELECT code,details_json FROM issue WHERE episode_id=? ORDER BY rowid DESC",
                             (uploaded["id"],))
    assert issue["code"] == "CHECKPOINT_INVALID"
    assert "tampered alignment" not in issue["details_json"]
