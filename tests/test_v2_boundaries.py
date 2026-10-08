from pathlib import Path

import pytest

from autodub.adapters.runtime_paths import run_file
from autodub.adapters.v2_mix import _fit_clip
from autodub.adapters.v2_vision import EventTimeline


def test_run_artifact_rejects_external_files_before_existence_probe(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setenv("AUTODUB_WORKER_RUN_ROOT", str(run))
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    for path in (outside, tmp_path / "missing.txt"):
        with pytest.raises(ValueError, match="outside the frozen run"):
            run_file(path)
    approved = run / "clip.wav"
    approved.touch()
    assert run_file(approved) == approved


@pytest.mark.parametrize("speed", ["1.1;exec", float("nan"), float("inf"), 0, 1.3])
def test_voice_fitting_rejects_invalid_filter_input_before_execution(speed, monkeypatch):
    monkeypatch.setattr("autodub.adapters.v2_mix.subprocess.run", lambda *a, **k: pytest.fail("Unsafe fitting must not execute"))
    with pytest.raises(ValueError, match="Invalid voice fitting speed"):
        _fit_clip(Path("source.wav"), Path("output.wav"), speed)


def test_restoration_lookup_respects_event_bounds_and_rejects_overlaps():
    events = [{"start_ms": 100, "end_ms": 200, "donors": []},
              {"start_ms": 300, "end_ms": 400, "donors": []}]
    timeline = EventTimeline(events, 500)
    assert timeline.at(99) is None
    assert timeline.at(100) is events[0]
    assert timeline.at(200) is None
    assert timeline.at(300) is events[1]
    assert timeline.at(400) is None
    with pytest.raises(ValueError, match="ordered"):
        EventTimeline([events[1], events[0]], 500)
    with pytest.raises(ValueError, match="ordered"):
        EventTimeline([{**events[0], "end_ms": 700}], 500)
    with pytest.raises(ValueError, match="supported bounds"):
        EventTimeline(events, 86400001)
