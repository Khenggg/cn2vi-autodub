from pathlib import Path
from types import ModuleType

import pytest

from autodub.model_runner import _INTERPRETER_STAGE, ModelRunner, ModelRunnerError
from autodub.model_worker import run_stage


def test_vision_render_uses_vision_interpreter_and_worker_dispatches(monkeypatch):
    assert _INTERPRETER_STAGE["VISION_RENDER"] == "vision"
    module = ModuleType("autodub.adapters.subtitle_video")
    called = {}

    def render(source, config):
        called["source"] = source
        called["config"] = config
        return {"artifacts": ["render.mp4"]}

    module.run = render
    monkeypatch.setitem(__import__("sys").modules, "autodub.adapters.subtitle_video", module)
    config = {"target_output": "target.mp4", "subtitle_mode": "off"}

    result = run_stage("VISION_RENDER", Path("source.mp4"), config)

    assert result["artifacts"] == ["render.mp4"]
    assert called == {"source": Path("source.mp4"), "config": config}


def test_model_runner_rejects_a_missing_explicit_vision_interpreter(tmp_path):
    runner = ModelRunner({"interpreters": {"vision": str(tmp_path / "missing-python")}})

    with pytest.raises(ModelRunnerError, match="Configured model interpreter is unavailable"):
        runner.run("VISION_RENDER", tmp_path / "source.mp4", {"output_dir": str(tmp_path)})
