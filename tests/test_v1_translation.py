import json
from io import BytesIO

import pytest

from autodub.adapters import v1_translation
from autodub.model_assets import _PrivateRedirect


def test_missing_key_never_activates_local_model(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no translation model fallback"):
        v1_translation.run(tmp_path, {"output_dir": str(tmp_path)})


def test_context_evidence_and_token_usage_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-test-key")
    monkeypatch.setenv("AUTODUB_WORKER_OUTPUT", str(tmp_path))
    segment = {"id": "s", "start_ms": 0, "end_ms": 1000, "zh_text": "你好", "action": "DUB"}
    content = {"segments": [{"id": "s1", "subtitle_vi": "Xin chào", "dub_vi": "Xin chào",
                             "emotion": "neutral", "punctuation": "."}],
               "character_context": [{"id": "s1", "character_id": "c1", "addressee_id": "c2",
                                      "addressing": {"self": "tôi", "other": "bạn"}, "evidence": "你好"}]}
    usage = {"prompt_tokens": 100, "completion_tokens": 50}
    payload = {"choices": [{"message": {"content": json.dumps(content)}}], "usage": usage, "model": "deepseek-chat"}
    monkeypatch.setattr(v1_translation.urllib.request, "urlopen", lambda request, timeout: BytesIO(json.dumps(payload).encode()))
    result = v1_translation.run(tmp_path, {"output_dir": str(tmp_path), "segments": [segment],
        "character_context": {"characters": {"c1": {}, "c2": {}}}})
    assert result["segments"][0]["addressee_id"] == "c2"
    assert result["api_calls"][0]["usage"] == usage
    assert "private-test-key" not in json.dumps(result)


def test_gated_asset_redirect_does_not_leak_hf_auth_to_cdn():
    from urllib.request import Request
    request = Request("https://huggingface.co/pyannote/model", headers={"Authorization": "Bearer private-test-token"})
    redirected = _PrivateRedirect().redirect_request(request, None, 302, "Found", {}, "https://cdn.example/model")
    assert redirected.get_header("Authorization") is None
    same_host = _PrivateRedirect().redirect_request(request, None, 302, "Found", {}, "https://huggingface.co/other")
    assert same_host.get_header("Authorization") == "Bearer private-test-token"


def test_request_json_cannot_override_worker_filesystem_roots(tmp_path, monkeypatch):
    from autodub.adapters import runtime_paths as common
    model_root = tmp_path / "models"
    output = tmp_path / "allowed-output"
    monkeypatch.setenv("AUTODUB_WORKER_MODELS", str(model_root))
    monkeypatch.setenv("AUTODUB_WORKER_OUTPUT", str(output))
    monkeypatch.setattr(common, "load_manifest", lambda path: {"id": path.name})
    config = {"production": True, "models_root": str(tmp_path / "other"),
              "output_dir": str(tmp_path / "forbidden")}
    path, _ = common.asset(config, "bandit-v2-cinematic")
    assert path == model_root / "bandit-v2-cinematic"
    assert common.output_folder(config) == output
    assert not (tmp_path / "forbidden").exists()
    with pytest.raises(ValueError, match="Unsafe"):
        common.asset(config, "../private")
