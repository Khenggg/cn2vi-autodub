import array
import json
import sys
import types
import wave
from types import SimpleNamespace

import pytest

from autodub.adapters import bandit, qwen, vieneu
from autodub.storage import sha256_file


def test_qwen_aligned_words_uses_integer_ms_and_rejects_out_of_chunk_times():
    result = [SimpleNamespace(text="你好", start_time="0.1254", end_time="0.6504")]
    assert qwen.aligned_words(result, offset_ms=500, duration_ms=1000) == [
        {"t": "你好", "s": 625, "e": 1150}]
    with pytest.raises(ValueError, match="outside the source chunk"):
        qwen.aligned_words([SimpleNamespace(text="超界", start_time="0.9", end_time="1.1")],
                           offset_ms=500, duration_ms=1000)


def test_qwen_invalid_interval_does_not_load_model(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(bytes(16000))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    provider = qwen.QwenAsrProvider({"start_ms": 0, "end_ms": 1001})
    monkeypatch.setattr(provider, "load", lambda: pytest.fail("invalid input loaded model"))
    with pytest.raises(ValueError, match="Invalid benchmark source interval"):
        provider.transcribe(str(source))


def test_qwen_asr_chunks_keep_counter_source_hash_and_honest_review_fields(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"local fake source")
    output = tmp_path / "transcripts"
    output.mkdir()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(qwen, "bounds", lambda *_: (100, 1900))
    monkeypatch.setattr(qwen, "chunk_windows", lambda *_: [(0, 1000), (900, 1800)])
    extracted = []

    def extract(_source, output_path, start, end, **_kwargs):
        extracted.append((start, end))
        return tmp_path / f"chunk-{len(extracted)}.wav"

    monkeypatch.setattr(qwen, "extract_audio", extract)

    class FakeAsr:
        calls = 0

        def transcribe(self, **kwargs):
            assert kwargs["language"] == "Chinese"
            self.calls += 1
            return [SimpleNamespace(text=f"识别{self.calls}")]

    provider = qwen.QwenAsrProvider({"output_dir": str(output), "device": "cpu"})
    provider.model = FakeAsr()
    provider.device = "cpu"
    provider.manifest = {"id": "qwen-asr", "model_revision": "rev", "weights_sha256": "hash"}
    monkeypatch.setattr(qwen, "QwenAsrProvider", lambda _config: provider)
    result = qwen.run_asr(source, {"output_dir": str(output), "reference_text": ""})

    transcript = json.loads((output / "transcript.zh.json").read_text(encoding="utf-8"))
    assert extracted == [(100, 1100), (1000, 1900)]
    assert [item["id"] for item in transcript["segments"]] == ["seg_000000000100", "seg_000000001000"]
    assert transcript["source_sha256"] == sha256_file(source)
    assert [item["confidence"] for item in transcript["segments"]] == [
        {"asr": None, "alignment": None}, {"asr": None, "alignment": None}]
    assert all(item["action"] == "NEEDS_REVIEW" and item["needs_review"]
               for item in transcript["segments"])
    assert result["quality_metrics"]["cer"] is None
    assert transcript["overlap_not_reconciled"] is True


def test_alignment_source_hash_mismatch_stops_before_model_call(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"source bytes")
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"schema_version": 1, "source_sha256": "0" * 64,
                                      "segments": []}), encoding="utf-8")
    calls = []

    class FakeAligner:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls.append("loaded")
            return cls()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ForcedAligner=FakeAligner))
    monkeypatch.setattr(qwen, "configure_offline", lambda *_: None)
    monkeypatch.setattr(qwen, "asset", lambda *_: (tmp_path, {"id": "aligner"}))
    monkeypatch.setattr(qwen, "output_folder", lambda _config: tmp_path)
    with pytest.raises(ValueError, match="Transcript source hash mismatch"):
        qwen.run_alignment(source, {"transcript_path": str(transcript), "device": "cpu"})
    assert calls == []


def test_vieneu_rejects_unavailable_preset_voice_before_inference(tmp_path, monkeypatch):
    infer_calls = []
    fake_soundfile = types.ModuleType("soundfile")
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    class FakeModel:
        sample_rate = 8000

        def list_preset_voices(self):
            return [("id-1", "Configured Voice")]

        def infer(self, **_kwargs):
            infer_calls.append(1)
            return []

    provider = vieneu.VieNeuProvider({"device": "cpu", "output_dir": str(tmp_path)})
    provider.model = FakeModel()
    provider.device = "cpu"
    with pytest.raises(ValueError, match="preset voice is unavailable"):
        provider.synthesize("Xin chào", "Mai Anh", "neutral", 1000)
    assert infer_calls == []


def test_bandit_rejects_non_pinned_code_before_upstream_import(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace())
    code = tmp_path / "code"
    weights = tmp_path / "weights"
    monkeypatch.setattr(bandit, "asset", lambda _config, identifier:
                        (code, {"model_revision": "0" * 40}) if identifier == "bandit-code"
                        else (weights, {"model_revision": "weights"}))
    monkeypatch.setattr(bandit.importlib, "import_module",
                        lambda _name: pytest.fail("upstream code must not import before revision validation"))
    with pytest.raises(ValueError, match="pinned upstream commit"):
        bandit.run(tmp_path / "source.mp4", {})


def test_vieneu_writes_pcm16_qc_duration_and_runner_reports_rtf(tmp_path, monkeypatch):
    output = tmp_path / "audio"
    output.mkdir()
    artifact = output / "tts.wav"
    soundfile_calls = []

    def write(path, _waveform, sample_rate, *, subtype):
        soundfile_calls.append((sample_rate, subtype))
        values = array.array("h", [1200] * 1600)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(sample_rate)
            stream.writeframes(values.tobytes())

    fake_soundfile = types.ModuleType("soundfile")
    fake_soundfile.write = write
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setattr(vieneu, "milliseconds", iter([1000, 1080]).__next__)

    class FakeModel:
        sample_rate = 8000

        def list_preset_voices(self):
            return [("id-1", "Mai Anh")]

        def infer(self, **kwargs):
            assert kwargs["voice"] == "Mai Anh"
            return [0.1] * 1600

    provider = vieneu.VieNeuProvider({"device": "cpu", "output_dir": str(output)})
    provider.model = FakeModel()
    provider.device = "cpu"
    result = provider.synthesize("Xin chào", "Mai Anh", "neutral", 500)
    assert soundfile_calls == [(8000, "PCM_16")]
    assert result["actual_ms"] == 200
    assert result["qc"]["sample_format"] == "pcm_s16le"
    assert provider.metrics["inference_ms"] == 80

    # Exercise the real benchmark runner's TTS RTF basis using the adapter output contract.
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr("autodub.benchmark.probe_media", lambda *_: {"duration_ms": 5000})
    monkeypatch.setattr("autodub.benchmark.gpu_status", lambda: {"available": False})
    monkeypatch.setattr("autodub.benchmark.Sampler.sample", lambda *_: None)
    manifest = {"id": "vieneu-turbo", "model_revision": "rev", "weights_sha256": "hash"}
    monkeypatch.setattr(vieneu, "VieNeuProvider", lambda _config: SimpleNamespace(
        manifests=[manifest], metrics={"inference_ms": 80}, synthesize=lambda *_args, **_kwargs: {
            "wav": str(artifact), "actual_ms": 200, "qc": result["qc"],
            "duration_action": "FIT"}))
    from autodub.benchmark import run_benchmark

    report = run_benchmark("tts", source, tmp_path / "report.json",
                           adapter="autodub.adapters.vieneu:run",
                           config={"output_dir": str(output), "text": "Xin chào", "target_ms": 500})
    assert report["status"] == "MEASURED"
    assert report["metrics"]["processed_media_ms"] == 200
    assert report["metrics"]["realtime_factor_basis_ms"] == 200
    assert report["metrics"]["realtime_factor"] == pytest.approx(0.4)
