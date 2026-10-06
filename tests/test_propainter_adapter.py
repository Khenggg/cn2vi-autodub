import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image

from autodub.adapters import propainter


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_config(tmp_path: Path) -> dict:
    return {"roi": {"x": 0.2, "y": 0.25, "w": 0.1, "h": 0.125},
            "start_ms": 0, "end_ms": 120, "crop_context_px": 64, "max_frames": 50,
            "subvideo_length": 40, "fp16": True, "ocr_manifest_path": str(tmp_path / "ocr" / "dense.json"),
            "models_root": str(tmp_path / "models"), "output_dir": str(tmp_path / "output")}


def fixture_manifest(tmp_path: Path, source: Path, *, temporal_coverage="DENSE_SOURCE_FRAMES",
                     source_hash: str | None = None, missing_mask: bool = False) -> Path:
    from PIL import ImageDraw

    folder = tmp_path / "ocr"
    folder.mkdir(exist_ok=True)
    frames = []
    for index, timestamp in enumerate((0, 40, 80)):
        image_path = folder / f"roi_{index:06d}.png"
        mask_path = folder / f"mask_{index:06d}.png"
        Image.new("RGB", (100, 100), (70, 80, 90)).save(image_path)
        mask = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(mask).rectangle((10, 10, 18, 18), fill=255)
        if not (missing_mask and index == 1):
            mask.save(mask_path)
        frames.append({"frame_index": index, "at_ms": timestamp,
                       "image": image_path.name, "mask": mask_path.name,
                       "image_sha256": sha(image_path),
                       "mask_sha256": sha(mask_path) if mask_path.exists() else "0" * 64})
    manifest = {"schema_version": 1, "source_sha256": source_hash or sha(source),
                "roi": {"x": 0.2, "y": 0.25, "w": 0.1, "h": 0.125},
                "ocr_scope": "ROI_ONLY", "mask_origin": "OCR_POLYGONS",
                "temporal_coverage": temporal_coverage, "frame_count": len(frames),
                "fps_num": 25, "fps_den": 1, "crop_context_px": 0,
                "start_ms": 0, "end_ms": 120, "frames": frames}
    manifest_path = folder / "dense.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def patch_source_probe(monkeypatch):
    monkeypatch.setattr(propainter, "_probe_source", lambda *_args: {
        "width": 1000, "height": 800, "frame_count": 1000,
        "fps_num": 25, "fps_den": 1, "start_ms": 0})


def test_crop_guard_refuses_oversized_expanded_roi():
    with pytest.raises(ValueError, match="720x480"):
        propainter.crop_geometry({"x": 0, "y": 0, "w": 0.8, "h": 0.8}, 1920, 1080, 128)


def test_config_requires_explicit_roi_interval_and_fp16(tmp_path):
    config = valid_config(tmp_path)
    config.pop("roi")
    with pytest.raises(ValueError, match="explicit normalized ROI"):
        propainter._config_values(config)
    config = valid_config(tmp_path)
    config["fp16"] = False
    with pytest.raises(ValueError, match="fp16=true"):
        propainter._config_values(config)


def test_sparse_ocr_manifest_is_refused(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture-source")
    fixture_manifest(tmp_path, source, temporal_coverage="SAMPLED_5_FPS")
    config = valid_config(tmp_path)
    source_info = {"fps_num": 25, "fps_den": 1, "frame_count": 1000, "start_ms": 0}
    with pytest.raises(ValueError, match="dense ROI-only"):
        propainter._read_dense_manifest(source, config, config["roi"], 0, 120, source_info, 50)


def test_input_hash_mismatch_fails_before_gpu_or_model_assets(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture-source")
    fixture_manifest(tmp_path, source, source_hash="f" * 64)
    patch_source_probe(monkeypatch)
    monkeypatch.setattr(propainter, "_check_gpu_reserve", lambda: pytest.fail("GPU check ran too early"))
    monkeypatch.setattr(propainter, "asset", lambda *_args: pytest.fail("asset load ran too early"))
    with pytest.raises(ValueError, match="source checksum"):
        propainter.run(source, valid_config(tmp_path))


def test_missing_mask_is_refused_before_inference(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture-source")
    path = fixture_manifest(tmp_path, source, missing_mask=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    geometry = propainter.crop_geometry({"x": 0.2, "y": 0.25, "w": 0.1, "h": 0.125}, 1000, 800, 64)
    with pytest.raises(ValueError, match="missing"):
        propainter._validate_ocr_files(path, manifest, manifest["frames"], geometry)


def test_configured_subwindow_selects_exact_dense_source_frames(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture-source")
    fixture_manifest(tmp_path, source)
    config = valid_config(tmp_path)
    config.update(start_ms=40, end_ms=80, max_frames=1)
    source_info = {"fps_num": 25, "fps_den": 1, "frame_count": 1000, "start_ms": 0}
    _path, _manifest, frames = propainter._read_dense_manifest(
        source, config, config["roi"], 40, 80, source_info, 1)
    assert [frame["at_ms"] for frame in frames] == [40]


def test_upstream_command_is_roi_folder_bounded_and_uses_pinned_flags(tmp_path):
    command = propainter._command("python", tmp_path / "runtime", tmp_path / "roi-frames",
                                  tmp_path / "roi-masks", tmp_path / "out", 232, 224, 3)
    assert command[command.index("--video") + 1] == str(tmp_path / "roi-frames")
    assert command[command.index("--mask") + 1] == str(tmp_path / "roi-masks")
    assert command[command.index("--width") + 1] == "232"
    assert command[command.index("--height") + 1] == "224"
    assert command[command.index("--subvideo_length") + 1] == "3"
    assert command[command.index("--neighbor_length") + 1] == "10"
    assert command[command.index("--ref_stride") + 1] == "10"
    assert "--fp16" in command and "--save_frames" in command


def test_failed_child_suppresses_stderr_and_timeout_detail(monkeypatch, tmp_path):
    monkeypatch.setattr(propainter.subprocess, "run", lambda *_args, **_kwargs:
                        subprocess.CompletedProcess([], 17, stdout=None, stderr="HF_TOKEN=do-not-leak"))
    with pytest.raises(RuntimeError, match="exit 17") as failure:
        propainter._invoke_upstream(["python", "entry.py"], tmp_path, 10)
    assert "do-not-leak" not in str(failure.value)


def test_cuda_peak_wrapper_records_child_allocator_metrics_without_gpu(monkeypatch, tmp_path):
    wrapper_source = propainter._inference_wrapper_source()
    assert "torch.cuda.reset_peak_memory_stats(device)" in wrapper_source
    assert "torch.cuda.max_memory_allocated(device)" in wrapper_source
    assert "torch.cuda.max_memory_reserved(device)" in wrapper_source
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    payload = {"schema_version": 1, "device": "cuda:0",
               "gpu_allocator_peak_bytes": 123456, "gpu_reserved_peak_bytes": 234567}

    def fake_subprocess(command, *, cwd, **_kwargs):
        assert Path(command[1]).name == "autodub_inference.py"
        assert cwd == runtime
        (runtime / "cuda_peak.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(propainter.subprocess, "run", fake_subprocess)
    elapsed_ms = propainter._invoke_upstream(["python", str(runtime / "autodub_inference.py")], runtime, 10)
    assert elapsed_ms >= 0
    assert propainter._read_peak_metrics(runtime) == {
        "gpu_allocator_peak_bytes": 123456, "gpu_reserved_peak_bytes": 234567}
    (runtime / "cuda_peak.json").write_text('{"device":"cpu"}', encoding="utf-8")
    assert propainter._read_peak_metrics(runtime) is None

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["python"], 1, stderr="HF_TOKEN=do-not-leak")
    monkeypatch.setattr(propainter.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out") as failure:
        propainter._invoke_upstream(["python", "entry.py"], tmp_path, 1)
    assert "do-not-leak" not in str(failure.value)


def test_run_reextracts_crops_and_returns_only_validated_pngs(monkeypatch, tmp_path):
    from PIL import Image

    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture-source")
    fixture_manifest(tmp_path, source)
    config = valid_config(tmp_path)
    patch_source_probe(monkeypatch)
    monkeypatch.setattr(propainter, "_check_gpu_reserve", lambda: None)

    code_dir = tmp_path / "models" / "propainter-code"
    upstream = code_dir / "source"
    upstream.mkdir(parents=True)
    (upstream / "inference_propainter.py").write_text("# fixture", encoding="utf-8")
    weights_dir = tmp_path / "models" / "propainter-weights"
    weights_dir.mkdir(parents=True)
    for filename in propainter.WEIGHT_FILES:
        (weights_dir / filename).write_bytes(b"checksum-verified-fixture")
    code_manifest = {"id": "propainter-code", "model_revision": propainter.UPSTREAM_COMMIT,
                     "weights_sha256": "a" * 64}
    weights_manifest = {"id": "propainter-weights", "model_revision": "1" * 40,
                        "weights_sha256": "b" * 64,
                        "files": [{"path": name} for name in propainter.WEIGHT_FILES]}
    monkeypatch.setattr(propainter, "asset", lambda _config, identifier:
                        (code_dir, code_manifest) if identifier == "propainter-code"
                        else (weights_dir, weights_manifest))

    def extract(_source, output, _timestamp, geometry, _ffmpeg):
        Image.new("RGB", (geometry["width"], geometry["height"]), (100, 110, 120)).save(output)
    monkeypatch.setattr(propainter, "_extract_crop_frame", extract)

    @contextmanager
    def staged(_repo, _weights, _root):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        yield runtime
    monkeypatch.setattr(propainter, "_staged_source", staged)

    def invoke(command, cwd, _timeout):
        assert "--fp16" in command and "--save_frames" in command
        (cwd / "cuda_peak.json").write_text(json.dumps({
            "schema_version": 1, "device": "cuda:0", "gpu_allocator_peak_bytes": 123456,
            "gpu_reserved_peak_bytes": 234567}), encoding="utf-8")
        input_frames = Path(command[command.index("--video") + 1])
        output_root = Path(command[command.index("--output") + 1])
        assert source not in [input_frames, *input_frames.iterdir()]
        output_frames = output_root / input_frames.name / "frames"
        output_frames.mkdir(parents=True)
        for index, input_frame in enumerate(sorted(input_frames.glob("*.png"))):
            Image.open(input_frame).save(output_frames / f"{index:04d}.png")
        return 10
    monkeypatch.setattr(propainter, "_invoke_upstream", invoke)

    result = propainter.run(source, config)
    report = json.loads((tmp_path / "output" / "propainter.json").read_text(encoding="utf-8"))
    assert result["weights_sha256"] and result["processed_media_ms"] == 120
    assert result["quality_metrics"]["processed_frames"] == 3
    assert result["metrics"] == {"gpu_allocator_peak_bytes": 123456, "gpu_reserved_peak_bytes": 234567}
    assert result["quality_evidence"]["measured_vram_peak"] is True
    assert result["quality_evidence"]["missing"] == ["temporal_warp_quality_review"]
    assert report["render_scope"] == "CROPPED_ROI_FRAME_SEQUENCE"
    assert report["final_video"] is False and report["full_frame_composition"] is False
    assert report["cuda_peak_measurement"]["gpu_allocator_peak_bytes"] == 123456
    assert len([path for path in result["artifacts"] if path.endswith(".png")]) == 3


def test_output_count_must_match_dense_input(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(RuntimeError, match="frame count"):
        propainter._copy_and_validate_outputs(tmp_path / "missing", output, 1, (16, 16))
