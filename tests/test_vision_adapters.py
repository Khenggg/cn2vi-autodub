from types import SimpleNamespace

import pytest

from autodub.adapters.inpaint import inpaint_masked
from autodub.adapters.ocr import frame_times, polygon_mask


def test_inpaint_empty_ocr_mask_cannot_claim_model_measurement(tmp_path, monkeypatch, vision_libs):
    import json
    import sys

    from autodub.adapters import inpaint
    from autodub.storage import sha256_file

    _, Image = vision_libs
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    image, mask = tmp_path / "image.png", tmp_path / "mask.png"
    Image.new("RGB", (24, 20), "green").save(image)
    Image.new("L", (24, 20), 0).save(mask)
    manifest = tmp_path / "ocr.json"
    manifest.write_text(json.dumps({"schema_version": 1, "ocr_scope": "ROI_ONLY",
        "source_sha256": sha256_file(source), "frames": [{"image": image.name, "mask": mask.name,
        "image_sha256": sha256_file(image), "mask_sha256": sha256_file(mask)}]}), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    monkeypatch.setattr(inpaint, "asset", lambda *_: pytest.fail("empty mask loaded model"))
    with pytest.raises(ValueError, match="non-empty OCR glyph mask"):
        inpaint.run(source, {"ocr_manifest_path": str(manifest)})


@pytest.mark.parametrize("config", [
    {"sample_fps": 5.5}, {"sample_fps": True}, {"start_ms": -1},
    {"end_ms": 1001}, {"dense_frames": "false"}, {"mask_dilate_px": 4.5},
])
def test_ocr_bad_request_stops_before_loading_engine(tmp_path, monkeypatch, config):
    from autodub.adapters import ocr
    monkeypatch.setattr(ocr, "probe_media", lambda *_: {"duration_ms": 1000,
                         "video": {"avg_frame_rate": "24/1"}})
    monkeypatch.setattr(ocr, "build_engine", lambda *_: pytest.fail("invalid input loaded OCR"))
    with pytest.raises(ValueError):
        ocr.run(tmp_path / "source.mp4", {"roi": {"x": 0.1, "y": 0.7, "w": 0.8, "h": 0.2}, **config})


@pytest.fixture
def vision_libs():
    np = pytest.importorskip("numpy", reason="vision tests require NumPy")
    image_module = pytest.importorskip("PIL.Image", reason="vision tests require Pillow")
    return np, image_module


def test_ocr_frame_times_are_fps_spaced_and_strictly_before_end():
    assert frame_times(100, 1000, 5) == [100, 300, 500, 700, 900]
    assert frame_times(0, 1000, 3) == [0, 333, 666]
    assert frame_times(900, 1000, 5) == [900]


@pytest.mark.parametrize("start,end,fps", [(-1, 100, 5), (100, 100, 5), (100, 50, 5), (0, 100, 0), (0, 100, 31)])
def test_ocr_frame_times_reject_invalid_interval_or_sample_rate(start, end, fps):
    with pytest.raises(ValueError, match="Invalid OCR sampling interval"):
        frame_times(start, end, fps)


def test_polygon_mask_clips_crop_boundaries_and_draws_polygon(vision_libs):
    np, _ = vision_libs
    mask = polygon_mask((20, 12), [[(-5, -4), (8, 0), (10, 8), (0, 11)]], dilate_px=0)
    values = np.asarray(mask)
    assert values.shape == (12, 20)
    assert values[0, 0] == 255
    assert values[11, 0] == 255
    assert values[0, 19] == 0 and values[11, 19] == 0


@pytest.mark.parametrize("dilation", [-1, 13, 100])
def test_polygon_mask_refuses_unsafe_dilation_before_rendering(dilation, vision_libs):
    with pytest.raises(ValueError, match="dilation exceeds safe limit"):
        polygon_mask((10, 10), [], dilate_px=dilation)


class _FakeSession:
    def __init__(self, np, output=None, input_names=("image", "mask")):
        self.np = np
        self.output = output if output is not None else np.full((1, 3, 512, 512), 0.5, dtype=np.float32)
        self.input_names = input_names
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self.input_names]

    def run(self, outputs, feeds):
        self.calls.append((outputs, feeds))
        return [self.output]


def test_inpaint_uses_fixed_512_model_inputs_and_preserves_every_unmasked_pixel(vision_libs):
    np, _ = vision_libs
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[14:22, 25:35] = 255
    session = _FakeSession(np)

    result = inpaint_masked(image, mask, session, context_px=5)

    assert result.shape == image.shape and result.dtype == np.uint8
    assert len(session.calls) == 1
    _, feeds = session.calls[0]
    assert set(feeds) == {"image", "mask"}
    assert feeds["image"].shape == (1, 3, 512, 512)
    assert feeds["mask"].shape == (1, 1, 512, 512)
    assert feeds["image"].dtype == np.float32
    assert 0 <= feeds["image"].min() <= feeds["image"].max() <= 1
    assert set(np.unique(feeds["mask"])) <= {0.0, 1.0}
    assert np.array_equal(result[mask == 0], image[mask == 0])
    assert np.all(result[mask > 0] == 127)


def test_inpaint_carve_export_does_not_scale_rgb_output_twice(vision_libs):
    np, _ = vision_libs
    image = np.full((20, 24, 3), 64, dtype=np.uint8)
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[5:9, 8:13] = 255
    session = _FakeSession(np, output=np.full((1, 3, 512, 512), 72.0, dtype=np.float32))
    result = inpaint_masked(image, mask, session, output_range="0_255")
    assert np.all(result[mask > 0] == 72)
    assert np.array_equal(result[mask == 0], image[mask == 0])


def test_inpaint_empty_mask_returns_copy_without_calling_onnx(vision_libs):
    np, _ = vision_libs
    image = np.arange(18 * 25 * 3, dtype=np.uint8).reshape((18, 25, 3))
    mask = np.zeros((18, 25), dtype=np.uint8)

    class MustNotRun:
        def get_inputs(self):
            pytest.fail("empty masks must bypass model inspection and inference")

        def run(self, *_args, **_kwargs):
            pytest.fail("empty masks must bypass model inference")

    result = inpaint_masked(image, mask, MustNotRun())
    assert np.array_equal(result, image)
    assert result is not image


@pytest.mark.parametrize("output", [
    pytest.param("wrong_shape", id="wrong-model-shape"),
    pytest.param("non_finite", id="non-finite-model-output"),
])
def test_inpaint_rejects_wrong_onnx_output_shape_or_non_finite_values(vision_libs, output):
    np, _ = vision_libs
    image = np.full((20, 24, 3), 64, dtype=np.uint8)
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[5:9, 8:13] = 255
    prediction = np.zeros((1, 3, 511, 512), dtype=np.float32)
    if output == "non_finite":
        prediction = np.zeros((1, 3, 512, 512), dtype=np.float32)
        prediction[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="LaMa ONNX output schema mismatch"):
        inpaint_masked(image, mask, _FakeSession(np, output=prediction))


def test_inpaint_rejects_onnx_input_name_mismatch(vision_libs):
    np, _ = vision_libs
    image = np.zeros((20, 24, 3), dtype=np.uint8)
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[3:8, 4:10] = 255
    with pytest.raises(ValueError, match="LaMa ONNX input schema mismatch"):
        inpaint_masked(image, mask, _FakeSession(np, input_names=("input", "mask")))


def test_inpaint_context_crop_clips_to_image_edges_and_preserves_unmasked(vision_libs):
    np, _ = vision_libs
    image = np.full((16, 18, 3), 30, dtype=np.uint8)
    mask = np.zeros((16, 18), dtype=np.uint8)
    mask[0:4, 0:5] = 255
    session = _FakeSession(np, output=np.full((1, 3, 512, 512), 0.25, dtype=np.float32))
    result = inpaint_masked(image, mask, session, context_px=128)
    assert result.shape == image.shape
    assert np.all(result[mask > 0] == 63)
    assert np.array_equal(result[mask == 0], image[mask == 0])
