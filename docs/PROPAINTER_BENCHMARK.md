# ProPainter cropped ROI benchmark

The ProPainter adapter measures a bounded frame-sequence inpainting run for subtitle removal. It consumes OCR polygon masks and emits cropped PNG frames plus `propainter.json`; it does not compose a full-frame video. Its result is benchmark evidence for review, not a quality pass.

## Inputs and bounds

Run it from the `bandit` environment, which contains the pinned PyTorch, torchvision, OpenCV, and image libraries. Configure an explicit normalized subtitle ROI, `start_ms`, `end_ms`, `fp16: true`, `crop_context_px` from 64 through 128 (default 64), `max_frames` no greater than 50, and `subvideo_length` no greater than 50. The expanded crop must fit within 720×480 pixels and the source interval must remain within the video timeline. At least 1800 MiB of free GPU memory is required at launch. The 14,200 MiB admission budget in the report is provisional. The child process records PyTorch allocated and reserved allocator peaks after resetting counters and synchronizing CUDA; those values include model loading and inference but do not measure every process or total device allocation.

OCR must provide a dense frame manifest covering the requested interval, with `temporal_coverage: "DENSE_SOURCE_FRAMES"`, `ocr_scope: "ROI_ONLY"`, `mask_origin: "OCR_POLYGONS"`, the source SHA-256 and ROI, exact source-frame timestamps, contiguous local frame indexes, and checksums for each ROI image and mask. The manifest's `crop_context_px` must be zero. Sparse OCR samples, masks without glyph pixels, source or ROI mismatches, and incomplete intervals are rejected. When polygon boxes are present, their rasterized/dilated mask must equal the supplied mask.

For each selected timestamp, the adapter extracts only the bounded ROI plus context from the source and places the checked OCR mask into that crop. The original full-frame video is never supplied to ProPainter. Upstream receives the cropped frame and mask folders with fp16, saved-frame output, a subvideo length capped to the selected sequence, neighbor length 10, and reference stride 10. Model code and checkpoint files must already be present in the checksum-verified model asset registry; upstream downloads are disabled by staging only the pinned inference modules and local checkpoints.

## Example job configuration

```json
{
  "roi": {"x": 0.10, "y": 0.78, "w": 0.80, "h": 0.12},
  "start_ms": 4000,
  "end_ms": 5600,
  "crop_context_px": 64,
  "max_frames": 40,
  "subvideo_length": 40,
  "fp16": true,
  "ocr_manifest_path": "/data/results/ocr/dense.json",
  "models_root": "/data/models",
  "output_dir": "/data/results/propainter"
}
```

Use the repository suite runner with a per-job interpreter pointing at `/opt/autodub/venvs/bandit/bin/python`. The job should consume the same source and interval as the dense OCR job. Store its output outside the source tree and keep the generated report and frame hashes with the benchmark record. The output folder contains `propainter.json` and `propainter_000000.png` onward, each retaining the bounded crop dimensions; no file is a final rendered video.

## Evidence limits and license

The adapter reports exact processed frame count, interval, crop geometry, mask provenance, aggregate code/checkpoint identity, and child-process allocator peaks when available. Missing measurements remain null and are listed for review. The suite's realtime factor uses adapter wall time because model-load and inference time are not separated. Quality review remains required for temporal-warp quality; inspect output sequences for artifacts and compare them with a baseline before drawing quality conclusions. A listed encoder or successful CUDA allocation does not replace an actual benchmark run.

The pinned upstream ProPainter source is distributed under the S-Lab License 1.0. Review its terms for the intended use before running or redistributing results; the benchmark tooling does not grant additional rights.
