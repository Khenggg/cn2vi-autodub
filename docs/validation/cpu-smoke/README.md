# CPU synthetic model smoke — 06/10/2026

Produced by `scripts/cpu_model_smoke.py`, using the hashed vision and TTS CPU profiles on Windows 11/Python 3.12.14. Command used explicit native FFmpeg/FFprobe 8.1.1 binaries, the local Chinese-capable Microsoft YaHei font, verified `.cache/models`, and isolated `.cache/vision-env` / `.cache/tts-cpu-env` interpreters. Installed distributions and profile lock hashes are recorded alongside the reports.

These are copies of raw reports and artifacts from `.cache/final-cpu-smoke`. Absolute paths in the raw suite summary describe the machine that ran the jobs. Report `artifacts` paths are relative to each stage output folder (`ocr/`, `inpaint/`, `tts/`). Git commit in reports is the prior core baseline; `worktree_dirty: true` records that benchmark changes were being validated before committing.

All three jobs were `MEASURED`; every report and summary remains `QUALITY_REVIEW_REQUIRED`, `corpus_kind: synthetic_smoke`, `representative_quality_pass: false`. The source is a generated static green image with Chinese overlay text, an OUTSIDE ROI label and a sine tone. It contains no representative speech or motion.

| Stage | Model load | Model inference | Stage wall | Sampled peak RSS |
| --- | ---: | ---: | ---: | ---: |
| RapidOCR, one cropped frame | 1,161 ms | 2,458 ms | 4,202 ms | 517,218,304 bytes |
| LaMa, one cropped frame | 8,894 ms | 4,179 ms | 17,436 ms | 1,152,733,184 bytes |
| VieNeu ONNX, Vietnamese text | 3,519 ms | 1,563 ms | 6,950 ms | 1,160,581,120 bytes |

LaMa wall time additionally includes residual OCR QC (2,622 ms), model/asset verification and preprocessing. Its sampled-frame RTF is deliberately null. A visual check found and fixed an output-scale bug: the pinned Carve export returns RGB values on the 0–255 scale; treating them as normalized floats created a white patch. Corrected output has zero recognized residual text and byte-identical pixels outside the mask on this static fixture. Motion/flicker and human residual-text review remain required.

TTS synthesized `Chạy mau! Đừng quay đầu lại.` with preset `Mai Anh`: WAV PCM16, mono 48 kHz, 2,000 ms, no clipped samples, inference RTF 0.7815. Target is 2,000 ms; `duration_action: STRETCH` records fitting advice, not an already fitted production voice track. Pronunciation has not been human-rated. Generated audio duration varies between runs.

`smoke-checks.json` records eight passing structural checks: recognized fixture text inside ROI; no outside label; at least one OCR frame; nonempty mask (22,270 pixels); changed masked pixels (22,139); zero changed unmasked pixels; one inpaint frame with zero residual OCR text; nonempty unclipped PCM16 48 kHz TTS. These assertions prevent an empty OCR mask and a no-op LaMa run from being reported as a successful smoke. They do not accept representative quality.

The NVIDIA device usage in reports is whole-device telemetry including other desktop processes; these model jobs used CPU ONNX and did not benchmark CUDA. The measurements cannot establish cloud performance, film quality, long-video throughput, SFX preservation or project cost targets.
