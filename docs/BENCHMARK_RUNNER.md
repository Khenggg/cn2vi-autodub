# Benchmark runner

The runner measures one stage in a fresh Python process. The suite loads a JSON plan, validates every input/configuration before starting, and then launches each job in its own process so an earlier model can be released before the next one loads. Adapter defaults are Qwen ASR/alignment, VieNeu TTS, and the Bandit, OCR, and inpainting adapters. A missing or unconfigured adapter is a failed measurement and still produces a report.

## Validate and run

Copy `benchmarks/suite.example.json` to a local plan, point `input` at a representative video, create the stage config JSON files, and set `hourly_rate_vnd` to the actual cloud rate if cost estimates are needed. Input, config, output, and config `output_dir` paths are resolved relative to the plan or config file as appropriate. Config files stay local and are never copied into reports.

```powershell
.\.venv\Scripts\python.exe -m autodub.bench_suite --plan benchmarks/suite.local.json --dry-run
.\.venv\Scripts\python.exe -m autodub.bench_suite --plan benchmarks/suite.local.json
```

`--dry-run` validates file paths and each configured Python interpreter without importing an adapter, probing media, checking a GPU, downloading weights, or running inference. Set an optional per-job `python` path when stages need separate virtual environments. The runner is sequential. `timeout_seconds` defaults to one hour and is bounded to 24 hours; after timeout or failure, later jobs still run. Use `--summary` to choose the aggregate summary path.

A job may define `output`; otherwise it writes to `benchmark-results/<job-id>.json` beside the plan. The adapter receives `output_dir`, resolved to the configured directory or to an `output` subdirectory beside that job report. Adapter-returned artifact paths must point to existing files under that directory.

## Adapter result contract

An adapter is a Python `module:function` returning a JSON-compatible mapping with non-empty `model_revision`, `weights_sha256`, and a `quality_metrics` object. Optional fields include:

- `quality_evidence`: evidence such as the ground-truth dataset identity and evaluation protocol.
- `quality_evidence_missing`: field names that still need to be supplied or reviewed.
- `metrics`: runner measurements such as `model_load_ms`, `inference_ms`, `processed_media_ms`, and device allocator peak bytes.
- `artifacts`: existing output paths inside `output_dir`.

The report keeps runner-observed process wall time separate from adapter model-load and inference timings. It calculates RTF against `processed_media_ms` when supplied and otherwise uses the source duration; the report records the basis. GPU allocator measurements belong to the adapter because the sampler sees device-wide usage and may miss brief peaks. The runner samples process and child RSS and first-device VRAM every 500 ms.

`MEASURED` means the adapter ran and supplied model provenance; it does not mean quality passed. Every report and suite summary keeps `validation: QUALITY_REVIEW_REQUIRED`. Missing or null quality evidence remains visible for human review. Suite `MEASURED` only means each planned job produced a measured report; it is not a release gate. `FAILED`, `TIMED_OUT`, or `INCOMPLETE` jobs make the suite status reflect that problem, while the suite continues with subsequent jobs where possible. Cost uses the actual elapsed process time and configured hourly VND rate. It is an estimate and is omitted when no rate is configured.

The example plan points at placeholder golden inputs and config paths. Copy it and replace those paths before running the dry-run.
