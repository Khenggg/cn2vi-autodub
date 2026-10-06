# Local CPU model smoke workflow

The local preparation script creates isolated environments from the committed hashed profiles. It does not install the core application into them, and adds only a `.pth` pointer to this repository's `src` directory. The optional TTS environment is separate from OCR/LaMa and can be omitted.

```powershell
.\scripts\local_prepare.ps1 -BasePython .\.venv\Scripts\python.exe -Uv .\.venv\Scripts\uv.exe
.\scripts\local_prepare.ps1 -BasePython .\.venv\Scripts\python.exe -Uv .\.venv\Scripts\uv.exe -TtsCpu
```

The second command also installs the optional `requirements/bench-tts-cpu.txt` profile. Both profiles use Python 3.12 and the existing interpreter and `uv` executable. The script refuses to fetch Python automatically, reuses existing profile environments, and verifies required modules. It checks that FastAPI is absent from the vision profile; the TTS profile may include FastAPI through upstream Gradio dependencies. Neither profile installs the core application. The script does not download model weights, install CUDA/Torch GPU profiles, or call cloud APIs.

After reviewing `benchmarks/models.lock.json`, fetch just the CPU assets used by this smoke suite explicitly:

```powershell
.\.venv\Scripts\python.exe -m autodub.model_assets fetch --lock benchmarks\models.lock.json --root .cache\models --only lama-onnx rapidocr-v6 vieneu-turbo-onnx moss-onnx
```

This command downloads the selected CPU assets; it is separate from environment preparation.

Create synthetic inputs and suite configs without loading model code:

```powershell
python .\scripts\cpu_model_smoke.py `
  --output-dir .\.cache\cpu-model-smoke `
  --models-root .\.cache\models `
  --vision-python .\.cache\vision-env\Scripts\python.exe `
  --tts-python .\.cache\tts-cpu-env\Scripts\python.exe `
  --font C:\path\to\a\Chinese-capable-font.ttf `
  --prepare-only
```

`--font` must point to a local font that contains Chinese glyphs. No platform-specific font path is assumed. The generated one-second video contains Chinese text inside the OCR ROI, a separate `OUTSIDE ROI` label above it, and a sine tone. Its manifest and every model config are marked `synthetic_smoke`, with `representative_quality_pass: false`. Without `--tts-python`, only OCR and LaMa jobs are written to the suite.

`--prepare-only` creates the frame/video, corpus validation, OCR/LaMa/TTS configs, and sequential benchmark plan, then runs only `bench_suite.dry_run` to check Python executables. The fixture uses a 36 px Chinese subtitle placed fully inside the ROI, with `OUTSIDE ROI` above it. The Vietnamese TTS text targets 2,000 ms, and the LaMa output range is `0_255` to match the pinned ONNX adapter.

Omit `--prepare-only` to run the selected local models. The script executes stages in plan order, writes `smoke-checks.json`, and exits nonzero if OCR misses the fixture text, sees the outside label, produces an empty mask, LaMa does not change masked pixels or changes unmasked pixels, residual text remains, or selected TTS output is not nonempty PCM16 48 kHz audio without clipped samples. These are fixture and I/O checks; the manifest and suite summary always keep `representative_quality_pass: false`.

The latest local synthetic run passed these checks: OCR recognized `中文字幕测试` in one sampled frame and did not report `OUTSIDE`; the mask contained 22,270 pixels. LaMa changed 22,139 masked pixels, preserved every unmasked pixel, and OCR found zero residual text. TTS produced 2,000 ms mono PCM16 at 48 kHz with a clipped-sample ratio of 0.0. Suite-observed process elapsed times were 4.7 s for OCR, 18.2 s for LaMa, and 7.4 s for TTS, including process startup/shutdown. This synthetic result verifies that the local model path and artifact I/O work; it does not establish representative quality, subtitle accuracy across real footage, natural pronunciation, motion stability, or production readiness. The benchmark reports continue to mark human review evidence as required.
