# Benchmark corpus and local media helpers

The corpus manifest uses `schema_version: 1` and `kind: representative` or `kind: synthetic_smoke`. Each case names a source file relative to the manifest, a source-relative `[start_ms, end_ms)` interval, and one category: `speech_clear`, `music_loud`, `nonverbal`, `overlap`, `subtitle_static`, `subtitle_motion`, or `dark`. Optional fields are a lowercase or uppercase `source_sha256`, Chinese `zh_reference`, source-relative word times (`words: [{"t": "...", "s": 100, "e": 400}]`), normalized `roi`, `tts_text`, and positive integer `target_ms`.

`python -m autodub.corpus --manifest benchmarks/corpus.example.json --output benchmark-results/corpus-validation.json` checks schema fields, local paths, supplied hashes, media streams and segment bounds with FFprobe. A representative corpus is marked ready for quality evaluation when its cases cover all seven categories and total at least ten minutes. Readiness only means the dataset is assembled; it does not assert any model quality pass. Synthetic smoke manifests always report `SYNTHETIC_SMOKE_ONLY` and keep `representative_quality_pass` false.

Copy `benchmarks/corpus.example.json` to a working manifest, put a locally licensed source at `benchmarks/media/representative.mp4`, replace the hash placeholder with that file's SHA-256, and adjust intervals to fit the probed duration. Keep the source media out of version control. To create a small generated I/O fixture, run:

```powershell
python -c "from pathlib import Path; from autodub.corpus import create_synthetic_smoke; print(create_synthetic_smoke(Path('benchmark-results/synthetic'), ffmpeg_bin='ffmpeg', ffprobe_bin='ffprobe'))"
```

Generated color bars and a sine tone validate file handling only. They contain no speech and provide no ASR, TTS, subtitle, nonverbal-audio, or overall model-quality evidence.

`autodub.benchmedia` provides adapter-oriented helpers:

```python
from autodub.benchmedia import chunk_windows, extract_audio, extract_roi_frame

windows = chunk_windows(duration_ms, chunk_ms=240_000, overlap_ms=1_000)
extract_audio(source, chunk_wav, start_ms=0, end_ms=240_000, sample_rate=16_000, channels=1)
extract_audio(source, production_wav, sample_rate=48_000, channels=2)
extract_roi_frame(source, frame_png, roi={"x": 0.1, "y": 0.2, "w": 0.7, "h": 0.6}, at_ms=5_000)
```

Arguments are passed directly to subprocess APIs; timestamps and normalized ROIs are validated before FFmpeg runs. `autodub.quality.cer(reference, hypothesis)` returns normalized character error rate, or `None` when the reference is blank. `word_boundary_error(reference_words, output_words)` reports median and p95 absolute boundary error for exact token matches only. `audio_qc(path)` accepts PCM16 WAV and reports duration, peak, clipping sample ratio and 100 ms silence-window ratio. These metrics do not infer confidence or replace human review.
