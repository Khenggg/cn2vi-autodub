# Video-to-transcript timing audit — 2026-10-07

The previous 31/33 duration-failure count used unreliable forced-alignment bounds. It cannot establish that 31 Vietnamese lines need rewriting.

## Verified on the rented GPU cloud

The audit used the existing 60-second test clip and pinned installed models. No source media was sent to an external API for this audit. Ngọc Huyền remains the selected default voice.

| Boundary | Observation |
| --- | --- |
| Video audio stream | AAC, 44.1 kHz, stereo |
| Explicit FFmpeg extraction | Mono PCM16, 16 kHz, 960,006 samples, 60.000375 seconds |
| Faster-Whisper/PyAV decoder | Exactly identical sample array to the explicit extraction |
| WhisperX/FFmpeg decoder | Exactly identical sample array to the explicit extraction |
| Original ASR speech slots | No segment shorter than 100 ms |
| Previous forced alignment | Five utterances collapsed to 20–42 ms; their CTC match scores were zero |

For example, a 600 ms ASR speech slot became a 21 ms aligned utterance. The failure appears after decoding, in forced alignment and the adapter's handling of that output. Decoder equality does not establish transcript accuracy or correct word timing.

## Problems and changes

The pinned Chinese model's feature-extractor configuration requires waveform normalization. WhisperX's Hugging Face inference path called the CTC model directly on cropped raw waveforms. The adapter now applies the pinned local feature extractor to each crop, including its configured attention mask. The upstream [model usage example](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn#usage) also passes audio through the processor before inference.

The old adapter discarded CTC scores and accepted any positive word span within the ASR window. It then used the first and last aligned word to replace the whole utterance window. That allowed failed 20 ms token supports to become translation and TTS duration targets.

The adapter now rejects unscored/very-low-score word matches, missing text coverage, overlapping word times, out-of-window words and collapsed utterances. The 0.01 raw-score threshold is a rejection heuristic, not calibrated confidence. Failed alignments keep their ASR text and window, clear unusable word times, and retain the review flag. There is no fabricated minimum-duration padding or guessed word timing.

Accepted groups keep the outer ASR speech window as a separate duration-planning slot. This window is still an ASR estimate, not independently verified speech ground truth. Word timing and dialogue masking remain separate from that slot. WhisperX is invoked per source segment so sentence splitting cannot shift results onto another source segment. Alignment policy revision 2 invalidates old alignment and downstream checkpoint inputs.

## Fresh rerun

The revised adapter processed the same clip in 6.385 seconds including worker startup. It produced 33 speech slots, none below 100 ms; the minimum ASR slot is 480 ms. Four source segments passed complete word-alignment validation and 29 were held for low CTC match scores. The artifact correctly reports `PARTIAL_REVIEW_REQUIRED`, with failed word lists empty and per-source diagnostics.

This fixes the acceptance and propagation of failed timings. It does **not** prove that forced alignment or transcription is accurate for this clip. Some text also changes when ASR is run on shorter context windows. The remaining issue is usable word alignment against the actual spoken source, including transcript mismatches. No full dubbing mix was generated from rejected timings, and the old 31/33 fitting result must not be reused as a verdict about the selected Ngọc Huyền voice.

Regression coverage includes zero-score/collapsed results, incomplete coverage, overlaps, sentence-split association, preserved ASR bounds and explicit partial-review output. Detailed transcript artifacts stay in the private cloud test workspace.
