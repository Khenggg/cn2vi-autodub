# Subtitle events and independent speech timelines

V2 scans every source frame inside the subtitle ROI using outlined-white-stroke
geometry. Saved/reviewed regions are reused; automatic localization also uses
geometry, without full OCR. Other colors and layouts still need quality review.
Short events survive. A scene cut does not automatically close unchanged text.
Each visual event supplies its sharpest representative to the existing PP-OCRv6
Medium CUDA recognizer, in batches of at most eight images. Equal adjacent text
fragments are merged; the same text after an absence is a separate event.
Unrecognized visible text is reported. Background donors never cross scenes.

OCR timestamps use original source PTS normalized by the shared source origin.
FFprobe supplies timestamps; OpenCV materializes each frame for the cheap scan.
This is CPU decoding, not a claim of NVDEC acceleration. Final restoration still
has its separate zero-origin CFR guard. OCR VFR support does not remove that guard.

Audio/OCR matches split eligible ASR paragraphs into individual sentences.
Unmatched audio text is preserved for review. Complete chronological native word
coverage owns precise speech bounds. Otherwise display-event bounds provide an
explicit OCR_EVENT_ESTIMATE, not forced alignment. No equally spaced word times
are fabricated. Accurate speech synchronization remains unverified for estimates.
Speech bounds and subtitle display bounds are independent contract fields.

The downloadable timeline.json joins results by segment ID and records source
text, polygons mapped back to source pixels, display/speech intervals, provenance,
estimated flags, WAVs and actual mix positions. Human review remains authoritative.
Voice placement uses absolute 48kHz PCM offsets. Only near-zero outer silence is
trimmed, preserving 10ms edge context. A clip up to 1.25 times its slot may be
fitted without changing pitch. Larger overruns and overlaps are reported; later
sentences never shift and speech is not truncated. Ngọc Huyền and models are fixed.

The previous 30-second trial took 86.142 seconds. Calibration cost 37.794 seconds;
recognition cost 16.036 seconds. This replacement removes full-OCR calibration.
Its performance, event recall, Chinese CER and synchronization accuracy require
a new frozen cloud run and labeled evidence. The five-second target is unverified.
