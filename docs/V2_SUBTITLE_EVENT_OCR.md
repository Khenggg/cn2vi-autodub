# Subtitle-line event OCR

The production V2 scanner now follows: locate a dialogue subtitle line, detect
changes inside that line, then batch recognition of the selected line images.
The PP-OCRv6 model, CUDA-only execution and fixed voice remain unchanged.

Initial localization uses the same OCR engine in the configured lower search
area, at most once per second until a suitable line is found. The automatic
rule selects a bottom-center white line separately from colored decoration and
corner logos. It is a style heuristic, not a calibrated dialogue classifier.
Other layouts should provide `subtitle_line_roi` as normalized full-frame
`x`, `y`, `w`, `h`. Invalid or out-of-search-area rectangles are rejected.

After localization, changes are measured on white strokes next to dark outlines
inside the selected line band. Selected images are cropped to their observed
stroke extent. Up to eight line images are submitted together to RapidOCR's
`text_rec` API. Its recognizer assembles a batched ONNX tensor; this is not a
loop of eight full detector/classifier/recognizer calls. The final partial batch
is flushed, and result order is checked against input order. Empty-line and
scene transitions flush pending recognition before closing events.

The decoder uses `grab` for intervening frames and `retrieve` only for sampled
frames. This avoids materializing every full BGR frame, but compressed video
still requires decoding dependencies. It does not claim that only sampled
frames are decoded. Restoration and final encoding still process the full
source timeline.

Reports distinguish decoded frames, materialized frames, calibration calls,
recognition batch calls and the selected subtitle rectangle. Calibration and
recognition have separate timing metrics. A missing line is reported for ROI
review. Subtitle positions that move outside the selected band, unusual styles,
two simultaneous lines and fades require further validation.

The previous trial took 102.002 seconds for 30 seconds of source video.
This replacement has not yet been benchmarked on cloud; neither a five-second
completion time nor the hour-in-ten-minutes target is verified.
