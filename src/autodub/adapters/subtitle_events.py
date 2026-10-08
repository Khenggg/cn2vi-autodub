"""Line geometry and bounded PP-OCRv6 recognition helpers."""
from __future__ import annotations


def select_line(image, lines):
    """Choose bottom-center white dialogue separately from colored decoration.

    This is an explicit style rule, not a calibrated dialogue classifier. Other
    subtitle styles can supply a reviewed line ROI instead.
    """
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    eligible = []
    for line in lines:
        box = np.asarray(line['box'])
        left, top = np.floor(box.min(axis=0)).astype(int)
        right, bottom = np.ceil(box.max(axis=0)).astype(int)
        center = (left + right) / 2
        if not width * 0.3 <= center <= width * 0.7 or bottom - top > height * 0.5:
            continue
        patch = image[max(0, top):min(height, bottom), max(0, left):min(width, right)]
        if not patch.size:
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        bright = hsv[:, :, 2] > 180
        white = bright & (hsv[:, :, 1] < 65)
        if np.count_nonzero(white) < 8 or np.count_nonzero(white) / max(1, np.count_nonzero(bright)) < 0.6:
            continue
        eligible.append((bottom, line))
    return max(eligible, key=lambda item: item[0])[1] if eligible else None


def line_signature(image):
    """White strokes adjacent to a dark outline; exclude colored titles."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_neighbor = cv2.erode(gray, np.ones((5, 5), dtype='uint8')) < 100
    return ((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 180) & dark_neighbor).astype('uint8')


def changed_line(previous, current):
    import numpy as np

    if previous is None:
        return True
    return np.count_nonzero(previous != current) / max(16, np.count_nonzero(previous | current)) > 0.15


def recognize_lines(engine, images, batch_size):
    """Width-bucketed ONNX batches with a bounded padded-tensor budget."""
    import cv2
    import numpy as np
    from rapidocr.ch_ppocr_rec.main import TextRecInput

    if not 1 <= len(images) <= batch_size <= 16:
        raise ValueError('Subtitle recognition batch exceeds memory bounds')
    prepared = []
    for index, image in enumerate(images):
        if image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError('Invalid subtitle line image')
        # Thin stroke bands must not be enlarged into extremely wide ONNX tensors.
        # Padding preserves glyph proportions, rather than stretching the text.
        height = image.shape[0]
        if height < 32:
            padding = 32 - height
            image = cv2.copyMakeBorder(image, padding // 2, padding - padding // 2, 0, 0,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
        prepared.append((index, np.ascontiguousarray(image)))
    shape = getattr(engine.text_rec, 'rec_image_shape', (3, 48, 320))
    rec_height, base_width = shape[1:3]

    def width(row):
        image = row[1]
        return max(base_width, round(rec_height * image.shape[1] / image.shape[0]))

    ordered = sorted(prepared, key=width)
    output_rows = [None] * len(images)
    while ordered:
        group = []
        while ordered and len(group) < batch_size:
            candidate = ordered[0]
            # Bound padded tensor area, so one wide line cannot inflate eight peers.
            if group and width(candidate) * (len(group) + 1) > 2048:
                break
            group.append(ordered.pop(0))
        engine.text_rec.rec_batch_num = batch_size
        batch = [row[1] for row in group]
        metadata = {'size': len(batch), 'normalized_width': max(width(row) for row in group),
                    'input_shapes': [list(image.shape) for image in batch]}
        if not hasattr(engine, 'autodub_rec_batches'):
            engine.autodub_rec_batches = []
        engine.autodub_rec_batches.append(metadata)
        try:
            output = engine.text_rec(TextRecInput(img=batch, return_word_box=False))
        except Exception as error:
            record_recognition_failure(error, metadata)
            raise
        if len(output.txts) != len(group) or len(output.scores) != len(group):
            raise ValueError('Subtitle batch lost frame association')
        for (index, _), text, score in zip(group, output.txts, output.scores, strict=True):
            output_rows[index] = (text, score)
    return output_rows


def glyph_signature(image):
    """Keep aligned, font-sized white components; reject large background edges."""
    import cv2
    import numpy as np

    raw = line_signature(image)
    joined = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 2), dtype='uint8'))
    joined = cv2.dilate(joined, np.ones((2, 2), dtype='uint8'))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    height, width = raw.shape
    components = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if max(6, height * 0.2) <= h <= height * 0.85 and 2 <= w <= h * 16 and area >= 12:
            components.append((label, x, y, w, h))
    groups = []
    for component in components:
        bottom = component[2] + component[4]
        group = next((g for g in groups if abs(g[0] - bottom) <= 4), None)
        if group is None:
            groups.append([bottom, [component]])
        else:
            group[1].append(component)
    eligible = []
    for _, components in groups:
        left = min(c[1] for c in components)
        right = max(c[1] + c[3] for c in components)
        if width * 0.2 <= (left + right) / 2 <= width * 0.8:
            eligible.append(components)
    if not eligible:
        return np.zeros_like(raw)
    best = max(eligible, key=lambda group: (len(group), sum(c[4] for c in group)))
    kept = [c[0] for group in eligible if len(group) >= max(1, len(best) // 2) for c in group]
    return raw & np.isin(labels, kept).astype('uint8')


def record_recognition_failure(error, metadata):
    """Keep safe numeric diagnosis, never raw exception/traceback/credentials."""
    import os
    import re
    from pathlib import Path

    from autodub.storage import atomic_json

    output = os.environ.get('AUTODUB_WORKER_OUTPUT')
    if not output:
        return
    match = re.search(r'Available memory of (\d+) is smaller than requested bytes of (\d+)', str(error))
    diagnostic = {'error_type': type(error).__name__, 'batch': metadata,
                  'category': 'OCR_RUNTIME_FAILURE'}
    if match:
        diagnostic.update(category='OCR_GPU_ARENA_EXHAUSTED', available_bytes=int(match[1]),
                          requested_bytes=int(match[2]))
    atomic_json(Path(output) / 'ocr-failure.json', diagnostic)
