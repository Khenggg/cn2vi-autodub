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
    """RapidOCR's recognizer makes one batched ONNX call, not N engine calls."""
    from rapidocr.ch_ppocr_rec.main import TextRecInput

    if not 1 <= len(images) <= batch_size <= 16:
        raise ValueError('Subtitle recognition batch exceeds memory bounds')
    engine.text_rec.rec_batch_num = batch_size
    output = engine.text_rec(TextRecInput(img=images, return_word_box=False))
    if len(output.txts) != len(images) or len(output.scores) != len(images):
        raise ValueError('Subtitle batch lost frame association')
    return list(zip(output.txts, output.scores, strict=True))
