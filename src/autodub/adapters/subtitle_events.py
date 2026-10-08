"""Locate a subtitle line, select changes, and recognize bounded GPU batches."""
from __future__ import annotations

import time
from pathlib import Path

from autodub.adapters.common import identity
from autodub.adapters.ocr import build_engine
from autodub.adapters.runtime_paths import asset, output_folder
from autodub.storage import atomic_json, sha256_file


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


def scan(source: Path, config: dict, *, engine_factory=build_engine) -> dict:
    import cv2
    import numpy as np

    folder = output_folder(config)
    tick = time.perf_counter()
    engine, manifest = engine_factory({**config, 'ocr_asset_id': 'rapidocr-v6-medium', 'ocr_require_cuda': True},
                                   resolve_asset=asset)
    load_ms = (time.perf_counter() - tick) * 1000
    batch_size = int(config.get('ocr_line_batch_size', 8))
    if not 1 <= batch_size <= 16:
        raise ValueError('OCR line batch size must be between 1 and 16')
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError('Video decode unavailable')
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError('Video frame rate unavailable')
    profile = config.get('subtitle_profile', {})
    top = max(0, int(height * float(profile.get('y', 0.55))))
    bottom = min(height, int(height * float(profile.get('y', 0.55) + profile.get('h', 0.43))))
    if bottom <= top:
        capture.release()
        raise ValueError('Invalid subtitle search area')
    scaled_width = min(640, width)
    scaled_height = max(32, round((bottom - top) * scaled_width / width))
    # A reviewed ROI is expressed in normalized full-frame coordinates.
    reviewed = config.get('subtitle_line_roi') or profile.get('line_roi')
    band = None
    if reviewed:
        x, y, w, h = (float(reviewed[key]) for key in ('x', 'y', 'w', 'h'))
        if not 0 <= x < x + w <= 1 or not 0 <= y < y + h <= 1:
            capture.release()
            raise ValueError('Reviewed subtitle line ROI exceeds frame')
        band = [round(x * scaled_width), round((y * height - top) * scaled_height / (bottom - top)),
                round((x + w) * scaled_width), round(((y + h) * height - top) * scaled_height / (bottom - top))]
        if not 0 <= band[1] < band[3] <= scaled_height:
            capture.release()
            raise ValueError('Reviewed line ROI lies outside subtitle search area')
    events, frames, pending, saved_paths = [], [], [], []
    current, last_clean, previous, previous_scene = None, None, None, None
    count = sampled = calibrations = batch_calls = scene = 0
    calibration_ms = recognition_ms = 0.0
    batch_sizes = []
    last_calibration = last_selected = -10000
    sample_step = max(1, round(fps * int(config.get('event_sample_ms', 200)) / 1000))
    heartbeat = int(config.get('ocr_heartbeat_ms', 4000))
    tick = time.perf_counter()

    def consume(row, text, score):
        nonlocal current, last_clean
        at, crop, box, scene_id = row
        lines = [{'text': str(text), 'score': float(score), 'box': box}] if text and score >= 0.5 else []
        frames.append({'at_ms': at, 'scene_id': scene_id, 'lines': lines})
        if current and current['scene_id'] == scene_id and current['text'] == text and lines:
            current['end_ms'] = min(config['duration_ms'], at + round(sample_step * 1000 / fps))
            return
        old = current
        if old:
            old['end_ms'] = at
        current = None
        if not lines:
            last_clean = (at, crop.copy(), scene_id)
            if old and old['scene_id'] == scene_id and at - old['start_ms'] <= int(config.get('donor_max_event_ms', 10000)):
                path = folder / f"donor-after-{old['id']}.png"
                if not cv2.imwrite(str(path), crop):
                    raise RuntimeError('Could not save clean donor')
                old['donors'].append(str(path))
                saved_paths.append(str(path))
            return
        from autodub.adapters.v2_vision import classify_text
        current = {'id': f'ocr-{len(events):06d}', 'start_ms': at,
                   'end_ms': min(config['duration_ms'], at + round(sample_step * 1000 / fps)),
                   'text': str(text), 'kind': classify_text(lines, scaled_height, scaled_width),
                   'score': float(score), 'boxes': [box], 'scene_id': scene_id, 'donors': []}
        if last_clean and last_clean[2] == scene_id and at - last_clean[0] <= int(config.get('donor_max_gap_ms', 2000)):
            path = folder / f"donor-before-{current['id']}.png"
            if not cv2.imwrite(str(path), last_clean[1]):
                raise RuntimeError('Could not save clean donor')
            current['donors'].append(str(path))
            saved_paths.append(str(path))
        events.append(current)

    def flush():
        nonlocal batch_calls, recognition_ms
        if not pending:
            return
        images = [row[4] for row in pending]
        started = time.perf_counter()
        results = recognize_lines(engine, images, batch_size)
        recognition_ms += (time.perf_counter() - started) * 1000
        batch_calls += 1
        batch_sizes.append(len(images))
        for row, (text, score) in zip(pending, results, strict=True):
            consume(row[:4], text, float(score))
        pending.clear()

    try:
        while capture.grab():
            at = round(count * 1000 / fps)
            selected = count % sample_step == 0
            count += 1
            if not selected:
                continue
            ok, image = capture.retrieve()
            if not ok:
                raise ValueError('Sampled frame unavailable')
            sampled += 1
            crop = cv2.resize(image[top:bottom], (scaled_width, scaled_height))
            small = cv2.cvtColor(cv2.resize(image, (64, 36)), cv2.COLOR_BGR2GRAY).astype('float32')
            if previous_scene is not None and float(np.mean(np.abs(small - previous_scene))) > 45:
                flush()
                if current:
                    current['end_ms'] = at
                current, last_clean, previous = None, None, None
                scene += 1
            previous_scene = small
            if band is None:
                if at - last_calibration < 1000 or np.count_nonzero(line_signature(crop)) < 16:
                    continue
                last_calibration = at
                started = time.perf_counter()
                output = engine(crop)
                calibration_ms += (time.perf_counter() - started) * 1000
                calibrations += 1
                lines = [] if output.boxes is None else [
                    {'box': box.tolist(), 'text': str(text), 'score': float(score)}
                    for box, text, score in zip(output.boxes, output.txts, output.scores, strict=True) if score >= 0.7]
                line = select_line(crop, lines)
                if line is None:
                    continue
                polygon = np.asarray(line['box'])
                band = [round(scaled_width * 0.15), max(0, int(polygon[:, 1].min()) - 4),
                        round(scaled_width * 0.85), min(scaled_height, int(polygon[:, 1].max()) + 5)]
                # The search-frame recognition locates the line; following calls use batched line recognition.
            left, upper, right, lower = band
            line_image = crop[upper:lower, left:right]
            signature = line_signature(line_image)
            if not changed_line(previous, signature) and at - last_selected < heartbeat:
                continue
            previous, last_selected = signature, at
            ys, xs = np.nonzero(signature)
            if len(xs) < 8:
                flush()
                consume((at, crop, [], scene), '', 0)
                continue
            x0, x1 = max(0, int(xs.min()) - 3), min(right - left, int(xs.max()) + 4)
            box = [[left + x0, upper], [left + x1, upper], [left + x1, lower], [left + x0, lower]]
            pending.append((at, crop.copy(), box, scene, line_image[:, x0:x1].copy()))
            if len(pending) == batch_size:
                flush()
        flush()
        if current:
            current['end_ms'] = config['duration_ms']
    finally:
        capture.release()
    issues = [] if band is not None else [{'code': 'SUBTITLE_LINE_NOT_LOCATED', 'action': 'REVIEW_ROI'}]
    roi = None if band is None else {'x': band[0] / scaled_width,
        'y': (top + band[1] * (bottom - top) / scaled_height) / height,
        'w': (band[2] - band[0]) / scaled_width,
        'h': (band[3] - band[1]) * (bottom - top) / scaled_height / height}
    target = folder / 'ocr-events.json'
    atomic_json(target, {'schema_version': 1, 'events': events, 'frames': frames,
        'source_sha256': sha256_file(source), 'width': width, 'height': height,
        'crop': [top, bottom, scaled_width, scaled_height], 'decoded_frames': count,
        'materialized_frames': sampled, 'ocr_calls': calibrations + batch_calls,
        'calibration_calls': calibrations, 'recognition_batch_calls': batch_calls,
        'recognized_line_images': sum(batch_sizes), 'recognition_batch_sizes': batch_sizes,
        'calibration_ms': calibration_ms, 'recognition_ms': recognition_ms,
        'recognition_batch_size': batch_size, 'subtitle_line_roi': roi,
        'provider': 'CUDAExecutionProvider', 'issues': issues,
        'subtitle_profile': {'y': top / height, 'h': (bottom - top) / height, 'line_roi': roi},
        'text_kind_policy': 'BOTTOM_CENTER_WHITE_LINE_UNCALIBRATED'})
    return {**identity([manifest]), 'artifacts': [str(target), *saved_paths],
        'metrics': {'model_load_ms': load_ms, 'inference_ms': (time.perf_counter() - tick) * 1000,
                    'calibration_ms': calibration_ms, 'recognition_ms': recognition_ms},
        'ocr_calls': calibrations + batch_calls, 'stage_status': 'DEGRADED' if issues else 'SUCCESS',
        'quality_evidence': {'issues': issues}}
