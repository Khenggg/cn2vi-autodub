"""PTS-based visual events: one representative per event, batched recognition."""
from __future__ import annotations

import json
import subprocess
import time
from decimal import Decimal
from pathlib import Path

from autodub.adapters import subtitle_events
from autodub.adapters.common import identity
from autodub.adapters.ocr import build_engine
from autodub.adapters.runtime_paths import asset, output_folder
from autodub.storage import atomic_json, sha256_file


def frame_timestamps(source, ffprobe='ffprobe'):
    result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'frame=best_effort_timestamp_time:format=start_time', '-of', 'json', str(source)],
        capture_output=True, text=True, check=True, timeout=180)
    data = json.loads(result.stdout)
    origin = Decimal(data.get('format', {}).get('start_time', '0'))
    stamps = [round((Decimal(frame['best_effort_timestamp_time']) - origin) * 1000) for frame in data['frames']]
    if not stamps or stamps[0] < 0 or any(b <= a for a, b in zip(stamps, stamps[1:], strict=False)):
        raise ValueError('Source presentation timestamps are unavailable or unordered')
    return stamps, float(origin)


def locate_band(mask):
    """Locate outlined white line geometry without running an OCR detector."""
    import numpy as np

    height, width = mask.shape
    left, right = round(width * 0.15), round(width * 0.85)
    rows = np.count_nonzero(mask[:, left:right], axis=1) >= 3
    import cv2

    _, _, components, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    runs, start = [], None
    # Join short vertical gaps inside glyphs, keeping separate distant lines.
    for index in range(height):
        if rows[index] and start is None:
            start = index
        if start is not None and (index == height - 1 or not rows[index:index + 3].any()):
            end = index + 1
            if 5 <= end - start <= height * 0.4:
                ys, xs = np.nonzero(mask[start:end, left:right])
                font_components = [c for c in components[1:] if c[4] >= 8 and c[2] >= 2
                    and max(5, (end - start) * 0.4) <= c[3] <= (end - start) * 1.3
                    and c[1] >= start - 2 and c[1] + c[3] <= end + 2]
                if len(font_components) >= 3 and len(xs) >= 24 and width * 0.25 <= left + float(np.mean(xs)) <= width * 0.75:
                    runs.append((start, end))
            start = None
    if not runs:
        return None
    upper, lower = runs[-1]
    if len(runs) > 1 and upper - runs[-2][1] <= lower - upper:
        upper = runs[-2][0]  # Keep a nearby second subtitle line.
    return [left, max(0, upper - 4), right, min(height, lower + 4)]


def same_shape(first, second):
    """Tolerate one-pixel motion, while checking local character changes."""
    import cv2
    import numpy as np

    if first.shape != second.shape:
        return False
    kernel = np.ones((3, 3), dtype='uint8')
    missed = ((first > 0) & (cv2.dilate(second, kernel) == 0)) | (
        (second > 0) & (cv2.dilate(first, kernel) == 0))
    union = (first | second) > 0
    if np.count_nonzero(missed) / max(1, np.count_nonzero(union)) > 0.1:
        return False
    for left in range(0, first.shape[1], 12):
        pixels = np.count_nonzero(union[:, left:left + 12])
        if pixels >= 12 and np.count_nonzero(missed[:, left:left + 12]) / pixels > 0.25:
            return False
    return True


def recognition_slices(image):
    """Give each horizontal line its own recognizer input, in display order."""
    import numpy as np

    mask = subtitle_events.line_signature(image)
    rows = np.count_nonzero(mask, axis=1) >= 3
    ranges, start = [], None
    for index in range(len(rows)):
        if rows[index] and start is None:
            start = index
        if start is not None and (index == len(rows) - 1 or not rows[index:index + 3].any()):
            if index + 1 - start >= 8:
                ranges.append((max(0, start - 3), min(len(rows), index + 4)))
            start = None
    if len(ranges) <= 1:
        return [(image, 0, len(image))]  # Retain complete line context for one-line subtitles.
    return [(image[upper:lower], upper, lower) for upper, lower in ranges]


def template_matches(reference, raw):
    """Compare existing glyph coordinates, ignoring unrelated background outside."""
    import numpy as np

    ys, xs = np.nonzero(reference)
    if not len(xs):
        return False
    region = np.zeros_like(raw)
    region[max(0, int(ys.min()) - 1):min(raw.shape[0], int(ys.max()) + 2),
           max(0, int(xs.min()) - 1):min(raw.shape[1], int(xs.max()) + 2)] = 1
    return same_shape(reference, raw & region)


class SubtitleTracker:
    """Bounded visual state; short events survive and cuts do not end text."""

    def __init__(self):
        self.current = None
        self.state = 'ABSENT'

    def update(self, at, mask, crop, box, line_image, scene):
        import cv2
        import numpy as np

        present = np.count_nonzero(mask) >= 8
        closed = None
        if self.current and (not present or not same_shape(self.current['mask'], mask)):
            closed = self.current
            closed['end_ms'] = at
            self.current = None
            self.state = 'CHANGING' if present else 'DISAPPEARING'
        if not present:
            self.state = 'ABSENT'
            return closed
        sharpness = float(cv2.Laplacian(cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
        if self.current is None:
            self.current = {'start_ms': at, 'end_ms': at, 'mask': mask.copy(), 'crop': crop.copy(),
                'box': box, 'line_image': line_image.copy(), 'representative_at_ms': at,
                'sharpness': sharpness, 'frame_count': 1, 'scene_id': scene, 'scene_ids': [scene]}
            self.state = 'APPEARING'
        else:
            event = self.current
            event['frame_count'] += 1
            if scene not in event['scene_ids']:
                event['scene_ids'].append(scene)
            if sharpness > event['sharpness']:
                event.update(crop=crop.copy(), box=box, line_image=line_image.copy(),
                             representative_at_ms=at, sharpness=sharpness)
            self.state = 'STABLE' if event['frame_count'] >= 3 else 'APPEARING'
        return closed

    def finish(self, at):
        result, self.current = self.current, None
        if result:
            result['end_ms'] = at
        self.state = 'ABSENT'
        return result


def scan(source: Path, config: dict, *, engine_factory=build_engine) -> dict:
    import cv2
    import numpy as np

    folder = output_folder(config)
    started = time.perf_counter()
    engine, manifest = engine_factory({**config, 'ocr_asset_id': 'rapidocr-v6-medium', 'ocr_require_cuda': True},
                                     resolve_asset=asset)
    load_ms = (time.perf_counter() - started) * 1000
    batch_size = int(config.get('ocr_line_batch_size', 8))
    if not 1 <= batch_size <= 16:
        raise ValueError('OCR batch exceeds bounded memory budget')
    stamps, origin = frame_timestamps(source, config.get('ffprobe_bin', 'ffprobe'))
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError('Video decode unavailable')
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    profile = config.get('subtitle_profile', {})
    top = max(0, round(height * float(profile.get('y', 0.55))))
    bottom = min(height, round(height * float(profile.get('y', 0.55) + profile.get('h', 0.43))))
    if bottom <= top:
        capture.release()
        raise ValueError('Invalid subtitle search area')
    scaled_width = min(640, width)
    scaled_height = max(32, round((bottom - top) * scaled_width / width))
    reviewed = config.get('subtitle_line_roi') or profile.get('line_roi')
    band = None
    if reviewed:
        x, y, w, h = (float(reviewed[key]) for key in ('x', 'y', 'w', 'h'))
        if not 0 <= x < x + w <= 1 or not 0 <= y < y + h <= 1:
            capture.release()
            raise ValueError('Reviewed subtitle ROI exceeds frame')
        band = [round(x * scaled_width), round((y * height - top) * scaled_height / (bottom - top)),
                round((x + w) * scaled_width), round(((y + h) * height - top) * scaled_height / (bottom - top))]
        if not 0 <= band[0] < band[2] <= scaled_width or not 0 <= band[1] < band[3] <= scaled_height:
            capture.release()
            raise ValueError('Reviewed subtitle ROI outside search crop')
    events, frames, pending, batches, issues, saved_paths = [], [], [], [], [], []
    tracker = SubtitleTracker()
    previous_scene, last_clean, scene, count = None, None, 0, 0
    recognition_ms = 0.0
    tick = time.perf_counter()

    def flush():
        nonlocal recognition_ms
        if not pending:
            return
        # Width sorting is internal to RapidOCR; its API returns original input order.
        started = time.perf_counter()
        groups = [recognition_slices(row['line_image']) for row in pending]
        images = [image for group in groups for image, _, _ in group]
        results = []
        for left in range(0, len(images), batch_size):
            chunk = images[left:left + batch_size]
            results.extend(subtitle_events.recognize_lines(engine, chunk, batch_size))
            batches.append(len(chunk))
        recognition_ms += (time.perf_counter() - started) * 1000
        from autodub.adapters.v2_vision import classify_text
        cursor = 0
        for row, group in zip(pending, groups, strict=True):
            outputs = results[cursor:cursor + len(group)]
            cursor += len(group)
            text = '\n'.join(str(value) for value, _ in outputs)
            score = min(float(score) for _, score in outputs)
            eid = f"ocr-{len(events):06d}"
            lines = [{'box': row['box'], 'text': str(text), 'score': float(score)}] if text and score >= 0.5 else []
            frames.append({'at_ms': row['representative_at_ms'], 'event_start_ms': row['start_ms'],
                           'event_end_ms': row['end_ms'], 'scene_id': row['scene_id'], 'lines': lines})
            if not lines:
                issues.append({'event_id': eid, 'code': 'VISIBLE_TEXT_UNRECOGNIZED', 'start_ms': row['start_ms'],
                               'end_ms': row['end_ms']})
                continue
            value = {'id': eid, 'start_ms': row['start_ms'], 'end_ms': row['end_ms'], 'text': str(text),
                'score': float(score), 'kind': classify_text(lines, scaled_height, scaled_width),
                'boxes': [row['box']], 'donors': row.get('donors', []), 'scene_id': row['scene_id'],
                'scene_ids': row['scene_ids'], 'observed_frames': row['frame_count'],
                'representative_at_ms': row['representative_at_ms'], 'timing_source': 'VIDEO_PTS'}
            if row['frame_count'] < 3:
                issues.append({'event_id': eid, 'code': 'SHORT_VISUAL_EVENT_PRESERVED'})
            if events and events[-1]['text'] == value['text'] and events[-1]['kind'] == value['kind'] and events[-1]['end_ms'] == value['start_ms']:
                # Adjacent visual fragments with verified equal text are one event.
                previous = events[-1]
                previous['end_ms'] = value['end_ms']
                previous['observed_frames'] += value['observed_frames']
                previous['scene_ids'] = sorted(set(previous['scene_ids'] + value['scene_ids']))
                if len(previous['scene_ids']) > 1:
                    previous['donors'] = []  # A donor from another shot is never borrowed.
            else:
                events.append(value)
        pending.clear()

    def enqueue(row, clean_after=None):
        if row is None or row['end_ms'] <= row['start_ms']:
            return
        row['donors'] = []
        if len(row['scene_ids']) == 1:
            donors = []
            if last_clean and last_clean[2] == row['scene_id'] and row['start_ms'] - last_clean[0] <= int(config.get('donor_max_gap_ms', 2000)):
                donors.append(('before', last_clean[1]))
            if clean_after is not None and row['end_ms'] - row['start_ms'] <= int(config.get('donor_max_event_ms', 10000)):
                donors.append(('after', clean_after))
            for role, pixels in donors:
                path = folder / f"donor-{role}-{row['start_ms']:012d}.png"
                if not cv2.imwrite(str(path), pixels):
                    raise RuntimeError('Could not save clean donor')
                row['donors'].append(str(path))
                saved_paths.append(str(path))
        row.pop('mask', None)
        pending.append(row)
        if len(pending) == batch_size:
            flush()

    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if count >= len(stamps):
                raise ValueError('Decoder has more frames than source PTS')
            at = stamps[count]
            count += 1
            small = cv2.cvtColor(cv2.resize(image, (64, 36)), cv2.COLOR_BGR2GRAY).astype('float32')
            if previous_scene is not None and float(np.mean(np.abs(small - previous_scene))) > 45:
                scene += 1
                last_clean = None
            previous_scene = small
            crop = cv2.resize(image[top:bottom], (scaled_width, scaled_height))
            if band is None:
                band = locate_band(subtitle_events.line_signature(crop))
                if band is None:
                    continue
            left, upper, right, lower = band
            line_image = crop[upper:lower, left:right]
            signature = subtitle_events.glyph_signature(line_image)
            if tracker.current and template_matches(tracker.current['mask'], subtitle_events.line_signature(line_image)):
                signature = tracker.current['mask'].copy()
            ys, xs = np.nonzero(signature)
            present = len(xs) >= 8
            x0, x1 = (max(0, int(xs.min()) - 3), min(right - left, int(xs.max()) + 4)) if present else (0, right - left)
            box = [[left + x0, upper], [left + x1, upper], [left + x1, lower], [left + x0, lower]]
            # The mask locates text; preserve original antialiasing for recognition.
            row = tracker.update(at, signature, crop, box, line_image[:, x0:x1], scene)
            enqueue(row, crop if not present else None)
            if not present:
                last_clean = (at, crop.copy(), scene)
        if count != len(stamps):
            raise ValueError('Decoder frame count differs from source PTS')
        enqueue(tracker.finish(config['duration_ms']))
        flush()
    finally:
        capture.release()
    if band is None:
        issues.append({'code': 'SUBTITLE_LINE_NOT_LOCATED', 'action': 'REVIEW_ROI'})
    roi = None if band is None else {'x': band[0] / scaled_width,
        'y': (top + band[1] * (bottom - top) / scaled_height) / height,
        'w': (band[2] - band[0]) / scaled_width, 'h': (band[3] - band[1]) * (bottom - top) / scaled_height / height}
    target = folder / 'ocr-events.json'
    actual_batches = getattr(engine, 'autodub_rec_batches', [])
    batch_sizes = [batch['size'] for batch in actual_batches] if actual_batches else batches
    atomic_json(target, {'schema_version': 1, 'events': events, 'frames': frames, 'issues': issues,
        'source_sha256': sha256_file(source), 'width': width, 'height': height,
        'crop': [top, bottom, scaled_width, scaled_height], 'decoded_frames': count, 'materialized_frames': count,
        'ocr_calls': len(batch_sizes), 'calibration_calls': 0, 'calibration_ms': 0,
        'recognition_batch_calls': len(batch_sizes), 'recognized_line_images': sum(batch_sizes),
        'recognition_batch_sizes': batch_sizes, 'recognition_batch_shapes': actual_batches,
        'recognition_ms': recognition_ms,
        'scan_policy': 'ALL_SOURCE_FRAMES_ROI_ONLY', 'timing_source': 'VIDEO_PTS', 'source_time_origin_seconds': origin,
        'subtitle_line_roi': roi, 'subtitle_profile': {'y': top / height, 'h': (bottom - top) / height, 'line_roi': roi},
        'provider': 'CUDAExecutionProvider', 'event_recall': None, 'character_error_rate': None,
        'timing_error_p95_ms': None, 'visual_event_tracker': 'OUTLINED_WHITE_TEXT_UNCALIBRATED'})
    return {**identity([manifest]), 'artifacts': [str(target), *saved_paths], 'ocr_calls': len(batch_sizes),
        'metrics': {'model_load_ms': load_ms, 'inference_ms': (time.perf_counter() - tick) * 1000,
                    'calibration_ms': 0, 'recognition_ms': recognition_ms},
        'stage_status': 'DEGRADED' if issues else 'SUCCESS', 'quality_evidence': {'issues': issues}}
