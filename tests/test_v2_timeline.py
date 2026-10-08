import json
from types import SimpleNamespace

import numpy as np
import pytest

from autodub.adapters.subtitle_tracking import (
    SubtitleTracker,
    frame_timestamps,
    recognition_slices,
    same_shape,
    template_matches,
)
from autodub.adapters.v1_audio import sentence_segments
from autodub.adapters.v2_mix import RATE, trim_silent_edges
from autodub.contracts import Segment
from autodub.timeline import sentence_timeline, timeline_manifest, validated_subtitle_profile
from autodub.v1_pipeline import write_subtitles


def parent(**changes):
    return {'id': 'parent', 'start_ms': 0, 'end_ms': 15000, 'zh_text': '你好再见',
            'action': 'DUB', 'timing_source': 'VAD_WINDOW', **changes}


def captions():
    return [{'id': 'a', 'start_ms': 1000, 'end_ms': 2000, 'text': '你好', 'score': 0.95, 'kind': 'DIALOGUE'},
            {'id': 'b', 'start_ms': 3000, 'end_ms': 4000, 'text': '再见', 'score': 0.95, 'kind': 'DIALOGUE'}]


def test_split_paragraph_has_separate_sentences_and_estimated_speech_labels():
    result = sentence_timeline([parent()], captions())
    assert [s['zh_text'] for s in result['segments']] == ['你好', '再见']
    assert [(s['start_ms'], s['end_ms']) for s in result['segments']] == [(1000, 2000), (3000, 4000)]
    assert all(s['timing_source'] == 'OCR_EVENT_ESTIMATE' and s['words'] == [] for s in result['segments'])
    assert all(s['dialogue_evidence']['speech_timing_estimated'] for s in result['segments'])
    assert len({s['id'] for s in result['segments']}) == 2
    assert result == sentence_timeline([parent()], list(reversed(captions())))


def test_native_speech_and_display_bounds_are_independent(tmp_path):
    words = [{'t': '你好', 's': 1100, 'e': 1800}, {'t': '再见', 's': 3200, 'e': 3800}]
    result = sentence_timeline([parent(words=words, timing_source='NATIVE_WORDS')], captions())
    first = result['segments'][0]
    assert (first['start_ms'], first['end_ms']) == (1100, 1800)
    assert (first['subtitle_start_ms'], first['subtitle_end_ms']) == (1000, 2000)
    assert result['issues'] == []
    first['subtitle_vi'] = 'Xin chào'
    path = tmp_path / 'subtitles.srt'
    write_subtitles(path, [first])
    assert '00:00:01,000 --> 00:00:02,000' in path.read_text()
    narrowed = sentence_timeline([parent(start_ms=1100, end_ms=3800, words=words,
                                         timing_source='NATIVE_WORDS')], captions())
    assert narrowed['segments'][0]['subtitle_start_ms'] == 1000
    assert narrowed['segments'][1]['subtitle_end_ms'] == 4000


def test_unmatched_audio_is_preserved_for_review_and_singing_is_not_split():
    result = sentence_timeline([parent(zh_text='你好请坐再见')], captions())
    unmatched = next(s for s in result['segments'] if s['id'].endswith(':unmatched'))
    assert unmatched['zh_text'] == '请坐' and unmatched['action'] == 'KEEP' and unmatched['needs_review']
    song = parent(action='KEEP', dialogue_kind='SINGING_OST')
    assert sentence_timeline([song], captions())['segments'][0]['id'] == 'parent'


def test_manifest_joins_by_id_and_maps_crop_boxes_back_to_source_pixels():
    segments = sentence_timeline([parent()], captions())['segments']
    event = {**captions()[0], 'boxes': [[[20, 10], [40, 10], [40, 20], [20, 20]]]}
    ocr = {'events': [event], 'width': 1280, 'crop': [400, 600, 640, 100]}
    sid = segments[0]['id']
    manifest = timeline_manifest(list(reversed(segments)), ocr, {sid: 'first.wav'}, {
        'voice_added': [{'segment_id': sid, 'start_ms': 1000, 'end_ms': 1700, 'start_sample': 48000}]})
    record = next(r for r in manifest['records'] if r['id'] == sid)
    assert record['speech']['audio_file'] == 'first.wav'
    assert record['speech']['placement']['start_sample'] == 48000
    assert record['subtitle']['polygons_source_pixels'][0][0] == [40, 420]


def test_subtitle_contract_rejects_half_interval():
    with pytest.raises(ValueError, match='both boundaries'):
        Segment.model_validate(parent(subtitle_start_ms=1000))


def test_short_event_is_not_lost_and_scene_cut_does_not_end_unchanged_text():
    tracker = SubtitleTracker()
    crop = np.zeros((32, 80, 3), dtype='uint8')
    mask = np.zeros((32, 80), dtype='uint8')
    mask[10:15, 20:40] = 1
    crop[mask > 0] = 255
    box = [[20, 10], [40, 10], [40, 15], [20, 15]]
    assert tracker.update(1000, mask, crop, box, crop, 0) is None
    assert tracker.update(1033, mask, crop, box, crop, 1) is None
    event = tracker.update(1100, np.zeros_like(mask), crop * 0, [], crop * 0, 1)
    assert (event['start_ms'], event['end_ms']) == (1000, 1100)
    assert event['scene_ids'] == [0, 1]
    assert tracker.state == 'ABSENT'


def test_character_shape_change_and_subpixel_motion_are_distinguished():
    first = np.zeros((32, 100), dtype='uint8')
    first[10:25, 20:23] = 1
    first[10:13, 20:35] = 1
    moved = np.roll(first, 1, axis=1)
    assert same_shape(first, moved)
    second = first.copy()
    second[10:25, 30:33] = 1
    assert not same_shape(first, second)


def test_silence_trim_preserves_signal_with_context():
    clip = np.zeros(RATE * 2, dtype='float32')
    clip[RATE // 2:RATE] = 0.1
    trimmed, leading, trailing = trim_silent_edges(clip)
    assert leading == RATE // 2 - RATE // 100
    assert trailing == RATE - RATE // 100
    assert np.array_equal(trimmed[RATE // 100:-RATE // 100], clip[RATE // 2:RATE])


def test_pts_preserves_variable_frame_intervals_and_shared_source_origin(monkeypatch):
    payload = {'format': {'start_time': '2.0'}, 'frames': [
        {'best_effort_timestamp_time': value} for value in ('2.0', '2.041', '2.127')]}
    monkeypatch.setattr('autodub.adapters.subtitle_tracking.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload)))
    assert frame_timestamps('video.mp4') == ([0, 41, 127], 2.0)


def test_two_subtitle_lines_are_separate_recognizer_inputs():
    image = np.zeros((64, 120, 3), dtype='uint8')
    image[10:22, 20:90] = 255
    image[40:52, 30:100] = 255
    lines = recognition_slices(image)
    assert len(lines) == 2 and lines[0][2] < lines[1][1]


def test_existing_glyph_template_ignores_unrelated_background_but_detects_changed_strokes():
    reference = np.zeros((32, 100), dtype='uint8')
    reference[10:25, 20:23] = 1
    reference[10:13, 20:35] = 1
    raw = reference.copy()
    raw[:, 80:85] = 1
    assert template_matches(reference, raw)
    raw[10:25, 30:33] = 1
    assert not template_matches(reference, raw)


def test_failed_or_ambiguous_ocr_cannot_destroy_the_previous_subtitle_layout():
    previous = {'line_roi': {'x': 0.15, 'y': 0.8, 'w': 0.7, 'h': 0.1}, 'validated_by': 'USER_REVIEW'}
    assert validated_subtitle_profile({}, [], previous) == previous
    bad = {'subtitle_profile': {'line_roi': {'x': 0.15, 'y': 0.55, 'w': 0.7, 'h': 0.06}}}
    assert validated_subtitle_profile(bad, [parent(action='KEEP')], previous) == previous
    good = {'subtitle_profile': {'line_roi': previous['line_roi']}}
    result = validated_subtitle_profile(good, [parent(dialogue_evidence={'audio_ocr_similarity': 0.95})], previous)
    assert result['validated_by'] == 'AUDIO_OCR_TEXT_MATCH_UNCALIBRATED'


def test_native_clause_boundaries_do_not_reuse_the_asr_processing_window():
    value = parent(zh_text='你好，再见。', words=[
        {'t': '你好', 's': 1100, 'e': 1800}, {'t': '再见', 's': 3200, 'e': 3800}])
    result = sentence_segments(value, split_commas=True)
    assert [(s['start_ms'], s['end_ms']) for s in result] == [(1100, 1800), (3200, 3800)]
    assert len({s['id'] for s in result}) == 2
    one = sentence_segments(parent(zh_text='你好。', words=[{'t': '你好', 's': 1100, 'e': 1800}]), split_commas=True)
    assert (one[0]['start_ms'], one[0]['end_ms']) == (1100, 1800)
