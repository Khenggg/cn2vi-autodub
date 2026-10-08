"""Identity-owned, independent subtitle and speech intervals with timing evidence."""
from __future__ import annotations

import hashlib
from difflib import SequenceMatcher

from autodub.contracts import Segment
from autodub.multimodal import lexical


def sentence_timeline(segments: list[dict], events: list[dict]) -> dict:
    output, issues, relations = [], [], []
    for raw in segments:
        parent = Segment.model_validate(raw).model_dump()
        if parent['action'] != 'DUB' or parent['context_provenance'].get('human_reviewed'):
            output.append(parent)
            continue
        source = lexical(parent['zh_text'])
        related = sorted((event for event in events if event.get('kind') == 'DIALOGUE'
            and event.get('score', 0) >= 0.7 and event['start_ms'] < parent['end_ms']
            and parent['start_ms'] < event['end_ms']), key=lambda e: (e['start_ms'], e['end_ms']))
        cursor, covered, children = 0, set(), []
        for event in related:
            caption = lexical(event['text'])
            if len(caption) < 2:
                continue
            match = SequenceMatcher(None, source[cursor:], caption, autojunk=False).find_longest_match()
            if match.size < 2 or match.size / len(caption) < 0.8:
                continue
            left_char, right_char = cursor + match.a, cursor + match.a + match.size
            left, right = event['start_ms'], event['end_ms']
            if right <= left:
                continue
            sid = parent['id'] + ':sentence:' + hashlib.sha256(
                f"{event['id']}:{left_char}:{right_char}".encode()).hexdigest()[:12]
            # Only native word timestamps whose text covers the complete parent can
            # own precise speech boundaries. OCR display times remain estimates.
            words, char_at = [], 0
            complete_words = (lexical(''.join(word['t'] for word in parent['words'])) == source
                and all(a['e'] <= b['s'] for a, b in zip(parent['words'], parent['words'][1:], strict=False)))
            if complete_words:
                for word in parent['words']:
                    word_end = char_at + len(lexical(word['t']))
                    if left_char <= char_at and word_end <= right_char:
                        words.append(word)
                    char_at = word_end
            native = bool(words) and lexical(''.join(w['t'] for w in words)) == source[left_char:right_char]
            speech_left, speech_right = (words[0]['s'], words[-1]['e']) if native else (left, right)
            evidence = {**parent['dialogue_evidence'], 'parent_segment_id': parent['id'],
                'ocr_event_ids': [event['id']], 'asr_char_range': [left_char, right_char],
                'speech_timing_estimated': not native, 'subtitle_timing_source': 'VIDEO_PTS',
                'source_asr_window_ms': [parent['start_ms'], parent['end_ms']]}
            child = {**parent, 'id': sid, 'zh_text': source[left_char:right_char],
                'asr_text': source[left_char:right_char], 'start_ms': speech_left, 'end_ms': speech_right,
                'subtitle_start_ms': left, 'subtitle_end_ms': right, 'words': words if native else [],
                'timing_source': 'NATIVE_WORDS' if native else 'OCR_EVENT_ESTIMATE',
                'dialogue_evidence': evidence}
            children.append(Segment.model_validate(child).model_dump())
            covered.update(range(left_char, right_char))
            relations.append({'segment_id': sid, 'parent_segment_id': parent['id'], 'ocr_event_id': event['id']})
            cursor = right_char
            if not native:
                issues.append({'segment_id': sid, 'code': 'SPEECH_TIMING_ESTIMATED_FROM_OCR',
                               'precise_speech_alignment': False})
        if not children:
            output.append(parent)
            if parent['timing_source'] != 'NATIVE_WORDS':
                issues.append({'segment_id': parent['id'], 'code': 'COARSE_SPEECH_WINDOW_REQUIRES_REVIEW'})
            continue
        output.extend(children)
        missing = ''.join(char for index, char in enumerate(source) if index not in covered)
        if missing:
            # Preserve unmatched audio text for review, rather than dubbing it at
            # the parent-window start or silently dropping part of the transcript.
            output.append({**parent, 'id': parent['id'] + ':unmatched', 'zh_text': missing, 'asr_text': missing,
                'action': 'KEEP', 'needs_review': True, 'words': [],
                'dialogue_evidence': {**parent['dialogue_evidence'], 'parent_segment_id': parent['id'],
                                      'ambiguity_reason': 'UNMATCHED_SENTENCE_TIMING'}})
            issues.append({'segment_id': parent['id'] + ':unmatched', 'code': 'UNMATCHED_SENTENCE_TIMING'})
    output.sort(key=lambda s: (s['start_ms'], s['id']))
    ids = [s['id'] for s in output]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate timeline segment identity')
    return {'schema_version': 1, 'segments': output, 'relations': relations, 'issues': issues,
            'speech_and_subtitle_timelines_independent': True, 'word_times_fabricated': False}


def timeline_manifest(segments: list[dict], ocr: dict, clips: dict, mix: dict) -> dict:
    events = {event['id']: event for event in ocr.get('events', [])}
    mixed = {row['segment_id']: row for row in mix.get('voice_added', [])}
    top, bottom, scaled_width, scaled_height = ocr.get('crop', [0, 1, 1, 1])
    scale_x, scale_y = ocr.get('width', 1) / scaled_width, (bottom - top) / scaled_height
    records = []
    for raw in segments:
        segment = Segment.model_validate(raw)
        boxes = []
        for eid in segment.dialogue_evidence.get('ocr_event_ids', []):
            for box in events.get(eid, {}).get('boxes', []):
                boxes.append([[round(x * scale_x, 3), round(top + y * scale_y, 3)] for x, y in box])
        records.append({'id': segment.id, 'source_text': segment.zh_text,
            'subtitle': {'start_ms': segment.subtitle_start_ms if segment.subtitle_start_ms is not None else segment.start_ms,
                         'end_ms': segment.subtitle_end_ms if segment.subtitle_end_ms is not None else segment.end_ms,
                         'text': segment.subtitle_vi, 'polygons_source_pixels': boxes},
            'speech': {'start_ms': segment.start_ms, 'end_ms': segment.end_ms,
                       'timing_source': segment.timing_source,
                       'estimated': segment.timing_source != 'NATIVE_WORDS',
                       'speaker_id': segment.speaker_id, 'audio_file': clips.get(segment.id),
                       'placement': mixed.get(segment.id)},
            'action': segment.action, 'evidence': segment.dialogue_evidence})
    return {'schema_version': 1, 'records': records, 'clock': 'SOURCE_PRESENTATION_TIMESTAMPS',
            'join_key': 'segment_id', 'voice_positions': 'ABSOLUTE_PCM_SAMPLE_OFFSETS'}


def validated_subtitle_profile(ocr, segments, previous):
    """A failed/ambiguous OCR pass must never erase or replace a known layout."""
    profile = ocr.get('subtitle_profile', {})
    supported = any(s.get('action') == 'DUB' and (
        s.get('dialogue_evidence', {}).get('audio_ocr_similarity', 0) >= 0.7
        or s.get('dialogue_evidence', {}).get('audio_ocr_sequence_coverage', 0) >= 0.7) for s in segments)
    if profile.get('line_roi') and supported:
        return {**profile, 'validated_by': 'AUDIO_OCR_TEXT_MATCH_UNCALIBRATED'}
    return dict(previous)
