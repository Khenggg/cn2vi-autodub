import json

import pytest

from autodub.adapters import local_translation
from autodub.contracts import Segment


def test_translation_uses_short_aliases_and_maps_them_back():
    segments = [Segment(id="seg_000000011440", start_ms=0, end_ms=2000, zh_text="你好"),
                Segment(id="seg_000000013000", start_ms=2000, end_ms=4000, zh_text="再见")]
    sent = json.loads(local_translation._messages(segments, {}, {})[1]["content"])["segments"]
    assert [item["id"] for item in sent] == ["s1", "s2"]
    reply = json.dumps({"segments": [
        {"id": "s2", "subtitle_vi": "Tạm biệt", "dub_vi": "Tạm biệt", "emotion": "calm", "punctuation": "."},
        {"id": "s1", "subtitle_vi": "Xin chào", "dub_vi": "Xin chào", "emotion": "calm", "punctuation": "!"}]},
        ensure_ascii=False)
    result = local_translation._parse_completion(segments, reply, {})
    assert [(s.id, s.dub_vi) for s in result] == [("seg_000000011440", "Xin chào!"),
                                                  ("seg_000000013000", "Tạm biệt.")]
    broken = reply.replace('"s1"', '"seg_00000000011440"')
    with pytest.raises(local_translation.LocalTranslationError):
        local_translation._parse_completion(segments, broken, {})
