import sys
from types import SimpleNamespace

import pytest

from autodub.adapters.v2_speech import native_alignment_float32


def test_native_alignment_converts_bfloat16_and_restores_function_on_error(monkeypatch):
    calls = []

    class Tensor:
        def __init__(self, dtype='bfloat16'):
            self.dtype = dtype

        def float(self):
            return Tensor('float32')

        def contiguous(self):
            return self

    def original(log_probs, targets):
        assert log_probs.dtype == 'float32'
        calls.append(targets)
        raise RuntimeError('Other alignment error')

    functional = SimpleNamespace(forced_align=original)
    monkeypatch.setitem(sys.modules, 'torchaudio', SimpleNamespace(functional=functional))
    with pytest.raises(RuntimeError, match='Other alignment error'), native_alignment_float32():
        functional.forced_align(Tensor(), 'token_ids')
    assert calls == ['token_ids']
    assert functional.forced_align is original
