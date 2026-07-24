"""Unit tests for the fused-interleave module's degradation behavior (the
kernel itself needs a GPU and is validated by the H200 probe/gate: numeric
parity against the strided path plus the pack checksum gate)."""

import torch
from flashpack import _interleave


def test_available_is_bool_and_cached() -> None:
    first = _interleave.fused_interleave_available()
    assert isinstance(first, bool)
    assert _interleave.fused_interleave_available() == first


def test_interleave_into_degrades_without_triton() -> None:
    if _interleave.fused_interleave_available():
        return  # covered by the GPU gate on hosts that have triton+CUDA
    out = torch.empty(8, dtype=torch.uint8)
    lo = torch.zeros(4, dtype=torch.uint8)
    hi = torch.ones(4, dtype=torch.uint8)
    assert _interleave.interleave_into(out, lo, hi) is False
