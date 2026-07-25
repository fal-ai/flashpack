"""Fused byte-plane interleave kernel (Triton) for the fpz GPU read paths.

The split-plane format stores a frame as separate low/high byte planes; the
reader must produce out[2i] = lo[i], out[2i+1] = hi[i]. The torch expression
of that -- two strided copies (``out[0::2] = lo; out[1::2] = hi``) -- makes
two passes of 2-byte-stride writes, a worst-case memory pattern that doubles
device traffic and dominates the GPU-side tail of the hot-tier load.

The fused kernel makes one pass: read lo[i] and hi[i] once, write one
little-endian uint16 ``lo | hi << 8`` (stored through an int16 view -- same
bit pattern, and torch's int16 has full op support where uint16 does not).

Triton ships inside the torch Linux wheels (pytorch-triton), so no extra
dependency; on hosts without it (or on any kernel failure) callers fall back
to the strided copies. Enable with FLASHPACK_FPZ_FUSED_INTERLEAVE=1.
"""

from __future__ import annotations

import threading

import torch

# 8 elements per thread (BLOCK / (32 * warps) == 8): two 8-byte loads feed one
# 16-byte store per thread, the measured-optimal shape for byte-plane joins on
# Hopper (dietgpu's FloatTypeInfo<kBFloat16> vectorization). Fixed config on
# purpose -- autotuning would compile extra variants on a fresh container's
# empty triton cache, which lands exactly on the cold-start path.
_BLOCK = 2048
_NUM_WARPS = 8

_lock = threading.Lock()
_kernel_cache: tuple | None = None


def _get_kernel():
    """Build (or fetch) the JIT'd kernel; None when triton is unavailable."""
    global _kernel_cache
    with _lock:
        if _kernel_cache is not None:
            return _kernel_cache[0]
        try:
            import triton
            import triton.language as tl

            @triton.jit
            def _interleave_u8(lo_ptr, hi_ptr, out_ptr, n_elem, BLOCK: tl.constexpr):
                # int16 (not uint16) throughout: identical bit pattern, and
                # triton's unsigned integer paths have known pointer-arith
                # bugs (triton#6043) while int16 bitcast is the in-house
                # precedent. .to(tl.int16) on a uint8 source zero-extends.
                pid = tl.program_id(0).to(tl.int64)
                offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
                mask = offs < n_elem
                lo = tl.load(lo_ptr + offs, mask=mask, other=0).to(tl.int16)
                hi = tl.load(hi_ptr + offs, mask=mask, other=0).to(tl.int16)
                tl.store(out_ptr + offs, lo | (hi << 8), mask=mask)

            def _launch(lo: torch.Tensor, hi: torch.Tensor, out_u8: torch.Tensor):
                n = lo.numel()
                out16 = out_u8.view(torch.int16)
                grid = (triton.cdiv(n, _BLOCK),)
                _interleave_u8[grid](
                    lo, hi, out16, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS
                )

            _kernel_cache = (_launch,)
        except Exception:
            _kernel_cache = (None,)
        return _kernel_cache[0]


def fused_interleave_available() -> bool:
    return _get_kernel() is not None


def interleave_into(out_u8: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> bool:
    """Fused single-pass interleave of ``lo``/``hi`` uint8 planes into
    ``out_u8`` (contiguous, even byte offset, ``2 * lo.numel()`` bytes) on
    the CURRENT stream. Returns False (having written nothing) when the
    kernel is unavailable so the caller can run the strided fallback."""
    launch = _get_kernel()
    if launch is None:
        return False
    launch(lo, hi, out_u8)
    return True
