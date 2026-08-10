"""Unit tests for the batched-nvcomp ("ll") groundwork that is testable
without a GPU: the v2 chunk-start alignment format change (padding, footer
field, back-compat) and the ctypes binding's loader/ABI guards. The foreign
calls themselves need a device and libnvcomp and are exercised by the H200
validation app, not CI.
"""

import ctypes

import torch
from flashpack import _nvcomp_ll, serialization
from flashpack.constants import FPZ_HI_CHUNK_ALIGN_BYTES
from flashpack.deserialization import (
    _align_up,
    get_flashpack_file_metadata,
    iterate_from_flash_tensor,
    read_flashpack_file,
)
from flashpack.serialization import pack_to_file

def _state_with_unaligned_tail() -> dict[str, torch.Tensor]:
    # An odd bf16 element count makes the (single, partial) frame's half-plane
    # length odd -- NOT a multiple of the chunk alignment -- so the test
    # covers the padded-lo-plane case, not just full 32 MiB frames.
    generator = torch.Generator().manual_seed(7)
    return {
        "w": torch.randn(100_001, generator=generator).to(torch.bfloat16),
    }


def _fpz_blocks(meta: dict) -> list[dict]:
    return [b for b in meta["macroblocks"] if b.get("fpz")]


def test_v2_footer_records_hi_align(tmp_path) -> None:
    dest = str(tmp_path / "pack.flashpack")
    pack_to_file(_state_with_unaligned_tail(), dest, None, compress="fpz-bf16")
    blocks = _fpz_blocks(get_flashpack_file_metadata(dest))
    assert blocks, "expected at least one fpz block"
    for block in blocks:
        assert block["fpz"]["hi_align"] == FPZ_HI_CHUNK_ALIGN_BYTES


def test_v2_chunk_starts_are_aligned_on_disk(tmp_path) -> None:
    """Every compressed chunk's absolute file offset must be a multiple of
    the recorded alignment (this is what the batched GPU decoder's device
    pointer table relies on), including after an unaligned tail lo plane."""
    dest = str(tmp_path / "pack.flashpack")
    pack_to_file(_state_with_unaligned_tail(), dest, None, compress="fpz-bf16")
    meta = get_flashpack_file_metadata(dest)
    saw_unaligned_lo = False
    for block in _fpz_blocks(meta):
        align = int(block["fpz"]["hi_align"])
        assert align > 1
        for frame in block["fpz"]["frames"]:
            lo_len = int(frame["lo_len"])
            saw_unaligned_lo |= lo_len % align != 0
            start = (
                int(block["offset_bytes"])
                + int(frame["payload_off"])
                + _align_up(lo_len, align)
            )
            off = start
            for clen in frame["hi_chunks"]:
                assert off % align == 0
                off += _align_up(int(clen), align)
    assert saw_unaligned_lo, "test state must produce an unaligned lo plane"


def test_v2_padded_pack_roundtrips_exactly(tmp_path) -> None:
    dest = str(tmp_path / "pack.flashpack")
    state = _state_with_unaligned_tail()
    pack_to_file(state, dest, None, compress="fpz-bf16")
    storage, meta = read_flashpack_file(dest, device="cpu")
    out = dict(iterate_from_flash_tensor(storage, meta))
    assert set(out) == set(state)
    for name, tensor in state.items():
        assert torch.equal(out[name], tensor), name


def test_v2_unpadded_layout_backcompat(tmp_path, monkeypatch) -> None:
    """A pack written with hi_align=1 (the pre-alignment packed layout, as
    older readers/writers produced) must still read byte-exactly: the reader
    takes the alignment from the footer, never from the current constant."""
    monkeypatch.setattr(serialization, "FPZ_HI_CHUNK_ALIGN_BYTES", 1)
    dest = str(tmp_path / "pack_old.flashpack")
    state = _state_with_unaligned_tail()
    pack_to_file(state, dest, None, compress="fpz-bf16")
    for block in _fpz_blocks(get_flashpack_file_metadata(dest)):
        assert block["fpz"]["hi_align"] == 1
    storage, meta = read_flashpack_file(dest, device="cpu")
    out = dict(iterate_from_flash_tensor(storage, meta))
    for name, tensor in state.items():
        assert torch.equal(out[name], tensor), name


def test_align_up() -> None:
    assert _align_up(0, 16) == 0
    assert _align_up(1, 16) == 16
    assert _align_up(16, 16) == 16
    assert _align_up(17, 16) == 32
    assert _align_up(123, 1) == 123


def test_nvcomp_ll_abi_struct_sizes() -> None:
    """The C structs are passed BY VALUE -- their byte sizes are load-bearing
    ABI facts (64-byte opts struct, three size_t alignment requirements)."""
    assert ctypes.sizeof(_nvcomp_ll._ZstdDecompressOpts) == 64
    assert ctypes.sizeof(_nvcomp_ll._AlignmentRequirements) == 3 * ctypes.sizeof(
        ctypes.c_size_t
    )


def test_nvcomp_ll_load_degrades_to_none(monkeypatch) -> None:
    """Without the libnvcomp wheel (or with a bad override path), load()
    returns None instead of raising, and the result is cached."""
    monkeypatch.setattr(_nvcomp_ll, "_loaded", None)
    monkeypatch.setenv("FLASHPACK_LIBNVCOMP_PATH", "/nonexistent/libnvcomp.so")
    assert _nvcomp_ll.load() is None
    # Cached: a second call must not re-probe (flip the env to a still-bad
    # value and confirm the cached None is returned without error).
    monkeypatch.delenv("FLASHPACK_LIBNVCOMP_PATH")
    assert _nvcomp_ll.load() is None
    monkeypatch.setattr(_nvcomp_ll, "_loaded", None)


def test_nvcomp_ll_find_library_env_override(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "libnvcomp.so.5"
    fake.write_bytes(b"not a real library")
    monkeypatch.setenv("FLASHPACK_LIBNVCOMP_PATH", str(fake))
    assert _nvcomp_ll._find_libnvcomp() == str(fake)
    # A real path that is not a loadable library must degrade to None too.
    monkeypatch.setattr(_nvcomp_ll, "_loaded", None)
    assert _nvcomp_ll.load() is None
    monkeypatch.setattr(_nvcomp_ll, "_loaded", None)
