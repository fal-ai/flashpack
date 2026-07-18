"""Tests for the opt-in CPU parallel read path (``FLASHPACK_CPU_PARALLEL_READ``).

Hermetic and CPU-only: they run in CI on every push/PR. The eager CPU reader
must produce byte-identical storage to the default lazy-mmap path for
mixed-dtype packs, across O_DIRECT/buffered and single-/multi-chunk plans,
and must stay OFF without the opt-in env.
"""

import os

import pytest
import torch
from flashpack.deserialization import read_flashpack_file
from flashpack.serialization import pack_to_file

posix_only = pytest.mark.skipif(
    os.name != "posix", reason="parallel reader is POSIX-only"
)


def _mixed_state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "a": torch.randn(1_000_003),  # fp32, odd length -> unaligned tail
        "b": torch.randn(2048, 512).to(torch.bfloat16),
        "c": torch.randint(0, 127, (777_777,), dtype=torch.int8),
        "d": torch.randint(0, 2, (12_345,)).bool(),
    }


@posix_only
@pytest.mark.parametrize("direct_io", ["1", "0"])
@pytest.mark.parametrize("chunk_bytes", ["8192", str(64 * 1024 * 1024)])
def test_cpu_parallel_matches_mmap(tmp_path, monkeypatch, direct_io, chunk_bytes):
    path = str(tmp_path / "pack.flashpack")
    pack_to_file(_mixed_state_dict(), path, target_dtype=None)

    mmap_storage, _ = read_flashpack_file(path, device="cpu")

    monkeypatch.setenv("FLASHPACK_CPU_PARALLEL_READ", "1")
    monkeypatch.setenv("FLASHPACK_DIRECT_IO", direct_io)
    monkeypatch.setenv("FLASHPACK_READ_CHUNK_BYTES", chunk_bytes)
    monkeypatch.setenv("FLASHPACK_READ_THREADS", "4")
    eager_storage, _ = read_flashpack_file(path, device="cpu")

    assert len(eager_storage.blocks) == len(mmap_storage.blocks)
    # materialized RAM tensors, not mmap views
    assert eager_storage.backing_arrays is None
    for i, (ref, got) in enumerate(zip(mmap_storage.blocks, eager_storage.blocks)):
        assert got.dtype == ref.dtype
        assert got.device.type == "cpu"
        # base addresses are 4K-aligned so O_DIRECT covers whole macroblocks
        assert got.view(torch.uint8).data_ptr() % 4096 == 0
        assert torch.equal(
            ref.view(torch.uint8), got.view(torch.uint8)
        ), f"macroblock {i} differs from mmap reference"


@posix_only
def test_cpu_parallel_storage_is_writable(tmp_path, monkeypatch):
    """Eager blocks own their memory — writing must not touch the file."""
    path = str(tmp_path / "pack.flashpack")
    pack_to_file({"w": torch.randn(4096)}, path, target_dtype=None)

    monkeypatch.setenv("FLASHPACK_CPU_PARALLEL_READ", "1")
    storage, _ = read_flashpack_file(path, device="cpu")
    before = os.path.getmtime(path)
    storage.blocks[0].zero_()
    assert os.path.getmtime(path) == before
    assert storage.blocks[0].abs().sum().item() == 0


def test_cpu_parallel_default_off(tmp_path, monkeypatch):
    """Without the opt-in env the CPU path must stay lazy mmap views."""
    monkeypatch.delenv("FLASHPACK_CPU_PARALLEL_READ", raising=False)
    path = str(tmp_path / "pack.flashpack")
    pack_to_file({"w": torch.randn(4096)}, path, target_dtype=None)
    storage, _ = read_flashpack_file(path, device="cpu")
    assert storage.backing_arrays is not None


@posix_only
def test_page_cache_fraction_detects_warm_file(tmp_path):
    """Regression: mincore was called without argtypes, so ctypes mangled the
    64-bit address and the fraction was 0.0 for EVERY file — the O_DIRECT
    gate never saw a warm page cache and warm loads paid the direct-IO
    penalty (~5x on a hot pack)."""
    from flashpack.parallel_read import _page_cache_resident_fraction

    path = str(tmp_path / "blob.bin")
    data = os.urandom(8 * 1024 * 1024)
    with open(path, "wb") as f:
        f.write(data)
    with open(path, "rb") as f:
        f.read()  # fault everything in
    frac = _page_cache_resident_fraction(path, len(data))
    assert frac > 0.9, f"warm file reported resident fraction {frac}"
