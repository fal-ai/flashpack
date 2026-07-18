"""Unit tests for the parallel O_DIRECT read path (CPU-testable pieces; the
full CUDA pipeline is exercised by scripts/bench_load_parallel.py on a GPU
host)."""

import os

import pytest
import torch
from flashpack.deserialization import MacroblockSpec
from flashpack.parallel_read import (
    _ALIGN,
    _plan_chunks,
    _read_chunk,
    parallel_read_supported,
)


def _spec(offset: int, nbytes: int) -> MacroblockSpec:
    return MacroblockSpec(
        dtype=torch.uint8, offset_bytes=offset, length_bytes=nbytes, length_elems=nbytes
    )


class TestPlanChunks:
    @pytest.mark.parametrize(
        "specs",
        [
            [_spec(0, 200 * 1024 * 1024)],  # aligned single block
            [_spec(100, 64 * 1024 * 1024)],  # unaligned head
            [_spec(0, 10), _spec(4096, 5000), _spec(9216, 3)],  # tiny blocks
            [_spec(128, 300_000_007)],  # unaligned + non-round size
        ],
    )
    def test_chunks_cover_each_block_exactly(self, specs) -> None:
        chunk_bytes = 64 * 1024 * 1024
        chunks = _plan_chunks(specs, chunk_bytes)
        for idx, spec in enumerate(specs):
            mine = [c for c in chunks if c[0] == idx]
            # contiguous in block space, starting at 0, covering length_bytes
            assert mine[0][2] == 0
            pos = 0
            for _, f_off, b_off, ln in mine:
                assert b_off == pos
                assert f_off == spec.offset_bytes + b_off
                assert 0 < ln <= chunk_bytes
                pos += ln
            assert pos == spec.length_bytes

    def test_odirect_alignment_invariant(self) -> None:
        # every chunk except a block's sub-page head starts 4K-aligned in file
        # space (the O_DIRECT requirement _read_chunk relies on)
        specs = [_spec(100, 300 * 1024 * 1024), _spec(4096, 5)]
        chunks = _plan_chunks(specs, 64 * 1024 * 1024)
        for idx, spec in enumerate(specs):
            mine = [c for c in chunks if c[0] == idx]
            head = (-spec.offset_bytes) % _ALIGN
            for k, (_, f_off, b_off, ln) in enumerate(mine):
                if head and k == 0:
                    assert ln == min(head, spec.length_bytes)
                else:
                    assert f_off % _ALIGN == 0

    def test_no_chunk_exceeds_chunk_bytes(self) -> None:
        chunks = _plan_chunks([_spec(0, 1_000_000)], 4096)
        assert all(c[3] <= 4096 for c in chunks)


@pytest.mark.skipif(
    os.name != "posix", reason="_read_chunk uses os.preadv (POSIX-only)"
)
class TestReadChunk:
    def test_buffered_read_exact_bytes(self, tmp_path) -> None:
        payload = bytes(range(256)) * 512  # 128 KiB
        f = tmp_path / "blob.bin"
        f.write_bytes(payload)
        fd = os.open(str(f), os.O_RDONLY)
        try:
            buf = torch.empty(65536, dtype=torch.uint8)
            view = memoryview(buf.numpy())
            _read_chunk(None, fd, view, 777, 65536)
            assert bytes(view[:65536]) == payload[777 : 777 + 65536]
        finally:
            os.close(fd)

    def test_short_file_raises(self, tmp_path) -> None:
        f = tmp_path / "short.bin"
        f.write_bytes(b"x" * 16)
        fd = os.open(str(f), os.O_RDONLY)
        try:
            buf = torch.empty(64, dtype=torch.uint8)
            with pytest.raises(IOError):
                _read_chunk(None, fd, memoryview(buf.numpy()), 0, 64)
        finally:
            os.close(fd)


class TestSupported:
    def test_env_opt_out(self, monkeypatch) -> None:
        monkeypatch.setenv("FLASHPACK_PARALLEL_READ", "0")
        assert not parallel_read_supported(torch.device("cuda"))

    def test_cuda_posix_default_on(self, monkeypatch) -> None:
        monkeypatch.delenv("FLASHPACK_PARALLEL_READ", raising=False)
        assert parallel_read_supported(torch.device("cuda")) == (os.name == "posix")

    def test_cpu_never(self, monkeypatch) -> None:
        monkeypatch.delenv("FLASHPACK_PARALLEL_READ", raising=False)
        assert not parallel_read_supported(torch.device("cpu"))
