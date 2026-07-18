"""Adversarial integrity tests for the parallel read path.

``parallel_read_into_storage`` targets CUDA, but every stage that could
*silently corrupt data* is CPU-visible: chunk planning, O_DIRECT/buffered
read assembly, staging-buffer reuse, thread/queue orchestration, and error
draining. These tests fake only the CUDA stream/event plumbing and the
pinned-buffer pool (plain CPU tensors), then run the real reader threads
against real files and require byte-identical results.

Every payload is position-dependent random data and every gap between
macroblocks is filled with a sentinel byte, so a chunk landing at the wrong
destination, a swapped staging buffer, an off-by-one file offset, or a
dropped tail all fail the equality check.

``test_parallel_matches_legacy_on_cuda`` additionally verifies the real
CUDA pipeline against the legacy reader on GPU hosts (marked ``gpu``).
"""

import os
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from flashpack import parallel_read
from flashpack.deserialization import MacroblockSpec, read_flashpack_file
from flashpack.parallel_read import (
    _ALIGN,
    _plan_chunks,
    _read_chunk,
    parallel_read_into_storage,
)
from flashpack.serialization import pack_to_file

GAP_SENTINEL = 0xEE

posix_only = pytest.mark.skipif(
    os.name != "posix", reason="the parallel reader uses os.preadv (POSIX-only)"
)


class _FakeEvent:
    def record(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass


class _FakeStream:
    def __init__(self, device=None) -> None:
        self.device = device

    def wait_event(self, event) -> None:
        pass

    def synchronize(self) -> None:
        pass


class _FakeCuda:
    Event = _FakeEvent
    Stream = _FakeStream

    @staticmethod
    def current_stream(device=None) -> _FakeStream:
        return _FakeStream(device)

    @staticmethod
    def synchronize(device=None) -> None:
        pass

    @staticmethod
    @contextmanager
    def stream(stream):
        yield


@pytest.fixture()
def cpu_parallel_read(monkeypatch):
    """Run the real parallel reader on CPU: fake the CUDA plumbing, use plain
    CPU staging buffers, and default O_DIRECT off (tests opt back in)."""

    def fake_pool(n_threads: int, chunk_bytes: int) -> list:
        return [
            [torch.empty(chunk_bytes, dtype=torch.uint8) for _ in range(2)]
            for _ in range(n_threads)
        ]

    monkeypatch.setattr(parallel_read.torch, "cuda", _FakeCuda)
    monkeypatch.setattr(parallel_read, "_get_pinned_pool", fake_pool)
    monkeypatch.setenv("FLASHPACK_DIRECT_IO", "0")
    # Small defaults keep the fake staging pool tiny; tests override as needed.
    monkeypatch.setenv("FLASHPACK_READ_THREADS", "4")
    monkeypatch.setenv("FLASHPACK_READ_CHUNK_BYTES", "8192")
    return monkeypatch


def _make_file(
    tmp_path, specs: list[MacroblockSpec], seed: int = 0
) -> tuple[str, bytes]:
    """Write a file whose payload regions hold position-dependent random
    bytes and whose gaps hold GAP_SENTINEL."""
    total = max((s.offset_bytes + s.length_bytes) for s in specs)
    data = np.full(total, GAP_SENTINEL, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for spec in specs:
        data[spec.offset_bytes : spec.offset_bytes + spec.length_bytes] = rng.integers(
            0, 256, spec.length_bytes, dtype=np.uint8
        )
    path = str(tmp_path / "payload.bin")
    with open(path, "wb") as f:
        f.write(data.tobytes())
    return path, data.tobytes()


def _uint8_spec(offset: int, nbytes: int) -> MacroblockSpec:
    return MacroblockSpec(
        dtype=torch.uint8, offset_bytes=offset, length_bytes=nbytes, length_elems=nbytes
    )


def _run_and_verify(path: str, data: bytes, specs: list[MacroblockSpec]) -> None:
    blocks = [
        torch.full((spec.length_elems,), 0xAB, dtype=torch.uint8) for spec in specs
    ]
    parallel_read_into_storage(path, specs, blocks, torch.device("cpu"))
    for spec, block in zip(specs, blocks):
        expected = data[spec.offset_bytes : spec.offset_bytes + spec.length_bytes]
        assert bytes(block.numpy()) == expected


@posix_only
class TestByteIdentical:
    def test_single_aligned_block_many_chunks(
        self, tmp_path, cpu_parallel_read
    ) -> None:
        """4 MiB through 4 KiB chunks: ~1000 chunks over 8 threads, hammering
        staging-buffer reuse and queue ordering."""
        cpu_parallel_read.setenv("FLASHPACK_READ_THREADS", "8")
        cpu_parallel_read.setenv("FLASHPACK_READ_CHUNK_BYTES", "4096")
        specs = [_uint8_spec(0, 4 * 1024 * 1024)]
        path, data = _make_file(tmp_path, specs)
        _run_and_verify(path, data, specs)

    def test_unaligned_multiblock_with_gaps(self, tmp_path, cpu_parallel_read) -> None:
        """Blocks at unaligned offsets separated by sentinel-filled gaps: any
        off-by-one in file offsets pulls sentinel bytes into a block."""
        cpu_parallel_read.setenv("FLASHPACK_READ_THREADS", "4")
        cpu_parallel_read.setenv("FLASHPACK_READ_CHUNK_BYTES", "8192")
        specs = [
            _uint8_spec(100, 70_003),
            _uint8_spec(80_001, 4096),
            _uint8_spec(90_000, 3),
            _uint8_spec(94_208, 130_001),  # 4K-aligned start, odd length
        ]
        path, data = _make_file(tmp_path, specs)
        _run_and_verify(path, data, specs)

    @pytest.mark.parametrize("n_threads", [1, 3, 16])
    @pytest.mark.parametrize("chunk_bytes", [4096, 1 << 20])
    def test_thread_chunk_matrix(
        self, tmp_path, cpu_parallel_read, n_threads: int, chunk_bytes: int
    ) -> None:
        cpu_parallel_read.setenv("FLASHPACK_READ_THREADS", str(n_threads))
        cpu_parallel_read.setenv("FLASHPACK_READ_CHUNK_BYTES", str(chunk_bytes))
        specs = [_uint8_spec(37, 1_000_003), _uint8_spec(1_003_520, 500_000)]
        path, data = _make_file(tmp_path, specs, seed=n_threads)
        _run_and_verify(path, data, specs)

    def test_more_threads_than_chunks(self, tmp_path, cpu_parallel_read) -> None:
        cpu_parallel_read.setenv("FLASHPACK_READ_THREADS", "16")
        specs = [_uint8_spec(0, 100)]
        path, data = _make_file(tmp_path, specs)
        _run_and_verify(path, data, specs)

    def test_zero_length_block_among_others(self, tmp_path, cpu_parallel_read) -> None:
        specs = [_uint8_spec(0, 8192), _uint8_spec(8192, 0), _uint8_spec(8192, 4096)]
        path, data = _make_file(tmp_path, specs)
        _run_and_verify(path, data, specs)

    def test_typed_blocks_roundtrip_bits(self, tmp_path, cpu_parallel_read) -> None:
        """Non-uint8 destination blocks (bfloat16/float32) must receive the
        exact payload bits through their uint8 views."""
        specs = [
            MacroblockSpec(
                dtype=torch.bfloat16,
                offset_bytes=0,
                length_bytes=8192,
                length_elems=4096,
            ),
            MacroblockSpec(
                dtype=torch.float32,
                offset_bytes=8192,
                length_bytes=40_000,
                length_elems=10_000,
            ),
        ]
        path, data = _make_file(tmp_path, specs)
        blocks = [torch.empty(spec.length_elems, dtype=spec.dtype) for spec in specs]
        parallel_read_into_storage(path, specs, blocks, torch.device("cpu"))
        for spec, block in zip(specs, blocks):
            expected = data[spec.offset_bytes : spec.offset_bytes + spec.length_bytes]
            assert bytes(block.view(torch.uint8).numpy()) == expected

    def test_direct_io_enabled_still_byte_identical(
        self, tmp_path, cpu_parallel_read
    ) -> None:
        """With O_DIRECT allowed, every fallback (unsupported filesystem,
        unaligned buffers, short direct reads) must stay byte-identical."""
        cpu_parallel_read.setenv("FLASHPACK_DIRECT_IO", "1")
        cpu_parallel_read.setenv("FLASHPACK_READ_THREADS", "4")
        cpu_parallel_read.setenv("FLASHPACK_READ_CHUNK_BYTES", "65536")
        specs = [_uint8_spec(100, 1_000_000), _uint8_spec(1_003_520, 250_001)]
        path, data = _make_file(tmp_path, specs)
        _run_and_verify(path, data, specs)

    def test_partial_reads_assemble_correctly(
        self, tmp_path, cpu_parallel_read
    ) -> None:
        """A filesystem returning short reads (e.g. 1000 bytes at a time) must
        never drop or misplace bytes — the assembly loop has to resume from
        the exact file offset it stopped at."""
        real_preadv = os.preadv

        def short_preadv(fd, buffers, offset):
            view = buffers[0]
            capped = [view[: min(1000, len(view))]]
            return real_preadv(fd, capped, offset)

        cpu_parallel_read.setattr(os, "preadv", short_preadv)
        specs = [_uint8_spec(37, 300_007)]
        path, data = _make_file(tmp_path, specs)
        _run_and_verify(path, data, specs)


@posix_only
class TestErrorsNeverSilent:
    def test_truncated_file_raises(self, tmp_path, cpu_parallel_read) -> None:
        """A file shorter than the specs claim must raise, not return
        partially-filled storage."""
        specs = [_uint8_spec(0, 100_000)]
        path, _ = _make_file(tmp_path, specs)
        with open(path, "r+b") as f:
            f.truncate(50_000)
        blocks = [torch.zeros(100_000, dtype=torch.uint8)]
        with pytest.raises(IOError):
            parallel_read_into_storage(path, specs, blocks, torch.device("cpu"))

    def test_mid_read_failure_propagates_and_joins(
        self, tmp_path, cpu_parallel_read
    ) -> None:
        """An I/O error on one chunk must propagate to the caller after all
        reader threads drain — never swallowed."""
        specs = [_uint8_spec(0, 512 * 1024)]
        path, _ = _make_file(tmp_path, specs)
        cpu_parallel_read.setenv("FLASHPACK_READ_THREADS", "4")
        cpu_parallel_read.setenv("FLASHPACK_READ_CHUNK_BYTES", "4096")

        real_preadv = os.preadv
        poison_offset = 256 * 1024

        def failing_preadv(fd, buffers, offset):
            if offset == poison_offset:
                raise OSError(5, "simulated I/O error")
            return real_preadv(fd, buffers, offset)

        cpu_parallel_read.setattr(os, "preadv", failing_preadv)
        blocks = [torch.zeros(512 * 1024, dtype=torch.uint8)]
        with pytest.raises(OSError, match="simulated I/O error"):
            parallel_read_into_storage(path, specs, blocks, torch.device("cpu"))

    def test_bogus_direct_fd_falls_back_to_buffered(self, tmp_path) -> None:
        """OSError on the O_DIRECT descriptor (EBADF here; EINVAL on real
        filesystems) must degrade to a byte-identical buffered read."""
        payload = np.random.default_rng(7).integers(0, 256, 128 * 1024, dtype=np.uint8)
        f = tmp_path / "blob.bin"
        f.write_bytes(payload.tobytes())
        fd_plain = os.open(str(f), os.O_RDONLY)
        bogus_fd_direct = 2**20  # certainly not an open descriptor
        try:
            buf = torch.empty(65536, dtype=torch.uint8)
            view = memoryview(buf.numpy())
            _read_chunk(bogus_fd_direct, fd_plain, view, 4096, 65536)
            assert bytes(view[:65536]) == payload.tobytes()[4096 : 4096 + 65536]
        finally:
            os.close(fd_plain)

    def test_short_direct_read_completes_buffered(self, tmp_path, monkeypatch) -> None:
        """A direct descriptor that returns part of the aligned body and then
        stalls must hand off to the buffered descriptor at the exact resume
        offset."""
        payload = np.random.default_rng(11).integers(0, 256, 64 * 1024, dtype=np.uint8)
        f = tmp_path / "blob.bin"
        f.write_bytes(payload.tobytes())
        fd_plain = os.open(str(f), os.O_RDONLY)
        fd_direct = os.open(str(f), os.O_RDONLY)  # plays the direct role

        real_preadv = os.preadv
        direct_calls = {"n": 0}

        def stalling_preadv(fd, buffers, offset):
            if fd == fd_direct:
                direct_calls["n"] += 1
                if direct_calls["n"] == 1:
                    capped = [buffers[0][:_ALIGN]]  # one aligned page, then stall
                    return real_preadv(fd, capped, offset)
                return 0
            return real_preadv(fd, buffers, offset)

        monkeypatch.setattr(os, "preadv", stalling_preadv)
        try:
            buf = torch.empty(64 * 1024, dtype=torch.uint8)
            view = memoryview(buf.numpy())
            _read_chunk(fd_direct, fd_plain, view, 0, 64 * 1024)
            assert bytes(view) == payload.tobytes()
            assert direct_calls["n"] >= 2
        finally:
            os.close(fd_plain)
            os.close(fd_direct)


class TestPlanChunksFuzz:
    @pytest.mark.parametrize("seed", range(50))
    def test_random_layouts_cover_exactly_once(self, seed: int) -> None:
        """Chunks must tile every macroblock exactly once — no gaps, no
        overlap, no bleed across blocks — for arbitrary offsets and sizes."""
        rng = np.random.default_rng(seed)
        chunk_bytes = int(rng.choice([4096, 65536, 1 << 20]))
        cursor = int(rng.integers(0, 5000))
        specs = []
        for _ in range(int(rng.integers(1, 8))):
            cursor += int(rng.integers(0, 10000))
            length = int(rng.integers(0, 3_000_000))
            specs.append(_uint8_spec(cursor, length))
            cursor += length

        chunks = _plan_chunks(specs, chunk_bytes)
        for idx, spec in enumerate(specs):
            mine = [c for c in chunks if c[0] == idx]
            pos = 0
            for _, f_off, b_off, ln in mine:
                assert b_off == pos, "gap or overlap in block coverage"
                assert f_off == spec.offset_bytes + b_off, "file offset drift"
                assert 0 < ln <= chunk_bytes
                pos += ln
            assert pos == spec.length_bytes, "block not fully covered"
            head = (-spec.offset_bytes) % _ALIGN
            for k, (_, f_off, _, ln) in enumerate(mine):
                if not (head and k == 0):
                    assert f_off % _ALIGN == 0, "O_DIRECT alignment violated"


@pytest.mark.gpu
def test_parallel_matches_legacy_on_cuda(tmp_path, monkeypatch) -> None:
    """On a real CUDA host, the parallel reader must produce bit-identical
    storage to the legacy chunked reader for a mixed-dtype pack."""
    torch.manual_seed(0)
    state_dict = {
        "a": torch.randn(1_000_003),
        "b": torch.randn(2048, 512).to(torch.bfloat16),
        "c": torch.randint(0, 127, (777_777,), dtype=torch.int8),
    }
    path = str(tmp_path / "pack.flashpack")
    pack_to_file(state_dict, path, target_dtype=None)

    monkeypatch.setenv("FLASHPACK_PARALLEL_READ", "1")
    parallel_storage, _ = read_flashpack_file(path, device="cuda")
    monkeypatch.setenv("FLASHPACK_PARALLEL_READ", "0")
    legacy_storage, _ = read_flashpack_file(path, device="cuda")

    assert len(parallel_storage) == len(legacy_storage)
    for parallel_block, legacy_block in zip(
        parallel_storage.blocks, legacy_storage.blocks
    ):
        assert parallel_block.dtype == legacy_block.dtype
        assert torch.equal(
            parallel_block.view(torch.uint8), legacy_block.view(torch.uint8)
        )
