"""Tests for the distributed (rank0-read + broadcast) load path.

This path ships in several production apps but had no coverage. It runs here
on the gloo backend with 2 CPU ranks (file:// rendezvous, no GPUs needed):
the collective plumbing, the byte-view broadcast, and the end-to-end
``assign_from_file(use_distributed_loading=True)`` flow are all
backend-agnostic; only the transport differs from prod's NCCL.

The byte-view broadcast (``_broadcast_storage``) is load-bearing: torch's
collective dtype maps do not cover every dtype flashpack stores (NCCL lacks
``float8_e8m0fnu`` -- the mxfp8 scale dtype -- and gloo lacks the whole
float8 family), so broadcasting native-dtype blocks crashes on quantized
packs. These tests broadcast float8/bfloat16 blocks through gloo, which only
works through the uint8 view.
"""

import os
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flashpack import deserialization
from flashpack.constants import DEFAULT_SHARD_STRATEGY, SHARD_STRATEGIES
from flashpack.deserialization import (
    FlashTensorStorage,
    MacroblockSpec,
    _broadcast_storage,
    _plan_windows,
    _shard_range,
    assign_from_file,
    iterate_from_flash_tensor,
    read_flashpack_file_distributed,
    resolve_shard_strategy,
)
from flashpack.serialization import pack_to_file

# The gloo file:// rendezvous with an explicit loopback interface is
# POSIX-shaped (Windows has no "lo" device: "Unable to find address for:
# lo"); the path is covered on Linux/macOS, same policy as the O_DIRECT
# integrity tests.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="gloo loopback rendezvous is POSIX-only in these tests",
)

_WORLD = 2


class _TwoParam(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(512, 33, dtype=torch.bfloat16))
        self.b = torch.nn.Parameter(torch.zeros(257))


def _source_state() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(3)
    return {
        "a": torch.randn(512, 33, generator=g).to(torch.bfloat16),
        "b": torch.randn(257, generator=g),
    }


def _init(rank: int, init_file: str) -> None:
    # Bind gloo to loopback explicitly: its default interface discovery
    # resolves the hostname, which fails on sandboxed/misconfigured hosts.
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD,
    )


def _bcast_worker(rank: int, init_file: str) -> None:
    _init(rank, init_file)
    try:
        dtypes = [torch.bfloat16, torch.float32]
        if hasattr(torch, "float8_e4m3fn"):
            dtypes.append(torch.float8_e4m3fn)
        blocks = []
        for i, dt in enumerate(dtypes):
            src = torch.arange(64, dtype=torch.int32) + 7 * i
            block = src.to(torch.uint8).view(torch.uint8).clone()
            if rank != 0:
                block.zero_()
            blocks.append(block.view(dt) if dt != torch.uint8 else block)
        storage = FlashTensorStorage(blocks=blocks)
        _broadcast_storage(storage, src=0)
        for i, block in enumerate(storage.blocks):
            expect = (torch.arange(64, dtype=torch.int32) + 7 * i).to(torch.uint8)
            got = block.view(torch.uint8)
            assert torch.equal(got, expect), f"rank {rank} block {i} mismatch"
    finally:
        dist.destroy_process_group()


def test_broadcast_storage_is_dtype_agnostic(tmp_path) -> None:
    mp.spawn(
        _bcast_worker,
        args=(str(tmp_path / "rdv1"),),
        nprocs=_WORLD,
        join=True,
    )


def _assign_worker(rank: int, init_file: str, pack_path: str) -> None:
    _init(rank, init_file)
    try:
        model = _TwoParam()
        assign_from_file(
            model,
            pack_path,
            device="cpu",
            use_distributed_loading=True,
        )
        state = _source_state()
        assert torch.equal(model.a.data, state["a"]), f"rank {rank} a mismatch"
        assert torch.equal(model.b.data, state["b"]), f"rank {rank} b mismatch"
    finally:
        dist.destroy_process_group()


def test_assign_from_file_distributed_end_to_end(tmp_path) -> None:
    pack = str(tmp_path / "pack.flashpack")
    pack_to_file(_source_state(), pack, None)
    mp.spawn(
        _assign_worker,
        args=(str(tmp_path / "rdv2"), pack),
        nprocs=_WORLD,
        join=True,
    )


def _read_dist_worker(rank: int, init_file: str, pack_path: str) -> None:
    _init(rank, init_file)
    try:
        storage, meta = read_flashpack_file_distributed(pack_path, device="cpu")
        got = dict(iterate_from_flash_tensor(storage, meta))
        state = _source_state()
        assert set(got) == set(state)
        for name, tensor in state.items():
            assert torch.equal(got[name], tensor), f"rank {rank} {name} mismatch"
    finally:
        dist.destroy_process_group()


def test_read_flashpack_file_distributed(tmp_path) -> None:
    pack = str(tmp_path / "pack.flashpack")
    pack_to_file(_source_state(), pack, None)
    mp.spawn(
        _read_dist_worker,
        args=(str(tmp_path / "rdv3"), pack),
        nprocs=_WORLD,
        join=True,
    )


def test_read_distributed_requires_process_group(tmp_path) -> None:
    pack = str(tmp_path / "pack.flashpack")
    pack_to_file(_source_state(), pack, None)
    assert not dist.is_initialized()
    with pytest.raises(RuntimeError, match="process group"):
        read_flashpack_file_distributed(pack, device="cpu")


def test_shard_range_covers_disjoint_aligned() -> None:
    """The contiguous strategy's shards must tile the block, stay aligned, and
    collapse empty for blocks too small to divide."""
    for length in (0, 1, 4095, 4096, 8192, 67_584, 1_000_000, 40 * 1024 * 1024):
        for world in (1, 2, 4, 8):
            ranges = [_shard_range(length, world, r) for r in range(world)]
            pos = 0
            for lo, hi in ranges:
                assert lo == pos or lo == hi  # empty shards collapse in place
                assert lo % 4096 == 0
                pos = max(pos, hi)
            assert pos == length
            for _, hi in ranges[:-1]:
                assert hi % 4096 == 0 or hi == length


def test_resolve_shard_strategy(monkeypatch) -> None:
    for name in SHARD_STRATEGIES:
        assert resolve_shard_strategy(name) == name
        assert resolve_shard_strategy(name.upper()) == name
    with pytest.raises(ValueError, match="unknown shard strategy"):
        resolve_shard_strategy("sideways")
    # the package default is pinned: flipping it is a deliberate decision that
    # must show up in this test, not ride in silently with a refactor
    monkeypatch.delenv("FLASHPACK_SHARD_STRATEGY", raising=False)
    assert resolve_shard_strategy(None) == DEFAULT_SHARD_STRATEGY == "contiguous"


def test_shard_strategy_env_override(monkeypatch) -> None:
    for name in SHARD_STRATEGIES:
        monkeypatch.setenv("FLASHPACK_SHARD_STRATEGY", name)
        assert resolve_shard_strategy(None) == name
        # an explicit argument still wins over the environment
        other = next(s for s in SHARD_STRATEGIES if s != name)
        assert resolve_shard_strategy(other) == other


def test_plan_windows_tiles_blocks_exactly() -> None:
    """The window plan must tile every block with no gap or overlap, keep every
    shard equal-sized within a window (AllGather requires that), stay 4096
    aligned, and leave only a sub-``world * 4096`` remainder unsharded."""
    lengths = (0, 1, 4095, 4096, 8192, 67_584, 1_000_000, 40 * 1024 * 1024)
    for length in lengths:
        for world in (1, 2, 4, 8):
            specs = [
                MacroblockSpec(
                    dtype=torch.uint8,
                    offset_bytes=0,
                    length_bytes=length,
                    length_elems=length,
                )
            ]
            windows = _plan_windows(specs, world, shard_bytes=8192)
            pos = 0
            for window in windows:
                assert window.base == pos, "windows must tile without gaps"
                assert window.base % 4096 == 0
                if window.shard_bytes:
                    assert window.shard_bytes % 4096 == 0
                    # equal shards, exactly covering the window
                    assert window.span == world * window.shard_bytes
                else:
                    # the only unsharded remainder, and it is small
                    assert window is windows[-1]
                    assert window.span < world * 4096
                pos += window.span
            assert pos == length, "windows must cover the block exactly"

    # every rank's shard of a window is disjoint and together they cover it
    specs = [
        MacroblockSpec(
            dtype=torch.uint8,
            offset_bytes=0,
            length_bytes=1_000_000,
            length_elems=1_000_000,
        )
    ]
    for world in (2, 4, 8):
        for window in _plan_windows(specs, world, shard_bytes=8192):
            if not window.shard_bytes:
                continue
            covered = [
                (window.base + r * window.shard_bytes, window.shard_bytes)
                for r in range(world)
            ]
            assert covered[0][0] == window.base
            assert (
                covered[-1][0] + covered[-1][1] == window.base + window.span
            ), "shards must cover the window"


def _sharded_source() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    state = {
        # big enough that both ranks get a real shard (>= 2 x 4096 bytes)
        "big.bf16": torch.randn(1024, 512, generator=g).to(torch.bfloat16),
        "big.fp32": torch.randn(600, 512, generator=g),
        # small enough that only rank 0 gets bytes (empty-shard path)
        "tiny.fp32": torch.randn(63, generator=g),
    }
    if hasattr(torch, "float8_e4m3fn"):
        state["q.fp8"] = torch.randn(4096, 64, generator=g).to(torch.float8_e4m3fn)
    return state


def _read_sharded_worker(
    rank: int, init_file: str, pack_path: str, strategy: str
) -> None:
    _init(rank, init_file)
    try:
        storage, meta = read_flashpack_file_distributed(
            pack_path, device="cpu", sharded=True, shard_strategy=strategy
        )
        got = dict(iterate_from_flash_tensor(storage, meta))
        state = _sharded_source()
        assert set(got) == set(state)
        for name, tensor in state.items():
            assert torch.equal(
                got[name].view(torch.uint8), tensor.contiguous().view(torch.uint8)
            ), f"rank {rank} {name} mismatch"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strategy", SHARD_STRATEGIES)
def test_read_flashpack_file_distributed_sharded(tmp_path, strategy) -> None:
    """Both shard strategies must deliver byte-identical payloads to every rank."""
    pack = str(tmp_path / "pack.flashpack")
    pack_to_file(_sharded_source(), pack, None)
    mp.spawn(
        _read_sharded_worker,
        args=(str(tmp_path / f"rdv4-{strategy}"), pack, strategy),
        nprocs=_WORLD,
        join=True,
    )


_real_inplace_probe = deserialization._supports_inplace_allgather


def _staged_sharded_worker(rank: int, init_file: str, pack_path: str) -> None:
    _init(rank, init_file)
    try:
        # gloo (and NCCL) accept the in-place gather, so the staged path -- the
        # fallback for a backend that does not -- would otherwise never run.
        # Force it and require byte-identical results from it too. The probe
        # is only consulted by the windows strategy, so it must be selected
        # explicitly: with the package default (contiguous) this test would
        # silently not exercise the staged branch at all.
        deserialization._supports_inplace_allgather = lambda device: False
        storage, meta = read_flashpack_file_distributed(
            pack_path, device="cpu", sharded=True, shard_strategy="windows"
        )
        got = dict(iterate_from_flash_tensor(storage, meta))
        for name, tensor in _sharded_source().items():
            assert torch.equal(
                got[name].view(torch.uint8), tensor.contiguous().view(torch.uint8)
            ), f"rank {rank} {name} mismatch on the staged path"
    finally:
        deserialization._supports_inplace_allgather = _real_inplace_probe
        dist.destroy_process_group()


def test_read_sharded_staged_fallback_matches(tmp_path) -> None:
    pack = str(tmp_path / "pack.flashpack")
    pack_to_file(_sharded_source(), pack, None)
    mp.spawn(
        _staged_sharded_worker,
        args=(str(tmp_path / "rdv6"), pack),
        nprocs=_WORLD,
        join=True,
    )


def _faulting_worker(
    rank: int, init_file: str, pack_path: str, fault: str, strategy: str
) -> None:
    """Rank 1 fails inside the synchronized envelope; both ranks must raise
    together, quickly -- not leave rank 0 blocked in a collective until the
    process-group timeout."""
    _init(rank, init_file)
    try:
        if rank == 1:

            def _boom(*args, **kwargs):
                raise RuntimeError("injected fault (test)")

            if fault == "read":
                deserialization.parallel_read_into_storage = _boom
            else:  # "alloc": the likeliest production failure (device OOM)
                deserialization._allocate_empty_storage = _boom
        start = time.monotonic()
        try:
            read_flashpack_file_distributed(
                pack_path, device="cpu", sharded=True, shard_strategy=strategy
            )
        except RuntimeError as exc:
            elapsed = time.monotonic() - start
            if rank == 1:
                assert "injected fault" in str(exc), f"rank 1 got {exc!r}"
            else:
                assert "failed on another rank" in str(exc), f"rank 0 got {exc!r}"
            assert elapsed < 60, f"rank {rank} took {elapsed:.0f}s -- not synchronized"
        else:
            raise AssertionError(f"rank {rank}: load unexpectedly succeeded")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strategy", SHARD_STRATEGIES)
@pytest.mark.parametrize("fault", ["read", "alloc"])
def test_sharded_one_rank_failure_is_synchronized(tmp_path, fault, strategy) -> None:
    pack = str(tmp_path / "pack.flashpack")
    pack_to_file(_sharded_source(), pack, None)
    mp.spawn(
        _faulting_worker,
        args=(str(tmp_path / f"rdv7-{fault}-{strategy}"), pack, fault, strategy),
        nprocs=_WORLD,
        join=True,
    )


def _divergent_env_worker(rank: int, init_file: str, pack_path: str) -> None:
    """Sharding knobs set on only one rank must not desynchronize the load:
    rank 0's resolved config is broadcast and wins everywhere."""
    if rank == 1:
        os.environ["FLASHPACK_SHARD_STRATEGY"] = "windows"
        os.environ["FLASHPACK_SHARD_BYTES"] = "8192"
    _init(rank, init_file)
    try:
        storage, meta = read_flashpack_file_distributed(
            pack_path, device="cpu", sharded=True
        )
        got = dict(iterate_from_flash_tensor(storage, meta))
        for name, tensor in _sharded_source().items():
            assert torch.equal(
                got[name].view(torch.uint8), tensor.contiguous().view(torch.uint8)
            ), f"rank {rank} {name} mismatch under divergent env"
    finally:
        dist.destroy_process_group()


def test_sharded_divergent_env_does_not_desynchronize(tmp_path) -> None:
    pack = str(tmp_path / "pack.flashpack")
    pack_to_file(_sharded_source(), pack, None)
    mp.spawn(
        _divergent_env_worker,
        args=(str(tmp_path / "rdv8"), pack),
        nprocs=_WORLD,
        join=True,
    )


def _kill_switch_worker(rank: int, init_file: str, pack_path: str) -> None:
    """FLASHPACK_PARALLEL_READ=0 must reach the sharded path: the load falls
    back to rank-src broadcast mode (and still returns correct bytes) instead
    of silently keeping the parallel reader on."""
    os.environ["FLASHPACK_PARALLEL_READ"] = "0"
    _init(rank, init_file)
    try:
        storage, meta = read_flashpack_file_distributed(
            pack_path, device="cpu", sharded=True
        )
        got = dict(iterate_from_flash_tensor(storage, meta))
        for name, tensor in _sharded_source().items():
            assert torch.equal(
                got[name].view(torch.uint8), tensor.contiguous().view(torch.uint8)
            ), f"rank {rank} {name} mismatch under kill switch"
    finally:
        dist.destroy_process_group()


def test_sharded_respects_parallel_read_kill_switch(tmp_path) -> None:
    pack = str(tmp_path / "pack.flashpack")
    pack_to_file(_sharded_source(), pack, None)
    mp.spawn(
        _kill_switch_worker,
        args=(str(tmp_path / "rdv9"), pack),
        nprocs=_WORLD,
        join=True,
    )


def _assign_sharded_worker(
    rank: int, init_file: str, pack_path: str, strategy: str
) -> None:
    _init(rank, init_file)
    try:
        model = _TwoParam()
        assign_from_file(
            model,
            pack_path,
            device="cpu",
            use_distributed_loading=True,
            distributed_sharded=True,
            distributed_shard_strategy=strategy,
        )
        state = _source_state()
        assert torch.equal(model.a.data, state["a"]), f"rank {rank} a mismatch"
        assert torch.equal(model.b.data, state["b"]), f"rank {rank} b mismatch"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strategy", SHARD_STRATEGIES)
def test_assign_from_file_distributed_sharded(tmp_path, strategy) -> None:
    pack = str(tmp_path / "pack.flashpack")
    pack_to_file(_source_state(), pack, None)
    mp.spawn(
        _assign_sharded_worker,
        args=(str(tmp_path / f"rdv5-{strategy}"), pack, strategy),
        nprocs=_WORLD,
        join=True,
    )
