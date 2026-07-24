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

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flashpack.deserialization import (
    FlashTensorStorage,
    _broadcast_storage,
    assign_from_file,
)
from flashpack.serialization import pack_to_file

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
