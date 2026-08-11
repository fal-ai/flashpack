"""Compare the sharded distributed-read strategies on one pack and one node.

Neither ``contiguous`` nor ``windows`` is known to dominate: which wins depends
on pack size, macroblock layout, world size, filesystem and fabric. This script
measures them where it matters -- the actual pack, on the actual node -- so the
choice can be made from data rather than from the default.

    torchrun --nproc_per_node=8 scripts/bench_distributed_shard_strategy.py \\
        --path /path/to/model.flashpack

Modes:

  every_rank   every rank reads the whole pack (no sharing; world x traffic)
  rank0_bcast  rank 0 reads it all, blocks broadcast from it
  contiguous   sharded, one contiguous 1/N slab per rank + owner broadcasts
  windows      sharded, interleaved superwindows + one AllGather per window

Every mode is checksummed over the full payload and compared across modes and
ranks: a faster mode that delivers different bytes is a failure, not a result.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import flashpack
import torch
import torch.distributed as dist
from flashpack.constants import SHARD_STRATEGY_CONTIGUOUS, SHARD_STRATEGY_WINDOWS
from flashpack.deserialization import (
    _build_macroblock_specs,
    get_flashpack_file_metadata,
    read_flashpack_file,
    read_flashpack_file_distributed,
)


def checksum(blocks: list[torch.Tensor]) -> int:
    """Full-coverage fingerprint: wrapping int64 sum over every byte."""
    total = 0
    for block in blocks:
        flat = block.reshape(-1).view(torch.uint8)
        n8 = flat.numel() - (flat.numel() % 8)
        acc = flat.narrow(0, 0, n8).view(torch.int64).sum().item()
        if n8 < flat.numel():
            acc += flat.narrow(0, n8, flat.numel() - n8).to(torch.int64).sum().item()
        total = (total + acc + flat.numel()) % (1 << 62)
    return total


def run_mode(mode: str, path: str, device: torch.device, meta: dict):
    if mode == "every_rank":
        storage, _ = read_flashpack_file(path, device=device, metadata=meta)
    elif mode == "rank0_bcast":
        storage, _ = read_flashpack_file_distributed(
            path, device=device, sharded=False, metadata=meta
        )
    else:
        storage, _ = read_flashpack_file_distributed(
            path, device=device, sharded=True, shard_strategy=mode, metadata=meta
        )
    return storage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument(
        "--modes",
        default=(f"rank0_bcast,{SHARD_STRATEGY_CONTIGUOUS},{SHARD_STRATEGY_WINDOWS}"),
        help="comma-separated; add every_rank for the no-sharing baseline",
    )
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl")
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo")

    meta = get_flashpack_file_metadata(args.path)
    specs = _build_macroblock_specs(meta)
    total = sum(spec.length_bytes for spec in specs)

    if rank == 0:
        print(
            f"[bench] flashpack {flashpack.__version__} torch {torch.__version__}\n"
            f"[bench] {args.path}\n"
            f"[bench] {len(specs)} macroblocks, payload {total / 1e9:.2f} GB, "
            f"world {world}, per-rank read {total / world / 1e9:.2f} GB",
            flush=True,
        )

    modes = [m for m in args.modes.split(",") if m]
    results: dict[str, list[float]] = {m: [] for m in modes}
    checksums: set[int] = set()

    for rnd in range(args.rounds):
        # rotate the order each round so cache warming does not always favour
        # whichever mode happens to run first
        order = modes[rnd % len(modes) :] + modes[: rnd % len(modes)]
        for mode in order:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
            dist.barrier()
            started = time.perf_counter()
            storage = run_mode(mode, args.path, device, meta)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dist.barrier()
            elapsed = time.perf_counter() - started

            digest = checksum(storage.blocks)
            del storage
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            worst = torch.tensor([elapsed], device=device, dtype=torch.float64)
            dist.all_reduce(worst, op=dist.ReduceOp.MAX)  # slowest rank gates
            elapsed = worst.item()

            low = torch.tensor([digest], device=device, dtype=torch.int64)
            high = low.clone()
            dist.all_reduce(low, op=dist.ReduceOp.MIN)
            dist.all_reduce(high, op=dist.ReduceOp.MAX)
            agree = low.item() == high.item()
            checksums.add(low.item() if agree else -1)

            results[mode].append(elapsed)
            if rank == 0:
                print(
                    f"[bench] round {rnd} {mode:12s} {elapsed:7.2f}s "
                    f"= {total / 1e9 / elapsed:6.2f} GB/s per rank"
                    f"{'' if agree else '  *** RANKS DISAGREE ***'}",
                    flush=True,
                )

    if rank == 0:
        best = {m: min(v) for m, v in results.items() if v}
        print("\n[bench] === best of rounds ===", flush=True)
        for mode, seconds in sorted(best.items(), key=lambda kv: kv[1]):
            print(f"[bench]   {mode:12s} {seconds:7.2f}s", flush=True)
        identical = len(checksums) == 1 and -1 not in checksums
        print(
            "[bench] checksum: "
            + ("ALL MODES IDENTICAL" if identical else f"MISMATCH {checksums}"),
            flush=True,
        )
        print(
            "[bench] " + json.dumps({"best": best, "payload_bytes": total}), flush=True
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
