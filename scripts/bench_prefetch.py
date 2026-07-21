"""Benchmark the loader-aware prefetch (``flashpack.prefetch``).

The prefetch exists for one production shape: model servers spend tens of
seconds on imports / env setup before their first pack load. Started early,
the prefetch overlaps the filesystem read with that work, so the load call
itself runs against a hot page cache (the mincore gate then routes it onto
buffered reads at memory speed). This script measures exactly that, plus
the honesty cases (sequential prefetch is NOT supposed to win; an
immediately-following load must cancel the warm and regress nothing).

Scenarios (every load bit-verified against a lazy-mmap reference):

  load-only            evict -> load                       (baseline)
  prefetch-seq         evict -> prefetch.wait() -> load    (worst case, documented)
  prefetch-overlap     evict -> prefetch -> W s of "setup work" -> load
  work-then-load       evict -> W s of "setup work" -> load (fair overlap baseline)
  settle-cancel        evict -> prefetch -> load immediately (in-flight cancel)
  load-hot             fully hot file -> load              (upper bound)

Usage:
    python scripts/bench_prefetch.py --size-gb 8 --dir /tmp --device cuda
    python scripts/bench_prefetch.py --pack /data/pack.flashpack --device cuda \
        --work-seconds 10 --repeats 2 --json-out results.json

On network mounts (JuiceFS) page-cache eviction leaves the mount's own
node-local disk cache warm — that "fs-cache warm, page-cache cold" tier is
the cold-start-relevant one (a runner rarely boots on a node that has
never seen the pack). Pass ``--fresh-copy`` to also measure true
mount-cold loads from a unique copy of the pack per repeat.
"""

import argparse
import json
import os
import shutil
import time

import torch
from flashpack.deserialization import read_flashpack_file
from flashpack.prefetch import consume_prefetch, prefetch_flashpack_file
from flashpack.serialization import pack_to_file

GB = 1024**3


def build_pack(path: str, size_gb: float) -> None:
    torch.manual_seed(0)
    bf16_bytes = int(size_gb * GB * 0.97)
    per = (bf16_bytes // 2) // 16
    sd: dict[str, torch.Tensor] = {
        f"blocks.{i}.w": torch.randn(per, dtype=torch.bfloat16) for i in range(16)
    }
    sd["norm.weight"] = torch.randn(4096, dtype=torch.float32)
    sd["ids"] = torch.randint(0, 1 << 30, (1 << 20,), dtype=torch.int64)
    print(f"packing {size_gb:.1f} GB synthetic checkpoint -> {path}", flush=True)
    pack_to_file(sd, path, target_dtype=None)


def evict_page_cache(path: str) -> bool:
    if not hasattr(os, "posix_fadvise"):  # macOS: dev smoke-runs only
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def make_hot(path: str) -> None:
    with open(path, "rb") as f:
        while f.read(64 * 1024 * 1024):
            pass


def checksum(storage) -> list[int]:
    return [
        int(block.view(torch.uint8).to(torch.int64).sum().item())
        for block in storage.blocks
    ]


def busy_work(seconds: float) -> None:
    """CPU-bound stand-in for imports/env setup (keeps the GIL churning the
    way real import work does, unlike a plain sleep)."""
    deadline = time.monotonic() + seconds
    x = torch.randn(512, 512)
    while time.monotonic() < deadline:
        x = (x @ x).clamp(-1, 1)


def timed_load(path: str, device: str, eager_cpu: bool | None):
    t0 = time.perf_counter()
    storage, _ = read_flashpack_file(path, device=device, eager_cpu=eager_cpu)
    if device == "cuda":
        torch.cuda.synchronize()
    elif storage.backing_arrays is not None:
        # lazy mmap: fault everything in so the measurement is honest
        for block in storage.blocks:
            block.view(torch.uint8)[:: 4096].sum()
    return storage, time.perf_counter() - t0


def run_scenario(
    name: str,
    path: str,
    device: str,
    eager_cpu: bool | None,
    work_seconds: float,
    reference: list[int],
) -> dict:
    size = os.path.getsize(path)
    consume_prefetch(path)  # isolation between scenarios

    if name == "load-hot":
        make_hot(path)
    else:
        evicted = evict_page_cache(path)
        if not evicted:
            print(f"WARNING: could not evict {path}; {name} tier is unreliable")

    t_start = time.perf_counter()
    handle = None
    if name in ("prefetch-seq", "prefetch-overlap", "settle-cancel"):
        handle = prefetch_flashpack_file(path)
    if name == "prefetch-seq":
        handle.wait()
    if name in ("prefetch-overlap", "work-then-load"):
        busy_work(work_seconds)

    storage, load_s = timed_load(path, device, eager_cpu)
    total_s = time.perf_counter() - t_start

    ok = checksum(storage) == reference
    row = {
        "scenario": name,
        "load_s": round(load_s, 3),
        "total_s": round(total_s, 3),
        "load_gbps": round(size / GB / load_s, 2) if load_s > 0 else None,
        "verified": ok,
    }
    if handle is not None:
        row["prefetch"] = {
            "done": handle.done,
            "cancelled": handle.cancelled,
            "progress": round(handle.progress, 3),
        }
    del storage
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", help="existing .flashpack file (else synthesize)")
    ap.add_argument("--size-gb", type=float, default=8.0)
    ap.add_argument("--dir", default="/tmp", help="where to synthesize the pack")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--eager-cpu",
        action="store_true",
        help="use eager_cpu=True for CPU loads (default: lazy mmap + touch)",
    )
    ap.add_argument("--work-seconds", type=float, default=10.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument(
        "--fresh-copy",
        action="store_true",
        help="also measure mount-cold loads from a unique copy per repeat",
    )
    ap.add_argument("--json-out")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")

    path = args.pack
    if not path:
        path = os.path.join(args.dir, f"bench_prefetch_{args.size_gb:g}gb.flashpack")
        if not os.path.exists(path):
            build_pack(path, args.size_gb)

    eager_cpu = True if (args.device == "cpu" and args.eager_cpu) else None

    print(f"reference load for bit-verification ({path})", flush=True)
    ref_storage, _ = read_flashpack_file(path, device="cpu")
    reference = checksum(ref_storage)
    del ref_storage

    scenarios = [
        "load-only",
        "work-then-load",
        "prefetch-seq",
        "prefetch-overlap",
        "settle-cancel",
        "load-hot",
    ]
    results: list[dict] = []
    for rep in range(args.repeats):
        for name in scenarios:
            row = run_scenario(
                name, path, args.device, eager_cpu, args.work_seconds, reference
            )
            row["repeat"] = rep
            results.append(row)
            print(json.dumps(row), flush=True)

    if args.fresh_copy:
        for rep in range(args.repeats):
            fresh = f"{path}.fresh{rep}"
            print(f"copying pack for mount-cold tier -> {fresh}", flush=True)
            shutil.copyfile(path, fresh)
            evict_page_cache(fresh)
            for name in ("load-only", "prefetch-overlap"):
                row = run_scenario(
                    name, fresh, args.device, eager_cpu, args.work_seconds, reference
                )
                row["repeat"] = rep
                row["tier"] = "mount-cold"
                results.append(row)
                print(json.dumps(row), flush=True)
                os.unlink(fresh)
                if name != "prefetch-overlap":
                    shutil.copyfile(path, fresh)
                    evict_page_cache(fresh)

    print("\n== summary (median load_s per scenario) ==")
    by: dict[tuple, list] = {}
    for r in results:
        by.setdefault((r.get("tier", "fadvise-cold"), r["scenario"]), []).append(r)
    for (tier, name), rows in by.items():
        loads = sorted(x["load_s"] for x in rows)
        totals = sorted(x["total_s"] for x in rows)
        med_l = loads[len(loads) // 2]
        med_t = totals[len(totals) // 2]
        flags = "" if all(x["verified"] for x in rows) else "  !! VERIFY FAILED"
        print(f"{tier:12s} {name:18s} load {med_l:7.2f}s  total {med_t:7.2f}s{flags}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
