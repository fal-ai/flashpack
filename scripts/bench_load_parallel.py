"""Benchmark + validate the CUDA load paths: legacy mmap walk vs parallel
O_DIRECT reader (FLASHPACK_PARALLEL_READ).

Run on a GPU host. Builds (or reuses) a synthetic multi-dtype pack, then for
each variant: loads it to CUDA, reports seconds + GB/s, and bit-verifies the
loaded macroblocks against a CPU mmap reference (every variant is verified
unless --no-verify).

Usage:
    python scripts/bench_load_parallel.py --size-gb 8 --dir /tmp
    python scripts/bench_load_parallel.py --size-gb 24 --dir /data/bench --cold
    python scripts/bench_load_parallel.py --pack /data/some/model.flashpack

Cold runs (--cold): every measurement gets its own fresh copy of the pack
(JuiceFS/network mounts cache per FILE, so a fresh copy is cold for the
mount too), and the copy's pages are evicted from the page cache (fsync +
POSIX_FADV_DONTNEED) right before the measurement. Rows are labeled
cold/warm so mixed tables can't be misread. Caveat: copies written by THIS
host may remain in the mount's local write cache — for strictly cold
network-FS numbers, seed the copies from a different host.
"""

import argparse
import os
import shutil
import sys
import time

import torch
from flashpack.deserialization import read_flashpack_file
from flashpack.serialization import pack_to_file

GB = 1024**3
VARIANTS = {
    "legacy": {"FLASHPACK_PARALLEL_READ": "0"},
    "parallel": {"FLASHPACK_PARALLEL_READ": "1"},
    "parallel-buffered": {"FLASHPACK_PARALLEL_READ": "1", "FLASHPACK_DIRECT_IO": "0"},
}


def build_pack(path: str, size_gb: float) -> None:
    """Synthetic pack shaped like a real DiT artifact: dominant bf16 block plus
    small fp32/int64/bool blocks (multiple macroblocks, V4 layout)."""
    torch.manual_seed(0)
    bf16_bytes = int(size_gb * GB * 0.97)
    per = (bf16_bytes // 2) // 16
    sd: dict[str, torch.Tensor] = {
        f"blocks.{i}.w": torch.randn(per, dtype=torch.bfloat16) for i in range(16)
    }
    sd["norm.weight"] = torch.randn(4096, dtype=torch.float32)
    sd["ids"] = torch.randint(0, 1 << 30, (1 << 20,), dtype=torch.int64)
    sd["mask"] = torch.randint(0, 2, (1 << 20,), dtype=torch.int64).bool()
    print(f"packing {size_gb:.1f} GB synthetic checkpoint -> {path}", flush=True)
    pack_to_file(sd, path, target_dtype=None)


def evict_page_cache(path: str) -> bool:
    """Drop the file's pages from the OS page cache (a freshly written copy is
    fully resident, so without this a "cold" measurement is actually warm)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def verify(path: str, storage) -> None:
    """Bit-verify every CUDA macroblock against the CPU mmap reference."""
    ref, _ = read_flashpack_file(path, device="cpu")
    assert len(ref.blocks) == len(storage.blocks)
    for i, (cpu_b, gpu_b) in enumerate(zip(ref.blocks, storage.blocks)):
        a = cpu_b.view(torch.uint8) if cpu_b.dtype.is_floating_point else cpu_b
        b = gpu_b.cpu()
        b = b.view(torch.uint8) if b.dtype.is_floating_point else b
        if not torch.equal(a, b):
            raise AssertionError(f"macroblock {i} differs from reference")
    print("    verify: all macroblocks bit-exact vs CPU reference", flush=True)


def bench_one(path: str, variant: str, cold: bool, do_verify: bool) -> float:
    size = os.path.getsize(path)
    old_env = {k: os.environ.get(k) for k in VARIANTS[variant]}
    os.environ.update(VARIANTS[variant])
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        storage, _meta = read_flashpack_file(path, device="cuda")
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    rate = size / dt / GB
    print(
        f"{variant:<18s} [{'cold' if cold else 'warm'}]: {dt:7.2f}s  {rate:6.2f} GB/s",
        flush=True,
    )
    if do_verify:
        # verification re-reads on CPU (warms the cache) — it runs AFTER the
        # timed load, and cold mode gives every row its own copy
        verify(path, storage)
    del storage
    torch.cuda.empty_cache()
    return rate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", help="existing .flashpack to load (skips synthesis)")
    ap.add_argument("--size-gb", type=float, default=8.0)
    ap.add_argument("--dir", default="/tmp", help="where to build the synthetic pack")
    ap.add_argument(
        "--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS)
    )
    ap.add_argument(
        "--cold",
        action="store_true",
        help="one fresh, page-cache-evicted copy per measurement",
    )
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required (run on a GPU host)"

    if args.pack:
        base = args.pack
    else:
        base = os.path.join(args.dir, f"bench_{args.size_gb:.0f}gb.flashpack")
        if not os.path.exists(base):
            build_pack(base, args.size_gb)

    print(
        f"\nfile={base} size={os.path.getsize(base) / GB:.2f} GB "
        f"gpu={torch.cuda.get_device_name(0)} mode={'cold' if args.cold else 'warm'}\n",
        flush=True,
    )

    if not args.cold:
        # warm-up pass so the first row isn't the one paying cache population
        with open(base, "rb") as f:
            while f.read(256 * 1024 * 1024):
                pass

    copies: list[str] = []
    try:
        for k, variant in enumerate(args.variants):
            path = base
            if args.cold:
                path = f"{base}.cold{k}"
                shutil.copyfile(base, path)
                copies.append(path)
                if not evict_page_cache(path):
                    sys.exit(f"failed to evict page cache for {path}; aborting")
            bench_one(path, variant, cold=args.cold, do_verify=not args.no_verify)
    finally:
        for c in copies:
            try:
                os.remove(c)
            except OSError:
                pass


if __name__ == "__main__":
    main()
