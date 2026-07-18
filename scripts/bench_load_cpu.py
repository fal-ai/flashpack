"""Benchmark CPU-target load methods for flashpack packs.

The current CPU path (``read_flashpack_file(device="cpu")``) returns lazy
mmap views: the call is instant but every byte is paged in serially on
first touch. This script measures the cost of actually getting the weights
into RAM under candidate methods, all bit-verified against the mmap
reference:

  mmap-touch        current lazy path + faulting every page in (1 byte/page)
  mmap-clone        torch.from_numpy(mm).clone() per macroblock (eager, 1 thread)
  pread-1t          single-thread preadv into the destination buffer
  pread-Nt          N threads preadv buffered into disjoint slices of the
                    destination CPU tensor (no staging, no pinned memory)
  pread-Nt-direct   same, with O_DIRECT for the 4K-aligned chunk bodies
                    (destination is allocated 4K-aligned)

Usage:
    python scripts/bench_load_cpu.py --size-gb 8 --dir /tmp
    python scripts/bench_load_cpu.py --pack /path/model.flashpack --threads 8,16
    python scripts/bench_load_cpu.py --size-gb 4 --dir /tmp --cold

Cold mode evicts the file from the page cache (fsync + POSIX_FADV_DONTNEED)
before every measurement. On network mounts (JuiceFS) prefer a fresh copy
per row; see bench_load_parallel.py for the caveats.
"""

import argparse
import os
import queue
import threading
import time

import torch

from flashpack.deserialization import (
    _build_macroblock_specs,
    _open_memmaps,
    get_flashpack_file_metadata,
    read_flashpack_file,
)
from flashpack.serialization import pack_to_file

GB = 1024**3
ALIGN = 4096


def build_pack(path: str, size_gb: float) -> None:
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
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def alloc_dest(specs, aligned: bool):
    """Destination uint8 views per macroblock (+ keepalives)."""
    keep, views = [], []
    for spec in specs:
        raw = torch.empty(spec.length_bytes + (ALIGN if aligned else 0), dtype=torch.uint8)
        off = (-raw.data_ptr()) % ALIGN if aligned else 0
        keep.append(raw)
        views.append(raw.narrow(0, off, spec.length_bytes))
    return keep, views


def plan_chunks(specs, chunk_bytes):
    chunks = []
    for idx, spec in enumerate(specs):
        b_off, remaining = 0, spec.length_bytes
        head = (-spec.offset_bytes) % ALIGN
        if head:
            head = min(head, remaining)
            chunks.append((idx, spec.offset_bytes, 0, head))
            b_off += head
            remaining -= head
        while remaining > 0:
            ln = min(chunk_bytes, remaining)
            chunks.append((idx, spec.offset_bytes + b_off, b_off, ln))
            b_off += ln
            remaining -= ln
    return chunks


def read_chunk(fd_direct, fd_plain, view, f_off, ln):
    got = 0
    if fd_direct is not None and f_off % ALIGN == 0:
        body = ln & ~(ALIGN - 1)
        while got < body:
            try:
                n = os.preadv(fd_direct, [view[got:body]], f_off + got)
            except OSError:
                break
            if n <= 0:
                break
            got += n
    while got < ln:
        n = os.preadv(fd_plain, [view[got:ln]], f_off + got)
        if n <= 0:
            raise IOError(f"short read at {f_off}+{got}")
        got += n


def parallel_fill(path, specs, dest_views, n_threads, chunk_bytes, direct):
    mvs = [memoryview(v.numpy()) for v in dest_views]
    work: queue.SimpleQueue = queue.SimpleQueue()
    chunks = plan_chunks(specs, chunk_bytes)
    for c in chunks:
        work.put(c)
    n_threads = min(n_threads, max(1, len(chunks)))
    for _ in range(n_threads):
        work.put(None)
    errors: list[BaseException] = []

    aligned_dest = all(v.data_ptr() % ALIGN == 0 for v in dest_views)

    def reader():
        try:
            fd_plain = os.open(path, os.O_RDONLY)
            fd_direct = None
            if direct and aligned_dest and hasattr(os, "O_DIRECT"):
                try:
                    fd_direct = os.open(path, os.O_RDONLY | os.O_DIRECT)
                except OSError:
                    fd_direct = None
            try:
                while True:
                    item = work.get()
                    if item is None:
                        break
                    blk, f_off, b_off, ln = item
                    read_chunk(fd_direct, fd_plain, mvs[blk][b_off : b_off + ln], f_off, ln)
            finally:
                os.close(fd_plain)
                if fd_direct is not None:
                    os.close(fd_direct)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def verify(path, specs, dest_views):
    memmaps = _open_memmaps(path, specs)
    for i, (mm, view) in enumerate(zip(memmaps, dest_views)):
        ref = torch.from_numpy(mm.view("uint8"))
        if not torch.equal(ref, view):
            raise AssertionError(f"macroblock {i} differs from mmap reference")
    print("    verify: bit-exact vs mmap reference", flush=True)


RESULTS: list[tuple[str, str, float, float]] = []


def bench(label, path, size, fn, cold, do_verify, specs=None, views=None):
    if cold:
        evict_page_cache(path)
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    state = "cold" if cold else "warm"
    print(
        f"{label:<24s} [{state}]: {dt:7.2f}s  {size / dt / GB:6.2f} GB/s",
        flush=True,
    )
    RESULTS.append((label, state, dt, size / dt / GB))
    if do_verify and specs is not None and views is not None:
        verify(path, specs, views)
    return dt


def write_markdown(out_path: str, size: int) -> None:
    """Write the collected results as a GitHub-flavored markdown table
    (e.g. for $GITHUB_STEP_SUMMARY)."""
    with open(out_path, "a") as f:
        f.write(f"### flashpack CPU load benchmark ({size / GB:.2f} GB pack)\n\n")
        f.write("| method | cache | seconds | GB/s |\n|---|---|---|---|\n")
        for label, state, dt, rate in RESULTS:
            f.write(f"| `{label}` | {state} | {dt:.2f} | {rate:.2f} |\n")
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", help="existing .flashpack (skips synthesis)")
    ap.add_argument("--size-gb", type=float, default=8.0)
    ap.add_argument("--dir", default="/tmp")
    ap.add_argument("--threads", default="1,2,4,8,16,32")
    ap.add_argument("--chunk-mb", default="8,64")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--markdown-out", help="append results as a markdown table (CI job summary)")
    args = ap.parse_args()

    if args.pack:
        path = args.pack
    else:
        path = os.path.join(args.dir, f"bench_cpu_{args.size_gb:g}gb.flashpack")
        if not os.path.exists(path):
            build_pack(path, args.size_gb)

    size = os.path.getsize(path)
    meta = get_flashpack_file_metadata(path)
    specs = _build_macroblock_specs(meta)
    do_verify = not args.no_verify
    print(f"file={path} size={size / GB:.2f} GB cpus={os.cpu_count()}", flush=True)

    # 1. current lazy path: open mmaps + fault every page in (1 byte per page)
    def mmap_touch():
        memmaps = _open_memmaps(path, specs)
        acc = 0
        for mm in memmaps:
            acc += int(mm.view("uint8")[:: ALIGN].sum())
        return acc

    bench("mmap-touch", path, size, mmap_touch, args.cold, False)

    # 2. eager single-thread clone through the page cache
    clones = []

    def mmap_clone():
        memmaps = _open_memmaps(path, specs)
        for mm in memmaps:
            clones.append(torch.from_numpy(mm.view("uint8")).clone())

    bench("mmap-clone", path, size, mmap_clone, args.cold, False)
    del clones

    # 3. the shipped API paths: default (lazy mmap + touch) vs the opt-in
    #    eager parallel reader (FLASHPACK_CPU_PARALLEL_READ=1)
    def api_default():
        storage, _ = read_flashpack_file(path, device="cpu")
        acc = 0
        for b in storage.blocks:
            acc += int(b.view(torch.uint8)[::ALIGN].long().sum())
        return acc

    bench("api-cpu-default+touch", path, size, api_default, args.cold, False)

    api_result = {}

    def api_parallel():
        old = os.environ.get("FLASHPACK_CPU_PARALLEL_READ")
        os.environ["FLASHPACK_CPU_PARALLEL_READ"] = "1"
        try:
            api_result["storage"], _ = read_flashpack_file(path, device="cpu")
        finally:
            if old is None:
                os.environ.pop("FLASHPACK_CPU_PARALLEL_READ", None)
            else:
                os.environ["FLASHPACK_CPU_PARALLEL_READ"] = old

    bench("api-cpu-parallel", path, size, api_parallel, args.cold, False)
    if do_verify and api_result.get("storage") is not None:
        verify(
            path,
            specs,
            [b.view(torch.uint8) for b in api_result["storage"].blocks],
        )
    del api_result

    threads = [int(t) for t in args.threads.split(",")]
    chunks = [int(c) * 1024 * 1024 for c in args.chunk_mb.split(",")]

    for direct in (False, True):
        for chunk_bytes in chunks:
            for n in threads:
                if direct and n == 1:
                    continue
                keep, views = alloc_dest(specs, aligned=direct)
                label = f"pread-{n}t{'-direct' if direct else ''}-c{chunk_bytes // (1024 * 1024)}m"
                bench(
                    label,
                    path,
                    size,
                    lambda: parallel_fill(path, specs, views, n, chunk_bytes, direct),
                    args.cold,
                    do_verify,
                    specs,
                    views,
                )
                del keep, views

    if args.markdown_out:
        write_markdown(args.markdown_out, size)


if __name__ == "__main__":
    main()
