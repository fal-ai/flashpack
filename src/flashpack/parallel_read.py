"""Fused parallel file->GPU reader for flashpack payloads.

The legacy CUDA read path (``_copy_memmaps_into_storage``) walks the file
with a single thread: each chunk is page-faulted through a read-only mmap
into a pinned staging buffer, then copied H2D on a CUDA stream. Only the
H2D leg overlaps, so throughput is capped by one thread's ability to fault
pages in — on network filesystems (JuiceFS/FUSE, NFS) that is 1-3 GB/s,
far below both the filesystem's parallel ceiling and PCIe.

This module replaces that walk with N reader threads that ``preadv`` file
ranges directly into per-thread pinned buffers and issue async H2D copies
on per-thread CUDA streams. Reads use ``O_DIRECT`` when the file offset is
4K-aligned and the page cache does not already hold the file, which
bypasses the kernel-page-cache copy inside FUSE (measured ~2x over
buffered parallel reads on JuiceFS). Every fallback keeps byte-identical
results: filesystems without ``O_DIRECT`` degrade to buffered parallel
reads, and unaligned heads/tails of each range are read buffered.

Measured on a production H100 (JuiceFS /data, 23.8 GB bf16 pack,
file->GPU wall time including sync):

======================  ==========  ============
state                   legacy      parallel
======================  ==========  ============
node-cold (object st.)  ~90-130 s   ~22 s
fs-cache warm           9.3-11.0 s  1.9-2.6 s
page-cache hot          1.25 s      1.2 s
======================  ==========  ============

CPU targets can opt in to the same machinery (``FLASHPACK_CPU_PARALLEL_READ=1``):
reader threads then ``preadv`` straight into the destination CPU tensors —
no staging, no pinned memory, no streams — eagerly materializing the payload
instead of returning lazy mmap views. Measured on local NVMe (8 GB pack):
page-cache cold 1.2 s / 6.4 GB/s vs 3.2 s / 2.4 GB/s for faulting the mmap
in; page-cache warm 0.4 s / 18 GB/s (on par with the mmap fast path).

Tunables (environment):
- ``FLASHPACK_PARALLEL_READ=0``      disable (use legacy path)
- ``FLASHPACK_CPU_PARALLEL_READ=1``  enable eager parallel reads for CPU targets
- ``FLASHPACK_READ_THREADS``         reader threads (default 16)
- ``FLASHPACK_READ_CHUNK_BYTES``     chunk size (default 64 MiB)
- ``FLASHPACK_CONDITIONAL_RAMP=0``   disable slow direct-read ramp
- ``FLASHPACK_RAMP_THREADS``         ramp target (default 32)
- ``FLASHPACK_RAMP_THRESHOLD_S``     first-read threshold (default 0.2 s)
- ``FLASHPACK_DIRECT_IO=0``          never use O_DIRECT
- ``FLASHPACK_CACHE_PINNED=0``       free pinned staging buffers after load (CUDA)
- ``FLASHPACK_SAMPLE_PROBE=0``       trust mincore alone for the O_DIRECT gate
- ``FLASHPACK_SAMPLE_PROBE_MIN_GBPS``  hot threshold for the sample probe (default 5.0)
"""

import ctypes
import mmap as mmap_module
import os
import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from .utils import effective_read_threads

if TYPE_CHECKING:
    from .deserialization import MacroblockSpec

__all__ = [
    "parallel_read_supported",
    "parallel_read_into_storage",
    "release_pinned_pool",
]

_ALIGN = 4096
_BUFFERS_PER_THREAD = 1
_DEFAULT_READ_THREADS = 16
_DEFAULT_RAMP_THREADS = 32
_DEFAULT_RAMP_THRESHOLD_SECONDS = 0.2
_RAMP_DECISION_READS = 4

_POSIX_FADV_DONTNEED = 4


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_flag(name: str, default: bool = True) -> bool:
    return os.environ.get(name, "1" if default else "0") != "0"


def sharded_read_available() -> bool:
    """Whether the sharded distributed read can run at all on this host.

    The sharded path is built on the parallel reader (``os.preadv`` +
    O_DIRECT), so it honors the same ``FLASHPACK_PARALLEL_READ=0`` kill switch
    and POSIX requirement. It deliberately does NOT require the CPU eager
    opt-in (``FLASHPACK_CPU_PARALLEL_READ``): that flag chooses eager-vs-mmap
    for plain CPU loads, while a sharded read has no mmap alternative --
    every rank must materialize its shard to broadcast it.

    Callers must not branch on this per rank: the distributed entry point
    resolves it on rank 0 and broadcasts the decision, so a kill switch set on
    only some ranks cannot desynchronize the collective schedule.
    """
    return os.name == "posix" and _env_flag("FLASHPACK_PARALLEL_READ")


def parallel_read_supported(device: torch.device) -> bool:
    """Whether the parallel reader applies to ``device`` (POSIX only).

    CUDA targets use it by default (``FLASHPACK_PARALLEL_READ=0`` disables).
    CPU targets are opt-in via ``FLASHPACK_CPU_PARALLEL_READ=1``: the default
    CPU path returns lazy mmap views (instant, zero RSS), while the parallel
    reader eagerly materializes the payload into RAM — measured on an H100
    node's local NVMe (8 GB pack, page-cache cold) it is ~2.6x faster than
    faulting the mmap in (1.2 s / 6.4 GB/s vs 3.2 s / 2.4 GB/s), so it is the
    right choice when the weights will all be read anyway (serving), but it
    trades away mmap laziness — hence opt-in.
    """
    if os.name != "posix" or not _env_flag("FLASHPACK_PARALLEL_READ"):
        return False
    if device.type == "cuda":
        return True
    if device.type == "cpu":
        return _env_flag("FLASHPACK_CPU_PARALLEL_READ", default=False)
    return False


# Pinned staging memory is expensive to allocate (~0.5 s/GB), so the pool is
# kept for the process lifetime by default: model servers load all their
# packs back-to-back at startup and the initial pool (threads x chunk, 1 GiB at
# defaults) amortizes across them. Slow large reads can grow it to 2 GiB. One
# buffer per thread still overlaps I/O and H2D across reader threads while
# halving the warm first-load pinning cost. Set
# FLASHPACK_CACHE_PINNED=0 or call
# release_pinned_pool() to free it.
_PINNED_POOL: dict = {}
_PINNED_POOL_LOCK = threading.Lock()


def _get_pinned_pool(n_threads: int, chunk_bytes: int) -> list:
    with _PINNED_POOL_LOCK:
        pool = next(
            (
                cached
                for (cached_threads, cached_bytes), cached in _PINNED_POOL.items()
                if cached_bytes == chunk_bytes and cached_threads >= n_threads
            ),
            None,
        )
        if pool is None:
            pool = [
                [
                    torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True)
                    for _ in range(_BUFFERS_PER_THREAD)
                ]
                for _ in range(n_threads)
            ]
            _PINNED_POOL.clear()  # hold at most one pool
            _PINNED_POOL[(n_threads, chunk_bytes)] = pool
        return pool


def _grow_pinned_pool(pool: list, n_threads: int, chunk_bytes: int) -> None:
    """Grow the active pool in place so existing readers keep their buffers."""
    with _PINNED_POOL_LOCK:
        if len(pool) >= n_threads:
            return
        extra = [
            [
                torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True)
                for _ in range(_BUFFERS_PER_THREAD)
            ]
            for _ in range(n_threads - len(pool))
        ]
        pool.extend(extra)
        _PINNED_POOL.clear()
        _PINNED_POOL[(n_threads, chunk_bytes)] = pool


def release_pinned_pool() -> None:
    """Free the cached pinned staging buffers."""
    with _PINNED_POOL_LOCK:
        _PINNED_POOL.clear()


def _page_cache_resident_fraction(path: str, size: int) -> float:
    """Fraction of the file resident in the page cache (0.0 on any failure).

    Used to keep the page-cache-hot fast path: buffered reads from a hot
    cache run at memory speed, while O_DIRECT would force a filesystem
    round-trip.
    """
    if size == 0:
        return 1.0
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            mm = mmap_module.mmap(
                fd,
                size,
                flags=mmap_module.MAP_PRIVATE,
                prot=mmap_module.PROT_READ | mmap_module.PROT_WRITE,
            )
        finally:
            os.close(fd)
        try:
            npages = (size + mmap_module.PAGESIZE - 1) // mmap_module.PAGESIZE
            vec = (ctypes.c_ubyte * npages)()
            buf = (ctypes.c_char * size).from_buffer(mm)
            libc = ctypes.CDLL(None, use_errno=True)
            # argtypes MUST be declared: without them ctypes converts the
            # 64-bit address through C int, mincore gets a truncated pointer
            # and fails, and this function reported 0.0 for every file — so
            # the O_DIRECT gate never detected a warm page cache and warm
            # loads paid the direct-IO path (measured ~5x on a hot 2.6 GB
            # pack: 1.63 s vs 0.30 s buffered).
            libc.mincore.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_ubyte),
            ]
            libc.mincore.restype = ctypes.c_int
            ret = libc.mincore(
                ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(size), vec
            )
            if ret != 0:
                return 0.0
            # vectorized: a Python-level sum over the per-page vector costs
            # ~0.3 s for an 8 GB file — real overhead on every load
            resident = int((np.frombuffer(vec, dtype=np.uint8) & 1).sum())
            del buf
            return resident / npages
        finally:
            mm.close()
    except Exception:
        return 0.0


_SAMPLE_PROBE_THREADS = 4
_SAMPLE_PROBE_BYTES = 16 * 1024 * 1024


def _sample_read_gbps(path: str, size: int) -> float:
    """Aggregate buffered read rate over samples spread across the file
    (GB/s; 0.0 on any failure).

    Discriminates page-hot from cache-cold where mincore cannot: hot files
    scale with reader threads (well above 5 GB/s aggregate), while cold
    network/NVMe-cache buffered reads saturate around 2-4 GB/s regardless
    of thread count. Cost: at most 64 MiB of buffered reads, which land in
    the page cache and are not wasted.
    """
    if size <= _SAMPLE_PROBE_THREADS * _SAMPLE_PROBE_BYTES:
        # Small file: buffered is the right choice either way.
        return float("inf")
    try:
        results = [0] * _SAMPLE_PROBE_THREADS
        step = (size - _SAMPLE_PROBE_BYTES) // (_SAMPLE_PROBE_THREADS - 1)

        def _sampler(idx: int) -> None:
            fd = os.open(path, os.O_RDONLY)
            try:
                off = (idx * step) & ~(_ALIGN - 1)
                want = min(_SAMPLE_PROBE_BYTES, size - off)
                buf = bytearray(want)
                got = os.preadv(fd, [buf], off)
                results[idx] = max(got, 0)
            finally:
                os.close(fd)

        threads = [
            threading.Thread(target=_sampler, args=(i,), daemon=True)
            for i in range(_SAMPLE_PROBE_THREADS)
        ]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.perf_counter() - t0
        total = sum(results)
        if dt <= 0 or total == 0:
            return 0.0
        return total / 1e9 / dt
    except Exception:
        return 0.0


def _sample_probe_min_gbps() -> float:
    try:
        return float(os.environ.get("FLASHPACK_SAMPLE_PROBE_MIN_GBPS", "5.0"))
    except ValueError:
        return 5.0


def _should_use_direct(path: str, size: int) -> bool:
    """Decide O_DIRECT vs buffered for this load.

    mincore is the fast positive signal, but under cgroup-managed runners it
    can report 0.0 residency for a demonstrably hot page cache (measured:
    buffered repeat rode the cache at 21.6 GB/s right after mincore read 0.0),
    which silently forces every warm reload onto the ~2x-slower direct path.
    A timed sample read verifies coldness before O_DIRECT is chosen.
    """
    if not (_env_flag("FLASHPACK_DIRECT_IO") and hasattr(os, "O_DIRECT")):
        return False
    if _page_cache_resident_fraction(path, size) >= 0.9:
        return False
    if not _env_flag("FLASHPACK_SAMPLE_PROBE", default=True):
        return True
    return _sample_read_gbps(path, size) < _sample_probe_min_gbps()


def _read_chunk(fd_direct, fd_plain: int, view, f_off: int, ln: int) -> None:
    """Fill ``view[:ln]`` from file offset ``f_off``.

    O_DIRECT covers the aligned body when the chunk starts 4K-aligned (chunk
    planning guarantees this for all but sub-page macroblock heads); the
    unaligned tail — and any chunk O_DIRECT cannot serve — is read through
    the buffered descriptor. Short direct reads (e.g. filesystems that
    accept O_DIRECT on open but not on read) also degrade to buffered.
    """
    got = 0
    if fd_direct is not None and f_off % _ALIGN == 0:
        body = ln & ~(_ALIGN - 1)
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
            raise IOError(f"short read: wanted {ln} bytes at offset {f_off}, got {got}")
        got += n


def _plan_chunks(
    specs: "list[MacroblockSpec]", chunk_bytes: int
) -> list[tuple[int, int, int, int]]:
    """(block_idx, file_offset, block_offset, length) chunks covering every
    macroblock. Every chunk after a macroblock's (sub-page) head starts
    4K-aligned in file space and at position 0 of its staging buffer —
    satisfying both O_DIRECT alignment requirements.
    """
    chunks: list[tuple[int, int, int, int]] = []
    for idx, spec in enumerate(specs):
        b_off = 0
        remaining = spec.length_bytes
        head = (-spec.offset_bytes) % _ALIGN
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


def _parallel_read_into_cpu_storage(
    path: str,
    specs: "list[MacroblockSpec]",
    blocks: list[torch.Tensor],
) -> None:
    """CPU variant: N reader threads ``preadv`` file ranges straight into the
    destination blocks' memory — no staging buffers, no pinned memory, no
    streams. ``preadv`` releases the GIL, so plain Python threads scale to
    the device ceiling. O_DIRECT is used under the same conditions as the
    CUDA path (cache-cold file, 4K-aligned file offset) plus a 4K-aligned
    destination address; every misaligned or failing chunk degrades to a
    buffered read of identical bytes.
    """
    n_threads = effective_read_threads(_env_int("FLASHPACK_READ_THREADS", 16))
    chunk_bytes = max(_ALIGN, _env_int("FLASHPACK_READ_CHUNK_BYTES", 64 * 1024 * 1024))

    byte_views = [b.view(torch.uint8) for b in blocks]
    mvs = [memoryview(v.numpy()) for v in byte_views]
    dest_ptrs = [v.data_ptr() for v in byte_views]

    work: queue.SimpleQueue = queue.SimpleQueue()
    chunks = _plan_chunks(specs, chunk_bytes)
    for chunk in chunks:
        work.put(chunk)
    n_threads = min(n_threads, max(1, len(chunks)))
    for _ in range(n_threads):
        work.put(None)

    size = os.path.getsize(path)
    use_direct = _should_use_direct(path, size)

    errors: list[BaseException] = []

    def _reader() -> None:
        try:
            fd_plain = os.open(path, os.O_RDONLY)
            fd_direct = None
            if use_direct:
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
                    # O_DIRECT also requires a 4K-aligned destination
                    # address; misaligned chunks read buffered.
                    fd_d = fd_direct if (dest_ptrs[blk] + b_off) % _ALIGN == 0 else None
                    _read_chunk(fd_d, fd_plain, mvs[blk][b_off : b_off + ln], f_off, ln)
            finally:
                os.close(fd_plain)
                if fd_direct is not None:
                    os.close(fd_direct)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_reader, daemon=True) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def parallel_read_into_storage(
    path: str,
    specs: "list[MacroblockSpec]",
    blocks: list[torch.Tensor],
    device: torch.device,
) -> None:
    """Fill pre-allocated device ``blocks`` from ``path`` with parallel reads.

    ``blocks[i]`` must be the flat device tensor for ``specs[i]`` (same
    allocation the legacy path uses). Raises on any integrity error; never
    returns partially-filled storage silently.
    """
    if device.type == "cpu":
        _parallel_read_into_cpu_storage(path, specs, blocks)
        return

    # IO-bound path: threads are the IO queue depth, so the affinity clamp
    # floors at the default -- only oversubscribed requests get capped.
    base_threads = effective_read_threads(
        _env_int("FLASHPACK_READ_THREADS", _DEFAULT_READ_THREADS),
        floor=_DEFAULT_READ_THREADS,
    )
    chunk_bytes = max(_ALIGN, _env_int("FLASHPACK_READ_CHUNK_BYTES", 64 * 1024 * 1024))

    byte_views = [b.view(torch.uint8) for b in blocks]

    # The destination blocks were allocated on the caller's current stream;
    # order every reader stream after that allocation so the caching
    # allocator cannot hand the readers memory whose prior (default-stream)
    # work is still pending — the documented wait_event pattern. Free: one
    # event, recorded once.
    alloc_ready = torch.cuda.Event()
    alloc_ready.record(torch.cuda.current_stream(device))

    work: queue.SimpleQueue = queue.SimpleQueue()
    chunks = _plan_chunks(specs, chunk_bytes)
    for chunk in chunks:
        work.put(chunk)
    n_chunks = len(chunks)
    payload_bytes = sum(chunk[3] for chunk in chunks)

    base_threads = min(base_threads, max(1, n_chunks))

    size = os.path.getsize(path)
    # A page-cache-hot file is faster through buffered reads; O_DIRECT
    # would bypass the cache and re-fetch from the filesystem.
    use_direct = _should_use_direct(path, size)

    ramp_threads = effective_read_threads(
        _env_int("FLASHPACK_RAMP_THREADS", _DEFAULT_RAMP_THREADS),
        # Like the base floor, this is I/O queue depth rather than a CPU
        # worker budget. Preserve the default ramp even on small cpusets.
        floor=_DEFAULT_RAMP_THREADS,
    )
    max_threads = min(max(base_threads, ramp_threads), max(1, n_chunks))
    can_ramp = (
        use_direct
        and _env_flag("FLASHPACK_CONDITIONAL_RAMP")
        and max_threads > base_threads
        # Do not allocate another pinned pool for a small tail of work. At
        # defaults this limits the ramp to payloads of at least 4 GiB.
        and payload_bytes >= 2 * max_threads * chunk_bytes
    )
    for _ in range(max_threads if can_ramp else base_threads):
        work.put(None)

    pool = _get_pinned_pool(base_threads, chunk_bytes)
    errors: list[BaseException] = []
    completed_reads = 0
    progress_lock = threading.Lock()
    ramp_decision = threading.Event()
    should_ramp = False
    decision_reads = min(_RAMP_DECISION_READS, n_chunks)
    ramp_threshold = max(
        0.0,
        _env_float("FLASHPACK_RAMP_THRESHOLD_S", _DEFAULT_RAMP_THRESHOLD_SECONDS),
    )
    read_started = time.perf_counter()

    def _reader(thread_idx: int) -> None:
        nonlocal completed_reads, should_ramp
        try:
            fd_plain = os.open(path, os.O_RDONLY)
            fd_direct = None
            if use_direct:
                try:
                    fd_direct = os.open(path, os.O_RDONLY | os.O_DIRECT)
                except OSError:
                    fd_direct = None
            stream = torch.cuda.Stream(device=device)
            stream.wait_event(alloc_ready)
            bufs = pool[thread_idx]
            # O_DIRECT also requires 4K-aligned user buffers; pinned
            # allocations are page-aligned in practice, but verify.
            if fd_direct is not None and any(b.data_ptr() % _ALIGN for b in bufs):
                os.close(fd_direct)
                fd_direct = None
            events = [torch.cuda.Event() for _ in range(_BUFFERS_PER_THREAD)]
            for ev in events:
                ev.record(stream)
            views = [memoryview(b.numpy()) for b in bufs]
            i = 0
            try:
                while True:
                    item = work.get()
                    if item is None:
                        break
                    blk, f_off, b_off, ln = item
                    slot = i % _BUFFERS_PER_THREAD
                    buf, ev, view = bufs[slot], events[slot], views[slot]
                    i += 1
                    ev.synchronize()  # buffer's previous H2D must be done
                    _read_chunk(fd_direct, fd_plain, view, f_off, ln)
                    if can_ramp:
                        with progress_lock:
                            completed_reads += 1
                            if completed_reads == decision_reads:
                                should_ramp = (
                                    time.perf_counter() - read_started >= ramp_threshold
                                )
                                ramp_decision.set()
                    with torch.cuda.stream(stream):
                        byte_views[blk].narrow(0, b_off, ln).copy_(
                            buf.narrow(0, 0, ln), non_blocking=True
                        )
                        ev.record(stream)
            finally:
                os.close(fd_plain)
                if fd_direct is not None:
                    os.close(fd_direct)
            stream.synchronize()
        except BaseException as e:
            errors.append(e)

    threads = [
        threading.Thread(target=_reader, args=(i,), daemon=True)
        for i in range(base_threads)
    ]
    extra_threads: list[threading.Thread] = []

    def _ramp_controller() -> None:
        ramp_decision.wait()
        if not should_ramp or errors:
            return
        try:
            _grow_pinned_pool(pool, max_threads, chunk_bytes)
        except (MemoryError, RuntimeError):
            # Ramping is optional: retain the working base readers if the
            # host cannot pin the additional staging memory.
            return
        for i in range(base_threads, max_threads):
            thread = threading.Thread(target=_reader, args=(i,), daemon=True)
            try:
                thread.start()
            except RuntimeError:
                break
            extra_threads.append(thread)

    controller = (
        threading.Thread(target=_ramp_controller, daemon=True) if can_ramp else None
    )
    if controller is not None:
        controller.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if controller is not None:
        # An early read failure can prevent the completion threshold.
        ramp_decision.set()
        controller.join()
        for t in extra_threads:
            t.join()
    torch.cuda.synchronize(device)
    if not _env_flag("FLASHPACK_CACHE_PINNED"):
        release_pinned_pool()
    if errors:
        raise errors[0]
