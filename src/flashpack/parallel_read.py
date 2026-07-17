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

Tunables (environment):
- ``FLASHPACK_PARALLEL_READ=0``      disable (use legacy path)
- ``FLASHPACK_READ_THREADS``         reader threads (default 16)
- ``FLASHPACK_READ_CHUNK_BYTES``     chunk size (default 64 MiB)
- ``FLASHPACK_DIRECT_IO=0``          never use O_DIRECT
- ``FLASHPACK_CACHE_PINNED=0``       free pinned staging buffers after load
"""

import ctypes
import mmap as mmap_module
import os
import queue
import threading
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .deserialization import MacroblockSpec

__all__ = [
    "parallel_read_supported",
    "parallel_read_into_storage",
    "release_pinned_pool",
]

_ALIGN = 4096
_BUFFERS_PER_THREAD = 2

_POSIX_FADV_DONTNEED = 4


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_flag(name: str, default: bool = True) -> bool:
    return os.environ.get(name, "1" if default else "0") != "0"


def parallel_read_supported(device: torch.device) -> bool:
    """Parallel fused reads only apply to CUDA targets on POSIX systems."""
    return (
        device.type == "cuda"
        and os.name == "posix"
        and _env_flag("FLASHPACK_PARALLEL_READ")
    )


# Pinned staging memory is expensive to allocate (~0.5 s/GB), so the pool is
# kept for the process lifetime by default: model servers load all their
# packs back-to-back at startup and the pool (threads x 2 x chunk, 2 GiB at
# defaults) amortizes across them. Set FLASHPACK_CACHE_PINNED=0 or call
# release_pinned_pool() to free it.
_PINNED_POOL: dict = {}
_PINNED_POOL_LOCK = threading.Lock()


def _get_pinned_pool(n_threads: int, chunk_bytes: int) -> list:
    key = (n_threads, chunk_bytes)
    with _PINNED_POOL_LOCK:
        pool = _PINNED_POOL.get(key)
        if pool is None:
            pool = [
                [
                    torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True)
                    for _ in range(_BUFFERS_PER_THREAD)
                ]
                for _ in range(n_threads)
            ]
            _PINNED_POOL.clear()  # hold at most one pool
            _PINNED_POOL[key] = pool
        return pool


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
            ret = libc.mincore(ctypes.addressof(buf), ctypes.c_size_t(size), vec)
            if ret != 0:
                return 0.0
            resident = sum(v & 1 for v in vec)
            del buf
            return resident / npages
        finally:
            mm.close()
    except Exception:
        return 0.0


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
    n_threads = max(1, _env_int("FLASHPACK_READ_THREADS", 16))
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

    n_threads = min(n_threads, max(1, n_chunks))
    for _ in range(n_threads):
        work.put(None)

    size = os.path.getsize(path)
    use_direct = (
        _env_flag("FLASHPACK_DIRECT_IO")
        and hasattr(os, "O_DIRECT")
        # A page-cache-hot file is faster through buffered reads; O_DIRECT
        # would bypass the cache and re-fetch from the filesystem.
        and _page_cache_resident_fraction(path, size) < 0.9
    )

    pool = _get_pinned_pool(n_threads, chunk_bytes)
    errors: list[BaseException] = []

    def _reader(thread_idx: int) -> None:
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
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize(device)
    if not _env_flag("FLASHPACK_CACHE_PINNED"):
        release_pinned_pool()
    if errors:
        raise errors[0]
