"""Loader-aware background prefetch (page-cache warm) for flashpack files.

This ports the useful half of fal's internal ``cattensors`` prewarmer into
flashpack itself, and couples it to the parallel reader so a warm can never
fight the O_DIRECT gate:

- ``prefetch_flashpack_file(path)`` starts a cancellable, chunk-parallel,
  buffered read of the file into the OS page cache. Start it early (before
  heavy imports / other setup work) and the file is cache-hot by the time
  the load runs — the load then reads at memory speed through the reader's
  page-cache-hot buffered path.
- The parallel reader settles the prefetch before choosing its I/O path
  (:func:`settle_prefetch`): a completed prefetch simply leaves a hot cache
  for the mincore gate to detect; an in-flight prefetch is either waited on
  (nearly done) or cancelled (barely started) — never raced for bandwidth.
- Residency-gated: prefetching an already-hot file is an immediate no-op,
  so repeated calls and multi-worker duplication cost nothing.

Why this replaces external prewarmers (``cattensors``, ``cat``-based
directory warms) for flashpack files: an *uncoordinated* warm makes pages
resident, which flips the reader's mincore gate onto the buffered path — at
best duplicating a full pass over the file, at worst (partial eviction,
warm still in flight) leaving the load on the slow path while competing
with it for filesystem bandwidth. A *coordinated* prefetch has neither
failure mode: the gate decision happens after the prefetch settles, and the
prefetch no-ops when the cache is already hot.

Reads are always buffered (never O_DIRECT) — populating the page cache is
the entire point.
"""

import os
import threading

from .parallel_read import _env_int, _page_cache_resident_fraction

__all__ = [
    "FlashpackPrefetch",
    "prefetch_flashpack_file",
    "consume_prefetch",
    "settle_prefetch",
]

# Fraction of the file that must already be resident for a prefetch to
# no-op, and for an in-flight prefetch to be worth waiting for at load
# time. Matches the parallel reader's O_DIRECT gate threshold so the two
# decisions can never disagree about what "hot" means.
_RESIDENT_FRACTION = 0.9

# Ceiling on how long a load will wait for a nearly-done prefetch before
# giving up and cancelling it (dead FUSE mounts should fail the load
# through its own reads, not hang it inside the prefetch join).
_SETTLE_WAIT_SECONDS = 120.0


def _read_chunk_buffered(fd: int, scratch: memoryview, offset: int, want: int) -> int:
    """Read up to ``want`` bytes at ``offset`` through the page cache.

    Uses ``preadv`` into the reusable scratch buffer where available
    (Linux/FreeBSD), falling back to ``pread`` (macOS). Returns bytes read
    (0 at EOF).
    """
    if hasattr(os, "preadv"):
        return os.preadv(fd, [scratch[:want]], offset)
    # pragma: no cover - macOS fallback, exercised on darwin only
    return len(os.pread(fd, want, offset))


class FlashpackPrefetch:
    """Handle for one background prefetch. Obtain via
    :func:`prefetch_flashpack_file`; never construct directly.

    Thread-safe. ``wait()`` blocks until the prefetch settles (finished,
    cancelled, or failed); ``cancel()`` stops it at the next chunk
    boundary. Failures never propagate — a prefetch is advisory, and the
    subsequent load performs (and error-checks) its own reads.
    """

    def __init__(self, path: str, size: int):
        self.path = path
        self.size = size
        self.error: BaseException | None = None
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._bytes_done = 0
        self._workers_left = 0
        self._threads: list[threading.Thread] = []

    # -- state ---------------------------------------------------------

    @property
    def done(self) -> bool:
        """Whether the prefetch has settled (finished, cancelled, or failed)."""
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def progress(self) -> float:
        """Fraction of the file read so far (1.0 for empty files)."""
        if self.size == 0:
            return 1.0
        with self._lock:
            return self._bytes_done / self.size

    # -- control -------------------------------------------------------

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the prefetch settles. Returns ``done``."""
        return self._done.wait(timeout)

    def cancel(self, join: bool = True) -> None:
        """Stop reading at the next chunk boundary.

        With ``join=True`` (default) returns only after every reader thread
        has exited, so no prefetch I/O competes with whatever runs next.
        """
        self._cancel.set()
        if join:
            for t in self._threads:
                t.join()
        self._done.set()

    # -- internal ------------------------------------------------------

    def _complete_immediately(self) -> None:
        with self._lock:
            self._bytes_done = self.size
        self._done.set()

    def _start(self, n_threads: int, chunk_bytes: int) -> None:
        n_threads = max(1, min(n_threads, max(1, self.size // chunk_bytes)))
        with self._lock:
            self._workers_left = n_threads

        bytes_per_thread = self.size // n_threads

        def _worker(start: int, end: int) -> None:
            try:
                scratch = memoryview(bytearray(chunk_bytes))
                fd = os.open(self.path, os.O_RDONLY)
                try:
                    off = start
                    while off < end and not self._cancel.is_set():
                        want = min(chunk_bytes, end - off)
                        got = _read_chunk_buffered(fd, scratch, off, want)
                        if got <= 0:
                            break
                        off += got
                        with self._lock:
                            self._bytes_done += got
                finally:
                    os.close(fd)
            except BaseException as e:  # advisory: record, never raise
                self.error = e
            finally:
                with self._lock:
                    self._workers_left -= 1
                    last = self._workers_left == 0
                if last:
                    self._done.set()

        for i in range(n_threads):
            start = i * bytes_per_thread
            end = self.size if i == n_threads - 1 else (i + 1) * bytes_per_thread
            t = threading.Thread(
                target=_worker,
                args=(start, end),
                name=f"flashpack-prefetch-{os.path.basename(self.path)}-{i}",
                daemon=True,
            )
            self._threads.append(t)
            t.start()


_REGISTRY: dict[str, FlashpackPrefetch] = {}
_REGISTRY_LOCK = threading.Lock()


def prefetch_flashpack_file(
    path: str,
    n_threads: int | None = None,
    chunk_bytes: int | None = None,
) -> FlashpackPrefetch:
    """Start (or reuse) a background page-cache prefetch of ``path``.

    Returns immediately with a :class:`FlashpackPrefetch` handle. Idempotent
    per path: concurrent/repeated calls share one live prefetch. If the file
    is already page-cache-hot the handle is returned pre-completed and no
    I/O happens.

    The parallel reader settles any registered prefetch before it picks its
    I/O path, so calling this is always safe — it can only move work off
    the load's critical path, never race it. Defaults follow the reader's
    tunables (``FLASHPACK_READ_THREADS``, capped at 8 for the buffered warm,
    and 16 MiB chunks via ``FLASHPACK_PREFETCH_CHUNK_BYTES``).
    """
    ap = os.path.abspath(path)
    size = os.path.getsize(ap)  # raises loudly on a bad path

    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(ap)
        if existing is not None and not existing.cancelled:
            return existing
        handle = FlashpackPrefetch(ap, size)
        _REGISTRY[ap] = handle

    # Buffered warm saturates well below the O_DIRECT thread count; 8 is
    # cattensors' long-serving production default.
    if n_threads is None:
        n_threads = min(8, _env_int("FLASHPACK_READ_THREADS", 16))
    if chunk_bytes is None:
        chunk_bytes = _env_int("FLASHPACK_PREFETCH_CHUNK_BYTES", 16 * 1024 * 1024)
    chunk_bytes = max(64 * 1024, chunk_bytes)

    if size == 0 or _page_cache_resident_fraction(ap, size) >= _RESIDENT_FRACTION:
        handle._complete_immediately()
        return handle

    handle._start(n_threads, chunk_bytes)
    return handle


def consume_prefetch(path: str) -> FlashpackPrefetch | None:
    """Pop and return the registered prefetch for ``path``, if any."""
    with _REGISTRY_LOCK:
        return _REGISTRY.pop(os.path.abspath(path), None)


def settle_prefetch(path: str) -> None:
    """Settle any registered prefetch for ``path`` before a load reads it.

    Policy (mirrors the O_DIRECT gate threshold):

    - finished: nothing to do — the mincore gate will see the hot cache and
      route the load onto buffered reads at memory speed;
    - nearly done (``progress >= 0.9``): wait for it — the remainder is
      cheaper than abandoning the warm bytes, and the gate then sees a
      stable hot cache;
    - barely started: cancel it (joining its readers) — a cold O_DIRECT
      load is faster than a buffered load racing its own warm.

    Never raises: on a wait timeout the prefetch is cancelled and the load
    proceeds on whatever the cache state is — the gate re-checks residency
    itself, so partial warms and evictions self-correct.
    """
    handle = consume_prefetch(path)
    if handle is None or handle.done:
        return
    if handle.progress >= _RESIDENT_FRACTION:
        if not handle.wait(timeout=_SETTLE_WAIT_SECONDS):
            handle.cancel()
    else:
        handle.cancel()
