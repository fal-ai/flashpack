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
  (nearly resident) or cancelled (barely started) — never raced for
  bandwidth. Lazy-mmap and legacy CUDA loads deliberately do NOT settle:
  their buffered page faults only benefit from a warm running ahead of
  them; only the O_DIRECT-capable parallel reader must not race one.
- Residency-gated: prefetching an already-hot file is an immediate no-op,
  so repeated calls and multi-worker duplication cost nothing. Files
  larger than the available physical memory are not warmed at all (the
  warm would evict its own head before the load arrives).

Why this replaces external prewarmers (``cattensors``, ``cat``-based
directory warms) for flashpack files: an *uncoordinated* warm makes pages
resident, which flips the reader's mincore gate onto the buffered path — at
best duplicating a full pass over the file, at worst (partial eviction,
warm still in flight) leaving the load on the slow path while competing
with it for filesystem bandwidth. A *coordinated* prefetch has neither
failure mode: the gate decision happens after the prefetch settles, and the
prefetch no-ops when the cache is already hot.

Reads are always buffered (never O_DIRECT) — populating the page cache is
the entire point. Built for the Linux runners flashpack serves on; no
platform fallbacks.
"""

import os
import threading
import time

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

# Ceiling on how long cancel() waits for reader threads when invoked from
# the settle path. A reader wedged inside a preadv on a dead mount must
# not convert the settle into an unbounded hang — the readers are daemon
# threads, and the load's own reads will surface the mount failure.
_SETTLE_JOIN_SECONDS = 15.0


def _available_memory_bytes() -> int | None:
    """MemAvailable from /proc/meminfo — includes reclaimable page cache.

    NOT ``sysconf(SC_AVPHYS_PAGES)``: that counts only free pages, and on a
    long-running node the page cache keeps free-RAM near zero, which made
    this guard spuriously no-op every prefetch (caught on a production
    runner against qwen-image-2512's real pack).
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _read_chunk_buffered(fd: int, scratch: memoryview, offset: int, want: int) -> int:
    """Read up to ``want`` bytes at ``offset`` through the page cache,
    ``preadv``-ing into the reusable scratch buffer. Returns bytes read
    (0 at EOF)."""
    return os.preadv(fd, [scratch[:want]], offset)


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
        #: why a pre-completed handle skipped the warm: "hot", "memory",
        #: "empty" — None for a prefetch that actually ran.
        self.skipped_reason: str | None = None
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        # Serializes thread spawning against cancel(): cancel must never
        # observe an appended-but-not-yet-started thread (join would raise),
        # and a cancelled handle must never spawn readers at all.
        self._spawn_lock = threading.Lock()
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
        """Fraction of the file read so far (1.0 for empty files).

        Bytes *read*, not bytes currently resident — under memory pressure
        the page cache may already have evicted part of a large warm.
        """
        if self.size == 0:
            return 1.0
        with self._lock:
            return self._bytes_done / self.size

    # -- control -------------------------------------------------------

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the prefetch settles. Returns ``done``."""
        return self._done.wait(timeout)

    def cancel(self, join: bool = True, join_timeout: float | None = None) -> None:
        """Stop reading at the next chunk boundary.

        With ``join=True`` (default) waits for the reader threads so no
        prefetch I/O competes with whatever runs next; ``join_timeout``
        bounds that wait (total seconds across all threads) — a reader
        wedged inside a read on a dead mount is abandoned as a daemon
        thread rather than hanging the caller.
        """
        with self._spawn_lock:
            self._cancel.set()
            threads = list(self._threads)
        if join:
            deadline = None if join_timeout is None else time.monotonic() + join_timeout
            for t in threads:
                if deadline is None:
                    t.join()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    t.join(remaining)
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

        with self._spawn_lock:
            if self._cancel.is_set():
                # A settle raced us between registration and start; the
                # handle is already done — never spawn readers for it.
                return
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


def _registry_key(path: str) -> str:
    # realpath: symlink aliases of the same pack must share one warm and
    # one settle, or a load via the alias would race the warm via the target.
    return os.path.realpath(path)


def _after_fork_in_child() -> None:
    """Forked children inherit the registry, but reader threads never
    survive a fork: nothing would ever complete an inherited handle, so a
    child's settle could stall (or deadlock on a lock that happened to be
    held at the fork instant). Mark everything done and start fresh."""
    global _REGISTRY_LOCK
    _REGISTRY_LOCK = threading.Lock()
    for handle in _REGISTRY.values():
        handle._spawn_lock = threading.Lock()
        handle._lock = threading.Lock()
        handle._cancel.set()
        handle._done.set()
    _REGISTRY.clear()


# hasattr is import-safety only (flashpack/__init__ imports this module and
# the repo's packing side still CI-tests Windows) — the prefetch engine
# itself assumes Linux runners.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def prefetch_flashpack_file(
    path: str,
    n_threads: int | None = None,
    chunk_bytes: int | None = None,
) -> FlashpackPrefetch:
    """Start (or reuse) a background page-cache prefetch of ``path``.

    Returns immediately with a :class:`FlashpackPrefetch` handle. Idempotent
    per (real)path while a prefetch is live: concurrent/repeated calls share
    one warm. A settled handle (finished, failed, or cancelled) is replaced
    by a fresh one — the residency gate makes re-prefetching a still-hot
    file a free no-op, while a genuinely evicted file gets re-warmed.

    No-ops (returns a pre-completed handle) when the file is already
    page-cache-hot, or when it is larger than the currently available
    physical memory (warming it would evict its own head before the load
    arrives — the O_DIRECT cold path handles that case better).

    The parallel reader settles any registered prefetch before it picks its
    I/O path, so calling this is always safe — it can only move work off
    the load's critical path, never race it. Tunables:
    ``FLASHPACK_PREFETCH_THREADS`` (default 2 — deliberately gentle, see
    inline note) and ``FLASHPACK_PREFETCH_CHUNK_BYTES`` (default 16 MiB).
    """
    key = _registry_key(path)
    size = os.path.getsize(key)  # raises loudly on a bad path

    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(key)
        if existing is not None and not existing.done:
            return existing
        handle = FlashpackPrefetch(key, size)
        _REGISTRY[key] = handle

    if n_threads is None:
        # Deliberately gentle: the warm has the entire import/setup window
        # to work with, and every block it pulls is also WRITTEN to the
        # mount's node-local cache — an aggressive warm competes with
        # co-located reads (cold venv imports, other tenants) for the same
        # NVMe. Measured on production H100s against a 25 GB pack: 2
        # threads delivered at least the load benefit of 8.
        n_threads = _env_int("FLASHPACK_PREFETCH_THREADS", 2)
    if chunk_bytes is None:
        chunk_bytes = _env_int("FLASHPACK_PREFETCH_CHUNK_BYTES", 16 * 1024 * 1024)
    chunk_bytes = max(64 * 1024, chunk_bytes)

    available = _available_memory_bytes()
    if size == 0:
        handle.skipped_reason = "empty"
    elif available is not None and size > available * 0.9:
        handle.skipped_reason = "memory"
    elif _page_cache_resident_fraction(key, size) >= _RESIDENT_FRACTION:
        handle.skipped_reason = "hot"
    if handle.skipped_reason is not None:
        handle._complete_immediately()
        return handle

    handle._start(n_threads, chunk_bytes)
    return handle


def consume_prefetch(path: str) -> FlashpackPrefetch | None:
    """Pop and return the registered prefetch for ``path``, if any."""
    with _REGISTRY_LOCK:
        return _REGISTRY.pop(_registry_key(path), None)


def settle_prefetch(path: str, wait_timeout: float = _SETTLE_WAIT_SECONDS) -> None:
    """Settle any registered prefetch for ``path`` before a load reads it.

    Policy (mirrors the O_DIRECT gate threshold):

    - finished: nothing to do — the mincore gate will see the hot cache and
      route the load onto buffered reads at memory speed;
    - nearly there — most bytes read AND actually still resident
      (``>= 0.9`` on both counts): wait for it (bounded), the remainder is
      cheaper than abandoning the warm bytes;
    - otherwise: cancel it (bounded join) — a cold O_DIRECT load is faster
      than a buffered load racing its own warm. Bytes *read* are checked
      against bytes *resident* because under memory pressure a large warm
      can already have evicted its own head, in which case waiting buys a
      stale cache and loses the O_DIRECT path.

    Never raises. All waits and joins are bounded: a prefetch wedged on a
    dead mount is abandoned (daemon threads) and the load proceeds to fail
    — or succeed — through its own error-checked reads. Partial warms and
    evictions self-correct because the gate re-checks residency itself.
    The handle stays registered until it is settled, so concurrent loads of
    the same path all settle the same warm.
    """
    with _REGISTRY_LOCK:
        handle = _REGISTRY.get(_registry_key(path))
    if handle is None:
        return
    try:
        if not handle.done:
            worth_waiting = (
                handle.progress >= _RESIDENT_FRACTION
                and _page_cache_resident_fraction(handle.path, handle.size)
                >= _RESIDENT_FRACTION
            )
            if worth_waiting:
                if not handle.wait(timeout=wait_timeout):
                    handle.cancel(join_timeout=_SETTLE_JOIN_SECONDS)
            else:
                handle.cancel(join_timeout=_SETTLE_JOIN_SECONDS)
    finally:
        with _REGISTRY_LOCK:
            if _REGISTRY.get(handle.path) is handle:
                del _REGISTRY[handle.path]
