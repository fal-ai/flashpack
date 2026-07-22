"""Tests for the loader-aware background prefetch (``flashpack.prefetch``).

Hermetic and CPU-only (one gpu-marked test). The contract under test:

* a settled prefetch never changes load results (bit-identity);
* the parallel reader settles a registered prefetch before choosing an I/O
  path — waiting on nearly-resident warms, cancelling barely-started ones,
  never racing them — with every wait and join bounded;
* prefetching is advisory: errors are recorded, never raised, and the load
  performs its own (error-checked) reads regardless;
* residency and memory gating make prefetch idempotent, hot-file-free, and
  safe against self-evicting warms;
* forked children never inherit a settleable warm.
"""

import os
import threading
import time

import pytest
import torch
from flashpack import prefetch as prefetch_mod
from flashpack.deserialization import read_flashpack_file, revert_from_file
from flashpack.prefetch import (
    FlashpackPrefetch,
    consume_prefetch,
    prefetch_flashpack_file,
    settle_prefetch,
)
from flashpack.serialization import pack_to_file

posix_only = pytest.mark.skipif(
    os.name != "posix", reason="parallel reader is POSIX-only"
)


@pytest.fixture(autouse=True)
def _clean_registry():
    with prefetch_mod._REGISTRY_LOCK:
        prefetch_mod._REGISTRY.clear()
    yield
    with prefetch_mod._REGISTRY_LOCK:
        handles = list(prefetch_mod._REGISTRY.values())
        prefetch_mod._REGISTRY.clear()
    for handle in handles:
        handle.cancel()


def _mixed_state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "a": torch.randn(1_000_003),  # fp32, odd length -> unaligned tail
        "b": torch.randn(2048, 512).to(torch.bfloat16),
        "c": torch.randint(0, 127, (777_777,), dtype=torch.int8),
        "d": torch.randint(0, 2, (12_345,)).bool(),
    }


def _pack(tmp_path) -> str:
    path = str(tmp_path / "pack.flashpack")
    pack_to_file(_mixed_state_dict(), path, target_dtype=None)
    return path


def _force_cold(monkeypatch) -> None:
    monkeypatch.setattr(prefetch_mod, "_page_cache_resident_fraction", lambda *a: 0.0)


def _throttled_reader(delay: float):
    """A slowed-down _read_chunk_buffered so tests can observe in-flight
    prefetches deterministically."""
    real = prefetch_mod._read_chunk_buffered

    def slow(fd, scratch, offset, want):
        time.sleep(delay)
        return real(fd, scratch, offset, min(want, 64 * 1024))

    return slow


# -- handle lifecycle ------------------------------------------------------


@posix_only
def test_prefetch_completes_and_load_matches(tmp_path, monkeypatch):
    path = _pack(tmp_path)
    reference, _ = read_flashpack_file(path, device="cpu")

    handle = prefetch_flashpack_file(path)
    assert handle.wait(timeout=30.0)
    assert handle.done and not handle.cancelled
    assert handle.error is None
    assert handle.progress == pytest.approx(1.0)

    monkeypatch.setenv("FLASHPACK_CPU_PARALLEL_READ", "1")
    monkeypatch.setenv("FLASHPACK_READ_THREADS", "4")
    eager, _ = read_flashpack_file(path, device="cpu")
    for i, (ref, got) in enumerate(zip(reference.blocks, eager.blocks)):
        assert torch.equal(
            ref.view(torch.uint8), got.view(torch.uint8)
        ), f"macroblock {i} differs after prefetch"
    # the load consumed the registry entry
    assert consume_prefetch(path) is None


@posix_only
def test_prefetch_cold_runs_workers_to_completion(tmp_path, monkeypatch):
    """Natural completion with real reads: worker accounting must converge
    (every byte counted once, last worker sets done)."""
    _force_cold(monkeypatch)
    path = _pack(tmp_path)
    handle = prefetch_flashpack_file(path, n_threads=3, chunk_bytes=64 * 1024)
    assert handle.wait(timeout=60.0)
    assert handle.error is None and not handle.cancelled
    assert handle.progress == pytest.approx(1.0)
    assert handle._bytes_done == handle.size


@posix_only
def test_prefetch_is_idempotent_while_live(tmp_path, monkeypatch):
    _force_cold(monkeypatch)
    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", _throttled_reader(0.02))
    path = _pack(tmp_path)
    a = prefetch_flashpack_file(path, n_threads=2, chunk_bytes=64 * 1024)
    b = prefetch_flashpack_file(path)
    assert a is b
    a.cancel()


@posix_only
def test_concurrent_prefetch_calls_share_one_handle(tmp_path, monkeypatch):
    _force_cold(monkeypatch)
    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", _throttled_reader(0.02))
    path = _pack(tmp_path)
    barrier = threading.Barrier(8)
    handles: list = [None] * 8

    def call(i):
        barrier.wait()
        handles[i] = prefetch_flashpack_file(path)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(h) for h in handles}) == 1
    handles[0].cancel()


def test_reprefetch_after_cancel_creates_new_handle(tmp_path, monkeypatch):
    _force_cold(monkeypatch)
    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", _throttled_reader(0.02))
    path = _pack(tmp_path)
    a = prefetch_flashpack_file(path, n_threads=2, chunk_bytes=64 * 1024)
    a.cancel()
    b = prefetch_flashpack_file(path, n_threads=2, chunk_bytes=64 * 1024)
    assert b is not a
    b.cancel()


@posix_only
def test_prefetch_hot_file_is_immediate_noop(tmp_path, monkeypatch):
    path = _pack(tmp_path)
    with open(path, "rb") as f:
        f.read()  # fault everything in
    from flashpack.parallel_read import _page_cache_resident_fraction

    if _page_cache_resident_fraction(path, os.path.getsize(path)) < 0.9:
        pytest.skip("page cache did not retain the file (mincore unavailable?)")

    def marker(*args, **kwargs):
        raise AssertionError("hot file must not be read")

    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", marker)
    handle = prefetch_flashpack_file(path)
    assert handle.done
    assert handle.skipped_reason == "hot"
    assert handle.progress == pytest.approx(1.0)


def test_prefetch_skips_files_larger_than_available_memory(tmp_path, monkeypatch):
    path = _pack(tmp_path)
    _force_cold(monkeypatch)
    monkeypatch.setattr(prefetch_mod, "_available_memory_bytes", lambda: 1024)

    def marker(*args, **kwargs):
        raise AssertionError("oversized file must not be warmed")

    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", marker)
    handle = prefetch_flashpack_file(path)
    assert handle.done and handle.error is None
    assert handle.skipped_reason == "memory"


@posix_only
def test_cancel_stops_promptly_and_settles(tmp_path, monkeypatch):
    path = _pack(tmp_path)
    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", _throttled_reader(0.02))
    _force_cold(monkeypatch)
    handle = prefetch_flashpack_file(path, n_threads=2, chunk_bytes=64 * 1024)
    assert not handle.done
    handle.cancel()
    assert handle.done and handle.cancelled
    assert handle.progress < 1.0


@posix_only
def test_prefetch_error_is_advisory(tmp_path, monkeypatch):
    path = _pack(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("injected")

    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", boom)
    _force_cold(monkeypatch)
    handle = prefetch_flashpack_file(path)
    assert handle.wait(timeout=30.0)
    assert handle.error is not None
    # the file is still perfectly loadable
    storage, _ = read_flashpack_file(path, device="cpu")
    assert len(storage.blocks) > 0


def test_prefetch_missing_file_raises_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        prefetch_flashpack_file(str(tmp_path / "nope.flashpack"))


def test_symlink_alias_shares_registry_key(tmp_path, monkeypatch):
    _force_cold(monkeypatch)
    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", _throttled_reader(0.02))
    path = _pack(tmp_path)
    alias = str(tmp_path / "alias.flashpack")
    os.symlink(path, alias)
    handle = prefetch_flashpack_file(alias)
    assert handle.path == os.path.realpath(path)
    assert consume_prefetch(path) is handle
    handle.cancel()


# -- settle policy ---------------------------------------------------------


def _registered_stub(path: str, size: int, progress_bytes: int) -> FlashpackPrefetch:
    key = prefetch_mod._registry_key(path)
    handle = FlashpackPrefetch(key, size)
    handle._bytes_done = progress_bytes
    with prefetch_mod._REGISTRY_LOCK:
        prefetch_mod._REGISTRY[key] = handle
    return handle


def test_settle_cancels_barely_started(tmp_path):
    path = _pack(tmp_path)
    size = os.path.getsize(path)
    handle = _registered_stub(path, size, progress_bytes=int(size * 0.1))

    settle_prefetch(path)
    assert handle.done and handle.cancelled
    assert consume_prefetch(path) is None


def test_settle_waits_for_nearly_done(tmp_path, monkeypatch):
    monkeypatch.setattr(prefetch_mod, "_page_cache_resident_fraction", lambda *a: 1.0)
    path = _pack(tmp_path)
    size = os.path.getsize(path)
    handle = _registered_stub(path, size, progress_bytes=int(size * 0.95))

    finished = threading.Timer(0.2, handle._complete_immediately)
    finished.start()
    t0 = time.monotonic()
    settle_prefetch(path)
    elapsed = time.monotonic() - t0
    assert handle.done and not handle.cancelled
    assert elapsed >= 0.15, "settle must actually wait for a nearly-done warm"
    assert consume_prefetch(path) is None


def test_settle_gate_uses_residency_not_bytes_read(tmp_path, monkeypatch):
    """A warm that has READ 95% of a pack whose pages were already evicted
    again must be cancelled, not waited for — waiting buys a stale cache
    and loses the O_DIRECT path."""
    monkeypatch.setattr(prefetch_mod, "_page_cache_resident_fraction", lambda *a: 0.2)
    path = _pack(tmp_path)
    size = os.path.getsize(path)
    handle = _registered_stub(path, size, progress_bytes=int(size * 0.95))

    t0 = time.monotonic()
    settle_prefetch(path)
    assert time.monotonic() - t0 < 5.0
    assert handle.done and handle.cancelled


def test_settle_wait_timeout_is_bounded(tmp_path, monkeypatch):
    """A nearly-done warm that never finishes (dead mount) must not hang the
    load: the wait times out and the cancel join is bounded."""
    monkeypatch.setattr(prefetch_mod, "_page_cache_resident_fraction", lambda *a: 1.0)
    path = _pack(tmp_path)
    size = os.path.getsize(path)
    handle = _registered_stub(path, size, progress_bytes=int(size * 0.95))

    t0 = time.monotonic()
    settle_prefetch(path, wait_timeout=0.2)
    elapsed = time.monotonic() - t0
    assert handle.done and handle.cancelled
    assert elapsed < 5.0, "settle must stay bounded on a stuck warm"
    assert consume_prefetch(path) is None


def test_settle_noop_without_registered_prefetch(tmp_path):
    settle_prefetch(str(tmp_path / "never-prefetched.flashpack"))


@posix_only
def test_eager_load_settles_inflight_prefetch(tmp_path, monkeypatch):
    """End to end: a load issued while its prefetch has barely started must
    cancel the warm (never race it) and still produce identical bytes."""
    path = _pack(tmp_path)
    reference, _ = read_flashpack_file(path, device="cpu")

    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", _throttled_reader(0.05))
    _force_cold(monkeypatch)
    handle = prefetch_flashpack_file(path, n_threads=2, chunk_bytes=64 * 1024)
    assert not handle.done

    monkeypatch.setenv("FLASHPACK_CPU_PARALLEL_READ", "1")
    monkeypatch.setenv("FLASHPACK_READ_THREADS", "4")
    eager, _ = read_flashpack_file(path, device="cpu")

    assert handle.done and handle.cancelled, "load must settle the warm first"
    for ref, got in zip(reference.blocks, eager.blocks):
        assert torch.equal(ref.view(torch.uint8), got.view(torch.uint8))


@posix_only
def test_prefetch_racing_load_never_raises(tmp_path, monkeypatch):
    """cancel() (from the load's settle) racing _start() must never throw —
    regression test for joining an appended-but-not-started thread."""
    monkeypatch.setenv("FLASHPACK_CPU_PARALLEL_READ", "1")
    monkeypatch.setenv("FLASHPACK_READ_THREADS", "4")
    path = _pack(tmp_path)
    reference, _ = read_flashpack_file(path, device="cpu", eager_cpu=False)
    _force_cold(monkeypatch)

    errors: list[BaseException] = []
    for _ in range(5):
        with prefetch_mod._REGISTRY_LOCK:
            prefetch_mod._REGISTRY.clear()
        barrier = threading.Barrier(2)

        def do_prefetch():
            try:
                barrier.wait()
                prefetch_flashpack_file(path, n_threads=2, chunk_bytes=64 * 1024)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        def do_load():
            try:
                barrier.wait()
                storage, _ = read_flashpack_file(path, device="cpu")
                for ref, got in zip(reference.blocks, storage.blocks):
                    assert torch.equal(ref.view(torch.uint8), got.view(torch.uint8))
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=do_prefetch),
            threading.Thread(target=do_load),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert not errors, f"race produced exceptions: {errors!r}"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_fork_child_settles_clean(tmp_path, monkeypatch):
    """A forked child inheriting an in-flight warm must settle instantly:
    the at-fork hook clears the registry (reader threads never survive a
    fork, so an inherited handle could otherwise stall settle for the full
    wait timeout)."""
    monkeypatch.setattr(prefetch_mod, "_page_cache_resident_fraction", lambda *a: 1.0)
    path = _pack(tmp_path)
    size = os.path.getsize(path)
    _registered_stub(path, size, progress_bytes=int(size * 0.95))

    pid = os.fork()
    if pid == 0:  # child
        try:
            t0 = time.monotonic()
            settle_prefetch(path, wait_timeout=30.0)
            ok = (time.monotonic() - t0) < 2.0 and not prefetch_mod._REGISTRY
            os._exit(0 if ok else 1)
        except BaseException:
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    assert (
        os.waitstatus_to_exitcode(status) == 0
    ), "child settle must no-op fast after fork"


@pytest.mark.gpu
def test_cuda_load_settles_prefetch(tmp_path, monkeypatch):
    path = _pack(tmp_path)
    reference, _ = read_flashpack_file(path, device="cpu")

    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", _throttled_reader(0.05))
    _force_cold(monkeypatch)
    handle = prefetch_flashpack_file(path, n_threads=2, chunk_bytes=64 * 1024)
    assert not handle.done

    storage, _ = read_flashpack_file(path, device="cuda")
    torch.cuda.synchronize()
    assert handle.done, "CUDA load must settle the warm first"
    for ref, got in zip(reference.blocks, storage.blocks):
        assert torch.equal(ref.view(torch.uint8), got.cpu().view(torch.uint8))


# -- eager_cpu plumbing ----------------------------------------------------


@posix_only
def test_read_flashpack_file_eager_cpu_param(tmp_path, monkeypatch):
    path = _pack(tmp_path)

    monkeypatch.delenv("FLASHPACK_CPU_PARALLEL_READ", raising=False)
    eager, _ = read_flashpack_file(path, device="cpu", eager_cpu=True)
    assert eager.backing_arrays is None, "explicit eager_cpu=True must materialize"

    monkeypatch.setenv("FLASHPACK_CPU_PARALLEL_READ", "1")
    lazy, _ = read_flashpack_file(path, device="cpu", eager_cpu=False)
    assert lazy.backing_arrays is not None, "explicit eager_cpu=False must stay lazy"

    monkeypatch.setenv("FLASHPACK_PARALLEL_READ", "0")
    downgraded, _ = read_flashpack_file(path, device="cpu", eager_cpu=True)
    assert (
        downgraded.backing_arrays is not None
    ), "the global kill switch overrides explicit eager_cpu"


@posix_only
def test_revert_from_file_eager_matches_lazy(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASHPACK_CPU_PARALLEL_READ", raising=False)
    path = _pack(tmp_path)
    eager = revert_from_file(path, eager_cpu=True)
    lazy = revert_from_file(path)  # default stays lazy (streaming-friendly)
    assert eager.keys() == lazy.keys()
    for name in eager:
        assert eager[name].dtype == lazy[name].dtype
        assert torch.equal(
            eager[name].view(torch.uint8).flatten(),
            lazy[name].view(torch.uint8).flatten(),
        ), f"tensor {name} differs between eager and lazy revert"
