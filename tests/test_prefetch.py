"""Tests for the loader-aware background prefetch (``flashpack.prefetch``).

Hermetic and CPU-only. The contract under test:

* a settled prefetch never changes load results (bit-identity);
* the loader settles a registered prefetch before choosing an I/O path —
  waiting on nearly-done warms, cancelling barely-started ones, never
  racing them;
* prefetching is advisory: errors are recorded, never raised, and the load
  performs its own (error-checked) reads regardless;
* residency gating makes prefetch idempotent and hot-file-free.
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
        for handle in prefetch_mod._REGISTRY.values():
            handle.cancel()
        prefetch_mod._REGISTRY.clear()


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


def test_prefetch_is_idempotent_per_path(tmp_path):
    path = _pack(tmp_path)
    a = prefetch_flashpack_file(path)
    b = prefetch_flashpack_file(path)
    assert a is b
    a.wait(timeout=30.0)


@posix_only
def test_prefetch_hot_file_is_immediate_noop(tmp_path, monkeypatch):
    path = _pack(tmp_path)
    with open(path, "rb") as f:
        f.read()  # fault everything in
    from flashpack.parallel_read import _page_cache_resident_fraction

    if _page_cache_resident_fraction(path, os.path.getsize(path)) < 0.9:
        pytest.skip("page cache did not retain the file (mincore unavailable?)")

    called = threading.Event()

    def marker(*args, **kwargs):
        called.set()
        raise AssertionError("hot file must not be read")

    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", marker)
    handle = prefetch_flashpack_file(path)
    assert handle.done
    assert handle.progress == pytest.approx(1.0)
    assert not called.is_set()


def test_cancel_stops_promptly_and_settles(tmp_path, monkeypatch):
    path = _pack(tmp_path)
    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", _throttled_reader(0.02))
    monkeypatch.setattr(prefetch_mod, "_page_cache_resident_fraction", lambda *a: 0.0)
    handle = prefetch_flashpack_file(path, n_threads=2, chunk_bytes=64 * 1024)
    assert not handle.done
    handle.cancel()
    assert handle.done and handle.cancelled
    assert handle.progress < 1.0


def test_prefetch_error_is_advisory(tmp_path, monkeypatch):
    path = _pack(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("injected")

    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", boom)
    monkeypatch.setattr(prefetch_mod, "_page_cache_resident_fraction", lambda *a: 0.0)
    handle = prefetch_flashpack_file(path)
    assert handle.wait(timeout=30.0)
    assert handle.error is not None
    # the file is still perfectly loadable
    storage, _ = read_flashpack_file(path, device="cpu")
    assert len(storage.blocks) > 0


def test_prefetch_missing_file_raises_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        prefetch_flashpack_file(str(tmp_path / "nope.flashpack"))


# -- settle policy ---------------------------------------------------------


def _registered_stub(path: str, size: int, progress_bytes: int) -> FlashpackPrefetch:
    handle = FlashpackPrefetch(os.path.abspath(path), size)
    handle._bytes_done = progress_bytes
    with prefetch_mod._REGISTRY_LOCK:
        prefetch_mod._REGISTRY[os.path.abspath(path)] = handle
    return handle


def test_settle_cancels_barely_started(tmp_path):
    path = _pack(tmp_path)
    size = os.path.getsize(path)
    handle = _registered_stub(path, size, progress_bytes=int(size * 0.1))

    settle_prefetch(path)
    assert handle.done and handle.cancelled
    assert consume_prefetch(path) is None


def test_settle_waits_for_nearly_done(tmp_path):
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


def test_settle_noop_without_registered_prefetch(tmp_path):
    settle_prefetch(str(tmp_path / "never-prefetched.flashpack"))


@posix_only
def test_eager_load_settles_inflight_prefetch(tmp_path, monkeypatch):
    """End to end: a load issued while its prefetch has barely started must
    cancel the warm (never race it) and still produce identical bytes."""
    path = _pack(tmp_path)
    reference, _ = read_flashpack_file(path, device="cpu")

    monkeypatch.setattr(prefetch_mod, "_read_chunk_buffered", _throttled_reader(0.05))
    monkeypatch.setattr(prefetch_mod, "_page_cache_resident_fraction", lambda *a: 0.0)
    handle = prefetch_flashpack_file(path, n_threads=2, chunk_bytes=64 * 1024)
    assert not handle.done

    monkeypatch.setenv("FLASHPACK_CPU_PARALLEL_READ", "1")
    monkeypatch.setenv("FLASHPACK_READ_THREADS", "4")
    eager, _ = read_flashpack_file(path, device="cpu")

    assert handle.done and handle.cancelled, "load must settle the warm first"
    for ref, got in zip(reference.blocks, eager.blocks):
        assert torch.equal(ref.view(torch.uint8), got.view(torch.uint8))


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
def test_revert_from_file_eager_matches_lazy(tmp_path):
    path = _pack(tmp_path)
    eager = revert_from_file(path)
    lazy = revert_from_file(path, eager_cpu=False)
    assert eager.keys() == lazy.keys()
    for name in eager:
        assert eager[name].dtype == lazy[name].dtype
        assert torch.equal(
            eager[name].view(torch.uint8).flatten(),
            lazy[name].view(torch.uint8).flatten(),
        ), f"tensor {name} differs between eager and lazy revert"
