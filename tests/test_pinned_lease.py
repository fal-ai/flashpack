"""Unit tests for the leased pinned staging cache: concurrent loads must
never share buffer objects (the flux-2 checkerboard corruption class)."""

import threading

import pytest

torch = pytest.importorskip("torch")

from flashpack import parallel_read as pr  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache():
    pr.release_pinned_pool()
    yield
    pr.release_pinned_pool()


def _ids(bundles):
    return {id(buf) for bundle in bundles for buf in bundle}


class TestPinnedLease:
    def test_concurrent_leases_are_disjoint(self, monkeypatch) -> None:
        # pin_memory needs CUDA on some builds; plain empty is fine for identity tests
        monkeypatch.setattr(
            torch,
            "empty",
            lambda *a, **kw: torch.zeros(a[0], dtype=kw.get("dtype", torch.uint8)),
        )
        results = []

        def worker():
            results.append(pr._lease_pinned_bundles(4, 1024))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        all_ids: set = set()
        for bundles in results:
            ids = _ids(bundles)
            assert not (all_ids & ids), "two concurrent leases shared a buffer"
            all_ids |= ids

    def test_release_then_lease_reuses(self, monkeypatch) -> None:
        monkeypatch.setattr(
            torch,
            "empty",
            lambda *a, **kw: torch.zeros(a[0], dtype=kw.get("dtype", torch.uint8)),
        )
        a = pr._lease_pinned_bundles(3, 2048)
        pr._release_pinned_bundles(2048, a)
        b = pr._lease_pinned_bundles(3, 2048)
        assert _ids(b) == _ids(a)

    def test_thread_count_change_still_reuses(self, monkeypatch) -> None:
        monkeypatch.setattr(
            torch,
            "empty",
            lambda *a, **kw: torch.zeros(a[0], dtype=kw.get("dtype", torch.uint8)),
        )
        a = pr._lease_pinned_bundles(8, 2048)
        pr._release_pinned_bundles(2048, a)
        b = pr._lease_pinned_bundles(2, 2048)
        assert _ids(b) <= _ids(a) and len(b) == 2

    def test_cache_cap_enforced(self, monkeypatch) -> None:
        monkeypatch.setattr(
            torch,
            "empty",
            lambda *a, **kw: torch.zeros(a[0], dtype=kw.get("dtype", torch.uint8)),
        )
        monkeypatch.setenv("FLASHPACK_PINNED_CACHE_BUNDLES", "2")
        a = pr._lease_pinned_bundles(5, 1024)
        pr._release_pinned_bundles(1024, a)
        assert len(pr._PINNED_FREE[1024]) == 2

    def test_chunk_size_change_retires_old_cache(self, monkeypatch) -> None:
        monkeypatch.setattr(
            torch,
            "empty",
            lambda *a, **kw: torch.zeros(a[0], dtype=kw.get("dtype", torch.uint8)),
        )
        a = pr._lease_pinned_bundles(2, 1024)
        pr._release_pinned_bundles(1024, a)
        b = pr._lease_pinned_bundles(2, 4096)
        pr._release_pinned_bundles(4096, b)
        assert list(pr._PINNED_FREE) == [4096]
