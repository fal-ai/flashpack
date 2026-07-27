"""Unit tests for the O_DIRECT sample-read gate (_should_use_direct)."""

import os

import pytest


class TestSampleProbe:
    def test_small_file_reports_hot(self, tmp_path) -> None:
        from flashpack.parallel_read import _sample_read_gbps

        p = tmp_path / "small.bin"
        p.write_bytes(b"x" * 1024)
        assert _sample_read_gbps(str(p), 1024) == float("inf")

    @pytest.mark.skipif(
        not hasattr(os, "preadv"),
        reason="the sample probe reads with preadv (POSIX-only); on platforms "
        "without it the probe reports cold and the reader keeps O_DIRECT off",
    )
    def test_probe_returns_rate_on_real_file(self, tmp_path) -> None:
        from flashpack.parallel_read import (
            _SAMPLE_PROBE_BYTES,
            _SAMPLE_PROBE_THREADS,
            _sample_read_gbps,
        )

        size = _SAMPLE_PROBE_THREADS * _SAMPLE_PROBE_BYTES + (1 << 20)
        p = tmp_path / "big.bin"
        with open(p, "wb") as f:
            f.truncate(size)
        rate = _sample_read_gbps(str(p), size)
        assert rate > 0.0 and rate != float("inf")

    def test_direct_io_env_kill_switch(self, tmp_path, monkeypatch) -> None:
        from flashpack.parallel_read import _should_use_direct

        monkeypatch.setenv("FLASHPACK_DIRECT_IO", "0")
        p = tmp_path / "f.bin"
        p.write_bytes(b"x" * 4096)
        assert _should_use_direct(str(p), 4096) is False

    def test_probe_disabled_falls_back_to_mincore_only(
        self, tmp_path, monkeypatch
    ) -> None:
        import flashpack.parallel_read as pr

        monkeypatch.setenv("FLASHPACK_SAMPLE_PROBE", "0")
        monkeypatch.delenv("FLASHPACK_DIRECT_IO", raising=False)
        monkeypatch.setattr(pr, "_page_cache_resident_fraction", lambda *_: 0.0)
        monkeypatch.setattr(
            pr,
            "_sample_read_gbps",
            lambda *_: pytest.fail("probe must not run when disabled"),
        )
        p = tmp_path / "f.bin"
        p.write_bytes(b"x" * 4096)
        if not hasattr(os, "O_DIRECT"):
            pytest.skip("no O_DIRECT on this platform")
        assert pr._should_use_direct(str(p), 4096) is True

    def test_hot_probe_blocks_direct(self, tmp_path, monkeypatch) -> None:
        import flashpack.parallel_read as pr

        monkeypatch.delenv("FLASHPACK_DIRECT_IO", raising=False)
        monkeypatch.delenv("FLASHPACK_SAMPLE_PROBE", raising=False)
        monkeypatch.setattr(pr, "_page_cache_resident_fraction", lambda *_: 0.0)
        monkeypatch.setattr(pr, "_sample_read_gbps", lambda *_: 21.6)
        p = tmp_path / "f.bin"
        p.write_bytes(b"x" * 4096)
        if not hasattr(os, "O_DIRECT"):
            pytest.skip("no O_DIRECT on this platform")
        assert pr._should_use_direct(str(p), 4096) is False

    def test_mincore_hot_short_circuits_probe(self, tmp_path, monkeypatch) -> None:
        import flashpack.parallel_read as pr

        monkeypatch.delenv("FLASHPACK_DIRECT_IO", raising=False)
        monkeypatch.setattr(pr, "_page_cache_resident_fraction", lambda *_: 1.0)
        monkeypatch.setattr(
            pr,
            "_sample_read_gbps",
            lambda *_: pytest.fail("probe must not run when mincore says hot"),
        )
        p = tmp_path / "f.bin"
        p.write_bytes(b"x" * 4096)
        assert pr._should_use_direct(str(p), 4096) is False
