"""Unit tests for the resource guardrails: the reader-thread affinity clamp
and the GPU-decoder warmup helper's graceful degradation."""

import os

from flashpack import utils
from flashpack.deserialization import fpz_gpu_warmup
from flashpack.utils import effective_read_threads


def test_clamps_to_affinity(monkeypatch) -> None:
    monkeypatch.delenv("FLASHPACK_NO_THREAD_CLAMP", raising=False)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False
    )
    assert effective_read_threads(16) == 4
    assert effective_read_threads(64) == 4


def test_floor_keeps_default_on_small_affinity(monkeypatch) -> None:
    # IO-bound call sites floor the clamp at the default thread count:
    # requests at or below the floor pass through untouched even on small
    # cpusets; oversubscribed requests are capped at max(affinity, floor).
    monkeypatch.delenv("FLASHPACK_NO_THREAD_CLAMP", raising=False)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False
    )
    assert effective_read_threads(16, floor=16) == 16
    assert effective_read_threads(64, floor=16) == 16
    assert effective_read_threads(8, floor=16) == 8


def test_floor_does_not_raise_large_affinity_cap(monkeypatch) -> None:
    monkeypatch.delenv("FLASHPACK_NO_THREAD_CLAMP", raising=False)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: set(range(32)), raising=False
    )
    assert effective_read_threads(64, floor=16) == 32


def test_request_below_budget_is_kept(monkeypatch) -> None:
    monkeypatch.delenv("FLASHPACK_NO_THREAD_CLAMP", raising=False)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: set(range(32)), raising=False
    )
    assert effective_read_threads(16) == 16
    assert effective_read_threads(1) == 1


def test_never_below_one(monkeypatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0}, raising=False)
    assert effective_read_threads(0) == 1
    assert effective_read_threads(-4) == 1


def test_escape_hatch(monkeypatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    monkeypatch.setenv("FLASHPACK_NO_THREAD_CLAMP", "1")
    assert effective_read_threads(64) == 64


def test_affinity_unavailable_falls_back_to_cpu_count(monkeypatch) -> None:
    monkeypatch.delenv("FLASHPACK_NO_THREAD_CLAMP", raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(utils.os, "cpu_count", lambda: 8)
    assert effective_read_threads(16) == 8


def test_fpz_gpu_warmup_degrades_without_gpu() -> None:
    """On a CUDA-less host the warmup must be a safe no-op returning False,
    never an exception (it is called from app setup paths)."""
    assert fpz_gpu_warmup() is False
