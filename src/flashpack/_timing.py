"""Dormant cold-start timing hook surface (fal telemetry).

Libraries expose hooks, never emit: everything here is a no-op unless a
collector registers via :func:`set_timing_hook` or the ``FAL_TIMING_COLLECTOR``
env pointer (``"module.path:attr"``, resolved lazily on first fire). Failures
are swallowed at the boundary; ``FAL_TIMING_DISABLED=1`` disables resolution.
"""

from __future__ import annotations

import importlib
import os

_hook = None
_resolve_attempted = False


def set_timing_hook(fn) -> None:
    global _hook, _resolve_attempted
    _hook = fn
    _resolve_attempted = True


def fire(phase: str, elapsed_s: float, **ctx) -> None:
    global _hook, _resolve_attempted
    try:
        if os.environ.get("FAL_TIMING_DISABLED") == "1":
            return
        if _hook is None:
            # Do NOT latch while the pointer is absent: on deployed runners
            # the collector advertises itself from its first emission, which
            # can land AFTER this library's first fire — a one-shot latch
            # made resolution silently permanent-dead (review, 2026-07-22).
            # One import attempt per present pointer.
            target = os.environ.get("FAL_TIMING_COLLECTOR")
            if target and not _resolve_attempted:
                _resolve_attempted = True
                mod, _, attr = target.partition(":")
                _hook = getattr(importlib.import_module(mod), attr, None)
        if _hook is not None:
            _hook(phase, elapsed_s, **ctx)
    except Exception:
        pass


_ordinals: dict = {}


def next_ordinal(name: str) -> int:
    """Per-name emission ordinal: keeps N same-shape events I2-distinct."""
    n = _ordinals.get(name, 0)
    _ordinals[name] = n + 1
    return n
