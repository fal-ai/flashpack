"""CPU-testable surface of the nvcomp GPU Zstd decode path for fpz.

The GPU decode itself needs a CUDA device and nvcomp, so it is exercised on the
H200, not in CI. What IS testable without a device -- and what these cover -- is
everything around it: the pure frame-batch planner, the env gating, and the
guarded-import fallback (including warn-once), plus a guard that turning the
flag on never disturbs the CPU decode path.
"""

import sys

import pytest
import torch
from flashpack import deserialization
from flashpack.deserialization import (
    _env_flag,
    _env_flag_default,
    _fpz_batch_signature,
    _fpz_gpu_decode_enabled,
    _load_nvcomp,
    iterate_from_flash_tensor,
    plan_fpz_gpu_batches,
    read_flashpack_file,
)
from flashpack.serialization import pack_to_file


def _frame_task(block_idx: int, n_out: int, out_pos: int = 0) -> tuple:
    # Matches _fpz_frame_tasks' ("frame", block_idx, frame, out_pos) shape.
    return ("frame", block_idx, {"n_out": n_out, "payload_off": 0}, out_pos)


# --------------------------------------------------------------------------
# plan_fpz_gpu_batches -- pure function
# --------------------------------------------------------------------------


def test_plan_empty_input_is_empty() -> None:
    assert plan_fpz_gpu_batches([], max_batch_frames=4, max_batch_bytes=1 << 30) == []


def test_plan_single_frame_is_one_batch() -> None:
    tasks = [_frame_task(0, 100)]
    assert plan_fpz_gpu_batches(tasks, 4, 1 << 30) == [tasks]


def test_plan_bounded_by_frame_count() -> None:
    tasks = [_frame_task(0, 10) for _ in range(10)]
    batches = plan_fpz_gpu_batches(tasks, max_batch_frames=3, max_batch_bytes=1 << 30)
    assert [len(b) for b in batches] == [3, 3, 3, 1]
    # Every frame appears exactly once, order preserved.
    assert [t for b in batches for t in b] == tasks


def test_plan_bounded_by_byte_budget() -> None:
    tasks = [_frame_task(0, 100) for _ in range(5)]
    # Budget of 250 bytes -> 2 frames per batch before the count cap bites.
    batches = plan_fpz_gpu_batches(tasks, max_batch_frames=99, max_batch_bytes=250)
    assert [len(b) for b in batches] == [2, 2, 1]
    for b in batches:
        assert sum(int(t[2]["n_out"]) for t in b) <= 250


def test_plan_oversized_frame_gets_its_own_batch() -> None:
    # A frame larger than the whole budget must not be dropped or merged away.
    tasks = [_frame_task(0, 50), _frame_task(0, 500), _frame_task(0, 50)]
    batches = plan_fpz_gpu_batches(tasks, max_batch_frames=99, max_batch_bytes=100)
    assert [[int(t[2]["n_out"]) for t in b] for b in batches] == [[50], [500], [50]]


def test_plan_preserves_frames_across_block_boundaries() -> None:
    # Frames from different macroblocks are grouped purely by order/size; the
    # per-frame block index and out_pos ride along untouched.
    tasks = [_frame_task(0, 10, 0), _frame_task(0, 10, 10), _frame_task(1, 10, 0)]
    batches = plan_fpz_gpu_batches(tasks, max_batch_frames=2, max_batch_bytes=1 << 30)
    assert [len(b) for b in batches] == [2, 1]
    assert [t for b in batches for t in b] == tasks


@pytest.mark.parametrize("bad", [0, -1])
def test_plan_rejects_nonpositive_frame_cap(bad: int) -> None:
    with pytest.raises(ValueError, match="max_batch_frames"):
        plan_fpz_gpu_batches(
            [_frame_task(0, 1)], max_batch_frames=bad, max_batch_bytes=1
        )


@pytest.mark.parametrize("bad", [0, -1])
def test_plan_rejects_nonpositive_byte_budget(bad: int) -> None:
    with pytest.raises(ValueError, match="max_batch_bytes"):
        plan_fpz_gpu_batches(
            [_frame_task(0, 1)], max_batch_frames=1, max_batch_bytes=bad
        )


# --------------------------------------------------------------------------
# _fpz_batch_signature -- pure config-cache key
# --------------------------------------------------------------------------


def test_signature_is_per_frame_half_sizes() -> None:
    batch = [_frame_task(0, 100), _frame_task(0, 40)]
    assert _fpz_batch_signature(batch) == (50, 20)


def test_signature_matches_for_same_shape_batches() -> None:
    # Two batches of identical full-frame shapes hash to the same key -> they
    # share one cached DecompressConfig (the whole point of the cache).
    a = [_frame_task(0, 128), _frame_task(0, 128)]
    b = [_frame_task(1, 128), _frame_task(2, 128)]
    assert _fpz_batch_signature(a) == _fpz_batch_signature(b) == (64, 64)


def test_signature_differs_when_a_tail_frame_changes_shape() -> None:
    full = [_frame_task(0, 128), _frame_task(0, 128)]
    with_tail = [_frame_task(0, 128), _frame_task(0, 40)]
    assert _fpz_batch_signature(full) != _fpz_batch_signature(with_tail)


# --------------------------------------------------------------------------
# env gating
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_env_flag_truthy(monkeypatch, value: str) -> None:
    monkeypatch.setenv("FLASHPACK_FPZ_GPU_DECODE", value)
    assert _env_flag("FLASHPACK_FPZ_GPU_DECODE") is True
    assert _fpz_gpu_decode_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "maybe"])
def test_env_flag_falsy(monkeypatch, value: str) -> None:
    monkeypatch.setenv("FLASHPACK_FPZ_GPU_DECODE", value)
    assert _env_flag("FLASHPACK_FPZ_GPU_DECODE") is False
    assert _fpz_gpu_decode_enabled() is False


def test_env_flag_unset_is_false(monkeypatch) -> None:
    monkeypatch.delenv("FLASHPACK_FPZ_GPU_DECODE", raising=False)
    assert _fpz_gpu_decode_enabled() is False


def test_env_flag_default_respects_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("FLASHPACK_FPZ_GPU_TORCH_ALLOC", raising=False)
    assert _env_flag_default("FLASHPACK_FPZ_GPU_TORCH_ALLOC", True) is True
    assert _env_flag_default("FLASHPACK_FPZ_GPU_TORCH_ALLOC", False) is False


@pytest.mark.parametrize(
    "value,expected", [("0", False), ("false", False), ("1", True), ("on", True)]
)
def test_env_flag_default_env_overrides(monkeypatch, value, expected) -> None:
    monkeypatch.setenv("FLASHPACK_FPZ_GPU_TORCH_ALLOC", value)
    # Env always wins over the default, in both directions.
    assert _env_flag_default("FLASHPACK_FPZ_GPU_TORCH_ALLOC", True) is expected
    assert _env_flag_default("FLASHPACK_FPZ_GPU_TORCH_ALLOC", False) is expected


# --------------------------------------------------------------------------
# torch caching allocator wiring into nvcomp (guarded, idempotent)
# --------------------------------------------------------------------------


class _FakeNvcompAllocOK:
    def __init__(self) -> None:
        self.installed = None

    def set_device_allocator(self, allocator) -> None:
        self.installed = allocator


class _FakeNvcompAllocRaises:
    def set_device_allocator(self, allocator) -> None:
        raise RuntimeError("no allocator hook here")


def _reset_alloc_latches(monkeypatch) -> None:
    monkeypatch.setattr(deserialization, "_FPZ_NVCOMP_ALLOC_INSTALLED", False)
    monkeypatch.setattr(deserialization, "_FPZ_NVCOMP_ALLOC_WARNED", False)


def test_install_allocator_success_registers_a_callable(monkeypatch) -> None:
    _reset_alloc_latches(monkeypatch)
    fake = _FakeNvcompAllocOK()
    ok = deserialization._install_torch_nvcomp_allocator(fake, torch.device("cuda:0"))
    assert ok is True
    assert callable(fake.installed)


def test_install_allocator_is_idempotent(monkeypatch) -> None:
    _reset_alloc_latches(monkeypatch)
    first = _FakeNvcompAllocOK()
    assert deserialization._install_torch_nvcomp_allocator(
        first, torch.device("cuda:0")
    )
    # Already installed globally: a second call is a no-op and does not
    # re-register on another (fake) module.
    second = _FakeNvcompAllocOK()
    assert deserialization._install_torch_nvcomp_allocator(
        second, torch.device("cuda:0")
    )
    assert second.installed is None


def test_install_allocator_failure_is_guarded_and_warns(monkeypatch) -> None:
    _reset_alloc_latches(monkeypatch)
    with pytest.warns(RuntimeWarning, match="set_device_allocator"):
        ok = deserialization._install_torch_nvcomp_allocator(
            _FakeNvcompAllocRaises(), torch.device("cuda:0")
        )
    assert ok is False


# --------------------------------------------------------------------------
# guarded import + warn-once fallback
# --------------------------------------------------------------------------


def test_load_nvcomp_missing_returns_none_and_warns_once(monkeypatch) -> None:
    # Force the import to fail regardless of what's installed, and reset the
    # process-level warn-once latch so the assertion is deterministic.
    monkeypatch.setitem(sys.modules, "nvidia", None)
    monkeypatch.setattr(deserialization, "_FPZ_GPU_DECODE_WARNED", False)

    with pytest.warns(RuntimeWarning, match="nvcomp"):
        assert _load_nvcomp() is None

    # Second call: still None, but no second warning (warn-once).
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning would raise
        assert _load_nvcomp() is None


# --------------------------------------------------------------------------
# the flag must never disturb the CPU decode path
# --------------------------------------------------------------------------


def test_gpu_flag_on_leaves_cpu_path_bit_identical(tmp_path, monkeypatch) -> None:
    # device="cpu" never touches the GPU branch, so enabling the flag (with no
    # nvcomp available) must produce exactly the same bytes as with it off.
    monkeypatch.setattr(deserialization, "_FPZ_GPU_DECODE_WARNED", False)
    source = {"w": torch.randn(1024, 512).to(torch.bfloat16)}
    comp = str(tmp_path / "comp.flashpack")
    pack_to_file(source, comp, target_dtype=None, compress="fpz-bf16")

    monkeypatch.delenv("FLASHPACK_FPZ_GPU_DECODE", raising=False)
    s_off, m_off = read_flashpack_file(comp, device="cpu")
    off = dict(iterate_from_flash_tensor(s_off, m_off))

    monkeypatch.setenv("FLASHPACK_FPZ_GPU_DECODE", "1")
    s_on, m_on = read_flashpack_file(comp, device="cpu")
    on = dict(iterate_from_flash_tensor(s_on, m_on))

    assert set(on) == set(off) == set(source)
    for name in source:
        assert torch.equal(
            on[name].contiguous().view(torch.uint16),
            off[name].contiguous().view(torch.uint16),
        )
