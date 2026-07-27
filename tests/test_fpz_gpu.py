"""CPU-testable surface of the fpz GPU decode path.

The GPU decode itself needs a CUDA device and libnvcomp, so it is exercised
on GPU hosts, not in CI. What IS testable without a device -- and what these
cover -- is everything around it: the pure frame-batch planner, the env
gating, the CPU-fallback warnings, and a guard that turning the flag on
never disturbs the CPU decode path.
"""

import pytest
import torch
from flashpack import deserialization
from flashpack.deserialization import (
    _env_flag,
    _fpz_gpu_decode_enabled,
    _fpz_hi_chunk_usizes,
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
# _fpz_hi_chunk_usizes -- pure v2 chunk sizing (matches the encoder)
# --------------------------------------------------------------------------


def test_hi_chunk_usizes_exact_multiple_has_no_tail() -> None:
    # 4 * chunk -> 4 equal chunks, no remainder (the full-frame case).
    assert _fpz_hi_chunk_usizes(4 * 64, 64) == [64, 64, 64, 64]


def test_hi_chunk_usizes_has_remainder_tail() -> None:
    assert _fpz_hi_chunk_usizes(200, 64) == [64, 64, 64, 8]


def test_hi_chunk_usizes_smaller_than_one_chunk() -> None:
    assert _fpz_hi_chunk_usizes(40, 64) == [40]


def test_hi_chunk_usizes_zero_is_empty() -> None:
    assert _fpz_hi_chunk_usizes(0, 64) == []


def test_hi_chunk_usizes_sum_equals_half() -> None:
    for half in (1, 63, 64, 65, 1000, 1 << 20):
        assert sum(_fpz_hi_chunk_usizes(half, 64)) == half


@pytest.mark.parametrize("bad", [0, -1])
def test_hi_chunk_usizes_rejects_bad_chunk(bad: int) -> None:
    with pytest.raises(ValueError):
        _fpz_hi_chunk_usizes(100, bad)


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


# --------------------------------------------------------------------------
# CPU-fallback warning (warn once)
# --------------------------------------------------------------------------


def test_gpu_decoder_missing_libnvcomp_warns_once_and_falls_back(
    monkeypatch,
) -> None:
    from flashpack import _nvcomp_ll

    monkeypatch.setattr(_nvcomp_ll, "_loaded", (None,))
    monkeypatch.setattr(deserialization, "_FPZ_GPU_DECODE_WARNED", False)

    with pytest.warns(RuntimeWarning, match="libnvcomp"):
        assert deserialization._fpz_gpu_decoder([]) is None

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning would raise
        assert deserialization._fpz_gpu_decoder([]) is None


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
