"""Unit tests for pack_to_file: file format, alignment, dtype handling, and
atomic-write behavior."""

import os

import pytest
import torch
from flashpack.constants import (
    DEFAULT_ALIGN_BYTES,
    FILE_FORMAT_V3,
    FILE_FORMAT_V4,
    MAGIC,
)
from flashpack.deserialization import (
    get_flashpack_file_metadata,
    revert_from_file,
)
from flashpack.serialization import pack_to_file


def _random_state_dict() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "large.weight": torch.randn(64, 32, generator=generator),
        "small.weight": torch.randn(8, generator=generator),
        "tiny.bias": torch.randn(3, generator=generator),
    }


def test_pack_writes_magic_trailer(tmp_path) -> None:
    path = str(tmp_path / "pack.flashpack")
    pack_to_file(_random_state_dict(), path, target_dtype=None)
    with open(path, "rb") as f:
        f.seek(-len(MAGIC), os.SEEK_END)
        assert f.read() == MAGIC


def test_single_dtype_produces_v3(tmp_path) -> None:
    path = str(tmp_path / "pack.flashpack")
    pack_to_file(_random_state_dict(), path, target_dtype=None)
    meta = get_flashpack_file_metadata(path)
    assert meta["format"] == FILE_FORMAT_V3
    assert meta["target_dtype"] == "float32"
    assert meta["align_bytes"] == DEFAULT_ALIGN_BYTES


def test_multiple_dtypes_produce_v4_macroblocks(tmp_path) -> None:
    state_dict = {
        "a": torch.randn(16),
        "b": torch.randn(16).to(torch.bfloat16),
        "c": torch.arange(16, dtype=torch.int32),
    }
    path = str(tmp_path / "pack.flashpack")
    pack_to_file(state_dict, path, target_dtype=None)
    meta = get_flashpack_file_metadata(path)
    assert meta["format"] == FILE_FORMAT_V4
    assert len(meta["macroblocks"]) == 3
    dtypes = {block["dtype"] for block in meta["macroblocks"]}
    assert dtypes == {"float32", "bfloat16", "int32"}
    # Macroblocks must not overlap and stay within the payload.
    blocks = sorted(meta["macroblocks"], key=lambda b: b["offset_bytes"])
    cursor = 0
    for block in blocks:
        assert block["offset_bytes"] >= cursor
        cursor = block["offset_bytes"] + block["length_bytes"]
    assert cursor == meta["total_payload_bytes"]


def test_target_dtype_converts_all_tensors(tmp_path) -> None:
    path = str(tmp_path / "pack.flashpack")
    pack_to_file(_random_state_dict(), path, target_dtype=torch.bfloat16)
    meta = get_flashpack_file_metadata(path)
    assert meta["format"] == FILE_FORMAT_V3
    assert meta["target_dtype"] == "bfloat16"
    state_dict = revert_from_file(path)
    assert all(t.dtype is torch.bfloat16 for t in state_dict.values())


def test_index_offsets_respect_alignment(tmp_path) -> None:
    align_bytes = 64
    state_dict = {"a": torch.randn(5), "b": torch.randn(7), "c": torch.randn(11)}
    path = str(tmp_path / "pack.flashpack")
    pack_to_file(state_dict, path, target_dtype=None, align_bytes=align_bytes)
    meta = get_flashpack_file_metadata(path)
    align_elems = align_bytes // torch.tensor([], dtype=torch.float32).element_size()
    for rec in meta["index"]:
        assert rec["offset"] % align_elems == 0


def test_zero_align_bytes_packs_densely(tmp_path) -> None:
    state_dict = {"a": torch.randn(5), "b": torch.randn(7)}
    path = str(tmp_path / "pack.flashpack")
    pack_to_file(state_dict, path, target_dtype=None, align_bytes=0)
    meta = get_flashpack_file_metadata(path)
    index = sorted(meta["index"], key=lambda r: r["offset"])
    cursor = 0
    for rec in index:
        assert rec["offset"] == cursor
        cursor += rec["length"]
    assert meta["total_elems"] == 12


def test_default_order_is_largest_first(tmp_path) -> None:
    path = str(tmp_path / "pack.flashpack")
    pack_to_file(_random_state_dict(), path, target_dtype=None)
    meta = get_flashpack_file_metadata(path)
    lengths = [rec["length"] for rec in meta["index"]]
    assert lengths == sorted(lengths, reverse=True)


def test_name_order_selects_and_orders_tensors(tmp_path) -> None:
    path = str(tmp_path / "pack.flashpack")
    # Unknown names are dropped silently; listed names keep their order.
    pack_to_file(
        _random_state_dict(),
        path,
        target_dtype=None,
        name_order=["tiny.bias", "large.weight", "not.a.tensor"],
    )
    meta = get_flashpack_file_metadata(path)
    assert [rec["name"] for rec in meta["index"]] == ["tiny.bias", "large.weight"]


def test_module_input_uses_state_dict(tmp_path) -> None:
    model = torch.nn.Linear(4, 4)
    path = str(tmp_path / "pack.flashpack")
    pack_to_file(model, path, target_dtype=None)
    state_dict = revert_from_file(path)
    assert set(state_dict) == {"weight", "bias"}
    assert torch.equal(state_dict["weight"], model.weight.detach())


def test_empty_state_dict_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="No tensors to pack"):
        pack_to_file({}, str(tmp_path / "pack.flashpack"), target_dtype=None)


def test_name_order_without_matches_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="No tensors to pack"):
        pack_to_file(
            {"a": torch.randn(2)},
            str(tmp_path / "pack.flashpack"),
            target_dtype=None,
            name_order=["missing"],
        )


def test_negative_align_bytes_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="align_bytes must be >= 0"):
        pack_to_file(
            {"a": torch.randn(2)},
            str(tmp_path / "pack.flashpack"),
            target_dtype=None,
            align_bytes=-1,
        )


def test_non_dtype_target_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        pack_to_file(
            {"a": torch.randn(2)},
            str(tmp_path / "pack.flashpack"),
            target_dtype="float32",  # type: ignore[arg-type]
        )


def test_overwrites_existing_destination(tmp_path) -> None:
    path = str(tmp_path / "pack.flashpack")
    pack_to_file({"a": torch.zeros(4)}, path, target_dtype=None)
    pack_to_file({"b": torch.ones(8)}, path, target_dtype=None)
    state_dict = revert_from_file(path)
    assert set(state_dict) == {"b"}
    assert torch.equal(state_dict["b"], torch.ones(8))


def test_no_temp_files_left_after_success(tmp_path) -> None:
    pack_to_file(
        _random_state_dict(), str(tmp_path / "pack.flashpack"), target_dtype=None
    )
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".packtmp_")]
    assert leftovers == []


def test_no_temp_files_left_after_failure(tmp_path, monkeypatch) -> None:
    """A failure after the temp file is created must clean it up and leave the
    destination untouched."""
    import flashpack.serialization as serialization

    def boom(*args, **kwargs):
        raise OSError("simulated memmap failure")

    monkeypatch.setattr(serialization.np, "memmap", boom)
    path = tmp_path / "pack.flashpack"
    with pytest.raises(OSError, match="simulated memmap failure"):
        pack_to_file({"a": torch.randn(4)}, str(path), target_dtype=None)
    assert not path.exists()
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".packtmp_")]
    assert leftovers == []


def test_creates_missing_destination_directory(tmp_path) -> None:
    path = tmp_path / "nested" / "dirs" / "pack.flashpack"
    pack_to_file({"a": torch.randn(4)}, str(path), target_dtype=None)
    assert path.exists()
