"""Unit tests for reading flashpack files: footer parsing, corrupt-file
handling, tensor iteration, and state-dict / module assignment."""

import json

import pytest
import torch
from flashpack.constants import MAGIC, U64LE
from flashpack.deserialization import (
    assign_from_file,
    get_flashpack_file_metadata,
    is_flashpack_file,
    iterate_from_flash_tensor,
    read_flashpack_file,
    revert_from_file,
)
from flashpack.serialization import pack_to_file


class SmallModel(torch.nn.Module):
    def __init__(self, features: int = 8) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(features, features)
        self.register_buffer("scale", torch.full((features,), 2.0))


def _pack(tmp_path, state_dict_or_model, **kwargs) -> str:
    path = str(tmp_path / "pack.flashpack")
    kwargs.setdefault("target_dtype", None)
    pack_to_file(state_dict_or_model, path, **kwargs)
    return path


class TestMetadata:
    def test_reads_back_index(self, tmp_path) -> None:
        path = _pack(tmp_path, {"a": torch.randn(4, 2), "b": torch.randn(3)})
        meta = get_flashpack_file_metadata(path)
        by_name = {rec["name"]: rec for rec in meta["index"]}
        assert by_name["a"]["shape"] == [4, 2]
        assert by_name["a"]["length"] == 8
        assert by_name["b"]["shape"] == [3]

    def test_file_too_small(self, tmp_path) -> None:
        path = tmp_path / "small.bin"
        path.write_bytes(b"tiny")
        with pytest.raises(ValueError, match="File too small"):
            get_flashpack_file_metadata(str(path))

    def test_bad_magic(self, tmp_path) -> None:
        path = tmp_path / "bad.bin"
        path.write_bytes(b"\0" * 64)
        with pytest.raises(ValueError, match="Bad magic"):
            get_flashpack_file_metadata(str(path))

    def test_corrupt_footer_length(self, tmp_path) -> None:
        path = tmp_path / "corrupt.bin"
        # Valid magic, but the footer length points beyond the file start.
        path.write_bytes(b"\0" * 8 + U64LE.pack(10_000) + MAGIC)
        with pytest.raises(ValueError, match="Corrupt footer length"):
            get_flashpack_file_metadata(str(path))

    def test_unexpected_format(self, tmp_path) -> None:
        payload = json.dumps({"format": "flashpack_v999"}).encode()
        path = tmp_path / "future.bin"
        path.write_bytes(payload + U64LE.pack(len(payload)) + MAGIC)
        with pytest.raises(ValueError, match="Unexpected format"):
            get_flashpack_file_metadata(str(path))

    def test_is_flashpack_file(self, tmp_path) -> None:
        packed = _pack(tmp_path, {"a": torch.randn(4)})
        garbage = tmp_path / "garbage.bin"
        garbage.write_bytes(b"\0" * 64)
        assert is_flashpack_file(packed) is True
        assert is_flashpack_file(str(garbage)) is False
        assert is_flashpack_file(str(tmp_path / "does-not-exist")) is False


class TestReadFlashpackFile:
    def test_cpu_read(self, tmp_path) -> None:
        path = _pack(tmp_path, {"a": torch.randn(4)})
        storage, meta = read_flashpack_file(path, device="cpu")
        assert storage.device.type == "cpu"
        assert len(storage) == 1

    def test_unsupported_device(self, tmp_path) -> None:
        path = _pack(tmp_path, {"a": torch.randn(4)})
        with pytest.raises(ValueError, match="Unsupported device"):
            read_flashpack_file(path, device="meta")


class TestIterate:
    def test_yields_original_values_and_shapes(self, tmp_path) -> None:
        source = {"a": torch.randn(4, 2), "b": torch.randn(3)}
        storage, meta = read_flashpack_file(_pack(tmp_path, source))
        tensors = dict(iterate_from_flash_tensor(storage, meta))
        assert set(tensors) == {"a", "b"}
        for name, tensor in source.items():
            assert tensors[name].shape == tensor.shape
            assert torch.equal(tensors[name], tensor)

    def test_ignore_filters(self, tmp_path) -> None:
        source = {
            "keep.weight": torch.randn(2),
            "rope.freqs": torch.randn(2),
            "drop.bias": torch.randn(2),
        }
        storage, meta = read_flashpack_file(_pack(tmp_path, source))
        names = {
            name
            for name, _ in iterate_from_flash_tensor(
                storage, meta, ignore_prefixes=["rope"], ignore_suffixes=[".bias"]
            )
        }
        assert names == {"keep.weight"}

    def test_invalid_macroblock_reference_rejected(self, tmp_path) -> None:
        storage, meta = read_flashpack_file(_pack(tmp_path, {"a": torch.randn(4)}))
        meta["index"][0]["macroblock"] = 5
        with pytest.raises(ValueError, match="macroblock"):
            list(iterate_from_flash_tensor(storage, meta))

    def test_misaligned_index_rejected(self, tmp_path) -> None:
        path = _pack(tmp_path, {"a": torch.randn(8), "b": torch.randn(8)})
        storage, meta = read_flashpack_file(path)
        meta["index"][1]["offset"] += 1
        with pytest.raises(ValueError, match="misaligned"):
            list(iterate_from_flash_tensor(storage, meta))

    def test_out_of_bounds_record_rejected(self, tmp_path) -> None:
        storage, meta = read_flashpack_file(_pack(tmp_path, {"a": torch.randn(4)}))
        meta["index"][0]["length"] = 10_000
        with pytest.raises(ValueError, match="Could not get tensor"):
            list(iterate_from_flash_tensor(storage, meta))


class TestRevert:
    def test_roundtrip_matches_source(self, tmp_path) -> None:
        source = {
            "a": torch.randn(16, 4),
            "b": torch.randn(5).to(torch.bfloat16),
            "c": torch.arange(6, dtype=torch.int64),
        }
        state_dict = revert_from_file(_pack(tmp_path, source))
        assert set(state_dict) == set(source)
        for name, tensor in source.items():
            assert state_dict[name].dtype == tensor.dtype
            assert torch.equal(state_dict[name], tensor)


class TestAssignFromFile:
    def test_assigns_params_and_buffers(self, tmp_path) -> None:
        torch.manual_seed(0)
        source = SmallModel()
        source.scale.mul_(3.0)
        torch.manual_seed(1)
        destination = SmallModel()
        path = _pack(tmp_path, source)

        assign_from_file(destination, path, device="cpu")

        assert torch.equal(destination.linear.weight, source.linear.weight)
        assert torch.equal(destination.linear.bias, source.linear.bias)
        assert torch.equal(destination.scale, source.scale)

    def test_device_inferred_from_model(self, tmp_path) -> None:
        source = SmallModel()
        destination = SmallModel()
        assign_from_file(destination, _pack(tmp_path, source), device=None)
        assert destination.linear.weight.device.type == "cpu"

    def test_requires_grad_preserved(self, tmp_path) -> None:
        source = SmallModel()
        destination = SmallModel()
        destination.linear.weight.requires_grad_(False)
        assign_from_file(destination, _pack(tmp_path, source), device="cpu")
        assert destination.linear.weight.requires_grad is False
        assert destination.linear.bias.requires_grad is True

    def test_missing_param_strict_rejected(self, tmp_path) -> None:
        class BiggerModel(SmallModel):
            def __init__(self, features: int = 8) -> None:
                super().__init__(features)
                self.extra = torch.nn.Parameter(torch.zeros(features))

        path = _pack(tmp_path, SmallModel())
        with pytest.raises(ValueError, match="Missing 1 parameters"):
            assign_from_file(BiggerModel(), path, device="cpu")

    def test_missing_param_allowed_when_not_strict(self, tmp_path) -> None:
        class BiggerModel(SmallModel):
            def __init__(self, features: int = 8) -> None:
                super().__init__(features)
                self.extra = torch.nn.Parameter(torch.zeros(features))

        path = _pack(tmp_path, SmallModel())
        destination = BiggerModel()
        assign_from_file(destination, path, device="cpu", strict_params=False)
        assert torch.equal(destination.extra, torch.zeros(8))

    def test_unknown_name_in_pack_rejected(self, tmp_path) -> None:
        source_dict = dict(SmallModel().state_dict())
        source_dict["unknown.weight"] = torch.randn(4)
        path = _pack(tmp_path, source_dict)
        with pytest.raises(ValueError, match="Could not assign"):
            assign_from_file(SmallModel(), path, device="cpu")

    def test_ignored_names_skipped_and_not_required(self, tmp_path) -> None:
        source = SmallModel()
        destination = SmallModel()
        original_bias = destination.linear.bias.detach().clone()
        assign_from_file(
            destination,
            _pack(tmp_path, source),
            device="cpu",
            ignore_names=["linear.bias"],
        )
        assert torch.equal(destination.linear.weight, source.linear.weight)
        assert torch.equal(destination.linear.bias, original_bias)

    def test_buffer_dtype_mismatch_rejected(self, tmp_path) -> None:
        class DoubleBufferModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)
                self.register_buffer("scale", torch.ones(8, dtype=torch.float64))

        path = _pack(tmp_path, SmallModel())
        with pytest.raises(ValueError, match="Error while assigning"):
            assign_from_file(DoubleBufferModel(), path, device="cpu")

    def test_buffer_dtype_mismatch_coerced(self, tmp_path) -> None:
        class DoubleBufferModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)
                self.register_buffer("scale", torch.ones(8, dtype=torch.float64))

        source = SmallModel()
        destination = DoubleBufferModel()
        assign_from_file(
            destination, _pack(tmp_path, source), device="cpu", coerce_dtype=True
        )
        assert destination.scale.dtype is torch.float64
        assert torch.equal(destination.scale, source.scale.to(torch.float64))

    def test_keep_flash_ref_on_model(self, tmp_path) -> None:
        destination = SmallModel()
        assign_from_file(
            destination,
            _pack(tmp_path, SmallModel()),
            device="cpu",
            keep_flash_ref_on_model=True,
        )
        assert hasattr(destination, "_flash_shared_storage")
        assert hasattr(destination, "_flash_shared_storage_meta")

    def test_no_flash_ref_by_default(self, tmp_path) -> None:
        destination = SmallModel()
        assign_from_file(destination, _pack(tmp_path, SmallModel()), device="cpu")
        assert not hasattr(destination, "_flash_shared_storage")
