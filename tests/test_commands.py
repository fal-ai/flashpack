"""Unit tests for the conversion helpers in flashpack.commands (file-based
paths only; repo-id paths need network access and are covered elsewhere)."""

import pytest
import safetensors.torch
import torch
from flashpack.commands import (
    convert_to_flashpack,
    convert_to_flashpack_from_state_dict,
    convert_to_flashpack_from_state_dict_file,
    filter_kwargs_for_method,
    revert_from_flashpack,
)
from flashpack.deserialization import (
    get_flashpack_file_metadata,
    is_flashpack_file,
    revert_from_file,
)
from flashpack.serialization import pack_to_file


def _state_dict() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "weight": torch.randn(8, 4, generator=generator),
        "bias": torch.randn(8, generator=generator),
    }


class TestFilterKwargsForMethod:
    def test_filters_to_signature(self) -> None:
        def method(a: int, b: int = 0) -> None:
            pass

        assert filter_kwargs_for_method({"a": 1, "b": 2, "c": 3}, method) == {
            "a": 1,
            "b": 2,
        }

    def test_var_keyword_passes_everything(self) -> None:
        def method(a: int, **kwargs) -> None:
            pass

        kwargs = {"a": 1, "anything": 2}
        assert filter_kwargs_for_method(kwargs, method) == kwargs


class TestConvertFromStateDict:
    def test_roundtrip(self, tmp_path) -> None:
        source = _state_dict()
        destination = str(tmp_path / "model.flashpack")
        result = convert_to_flashpack_from_state_dict(source, destination)
        assert result == destination
        state_dict = revert_from_file(destination)
        for name, tensor in source.items():
            assert torch.equal(state_dict[name], tensor)

    def test_dtype_as_string(self, tmp_path) -> None:
        destination = str(tmp_path / "model.flashpack")
        convert_to_flashpack_from_state_dict(
            _state_dict(), destination, dtype="bfloat16"
        )
        meta = get_flashpack_file_metadata(destination)
        assert meta["target_dtype"] == "bfloat16"

    def test_ignore_filters_applied(self, tmp_path) -> None:
        destination = str(tmp_path / "model.flashpack")
        convert_to_flashpack_from_state_dict(
            _state_dict(), destination, ignore_names=["bias"]
        )
        meta = get_flashpack_file_metadata(destination)
        assert [rec["name"] for rec in meta["index"]] == ["weight"]


class TestConvertFromStateDictFile:
    def test_safetensors_input(self, tmp_path) -> None:
        source = _state_dict()
        source_path = str(tmp_path / "model.safetensors")
        safetensors.torch.save_file(source, source_path)

        destination = str(tmp_path / "model.flashpack")
        convert_to_flashpack_from_state_dict_file(source_path, destination)
        state_dict = revert_from_file(destination)
        for name, tensor in source.items():
            assert torch.equal(state_dict[name], tensor)

    def test_torch_save_input(self, tmp_path) -> None:
        source = _state_dict()
        source_path = str(tmp_path / "model.pt")
        torch.save(source, source_path)

        destination = str(tmp_path / "model.flashpack")
        convert_to_flashpack_from_state_dict_file(source_path, destination)
        state_dict = revert_from_file(destination)
        for name, tensor in source.items():
            assert torch.equal(state_dict[name], tensor)


class TestConvertDispatch:
    def test_file_input_defaults_destination(self, tmp_path) -> None:
        source_path = str(tmp_path / "model.safetensors")
        safetensors.torch.save_file(_state_dict(), source_path)

        result = convert_to_flashpack(source_path)
        assert result == str(tmp_path / "model.flashpack")
        assert is_flashpack_file(result)

    def test_repo_id_without_destination_rejected(self) -> None:
        with pytest.raises(AssertionError, match="destination_path is required"):
            convert_to_flashpack("not-a-file/definitely-not-a-repo")


class TestRevertFromFlashpack:
    def test_default_safetensors_destination(self, tmp_path) -> None:
        source = _state_dict()
        pack_path = str(tmp_path / "model.flashpack")
        pack_to_file(source, pack_path, target_dtype=None)

        result = revert_from_flashpack(pack_path)
        assert result == str(tmp_path / "model.safetensors")
        state_dict = safetensors.torch.load_file(result)
        for name, tensor in source.items():
            assert torch.equal(state_dict[name], tensor)

    def test_torch_save_destination(self, tmp_path) -> None:
        source = _state_dict()
        pack_path = str(tmp_path / "model.flashpack")
        pack_to_file(source, pack_path, target_dtype=None)

        result = revert_from_flashpack(pack_path, str(tmp_path / "model.pt"))
        state_dict = torch.load(result, weights_only=True)
        for name, tensor in source.items():
            assert torch.equal(state_dict[name], tensor)
