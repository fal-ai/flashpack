"""Unit tests for flashpack.utils helpers."""

import numpy as np
import pytest
import torch
from flashpack.utils import (
    dtype_to_string,
    filter_state_dict,
    get_module_and_attribute,
    get_packing_dtype,
    human_duration,
    human_num_elements,
    is_ignored_tensor_name,
    string_to_dtype,
    torch_dtype_to_numpy_dtype,
)


class TestDtypeStrings:
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.uint8,
            torch.int64,
            torch.bool,
            torch.complex64,
            torch.float8_e4m3fn,
        ],
    )
    def test_roundtrip(self, dtype: torch.dtype) -> None:
        assert string_to_dtype(dtype_to_string(dtype)) is dtype

    def test_dtype_to_string_values(self) -> None:
        assert dtype_to_string(torch.float32) == "float32"
        assert dtype_to_string(torch.bfloat16) == "bfloat16"

    def test_dtype_to_string_rejects_non_dtype(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            dtype_to_string("float32")  # type: ignore[arg-type]

    def test_string_to_dtype_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype string"):
            string_to_dtype("not_a_dtype")

    def test_string_to_dtype_rejects_non_dtype_torch_attribute(self) -> None:
        # ``torch.Tensor`` exists but is not a dtype; it must not slip through.
        with pytest.raises(ValueError, match="Unsupported dtype string"):
            string_to_dtype("Tensor")


class TestPackingDtype:
    def test_fp8_packs_as_uint8(self) -> None:
        assert get_packing_dtype(torch.float8_e4m3fn) is torch.uint8
        assert get_packing_dtype(torch.float8_e5m2) is torch.uint8

    def test_bfloat16_packs_as_uint16(self) -> None:
        assert get_packing_dtype(torch.bfloat16) is torch.uint16

    def test_complex32_packs_as_uint32(self) -> None:
        assert get_packing_dtype(torch.complex32) is torch.uint32

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.int8, torch.int64, torch.bool]
    )
    def test_native_dtypes_pack_as_themselves(self, dtype: torch.dtype) -> None:
        assert get_packing_dtype(dtype) is dtype

    def test_float4_rejected(self) -> None:
        float4 = getattr(torch, "float4_e2m1fn", None)
        if float4 is None:
            pytest.skip("this torch build has no float4_e2m1fn")
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_packing_dtype(float4)


class TestTorchToNumpyDtype:
    def test_direct_mappings(self) -> None:
        assert torch_dtype_to_numpy_dtype(torch.float32) is np.float32
        assert torch_dtype_to_numpy_dtype(torch.int64) is np.int64
        assert torch_dtype_to_numpy_dtype(torch.bool) is np.bool_

    def test_bit_reinterpreted_mappings(self) -> None:
        assert torch_dtype_to_numpy_dtype(torch.bfloat16) is np.uint16
        assert torch_dtype_to_numpy_dtype(torch.complex32) is np.uint32
        assert torch_dtype_to_numpy_dtype(torch.float8_e4m3fn) is np.uint8

    def test_unsupported_dtype_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            torch_dtype_to_numpy_dtype(torch.qint8)


class TestHumanReadable:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (5e-7, "500.00ns"),
            (5e-4, "500.00µs"),
            (0.5, "500.00ms"),
            (5.0, "5.00s"),
            (65.0, "1m5s"),
            (3665.0, "1h1m5s"),
        ],
    )
    def test_human_duration(self, seconds: float, expected: str) -> None:
        assert human_duration(seconds) == expected

    @pytest.mark.parametrize(
        ("elements", "expected"),
        [
            (5, "5"),
            (5_000, "5.00K"),
            (5_000_000, "5.00M"),
            (5_000_000_000, "5.00B"),
        ],
    )
    def test_human_num_elements(self, elements: int, expected: str) -> None:
        assert human_num_elements(elements) == expected


class TestIgnoreFilters:
    def test_no_filters_ignores_nothing(self) -> None:
        assert is_ignored_tensor_name("layer.weight") is False

    def test_ignore_by_name(self) -> None:
        assert is_ignored_tensor_name("a.b", ignore_names=["a.b"]) is True
        assert is_ignored_tensor_name("a.c", ignore_names=["a.b"]) is False

    def test_ignore_by_prefix(self) -> None:
        assert is_ignored_tensor_name("rope.freqs", ignore_prefixes=["rope"]) is True
        assert is_ignored_tensor_name("attn.rope", ignore_prefixes=["rope"]) is False

    def test_ignore_by_suffix(self) -> None:
        assert is_ignored_tensor_name("a.bias", ignore_suffixes=[".bias"]) is True
        assert is_ignored_tensor_name("a.weight", ignore_suffixes=[".bias"]) is False

    def test_filter_state_dict(self) -> None:
        state_dict = {
            "keep.weight": torch.zeros(1),
            "rope.freqs": torch.zeros(1),
            "drop.bias": torch.zeros(1),
        }
        filtered = filter_state_dict(
            state_dict, ignore_prefixes=["rope"], ignore_suffixes=[".bias"]
        )
        assert list(filtered) == ["keep.weight"]


class TestGetModuleAndAttribute:
    def test_nested_parameter(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        module, attr = get_module_and_attribute(model, "0.weight")
        assert module is model[0]
        assert attr == "weight"

    def test_top_level_attribute(self) -> None:
        model = torch.nn.Linear(2, 2)
        module, attr = get_module_and_attribute(model, "weight")
        assert module is model
        assert attr == "weight"

    def test_missing_module_raises(self) -> None:
        model = torch.nn.Linear(2, 2)
        with pytest.raises(AttributeError):
            get_module_and_attribute(model, "does.not.exist")
