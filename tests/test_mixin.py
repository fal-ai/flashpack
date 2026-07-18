"""Unit tests for FlashPackMixin.from_flashpack / save_flashpack behavior."""

import pytest
import torch
from flashpack import FlashPackMixin


class Model(torch.nn.Module, FlashPackMixin):
    def __init__(self, features: int = 8) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(features, features)
        self.register_buffer("scale", torch.full((features,), 2.0))


def _save(model: torch.nn.Module, tmp_path) -> str:
    path = str(tmp_path / "model.flashpack")
    model.save_flashpack(path, target_dtype=None, silent=True)
    return path


def test_roundtrip_preserves_values(tmp_path) -> None:
    torch.manual_seed(0)
    source = Model()
    loaded = Model.from_flashpack(_save(source, tmp_path), features=8, silent=True)
    assert torch.equal(loaded.linear.weight, source.linear.weight)
    assert torch.equal(loaded.linear.bias, source.linear.bias)
    assert torch.equal(loaded.scale, source.scale)


def test_device_accepts_string(tmp_path) -> None:
    path = _save(Model(), tmp_path)
    loaded = Model.from_flashpack(path, features=8, device="cpu", silent=True)
    assert loaded.linear.weight.device.type == "cpu"


def test_unexpected_kwargs_are_filtered(tmp_path) -> None:
    """Kwargs not accepted by the init function must be dropped, not raise
    TypeError (mirrors how integrations forward loose ``from_pretrained``
    kwargs)."""
    path = _save(Model(), tmp_path)
    loaded = Model.from_flashpack(
        path, features=8, silent=True, not_a_real_kwarg=123, another_one="x"
    )
    assert isinstance(loaded, Model)


def test_flashpack_init_method_is_used(tmp_path) -> None:
    class InitMethodModel(torch.nn.Module, FlashPackMixin):
        flashpack_init_method = "build"

        def __init__(self, features: int = 8, built_via_build: bool = False) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(features, features)
            self.built_via_build = built_via_build

        @classmethod
        def build(cls, features: int = 8) -> "InitMethodModel":
            return cls(features=features, built_via_build=True)

    source = InitMethodModel()
    path = str(tmp_path / "model.flashpack")
    source.save_flashpack(path, target_dtype=None, silent=True)

    loaded = InitMethodModel.from_flashpack(path, features=8, silent=True)
    assert loaded.built_via_build is True


def test_explicit_init_fn_wins(tmp_path) -> None:
    path = _save(Model(), tmp_path)
    calls: list[int] = []

    def init_fn(features: int = 8) -> Model:
        calls.append(features)
        return Model(features=features)

    loaded = Model.from_flashpack(path, features=8, silent=True, init_fn=init_fn)
    assert isinstance(loaded, Model)
    assert calls == [8]


def test_class_level_ignore_names(tmp_path) -> None:
    class IgnoringModel(Model):
        flashpack_ignore_names = ["scale"]

    source = IgnoringModel()
    source.scale.mul_(21.0)  # pack a value that must NOT be restored
    path = _save(source, tmp_path)

    loaded = IgnoringModel.from_flashpack(path, features=8, silent=True)
    # The ignored buffer keeps its __init__ value instead of the packed one.
    assert torch.equal(loaded.scale, torch.full((8,), 2.0))
    assert torch.equal(loaded.linear.weight, source.linear.weight)


def test_buffer_dtype_mismatch_rejected_without_coercion(tmp_path) -> None:
    class DoubleBufferModel(torch.nn.Module, FlashPackMixin):
        def __init__(self, features: int = 8) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(features, features)
            self.register_buffer("scale", torch.ones(features, dtype=torch.float64))

    path = _save(Model(), tmp_path)
    with pytest.raises(ValueError, match="Error while assigning"):
        DoubleBufferModel.from_flashpack(path, features=8, silent=True)


def test_flashpack_coerce_dtype_class_flag(tmp_path) -> None:
    class CoercingModel(torch.nn.Module, FlashPackMixin):
        flashpack_coerce_dtype = True

        def __init__(self, features: int = 8) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(features, features)
            self.register_buffer("scale", torch.ones(features, dtype=torch.float64))

    source = Model()
    path = _save(source, tmp_path)

    loaded = CoercingModel.from_flashpack(path, features=8, silent=True)
    assert loaded.scale.dtype is torch.float64
    assert torch.equal(loaded.scale, source.scale.to(torch.float64))


def test_keep_flash_ref_on_model(tmp_path) -> None:
    path = _save(Model(), tmp_path)
    loaded = Model.from_flashpack(
        path, features=8, silent=True, keep_flash_ref_on_model=True
    )
    assert hasattr(loaded, "_flash_shared_storage")
    assert hasattr(loaded, "_flash_shared_storage_meta")


def test_no_flash_ref_by_default(tmp_path) -> None:
    path = _save(Model(), tmp_path)
    loaded = Model.from_flashpack(path, features=8, silent=True)
    assert not hasattr(loaded, "_flash_shared_storage")


def test_missing_param_strict_by_default(tmp_path) -> None:
    class BiggerModel(Model):
        def __init__(self, features: int = 8) -> None:
            super().__init__(features)
            self.extra = torch.nn.Parameter(torch.zeros(features))

    path = _save(Model(), tmp_path)
    with pytest.raises(ValueError, match="Missing 1 parameters"):
        BiggerModel.from_flashpack(path, features=8, silent=True)
