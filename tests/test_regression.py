"""Regression tests for flashpack loading correctness.

Each test targets a bug found while loading a diffusers/transformers stack via
flashpack (see fal-ai/registry#11335):

* ``from_flashpack`` returning models in training mode (dropout active).
* the dtype helpers raising ``AttributeError`` on torch builds without the newer
  fp8 dtypes.
* non-persistent buffers being left on the meta device.
"""

import importlib

import numpy as np
import pytest
import torch
from flashpack import FlashPackMixin


class ModelWithDropout(torch.nn.Module, FlashPackMixin):
    def __init__(self, features: int = 8) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(features, features)
        self.dropout = torch.nn.Dropout(p=0.5)


class ModelWithBuffers(torch.nn.Module, FlashPackMixin):
    """Mirrors the CLIP/rotary pattern: a persistent buffer that lands in the
    pack and a non-persistent buffer that does not."""

    def __init__(self, features: int = 8) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(features, features)
        # Persistent buffer: written to the pack via ``state_dict()``.
        self.register_buffer("scale", torch.full((features,), 3.0))
        # Non-persistent buffer: absent from ``state_dict()``, like CLIP
        # ``position_ids`` or rotary ``inv_freq``. Must be materialized by
        # ``__init__`` rather than restored from the pack.
        self.register_buffer(
            "position_ids",
            torch.arange(features, dtype=torch.long),
            persistent=False,
        )


def _save(model: torch.nn.Module, tmp_path) -> str:
    path = str(tmp_path / "model.flashpack")
    model.save_flashpack(path, target_dtype=torch.float32)
    return path


def test_from_flashpack_returns_eval_mode(tmp_path) -> None:
    """Bug 1: loaded models must be in eval mode, matching diffusers/transformers
    ``from_pretrained`` (dropout deactivated)."""
    source = ModelWithDropout()
    source.train()  # ensure the source is in train mode before packing
    path = _save(source, tmp_path)

    loaded = ModelWithDropout.from_flashpack(path, features=8)

    assert loaded.training is False
    # ``eval()`` must propagate to submodules so dropout is deactivated.
    assert loaded.dropout.training is False


def test_dtype_helpers_without_float8_e8m0fnu(monkeypatch) -> None:
    """Bug 2: the packing dtype helpers must import and run on torch builds that
    predate ``torch.float8_e8m0fnu`` (added in torch 2.7)."""
    import flashpack.utils as utils

    # Simulate an older torch build (e.g. torch 2.6) lacking the newest fp8 dtype.
    monkeypatch.delattr(torch, "float8_e8m0fnu", raising=False)
    try:
        importlib.reload(utils)

        assert not hasattr(torch, "float8_e8m0fnu")
        # The resolved set drops the missing dtype but keeps the present ones...
        assert torch.float8_e4m3fn in utils._FP8_DTYPES
        # ... and both helpers run without raising AttributeError.
        assert utils.get_packing_dtype(torch.float8_e4m3fn) is torch.uint8
        assert utils.torch_dtype_to_numpy_dtype(torch.float8_e4m3fn) is np.uint8
        # Unrelated dtypes are unaffected.
        assert utils.get_packing_dtype(torch.bfloat16) is torch.uint16
        assert utils.torch_dtype_to_numpy_dtype(torch.float32) is np.float32
    finally:
        monkeypatch.undo()
        importlib.reload(utils)


@pytest.mark.parametrize("include_buffers_env", [None, "true"])
def test_nonpersistent_buffer_is_materialized(
    tmp_path, monkeypatch, include_buffers_env
) -> None:
    """Bug 3: a non-persistent buffer must round-trip materialized (not meta) with
    its ``__init__`` value.

    With ``ACCELERATE_INIT_INCLUDE_BUFFERS=true`` (the condition that reproduced
    the fal-ai/registry#11335 failure) the unfixed loader leaves the buffer on the
    meta device; the explicit ``include_buffers=False`` in ``from_flashpack``
    overrides the env var and keeps it materialized.
    """
    if include_buffers_env is None:
        monkeypatch.delenv("ACCELERATE_INIT_INCLUDE_BUFFERS", raising=False)
    else:
        monkeypatch.setenv("ACCELERATE_INIT_INCLUDE_BUFFERS", include_buffers_env)

    source = ModelWithBuffers()
    expected_position_ids = source.position_ids.clone()
    path = _save(source, tmp_path)

    loaded = ModelWithBuffers.from_flashpack(path, features=8)

    # Non-persistent buffer: materialized by __init__, never on meta.
    assert loaded.position_ids.is_meta is False
    assert torch.equal(loaded.position_ids, expected_position_ids)
    # Persistent buffer: restored from the pack.
    assert loaded.scale.is_meta is False
    assert torch.equal(loaded.scale, source.scale)
    # Parameters: restored from the pack.
    assert loaded.linear.weight.is_meta is False
    assert torch.equal(loaded.linear.weight, source.linear.weight)


def test_leftover_meta_tensor_warns(tmp_path) -> None:
    """Bug 3 (defensive): a parameter missing from the pack under a non-strict
    flag stays on meta; the loader must warn loudly instead of silently returning
    a broken model."""

    class SmallModel(torch.nn.Module, FlashPackMixin):
        def __init__(self, features: int = 8) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(features, features)

    class BiggerModel(torch.nn.Module, FlashPackMixin):
        def __init__(self, features: int = 8) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(features, features)
            # Extra parameter with no counterpart in the pack.
            self.extra = torch.nn.Parameter(torch.zeros(features))

    path = _save(SmallModel(), tmp_path)

    with pytest.warns(UserWarning, match="meta device"):
        loaded = BiggerModel.from_flashpack(path, features=8, strict_params=False)

    assert loaded.extra.is_meta is True
