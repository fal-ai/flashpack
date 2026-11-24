import gc
import weakref

import torch
from flashpack import assign_from_file, compact, pack_to_file
from torch import nn


def test_reclaims_replaced_layers_pack_without_changing_surviving_tensors(tmp_path):
    model = nn.Sequential(nn.Linear(1024, 1024), nn.LayerNorm(1024))
    model.register_buffer("scale", torch.ones(64))
    path = str(tmp_path / "model.flashpack")
    pack_to_file(model, path, target_dtype=None)
    assign_from_file(model, path, device="cpu", keep_flash_ref_on_model=True)
    original = weakref.ref(model.scale._base)
    model[0] = nn.Identity()
    inputs = torch.ones(1, 1024)
    with torch.no_grad():
        expected = model(inputs)

    compact(model)
    gc.collect()

    assert original() is None
    with torch.no_grad():
        assert torch.equal(model(inputs), expected)
    assert torch.equal(model.scale, torch.ones(64))
