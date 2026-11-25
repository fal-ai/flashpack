"""The diffusers integration must import, and load a pipeline, on every
diffusers release flashpack supports. No network: the pipeline is a tiny
randomly initialised DDPM built in the test.

Skipped when diffusers is not installed (the CI tier), so run it locally or in
the gpu/network tier."""

import pytest

pytest.importorskip("diffusers")

import torch  # noqa: E402
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel  # noqa: E402
from flashpack.integrations.diffusers import (  # noqa: E402
    FlashPackDiffusersModelMixin,
    FlashPackDiffusionPipeline,
)


# Module level: the pipeline loader imports component classes by module path
# and name from ``model_index.json``
class FPUNet(UNet2DModel, FlashPackDiffusersModelMixin):
    pass


class FPDDPM(DDPMPipeline, FlashPackDiffusionPipeline):
    def __init__(self, unet: FPUNet, scheduler: DDPMScheduler) -> None:
        super().__init__(unet=unet, scheduler=scheduler)


def test_cli_module_imports() -> None:
    """``flashpack.__main__`` patches the integrations at import time, so a
    broken diffusers import takes every CLI command down with it."""
    import flashpack.__main__  # noqa: F401


def test_local_pipeline_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    unet = FPUNet(
        sample_size=8,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(8, 8),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )
    pipeline = FPDDPM(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=10))
    pipeline.save_pretrained_flashpack(str(tmp_path))

    loaded = FPDDPM.from_pretrained_flashpack(str(tmp_path))

    assert isinstance(loaded.unet, UNet2DModel)
    assert type(loaded.unet).__name__ == "FPUNet"
    for name, tensor in unet.state_dict().items():
        assert torch.equal(loaded.unet.state_dict()[name], tensor), name
