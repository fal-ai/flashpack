import gc
import os
import tempfile

import torch
import tqdm
from flashpack import FlashPackMixin
from flashpack.utils import timer

HERE = os.path.dirname(os.path.abspath(__file__))


def test_load_unload() -> None:
    """
    Test that we can load and unload a model from a flash pack file.
    """

    class TestModel(torch.nn.Module, FlashPackMixin):
        def __init__(self, num_blocks: int) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(1024, 1024, 3, 1, 1)
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(1024, 1024) for _ in range(num_blocks)]
            )

    model = TestModel(10)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.flashpack")
        model.save_flashpack(path, target_dtype=torch.float32)
        model2 = TestModel.from_flashpack(path, num_blocks=10)
        assert model2.conv.weight.shape == model.conv.weight.shape
        assert torch.allclose(model2.conv.weight, model.conv.weight)
        assert model2.blocks[0].weight.shape == model.blocks[0].weight.shape
        assert torch.allclose(model2.blocks[0].weight, model.blocks[0].weight)


def test_wan_transformer() -> None:
    """
    Tests a WanTransformer3D model.
    """
    try:
        from diffusers.models import WanTransformer3DModel
    except ImportError:
        print("Diffusers not installed, skipping WanTransformer3D model test")
        return

    from flashpack.integrations.diffusers import FlashPackDiffusersModelMixin

    class FlashPackWanTransformer3DModel(
        WanTransformer3DModel, FlashPackDiffusersModelMixin
    ):
        flashpack_ignore_prefixes = ["rope"]

    with timer("baseline"):
        initial_model = FlashPackWanTransformer3DModel.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            subfolder="transformer",
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        ).to("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)

    save_dir = os.path.join(HERE, "wan_transformer_flashpack")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.flashpack")

    if not os.path.exists(model_path):
        initial_model.save_pretrained_flashpack(save_dir, target_dtype=torch.bfloat16)

    flashpack_model = FlashPackWanTransformer3DModel.from_pretrained_flashpack(
        save_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Build lookups
    initial_model_params = {
        name: param for name, param in initial_model.named_parameters()
    }
    flashpack_model_params = {
        name: param for name, param in flashpack_model.named_parameters()
    }
    initial_model_buffers = {name: buf for name, buf in initial_model.named_buffers()}
    flashpack_model_buffers = {
        name: buf for name, buf in flashpack_model.named_buffers()
    }

    for name, param in tqdm.tqdm(
        initial_model_params.items(),
        total=len(initial_model_params),
        desc="Checking parameters",
    ):
        assert torch.allclose(
            param, flashpack_model_params[name].to(param.device, param.dtype)
        )
    for name, buf in tqdm.tqdm(
        initial_model_buffers.items(),
        total=len(initial_model_buffers),
        desc="Checking buffers",
    ):
        assert torch.allclose(
            buf, flashpack_model_buffers[name].to(buf.device, buf.dtype)
        )

    print("All checks passed")
    del initial_model
    del flashpack_model
    torch.cuda.empty_cache()
    gc.collect()


def test_wan_text_encoder() -> None:
    """
    Tests a WanTransformer3D model.
    """
    try:
        from transformers import UMT5EncoderModel
    except ImportError:
        print("Transformers not installed, skipping UMT5EncoderModel model test")
        return

    from flashpack.integrations.transformers import FlashPackTransformersModelMixin

    class FlashPackWanTextEncoderModel(
        UMT5EncoderModel, FlashPackTransformersModelMixin
    ):
        flashpack_ignore_names = ["encoder.embed_tokens.weight"]

    with timer("baseline"):
        initial_model = FlashPackWanTextEncoderModel.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            subfolder="text_encoder",
            # device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        ).to("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)

    save_dir = os.path.join(HERE, "wan_text_encoder_flashpack")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.flashpack")

    if not os.path.exists(model_path):
        initial_model.save_pretrained_flashpack(save_dir, target_dtype=torch.bfloat16)

    flashpack_model = FlashPackWanTextEncoderModel.from_pretrained_flashpack(
        save_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Build lookups
    initial_model_params = {
        name: param for name, param in initial_model.named_parameters()
    }
    flashpack_model_params = {
        name: param for name, param in flashpack_model.named_parameters()
    }
    initial_model_buffers = {name: buf for name, buf in initial_model.named_buffers()}
    flashpack_model_buffers = {
        name: buf for name, buf in flashpack_model.named_buffers()
    }

    for name, param in tqdm.tqdm(
        initial_model_params.items(),
        total=len(initial_model_params),
        desc="Checking parameters",
    ):
        assert torch.allclose(
            param, flashpack_model_params[name].to(param.device, param.dtype)
        )
    for name, buf in tqdm.tqdm(
        initial_model_buffers.items(),
        total=len(initial_model_buffers),
        desc="Checking buffers",
    ):
        assert torch.allclose(
            buf, flashpack_model_buffers[name].to(buf.device, buf.dtype)
        )

    print("All checks passed")
    del initial_model
    del flashpack_model
    torch.cuda.empty_cache()
    gc.collect()
