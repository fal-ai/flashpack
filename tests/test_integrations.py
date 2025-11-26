import os
from tempfile import TemporaryDirectory


def test_diffusers() -> None:
    """
    Tests the diffusers integration.
    """
    from flashpack.integrations.diffusers import patch_diffusers_auto_model

    patch_diffusers_auto_model()

    from diffusers.models import AutoModel
    from flashpack.integrations.diffusers.model import FlashPackDiffusersModelMixin

    model = AutoModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="vae",
    )
    assert model is not None
    assert isinstance(model, FlashPackDiffusersModelMixin)

    with TemporaryDirectory() as tmpdir:
        model.save_pretrained_flashpack(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "model.flashpack"))
        assert AutoModel.from_pretrained_flashpack(tmpdir) is not None


def test_transformers() -> None:
    """
    Tests the transformers integration.
    """
    from flashpack.integrations.transformers import patch_transformers_auto_model

    patch_transformers_auto_model()

    from flashpack.integrations.transformers.model import (
        FlashPackTransformersModelMixin,
    )
    from transformers.models import AutoModel

    model = AutoModel.from_pretrained(
        "openai/clip-vit-base-patch32",
    )
    assert model is not None
    assert isinstance(model, FlashPackTransformersModelMixin)

    with TemporaryDirectory() as tmpdir:
        model.save_pretrained_flashpack(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "model.flashpack"))
        assert AutoModel.from_pretrained_flashpack(tmpdir) is not None
