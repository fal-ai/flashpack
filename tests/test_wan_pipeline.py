import os
from typing import Optional

import pytest
import torch
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.pipelines import WanPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from flashpack.integrations.diffusers import (
    FlashPackDiffusersModelMixin,
    FlashPackDiffusionPipeline,
)
from flashpack.integrations.transformers import FlashPackTransformersModelMixin
from flashpack.utils import timer
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, UMT5EncoderModel


class FlashPackWanTransformer3DModel(
    WanTransformer3DModel, FlashPackDiffusersModelMixin
):
    flashpack_ignore_prefixes = ["rope"]


class FlashPackAutoencoderKLWan(AutoencoderKLWan, FlashPackDiffusersModelMixin):
    pass


class FlashPackUMT5EncoderModel(UMT5EncoderModel, FlashPackTransformersModelMixin):
    flashpack_ignore_names = ["encoder.embed_tokens.weight"]


class FlashPackWanPipeline(WanPipeline, FlashPackDiffusionPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: FlashPackUMT5EncoderModel,
        vae: FlashPackAutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        transformer: Optional[FlashPackWanTransformer3DModel] = None,
        transformer_2: Optional[FlashPackWanTransformer3DModel] = None,
        boundary_ratio: float | None = None,
        expand_timesteps: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            transformer_2=transformer_2,
            scheduler=scheduler,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
        )


HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(HERE, "wan_pipeline")


@pytest.fixture(scope="module")
def repo_dir():
    """Download and cache the Wan model repository."""
    return snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")


@pytest.fixture(scope="module")
def pipeline_dir():
    """Return the directory for saving/loading the flashpack pipeline."""
    os.makedirs(PIPELINE_DIR, exist_ok=True)
    return PIPELINE_DIR


@pytest.fixture(scope="module")
def saved_pipeline(repo_dir, pipeline_dir):
    """Save the pipeline using flashpack and return the path."""
    transformer = FlashPackWanTransformer3DModel.from_pretrained(
        os.path.join(repo_dir, "transformer"),
        torch_dtype=torch.bfloat16,
    ).to(dtype=torch.bfloat16)
    vae = FlashPackAutoencoderKLWan.from_pretrained(
        os.path.join(repo_dir, "vae"),
        torch_dtype=torch.float32,
    ).to(dtype=torch.float32)
    text_encoder = FlashPackUMT5EncoderModel.from_pretrained(
        os.path.join(repo_dir, "text_encoder"),
        torch_dtype=torch.bfloat16,
    ).to(dtype=torch.bfloat16)
    scheduler = UniPCMultistepScheduler.from_pretrained(
        os.path.join(repo_dir, "scheduler"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(repo_dir, "tokenizer"),
    )

    pipeline = FlashPackWanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )

    with timer("save"):
        pipeline.save_pretrained_flashpack(pipeline_dir)

    return pipeline_dir


def test_save_pipeline(saved_pipeline):
    """Test that the pipeline can be saved using flashpack."""
    assert os.path.exists(saved_pipeline)
    # Check that the expected files exist
    assert os.path.isdir(saved_pipeline)


def test_load_and_inference_accelerate(repo_dir):
    """Test loading and running inference with accelerate."""
    with timer("load_and_inference_accelerate"):
        pipeline = FlashPackWanPipeline.from_pretrained(
            repo_dir,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        output = pipeline(
            prompt="A beautiful sunset over a calm ocean.",
            width=832,
            height=480,
            num_inference_steps=28,
        )

    assert output is not None


def test_load_and_inference_flashpack(saved_pipeline):
    """Test loading and running inference with flashpack."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(42)

    with timer("load_and_inference_flashpack"):
        pipeline = FlashPackWanPipeline.from_pretrained_flashpack(
            saved_pipeline, device_map=device, silent=False
        )
        output = pipeline(
            prompt="A beautiful sunset over a calm ocean.",
            width=832,
            height=480,
            num_inference_steps=28,
            generator=generator,
        )

    assert output is not None
