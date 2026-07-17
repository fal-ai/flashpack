import os

import pytest
import torch
from diffusers.pipelines import WanPipeline
from flashpack.integrations.diffusers import FlashPackDiffusionPipeline
from flashpack.utils import timer
from huggingface_hub import snapshot_download


class FlashPackWanPipeline(WanPipeline, FlashPackDiffusionPipeline):
    pass


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
    pipeline = FlashPackWanPipeline.from_pretrained_flashpack(
        repo_dir,
        convert_diffusers_models=True,
        convert_transformers_models=True,
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
