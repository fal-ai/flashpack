import os
import sys

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
from typing import Optional


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
pipeline_dir = os.path.join(HERE, "wan_pipeline")
os.makedirs(pipeline_dir, exist_ok=True)

if len(sys.argv) < 2:
    raise ValueError("Usage: python wan_pipe.py <save|load>")

if sys.argv[1] not in ["save", "load"]:
    raise ValueError("Usage: python wan_pipe.py <save|load>")

is_save = sys.argv[1] == "save"
is_load = sys.argv[1] == "load"

repo_dir = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

if is_save:
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
        pipeline.save_pretrained_flashpack(
            pipeline_dir,
        )

elif is_load:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(42)
    with timer("load_and_inference_accelerate"):
        pipeline = FlashPackWanPipeline.from_pretrained(
            repo_dir,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        pipeline(
            prompt="A beautiful sunset over a calm ocean.",
            width=832,
            height=480,
            num_inference_steps=28,
        )

    with timer("load_and_inference_flashpack"):
        pipeline = FlashPackWanPipeline.from_pretrained_flashpack(
            pipeline_dir, device_map=device, silent=False
        )
        pipeline(
            prompt="A beautiful sunset over a calm ocean.",
            width=832,
            height=480,
            num_inference_steps=28,
            generator=generator,
        )
    generator.manual_seed(42)