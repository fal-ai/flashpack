import csv
import gc
import os
import shutil
import tempfile
import time

import torch
from flashpack.integrations.diffusers import patch_diffusers_auto_model
from flashpack.integrations.transformers import patch_transformers_auto_model
from huggingface_hub import snapshot_download

patch_diffusers_auto_model()
patch_transformers_auto_model()

from diffusers.models import AutoModel as DiffusersAutoModel  # noqa: E402
from transformers import AutoModel as TransformersAutoModel  # noqa: E402


def test_model(
    repo_id: str,
    subfolder: str | None = None,
    accelerate_device: str | torch.device = "cuda",
    flashpack_device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    use_transformers: bool = False,
) -> tuple[float, float, int]:
    """
    Test a model from a repository.
    """
    repo_dir = snapshot_download(
        repo_id, allow_patterns=None if subfolder is None else [f"{subfolder}/*"]
    )
    model_dir = repo_dir if subfolder is None else os.path.join(repo_dir, subfolder)
    saved_flashpack_path = os.path.join(model_dir, "model.flashpack")
    saved_flashpack_config_path = os.path.join(model_dir, "flashpack_config.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Make a new model directory with the model in it so it isn't cached
        temp_model_dir = os.path.join(tmpdir, "model")
        flashpack_dir = os.path.join(tmpdir, "flashpack")
        os.makedirs(flashpack_dir, exist_ok=True)
        print("Copying model to temporary directory")
        shutil.copytree(model_dir, temp_model_dir)

        # Load from the temporary model directory
        print("Loading model from temporary directory using from_pretrained")
        start_time = time.time()
        if use_transformers:
            model = TransformersAutoModel.from_pretrained(
                temp_model_dir,
                torch_dtype=dtype,
                device_map={"": accelerate_device},
            )
        else:
            model = DiffusersAutoModel.from_pretrained(
                temp_model_dir,
                torch_dtype=dtype,
                device_map={"": accelerate_device},
            )

        end_time = time.time()
        accelerate_time = end_time - start_time
        print(f"Time taken with from_pretrained: {accelerate_time} seconds")

        if os.path.exists(saved_flashpack_path) and os.path.exists(
            saved_flashpack_config_path
        ):
            print("Copying flashpack to temporary directory")
            shutil.copy(
                saved_flashpack_path, os.path.join(flashpack_dir, "model.flashpack")
            )
            shutil.copy(
                saved_flashpack_config_path, os.path.join(flashpack_dir, "config.json")
            )
        else:
            print("Packing model to flashpack")
            pack_start_time = time.time()
            model.save_pretrained_flashpack(
                flashpack_dir, target_dtype=dtype, silent=True
            )
            pack_end_time = time.time()
            print(
                f"Time taken with save_pretrained_flashpack: {pack_end_time - pack_start_time} seconds"
            )
            # Copy back to the original model directory
            shutil.copy(
                os.path.join(flashpack_dir, "model.flashpack"), saved_flashpack_path
            )
            shutil.copy(
                os.path.join(flashpack_dir, "config.json"), saved_flashpack_config_path
            )

        del model
        sync_and_flush()

        print("Loading model from flashpack directory using from_pretrained_flashpack")
        flashpack_start_time = time.time()
        if use_transformers:
            flashpack_model = TransformersAutoModel.from_pretrained_flashpack(
                flashpack_dir, device=flashpack_device, target_dtype=dtype
            )
        else:
            flashpack_model = DiffusersAutoModel.from_pretrained_flashpack(
                flashpack_dir, device=flashpack_device, target_dtype=dtype
            )

        flashpack_end_time = time.time()
        flashpack_time = flashpack_end_time - flashpack_start_time
        print(f"Time taken with from_pretrained_flashpack: {flashpack_time} seconds")

        total_numel = 0
        for param in flashpack_model.parameters():
            total_numel += param.numel()

        total_bytes = total_numel * dtype.itemsize

        del flashpack_model
        sync_and_flush()

        return accelerate_time, flashpack_time, total_bytes


def test_wan_small_transformer() -> tuple[float, float, int]:
    return test_model(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="transformer",
        accelerate_device="cuda:0" if torch.cuda.is_available() else "cpu",
        flashpack_device="cuda:1" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )


def test_wan_large_transformer() -> tuple[float, float, int]:
    return test_model(
        repo_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        subfolder="transformer",
        accelerate_device="cuda:0" if torch.cuda.is_available() else "cpu",
        flashpack_device="cuda:1" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )


def test_flux_transformer() -> tuple[float, float, int]:
    return test_model(
        repo_id="black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        accelerate_device="cuda:0" if torch.cuda.is_available() else "cpu",
        flashpack_device="cuda:1" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )


def test_qwen_text_encoder() -> tuple[float, float, int]:
    return test_model(
        repo_id="Qwen/Qwen-Image-Edit",
        subfolder="text_encoder",
        accelerate_device="cuda:0" if torch.cuda.is_available() else "cpu",
        flashpack_device="cuda:1" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        use_transformers=True,
    )


def print_test_result(
    model_name: str,
    accelerate_time: float,
    flashpack_time: float,
    total_bytes: int,
) -> None:
    print(f"{model_name}: Accelerate time: {accelerate_time} seconds")
    print(f"{model_name}: Flashpack time: {flashpack_time} seconds")
    accelerate_gbps = (total_bytes / 1000**3) / accelerate_time
    flashpack_gbps = (total_bytes / 1000**3) / flashpack_time
    print(f"{model_name}: Accelerate GB/s: {accelerate_gbps} GB/s")
    print(f"{model_name}: Flashpack GB/s: {flashpack_gbps} GB/s")


def sync_and_flush() -> None:
    torch.cuda.empty_cache()
    gc.collect()
    os.system("sync")
    if os.geteuid() == 0:
        os.system("echo 3 | tee /proc/sys/vm/drop_caches")


if __name__ == "__main__":
    with open("benchmark_results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accelerate_time", "flashpack_time", "total_bytes"])
        for i in range(10):
            for test_model_name, test_func in [
                ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", test_wan_small_transformer),
                ("Wan-AI/Wan2.1-T2V-14B-Diffusers", test_wan_large_transformer),
                ("black-forest-labs/FLUX.1-dev", test_flux_transformer),
                ("Qwen/Qwen-Image-Edit", test_qwen_text_encoder),
            ]:
                accelerate_time, flashpack_time, total_bytes = test_func()
                writer.writerow(
                    [test_model_name, accelerate_time, flashpack_time, total_bytes]
                )
                print_test_result(
                    test_model_name, accelerate_time, flashpack_time, total_bytes
                )
