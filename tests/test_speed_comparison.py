import os
import time

import safetensors.torch
import torch
from flashpack import assign_from_file, pack_to_file
from huggingface_hub import snapshot_download
from transformers import GPT2Model

repo_dir = snapshot_download("gpt2")
pt_filename = os.path.join(repo_dir, "pytorch_model.bin")
sf_filename = os.path.join(repo_dir, "model.safetensors")
flashpack_filename = os.path.join(repo_dir, "model.flashpack")

print(f"Preparing model")
model = GPT2Model.from_pretrained("gpt2", device_map="cuda")
if not os.path.exists(flashpack_filename):
    pack_to_file(model, flashpack_filename, target_dtype=model.dtype)

print("Running load time comparison")
pt_load_start_time = time.time()
state_dict = torch.load(pt_filename, map_location="cuda")
model.load_state_dict(state_dict, strict=False)
pt_load_end_time = time.time()
pt_load_time = pt_load_end_time - pt_load_start_time
print(f"PT load time: {pt_load_time} seconds")

sf_load_start_time = time.time()
state_dict = safetensors.torch.load_file(sf_filename, device="cuda")
model.load_state_dict(state_dict, strict=False)
sf_load_end_time = time.time()
sf_load_time = sf_load_end_time - sf_load_start_time
print(f"SF load time: {sf_load_time} seconds")

os.environ["SAFETENSORS_FAST_GPU"] = "1"
sf_fast_gpu_start_time = time.time()
state_dict = safetensors.torch.load_file(sf_filename, device="cuda")
sf_fast_gpu_end_time = time.time()
sf_fast_gpu_time = sf_fast_gpu_end_time - sf_fast_gpu_start_time
print(f"SF fast GPU time: {sf_fast_gpu_time} seconds")

fp_load_start_time = time.time()
assign_from_file(model, flashpack_filename, device="cuda")
fp_load_end_time = time.time()
fp_load_time = fp_load_end_time - fp_load_start_time
print(f"FP load time: {fp_load_time} seconds")
