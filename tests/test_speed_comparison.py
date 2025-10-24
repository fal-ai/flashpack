import os
import time

import safetensors.torch
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

print("Running load time comparison (10 runs each)")

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

num_runs = 10
times_pt = []
times_sf = []
times_sf_fast = []
times_fp = []

# Repeat timings
for i in range(num_runs):
    # PyTorch .bin
    os.environ.pop("SAFETENSORS_FAST_GPU", None)
    start = time.time()
    state_dict = torch.load(pt_filename, map_location="cuda")
    model.load_state_dict(state_dict, strict=False)
    cuda_sync()
    end = time.time()
    times_pt.append(end - start)

    # Safetensors
    start = time.time()
    state_dict = safetensors.torch.load_file(sf_filename, device="cuda")
    model.load_state_dict(state_dict, strict=False)
    cuda_sync()
    end = time.time()
    times_sf.append(end - start)

    # Safetensors (fast gpu)
    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    start = time.time()
    state_dict = safetensors.torch.load_file(sf_filename, device="cuda")
    model.load_state_dict(state_dict, strict=False)
    cuda_sync()
    end = time.time()
    times_sf_fast.append(end - start)

    # Flashpack
    start = time.time()
    assign_from_file(model, flashpack_filename, device="cuda")
    cuda_sync()
    end = time.time()
    times_fp.append(end - start)

print("Timing complete. Means (s):")
print(f"  pytorch: {np.mean(times_pt):.3f}")
print(f"  safetensors: {np.mean(times_sf):.3f}")
print(f"  safetensors (fast gpu): {np.mean(times_sf_fast):.3f}")
print(f"  flashpack: {np.mean(times_fp):.3f}")

# Plot configuration (aligned with scripts/plot-benchmark.py)
accelerate_color = "#0f5ef3"
flashpack_color = "#adff02"
label_color = "#111111"

labels = [
    "pytorch",
    "safetensors",
    "safetensors (fast gpu)",
    "flashpack",
]
means = [
    float(np.mean(times_pt)),
    float(np.mean(times_sf)),
    float(np.mean(times_sf_fast)),
    float(np.mean(times_fp)),
]

colors = [accelerate_color, accelerate_color, accelerate_color, flashpack_color]

fig, ax = plt.subplots(figsize=(10, 4))
ax.patch.set_facecolor((0, 0, 0, 0))

# Style spines and ticks
for spine in ax.spines.values():
    spine.set_color(label_color)
ax.xaxis.label.set_color(label_color)
ax.yaxis.label.set_color(label_color)
ax.tick_params(axis="x", colors=label_color)
ax.tick_params(axis="y", colors=label_color)

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, means, color=colors, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, color=label_color)
ax.invert_yaxis()  # top-to-bottom order as specified

ax.set_xlabel("Loading Time (seconds)", fontsize=12, fontweight="bold", color=label_color)
ax.set_title(
    "load_state_dict() Time Comparison",
    fontsize=14,
    fontweight="bold",
    pad=16,
    color=label_color,
)
ax.grid(axis="x", alpha=0.3, linestyle="--")

# Add value labels at the end of each bar
for bar, val in zip(bars, means):
    ax.text(
        bar.get_width() + max(means) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}s",
        va="center",
        ha="left",
        fontsize=9,
        color=label_color,
    )

plt.tight_layout()
output_path = "./speed_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
print(f"Graph saved to: {output_path}")
