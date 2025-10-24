#!/usr/bin/env python3
"""
Script to visualize benchmark results comparing accelerate vs flashpack loading times.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# configuration
accelerate_color = "#0f5ef3"
flashpack_color = "#adff02"
label_color = "#111111"
model_labels = {
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "Wan2.1 1.3B DiT",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": "Wan2.1 14B DiT",
    "black-forest-labs/FLUX.1-dev": "FLUX.1[dev] 12B DiT",
    "Qwen/Qwen-Image-Edit": "Qwen-Image-Edit 8B TE",
}

# Read the CSV file
df = pd.read_csv("./benchmark_results.csv")

# Add model labels
df["model"] = df["model"].map(model_labels)

# Calculate GB/s for each measurement
df["accelerate_gbps"] = (df["total_bytes"] / df["accelerate_time"]) / (1024**3)
df["flashpack_gbps"] = (df["total_bytes"] / df["flashpack_time"]) / (1024**3)

# Group by model and calculate statistics
grouped = (
    df.groupby(["model", "total_bytes"])
    .agg(
        {
            "accelerate_time": ["mean", "min", "max"],
            "flashpack_time": ["mean", "min", "max"],
            "accelerate_gbps": ["mean", "min", "max"],
            "flashpack_gbps": ["mean", "min", "max"],
        }
    )
    .reset_index()
)

# Flatten column names
grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]

# Sort by total_bytes (model size)
grouped = grouped.sort_values("total_bytes")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
# Set axes to be transparent
for ax in [ax1, ax2]:
    ax.patch.set_facecolor((0, 0, 0, 0))

# Set border (spine) and y-axis colors to label_color for both axes
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_color(label_color)
    ax.yaxis.label.set_color(label_color)
    ax.tick_params(axis="y", colors=label_color)
    ax.tick_params(axis="x", colors=label_color)

# ============================================================
# GRAPH 1: Paired bar graph of loading times
# ============================================================

x = np.arange(len(grouped))
width = 0.35

bars1 = ax1.bar(
    x - width / 2,
    grouped["accelerate_time_mean"],
    width,
    label="Accelerate",
    alpha=0.8,
    color=accelerate_color,
)
bars2 = ax1.bar(
    x + width / 2,
    grouped["flashpack_time_mean"],
    width,
    label="Flashpack",
    alpha=0.8,
    color=flashpack_color,
)

ax1.set_ylabel(
    "Loading Time (seconds)", fontsize=12, fontweight="bold", color=label_color
)
ax1.set_title(
    "Model Loading Time Comparison: Accelerate vs. FlashPack (lower is better)",
    fontsize=14,
    fontweight="bold",
    pad=20,
    color=label_color,
)
ax1.set_xticks(x)
ax1.set_xticklabels(grouped["model"], rotation=45, ha="right", color=label_color)
ax1.legend(fontsize=11)
ax1.grid(axis="y", alpha=0.3, linestyle="--")

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            color=label_color,
        )

# ============================================================
# GRAPH 2: Line graph of transfer speed (GB/s) vs model size
# ============================================================

# Convert bytes to GB for x-axis labels
grouped["total_gb"] = grouped["total_bytes"] / (1024**3)

# Plot mean lines
line1 = ax2.plot(
    grouped["total_gb"],
    grouped["accelerate_gbps_mean"],
    marker="o",
    linewidth=2.5,
    markersize=8,
    label="Accelerate (mean)",
    color=accelerate_color,
)

line2 = ax2.plot(
    grouped["total_gb"],
    grouped["flashpack_gbps_mean"],
    marker="s",
    linewidth=2.5,
    markersize=8,
    label="Flashpack (mean)",
    color=flashpack_color,
)

# Add shaded areas for bounds (min to max)
ax2.fill_between(
    grouped["total_gb"],
    grouped["accelerate_gbps_min"],
    grouped["accelerate_gbps_max"],
    alpha=0.2,
    color=accelerate_color,
    label="Accelerate (range)",
)

ax2.fill_between(
    grouped["total_gb"],
    grouped["flashpack_gbps_min"],
    grouped["flashpack_gbps_max"],
    alpha=0.2,
    color=flashpack_color,
    label="Flashpack (range)",
)

ax2.set_xlabel("Model Size (GB)", fontsize=12, fontweight="bold", color=label_color)
ax2.set_ylabel(
    "Transfer Speed (GB/s)", fontsize=12, fontweight="bold", color=label_color
)
ax2.set_title(
    "Data Transfer Speed vs. Model Size (higher is better)",
    fontsize=14,
    fontweight="bold",
    pad=20,
    color=label_color,
)
ax2.legend(fontsize=10, loc="best")
ax2.grid(True, alpha=0.3, linestyle="--")

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
output_path = "./benchmark_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
print(f"Graphs saved to: {output_path}")

# Also display summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(
    f"\n{'Model':<40} {'Size (GB)':<12} {'Accel (s)':<12} {'Flash (s)':<12} {'Speedup':<10}"
)
print("-" * 70)
for _, row in grouped.iterrows():
    speedup = row["accelerate_time_mean"] / row["flashpack_time_mean"]
    print(
        f"{row['model']:<40} {row['total_gb']:>10.2f}  {row['accelerate_time_mean']:>10.3f}  "
        f"{row['flashpack_time_mean']:>10.3f}  {speedup:>8.2f}x"
    )

print("\n" + "=" * 70)
print(f"\n{'Model':<40} {'Accel GB/s':<15} {'Flash GB/s':<15}")
print("-" * 70)
for _, row in grouped.iterrows():
    print(
        f"{row['model']:<40} {row['accelerate_gbps_mean']:>13.2f}  {row['flashpack_gbps_mean']:>13.2f}"
    )
print("=" * 70)

plt.show()
