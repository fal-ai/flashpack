"""Render media/coldstart-{black,white}.png — true-cold fleet A/B chart.

Data provenance: fal production H100 fleet A/B (2026-07), single-use copies of
two pinned production packs on a JuiceFS-backed network volume so every read
was guaranteed page-cache- AND node-cache-cold; 12 fresh runners; every load
bit-verified against the source (stripe hash). Baseline is the pre-0.3
single-threaded mmap walk; FlashPack is the v0.3+ parallel reader (O_DIRECT,
pinned staging, overlapped H2D). Times are the full read-to-GPU wall.

Colors validated with the dataviz palette validator (CVD + contrast):
light #2062E8/#84CC16, dark #5B8DEF/#65A30D. Bright-lime light-mode contrast
WARN is satisfied by direct value labels on every bar.

Usage: uv run --with matplotlib python scripts/plot_fleet_coldstart.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

PACKS = ["15.35 GB pack", "25.41 GB pack"]
MMAP_S = [255.0, 245.0]
FLASH_S = [12.5, 18.0]

MODES = {
    "black": {  # light GitHub theme (dark ink)
        "ink": "#1a1a19",
        "muted": "#57534E",
        "grid": "#D6D3D1",
        "baseline": "#2062E8",
        "flash": "#84CC16",
    },
    "white": {  # dark GitHub theme (light ink)
        "ink": "#E7E5E4",
        "muted": "#A8A29E",
        "grid": "#44403C",
        "baseline": "#5B8DEF",
        "flash": "#65A30D",
    },
}


def render(mode: str, c: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.4), dpi=220)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    y = np.arange(len(PACKS), dtype=float)
    h = 0.34
    gap = 0.04  # 2px-equivalent surface gap between adjacent bars

    b1 = ax.barh(
        y - h / 2 - gap / 2,
        MMAP_S,
        height=h,
        color=c["baseline"],
        label="mmap loader (single-threaded page-fault walk)",
        zorder=3,
    )
    b2 = ax.barh(
        y + h / 2 + gap / 2,
        FLASH_S,
        height=h,
        color=c["flash"],
        label="FlashPack parallel read (v0.3+)",
        zorder=3,
    )

    for rect, secs, ratio in (
        (b1[0], MMAP_S[0], None),
        (b1[1], MMAP_S[1], None),
        (b2[0], FLASH_S[0], MMAP_S[0] / FLASH_S[0]),
        (b2[1], FLASH_S[1], MMAP_S[1] / FLASH_S[1]),
    ):
        label = f"{secs:g} s" if ratio is None else f"{secs:g} s  ({ratio:.0f}x faster)"
        ax.text(
            rect.get_width() + 4,
            rect.get_y() + rect.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=11.5,
            color=c["ink"],
            fontweight="bold",
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(PACKS, fontsize=12, color=c["ink"])
    ax.invert_yaxis()
    ax.set_xlim(0, 320)
    ax.set_xlabel(
        "true-cold load time, disk -> GPU (seconds, lower is better)",
        fontsize=11,
        color=c["muted"],
    )
    ax.tick_params(axis="x", colors=c["muted"], labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=c["grid"], linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "True-cold model load from a network-backed weight store\n",
        fontsize=14,
        color=c["ink"],
        fontweight="bold",
        loc="left",
        pad=18,
    )
    ax.text(
        0,
        1.06,
        "fal production H100 fleet - page cache and node cache cold - "
        "all loads bit-verified",
        transform=ax.transAxes,
        fontsize=10.5,
        color=c["muted"],
        va="top",
    )

    leg = ax.legend(loc="lower right", fontsize=10, frameon=False)
    for t in leg.get_texts():
        t.set_color(c["ink"])

    fig.tight_layout()
    fig.savefig(f"media/coldstart-{mode}.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for mode, colors in MODES.items():
        render(mode, colors)
    print("wrote media/coldstart-black.png, media/coldstart-white.png")
