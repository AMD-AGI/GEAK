#!/usr/bin/env python3
"""Draw what is actually tunable on each backend, and how you prove it engaged.

The point of this figure is the asymmetry. The backends do not offer the same
surface: two let you author a search space, three only let you select from a
pre-compiled set, one gives you nothing but launch geometry and a profiler.
Treating them as interchangeable is how a tuning session ends up sweeping a
parameter the backend never reads.

The right-hand column is the one that matters most. Every backend has a
different way of lying about success, and a different string that proves it
didn't.

Usage:  python3 tools/surface_map.py [-o docs/surface_map.png]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Columns: what the backend exposes as a tunable.
COLS = [
    ("tile\ngeometry", "BLOCK_M/N/K, tile shape"),
    ("wave\nscheduling", "num_warps, num_stages, waves_per_eu"),
    ("MFMA\nlayout", "matrix_instr_nonkdim, kpack"),
    ("split-K\n/ GSU", "K-dimension partitioning"),
    ("launch\ngeometry", "grid, block dim, LDS budget"),
    ("solution\nselect", "pick from a pre-compiled set"),
    ("backend\nrace", "choose the language per shape"),
]

# 2 = you author/choose it directly, 1 = exposed but indirect, 0 = not a surface.
#
# Rows are ordered by how much of the space you control: authoring backends
# first, selection backends next, then the integration and framework layers
# that tune by delegating downward.
ROWS = [
    ("tuning-triton",   [2, 2, 2, 2, 1, 0, 0], "author a config list",
     "@autotune key must name every shape\nthe kernel specializes on"),
    ("tuning-flydsl",   [2, 2, 1, 2, 1, 0, 0], "author a Config space",
     "libtype selection — confirm FlyDSL\nand not the torch fallback ran"),
    ("tuning-hip",      [1, 1, 0, 0, 2, 0, 0], "hand-choose geometry",
     "rocprofv3 kernel trace — the only\nground truth on this row"),
    ("tuning-hipblaslt",[0, 0, 0, 2, 0, 2, 0], "race compiled solutions",
     "replay by solution index; bracket\nnumber is not the index"),
    ("tuning-ck",       [0, 0, 0, 1, 0, 2, 0], "race compiled instances",
     "instance name encodes the tile —\nread it, don't assume"),
    ("tuning-aiter",    [1, 1, 1, 2, 0, 2, 2], "gradlib races all backends",
     "AITER_LOG_TUNED_CONFIG=1 —\nno line means untuned dispatch"),
    ("tuning-in-vllm",  [1, 1, 1, 1, 0, 1, 1], "delegates down, owns deploy",
     "VLLM_TUNED_CONFIG_FOLDER +\nserver-log load line"),
    ("tuning-in-sglang",[1, 1, 1, 1, 0, 1, 1], "delegates down, owns deploy",
     "config dir is keyed by Triton\nversion — a fallback looks like a hit"),
]

FILL = {2: "#2a9d5c", 1: "#a8d5ba", 0: "#f0f2f4"}
MARK = {2: "●", 1: "○", 0: ""}


def draw(out_path):
    fig, ax = plt.subplots(figsize=(15.2, 8.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    fig.suptitle("What each backend lets you tune — and what proves it engaged",
                 fontsize=14, y=0.972)
    ax.text(50, 93.2,
            "the backends are not interchangeable: two let you author a search "
            "space, three only let you select from a pre-compiled set",
            ha="center", fontsize=9.5, color="#444")

    x_lab, w_lab = 1.0, 15.5
    x_grid = x_lab + w_lab + 0.5
    w_cell = 6.4
    x_how = x_grid + len(COLS) * w_cell + 1.5
    w_how = 15.5
    x_proof = x_how + w_how + 1.0

    y_top, h_row = 78.0, 8.4

    # column headers
    for j, (short, long) in enumerate(COLS):
        cx = x_grid + j * w_cell + w_cell / 2
        ax.text(cx, y_top + 5.0, short, ha="center", va="bottom", fontsize=8.4,
                fontweight="bold", color="#333", linespacing=1.4)
    # A banner over the grid would collide with the subtitle once the tight
    # bbox rescales; the column headers already say what these are.
    ax.plot([x_grid + 0.45, x_grid + len(COLS) * w_cell - 0.45],
            [y_top + 3.2] * 2, color="#2a9d5c", lw=1.6, solid_capstyle="butt")
    ax.text(x_how + w_how / 2, y_top + 5.0, "how you search",
            ha="center", va="bottom", fontsize=8.4, fontweight="bold",
            color="#333")
    ax.text(x_proof + 14, y_top + 5.0, "what proves it engaged",
            ha="center", va="bottom", fontsize=8.4, fontweight="bold",
            color="#b0453a")

    for i, (name, vals, how, proof) in enumerate(ROWS):
        y = y_top - (i + 1) * h_row
        if i % 2 == 0:
            ax.add_patch(Rectangle((x_lab, y), 99 - x_lab, h_row,
                                   fc="#fafbfc", ec="none", zorder=0))
        ax.text(x_lab + w_lab, y + h_row / 2, name, ha="right", va="center",
                fontsize=9.3, family="monospace", fontweight="bold",
                color="#2f3a45")
        for j, v in enumerate(vals):
            cx = x_grid + j * w_cell
            ax.add_patch(Rectangle((cx + 0.45, y + 1.0), w_cell - 0.9,
                                   h_row - 2.0, fc=FILL[v], ec="none", zorder=1))
            if MARK[v]:
                ax.text(cx + w_cell / 2, y + h_row / 2, MARK[v], ha="center",
                        va="center", fontsize=11,
                        color="#ffffff" if v == 2 else "#2a7a48", zorder=2)
        ax.text(x_how + w_how / 2, y + h_row / 2, how, ha="center", va="center",
                fontsize=7.8, color="#444", style="italic")
        ax.text(x_proof, y + h_row / 2, proof, ha="left", va="center",
                fontsize=7.5, color="#7a3b34", linespacing=1.5)

    y_end = y_top - len(ROWS) * h_row

    ax.text(x_lab, y_end - 4.0,
            "●  you set it directly       ○  exposed, but chosen for you by a "
            "tuner or a solution table       (blank)  not a surface on this backend",
            fontsize=8.2, color="#555")
    ax.text(x_lab, y_end - 9.5,
            "Read the blanks as hard constraints, not gaps. Sweeping BLOCK_M on "
            "hipBLASLt does nothing — the tile is baked into the compiled\n"
            "solution. Most wasted tuning time on this platform goes into "
            "parameters the chosen backend never reads.",
            fontsize=8.4, color="#333", linespacing=1.7)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}  ({len(ROWS)} backends x {len(COLS)} surfaces)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out",
                    default=os.path.join(ROOT, "docs", "surface_map.png"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    draw(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
