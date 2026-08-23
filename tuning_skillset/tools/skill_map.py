#!/usr/bin/env python3
"""Draw the skillset's coverage: which source languages and frameworks are
covered, and where each skill sits in the flow from authoring to a live server.

Reads the SKILL.md frontmatter of every skill directory, so adding a skill
updates the diagram. A typed-out figure would go stale the first time the set
changes, and a stale map of your own skills is worse than none.

Usage:  python3 tools/skill_map.py [-o docs/skill_map.png]
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

C_CORE = "#2f6f9f"
C_LANG = "#2a9d5c"
C_FW = "#7a4fa3"
C_SUP = "#8a8f96"
C_TEXT = "#ffffff"

# Where each skill sits in the flow. Tier drives both colour and row.
TIER = {
    "tuning-core": ("core", "the loop every other skill specializes"),
    "env-setup": ("support", "get the tools into the container"),
    "benchmark": ("support", "real shapes to tune against"),
    "tuning-triton": ("language", "author a config space"),
    "tuning-flydsl": ("language", "author a Config space"),
    "tuning-hip": ("language", "launch geometry + profiler ref"),
    "tuning-hipblaslt": ("language", "select a pre-compiled solution"),
    "tuning-ck": ("language", "select a pre-compiled instance"),
    "tuning-aiter": ("integration", "race backends, own the deploy path"),
    "tuning-in-vllm": ("framework", "live vLLM server"),
    "tuning-in-sglang": ("framework", "live SGLang server"),
}


def discover():
    """Every skill dir with a SKILL.md, plus its supporting .md count."""
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "SKILL.md"))):
        d = os.path.dirname(p)
        name = os.path.basename(d)
        txt = open(p).read()
        m = re.search(r"^description:\s*(.+)$", txt, re.M)
        refs = [f for f in os.listdir(d)
                if f.endswith(".md") and f != "SKILL.md"]
        out[name] = {
            "desc": (m.group(1) if m else ""),
            "refs": sorted(refs),
            "lines": txt.count("\n"),
        }
    # benchmark/ is a README-based skill, not a SKILL.md one
    b = os.path.join(ROOT, "benchmark", "README.md")
    if os.path.exists(b) and "benchmark" not in out:
        out["benchmark"] = {"desc": "shapes to tune against", "refs": [],
                            "lines": open(b).read().count("\n")}
    return out


def box(ax, x, y, w, h, label, sub, color, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=color, ec="none", zorder=2))
    ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
            fontsize=fs, color=C_TEXT, family="monospace",
            fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=6.9, color="#eaf2f8", zorder=3)


def arrow(ax, a, b, style="-|>", color="#98a2ad", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, color=color,
                                 lw=lw, mutation_scale=12, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def draw(out_path):
    sk = discover()
    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    fig.suptitle("Tuning skillset — coverage and flow", fontsize=14, y=0.975)
    ax.text(50, 94.5,
            "every source language a serving stack dispatches, from authoring "
            "a config space to proving a live server picked it up",
            ha="center", fontsize=9.5, color="#444")

    # --- core -----------------------------------------------------------
    box(ax, 34, 80, 32, 8.5, "tuning-core",
        "6-step loop · measurement · correctness gates · engagement", C_CORE, 11)
    n_refs = len(sk.get("tuning-core", {}).get("refs", []))
    ax.text(67.4, 84.2, f"+{n_refs} reference docs", fontsize=7.6,
            color="#2f6f9f", va="center", style="italic")

    # --- support ---------------------------------------------------------
    box(ax, 3, 80, 26, 8.5, "env-setup  ·  benchmark",
        "tools in the container · real shapes to tune", C_SUP, 9.5)
    arrow(ax, (29.2, 84.2), (33.8, 84.2))

    # --- languages -------------------------------------------------------
    langs = [k for k, v in TIER.items() if v[0] == "language" and k in sk]
    ax.text(1.2, 64.7, "one per source language", rotation=90,
            ha="center", va="center", fontsize=8.5, color="#2a9d5c",
            fontweight="bold")
    w, gap = 17.0, 1.6
    total = len(langs) * w + (len(langs) - 1) * gap
    x0 = (100 - total) / 2
    for i, k in enumerate(sorted(langs)):
        x = x0 + i * (w + gap)
        box(ax, x, 60, w, 9.5, k.replace("tuning-", ""), TIER[k][1], C_LANG, 9)
        arrow(ax, (50, 79.8), (x + w / 2, 69.7), rad=0.06 * (i - 2))

    # --- integration -----------------------------------------------------
    box(ax, 32, 43, 36, 9, "tuning-aiter",
        "races every backend on your shapes · owns the deploy path", C_FW, 11)
    for i in range(len(langs)):
        x = x0 + i * (w + gap) + w / 2
        arrow(ax, (x, 59.8), (50, 52.2), rad=0.05 * (2 - i))

    # --- frameworks ------------------------------------------------------
    fws = [k for k in ("tuning-in-vllm", "tuning-in-sglang") if k in sk]
    for i, k in enumerate(fws):
        x = 26 + i * 26
        box(ax, x, 26, 22, 8.5, k.replace("tuning-", ""),
            "capture shapes · deploy · verify engagement", C_FW, 9.5)
        arrow(ax, (50, 42.8), (x + 11, 34.7), rad=0.1 * (0.5 - i))

    # --- the through-line ------------------------------------------------
    ax.text(50, 21.0,
            "every path ends at the same question: did the live dispatch "
            "actually pick up the artifact?",
            ha="center", fontsize=10, color="#b0453a", fontweight="bold")
    ax.text(50, 14.0,
            "Tuning on this platform fails silently far more often than it "
            "fails loudly. A config lands where nothing reads it, a lookup key\n"
            "misses on an arch field, a stale entry loads through a fallback. "
            "Nothing raises. The tuner still prints a win. So every skill\n"
            "here ends with an engagement check rather than a timing.",
            ha="center", fontsize=8.2, color="#555", linespacing=1.65)

    ax.text(50, 5.2,
            f"{len(sk)} skills · validated on gfx942 / MI300X, ROCm 7.2.x · "
            f"gfx950 hooks marked throughout, method transfers, artifacts do not",
            ha="center", fontsize=8, color="#777")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}  ({len(sk)} skills)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out",
                    default=os.path.join(ROOT, "docs", "skill_map.png"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    draw(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
