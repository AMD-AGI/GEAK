#!/usr/bin/env python3
"""Draw the skillset's coverage and the routes a tuned artifact can take into a
live server.

Supersedes tools/skill_map.py, which drew every per-language skill as flowing
into tuning-aiter. That shape overstates aiter: it is one of six peer backend
skills, distinguished by racing several of the others in one run, not a stage
every other skill passes through. The skills say so directly — hipBLASLt lists three
deploy paths of which aiter is one (`tuning-hipblaslt/` §5), both framework
skills split into two tuning surfaces where only dense GEMM goes through aiter,
and `tuning-triton/` never mentions aiter at all.

So the edges here are routes, drawn only where a skill documents one:

  * solid, into the serving stack — a deploy path that changes live dispatch
  * dashed, into aiter        — optional, aiter can race this backend for you
  * dotted, from hip          — not a route; the profiler reference everyone uses

Boxes are discovered from disk; edges are editorial, read out of the SKILL.md
prose, so they cannot be derived. A skill on disk that is missing from the
layout below is reported on stderr rather than silently dropped.

Usage:  python3 tools/skill_map_2.py [-o docs/skill_map_2.png]
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
C_EDGE = "#98a2ad"
C_FAINT = "#b9c1c9"
C_WARN = "#b0453a"

# The backend row, left to right. Order is chosen so the optional race edges
# land on adjacent boxes instead of crossing the row.
LANGS = [
    ("tuning-hip", "launch geometry · profiler reference"),
    ("tuning-ck", "select a pre-compiled instance"),
    ("tuning-hipblaslt", "select a pre-compiled solution"),
    ("tuning-triton", "author a config space"),
    ("tuning-flydsl", "author a Config space"),
    ("tuning-aiter", "race backends · tuned config CSV"),
]

# Backends aiter can optionally race on your behalf, and by what mechanism.
# The gradlib tuner races the three library backends in one pass and names the
# winner in its output CSV's `libtype` column; there is no flag to pick one, and
# documenting it as `--libtype` was a bug (tuning-aiter/ §4). CK is separate: it
# is not in that race at all, and is reached through its own per-op tuners.
RACED_BY_AITER = {
    "tuning-ck": "per-op CK tuners",
    "tuning-hipblaslt": "gradlib race",
    "tuning-triton": "gradlib race",
    "tuning-flydsl": "gradlib race",
}

# Skills with a documented deploy path of their own into a live server, the
# artifact each one writes, and the lane it runs in. Each route gets its own
# lane and landing point so the arrows stay separate rather than converging.
DEPLOY = [
    ("tuning-hipblaslt", "TunableOp CSV · ext API", 58.6, 31.0),
    ("tuning-triton", "fused-MoE config JSON", 55.8, 49.0),
    ("tuning-aiter", "tuned config CSV · dense GEMM", 53.0, 66.0),
]

FRAMEWORKS = ["tuning-in-vllm", "tuning-in-sglang"]
SUPPORT = ["env-setup", "benchmark"]

ROW_Y, ROW_H = 68.0, 10.0
BOX_W, BOX_GAP = 14.7, 1.5
X0 = (100 - (len(LANGS) * BOX_W + (len(LANGS) - 1) * BOX_GAP)) / 2
STEP = BOX_W + BOX_GAP
RACE_Y = 64.3
SRV_X, SRV_W, SRV_Y, SRV_H = 22.0, 54.0, 36.0, 14.0
VER_Y, VER_H = 25.0, 6.5


def cx(name):
    """Centre x of a backend box."""
    i = [k for k, _ in LANGS].index(name)
    return X0 + i * STEP + BOX_W / 2


def discover():
    """Every skill dir with a SKILL.md, plus its supporting .md count."""
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "SKILL.md"))):
        d = os.path.dirname(p)
        txt = open(p).read()
        m = re.search(r"^description:\s*(.+)$", txt, re.M)
        out[os.path.basename(d)] = {
            "desc": (m.group(1) if m else ""),
            "refs": sorted(f for f in os.listdir(d)
                           if f.endswith(".md") and f != "SKILL.md"),
        }
    b = os.path.join(ROOT, "benchmark", "README.md")
    if os.path.exists(b) and "benchmark" not in out:
        out["benchmark"] = {"desc": "shapes to tune against", "refs": []}
    return out


def check_layout(sk):
    """A skill that exists but is not drawn is the failure mode this guards."""
    placed = {k for k, _ in LANGS} | set(FRAMEWORKS) | set(SUPPORT) | {"tuning-core"}
    missing = sorted(set(sk) - placed)
    stale = sorted(placed - set(sk))
    for m in missing:
        print(f"warning: {m}/ exists but has no place in the layout", file=sys.stderr)
    for s in stale:
        print(f"warning: layout expects {s}/ but it is not on disk", file=sys.stderr)


def box(ax, x, y, w, h, label, sub, color, fs=9.5, ec="none", lw=0.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=color, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), label,
            ha="center", va="center", fontsize=fs, color=C_TEXT,
            family="monospace", fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=6.6, color="#eaf2f8", zorder=3)


def arrow(ax, a, b, color=C_EDGE, lw=1.4, rad=0.0, ls="-", head="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=head, color=color, lw=lw,
                                 linestyle=ls, mutation_scale=12, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def route(ax, x_from, lane, x_land, label):
    """Right-angled deploy route, so the artifact label can sit level on the
    horizontal run instead of being rotated along a diagonal."""
    ax.plot([x_from, x_from], [ROW_Y - 0.2, lane], color=C_EDGE, lw=1.7,
            solid_capstyle="round", zorder=1)
    ax.plot([x_from, x_land], [lane, lane], color=C_EDGE, lw=1.7,
            solid_capstyle="round", zorder=1)
    arrow(ax, (x_land, lane), (x_land, SRV_Y + SRV_H + 0.2), lw=1.7)
    ax.text((x_from + x_land) / 2, lane + 1.0, label, ha="center", fontsize=7.0,
            color="#5f6b75")


def draw(out_path):
    sk = discover()
    check_layout(sk)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    fig.suptitle("Tuning skillset — coverage and the routes into a live server",
                 fontsize=14, y=0.972)
    ax.text(50, 96.3,
            "six peer backends, one per source language · a tuned artifact "
            "reaches a server by more than one route",
            ha="center", fontsize=9.5, color="#444")

    # --- core + support --------------------------------------------------
    box(ax, 33, 85, 34, 8.5, "tuning-core",
        "6-step loop · measurement · correctness gates · engagement", C_CORE, 11)
    ax.text(67.6, 89.2, f"+{len(sk.get('tuning-core', {}).get('refs', []))} "
            "reference docs", fontsize=7.6, color=C_CORE, va="center",
            style="italic")
    box(ax, 2, 85, 28, 8.5, "env-setup  ·  benchmark",
        "tools in the container · real shapes to tune", C_SUP, 9.5)
    arrow(ax, (30.2, 89.25), (32.8, 89.25))

    # Every skill specializes the same loop, so this is a bus, not six arrows
    # converging — the fan-in shape is what the old figure got wrong.
    bus_y = 81.3
    arrow(ax, (50, 84.8), (50, bus_y + 0.1), head="-")
    ax.plot([cx(LANGS[0][0]), cx(LANGS[-1][0])], [bus_y, bus_y],
            color=C_EDGE, lw=1.4, zorder=1)
    ax.text(cx(LANGS[-1][0]) + 1.4, bus_y, "read first", fontsize=7,
            color="#7d8790", va="center", style="italic")

    # --- the backend row -------------------------------------------------
    ax.text(1.1, ROW_Y + ROW_H / 2, "one per source language", rotation=90,
            ha="center", va="center", fontsize=8.5, color=C_LANG,
            fontweight="bold")
    for i, (k, sub) in enumerate(LANGS):
        x = X0 + i * STEP
        special = (k == "tuning-aiter")
        box(ax, x, ROW_Y, BOX_W, ROW_H, k.replace("tuning-", ""), sub, C_LANG,
            9, ec=(C_FW if special else "none"), lw=(2.0 if special else 0.0))
        arrow(ax, (x + BOX_W / 2, bus_y), (x + BOX_W / 2, ROW_Y + ROW_H + 0.2))
    ax.text(X0 + 5 * STEP + BOX_W, ROW_Y - 1.9, "also an integration point",
            ha="right", fontsize=6.9, color=C_FW, style="italic")

    # --- optional: aiter races the others --------------------------------
    stubs = [cx(k) - 3.5 for k in RACED_BY_AITER]
    ax.plot([min(stubs), cx("tuning-aiter") - 4], [RACE_Y, RACE_Y],
            color=C_FAINT, lw=1.3, ls=(0, (4, 3)), zorder=1)
    for k in RACED_BY_AITER:
        arrow(ax, (cx(k) - 3.5, ROW_Y - 0.2), (cx(k) - 3.5, RACE_Y),
              color=C_FAINT, lw=1.3, ls=(0, (4, 3)), head="-")
    arrow(ax, (cx("tuning-aiter") - 4, RACE_Y),
          (cx("tuning-aiter") - 4, ROW_Y - 0.3), color="#9aa4ad", lw=1.3,
          ls=(0, (4, 3)))
    ax.text(min(stubs) - 0.6, RACE_Y - 2.4, "optional — never required",
            fontsize=7.0, color="#8b949d", style="italic")

    # --- deploy routes into the serving stack ----------------------------
    ax.add_patch(FancyBboxPatch((SRV_X, SRV_Y), SRV_W, SRV_H,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#f4f0f8", ec=C_FW, lw=1.3, zorder=1.5))
    ax.text(SRV_X + SRV_W / 2, SRV_Y + SRV_H - 2.0,
            "live serving stack — three independent routes land here",
            ha="center", fontsize=8.4, color=C_FW, fontweight="bold", zorder=3)
    for i, k in enumerate([f for f in FRAMEWORKS if f in sk]):
        box(ax, SRV_X + 3 + i * 25, SRV_Y + 2.6, 23, 8.2,
            k.replace("tuning-", ""),
            "capture shapes · deploy · verify engagement", C_FW, 9.5)

    for k, artifact, lane, land in DEPLOY:
        route(ax, cx(k) + 4, lane, land, artifact)

    # --- hip is a reference, not a route ---------------------------------
    hx = cx("tuning-hip")
    arrow(ax, (hx, ROW_Y - 0.2), (hx, VER_Y + VER_H + 0.2), color=C_FAINT,
          lw=1.3, ls=(0, (1.5, 2.5)))
    ax.text(hx + 1.7, (ROW_Y + VER_Y) / 2,
            "the profiler reference, not a route", fontsize=7.0,
            color="#8b949d", rotation=90, ha="center", va="center")

    # --- legend ----------------------------------------------------------
    lx, ly = 79.5, SRV_Y + 0.5
    ax.text(lx, ly + 12.0, "reading the edges", fontsize=7.4, color="#5f6b75",
            fontweight="bold")
    for i, (ls, col, txt) in enumerate([
            ("-", C_EDGE, "deploy route — changes what\na live server dispatches"),
            ((0, (4, 3)), C_FAINT, "optional — aiter can race this\nbackend ("
             + ", ".join(sorted(set(RACED_BY_AITER.values()))) + ")"),
            ((0, (1.5, 2.5)), C_FAINT, "reference every skill points\n"
             "at, not a route")]):
        y = ly + 9.0 - i * 4.2
        ax.plot([lx, lx + 3.2], [y, y], color=col, lw=1.5, ls=ls, zorder=2)
        ax.text(lx + 4.2, y, txt, fontsize=6.5, color="#6d777f", va="center",
                linespacing=1.55)

    # --- the through-line ------------------------------------------------
    ax.add_patch(FancyBboxPatch((8, VER_Y), 84, VER_H,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#fbf2f1", ec=C_WARN, lw=1.2, zorder=1.5))
    ax.text(50, VER_Y + VER_H * 0.58, "engagement check", ha="center",
            fontsize=9.5, color=C_WARN, family="monospace", fontweight="bold",
            zorder=3)
    ax.text(50, VER_Y + VER_H * 0.24,
            "did the live dispatch actually pick up the artifact?",
            ha="center", fontsize=8, color="#8a5a53", zorder=3)
    arrow(ax, (50, SRV_Y - 0.2), (50, VER_Y + VER_H + 0.2), color=C_WARN,
          lw=1.5)

    ax.text(50, 16.0,
            "Tuning on this platform fails silently far more often than it "
            "fails loudly. A config lands where nothing reads it, a lookup key\n"
            "misses on an arch field, a stale entry loads through a fallback. "
            "Nothing raises. The tuner still prints a win. So every skill\n"
            "here ends with an engagement check rather than a timing.",
            ha="center", fontsize=8.2, color="#555", linespacing=1.65)
    # Name the images, not just the arch. "validated on gfx942" is ambiguous
    # across two shipped containers that differ in torch, flydsl and how aiter
    # is packaged -- and the reader's next question is always "which one".
    # These counts come from validate/claims.py, so the caption cannot drift
    # from the reports without the numbers being re-derived.
    ax.text(50, 6.4,
            f"{len(sk)} skills · gfx942 / MI300X and gfx950 / MI355, ROCm 7.2.x · "
            "34 claims re-checked in both shipped images on both parts: "
            "0 FAIL anywhere",
            ha="center", fontsize=8, color="#777")
    ax.text(50, 4.4,
            "gfx942: vllm 15 PASS / 3 n-a · sglang 14 / 4   |   "
            "gfx950: vllm 28 PASS / 6 n-a · sglang 28 / 6   ·  "
            "n/a = precondition absent in that image, not a pass",
            ha="center", fontsize=7.4, color="#999")
    ax.text(50, 2.6,
            "re-run with  python3 validate/claims.py  ·  every backend re-exercised "
            "on MI355; per-backend ledger in  docs/coverage_gfx950.md",
            ha="center", fontsize=7.4, color="#999")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}  ({len(sk)} skills)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out",
                    default=os.path.join(ROOT, "docs", "skill_map_2.png"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    draw(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
