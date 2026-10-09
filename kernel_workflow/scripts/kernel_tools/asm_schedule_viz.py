#!/usr/bin/env python3
"""Static hot-loop ISSUE-ORDER schedule visualizer.

Without ATT (no rocprof-trace-decoder), we cannot get wall-clock per-instruction
timestamps. But the assembler hot loop IS the compiler/author's issue ORDER, so
plotting each instruction (x = program index) on its op-class lane directly shows
the interleave / ping-pong pattern:

  * MFMA and VALU lanes alternating tightly  -> interleaved (ping-pong / good overlap)
  * MFMA lane empty over a long run while VALU is solid -> exposed serial VALU (bubble)

Optionally weights each instruction by an approximate issue latency (cyc/op) to get
an analytic cycle-x-axis. Requires matplotlib; reuses the skill's *_loop_audit.py.

Usage:
  python3 asm_schedule_viz.py <a.s> [--loop-label L] [--max 160] [--cycles] --out sched.png [--label NAME]
  python3 asm_schedule_viz.py --compare a.s=ASM b.s=TRITON --out cmp.png [--cycles]
  python3 asm_schedule_viz.py --selftest     # offline; renders only when matplotlib imports
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import asm_loop_audit as _A  # gluon / tilelang
except ModuleNotFoundError:
    import isa_loop_audit as _A  # flydsl

# approximate issue/occupy latency per op-class (gfx9xx, wave64), cycles.
_CYC = {"mfma": 16, "valu": 4, "exp": 4, "lds_read": 4, "lds_write": 4,
        "gld": 4, "gst": 4, "waitcnt": 1, "barrier": 4, "nop": 1,
        "scalar": 1, "other": 1}
_LANES = ["mfma", "valu", "exp", "lds_read", "lds_write", "gld", "gst",
          "waitcnt", "barrier", "nop", "scalar", "other"]
_COLOR = {"mfma": "#d62728", "valu": "#1f77b4", "exp": "#9467bd",
          "lds_read": "#2ca02c", "lds_write": "#98df8a", "gld": "#ff7f0e",
          "gst": "#ffbb78", "waitcnt": "#c7c7c7", "barrier": "#8c564b",
          "nop": "#e377c2", "scalar": "#7f7f7f", "other": "#cccccc"}


def hot_loop_ops(asm_path, loop_label, max_instr):
    lines = open(asm_path).read().splitlines()
    loops = _A.find_loops(lines)
    picked = _A.pick_loop(lines, loops, loop_label)
    if picked is None:
        raise SystemExit(f"no loop in {asm_path}")
    lab, s, e = picked
    ops = []
    for ln in lines[s:e]:
        mn = _A._mnemonic(ln)
        if mn is None:
            continue
        name, _sym = _A.classify(mn)
        ops.append(name)
        if len(ops) >= max_instr:
            break
    return lab, ops


def _plot_one(ax, ops, title, cycles):
    x = 0.0
    trans = 0
    prev_mv = None
    for name in ops:
        w = _CYC.get(name, 1) if cycles else 1.0
        y = len(_LANES) - 1 - _LANES.index(name if name in _LANES else "other")
        ax.barh(y, w, left=x, height=0.8, color=_COLOR.get(name, "#ccc"), edgecolor="none", zorder=3)
        x += w
        if name in ("mfma", "valu"):
            if prev_mv is not None and prev_mv != name:
                trans += 1
            prev_mv = name
    ax.set_yticks(range(len(_LANES)))
    ax.set_yticklabels(list(reversed(_LANES)), fontsize=7)
    ax.set_xlim(0, x)
    ax.set_title(f"{title}   (n={len(ops)} instr, MFMA<->VALU transitions={trans}, "
                 f"x={'cycles(approx)' if cycles else 'issue index'})", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)


def render(specs, out, cycles, loop_label=None, max_instr=160):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    n = len(specs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.6 * n + 0.6), squeeze=False)
    for i, (path, label) in enumerate(specs):
        lab, ops = hot_loop_ops(path, loop_label, max_instr)
        _plot_one(axes[i][0], ops, f"{label}  [loop {lab}]", cycles)
        if i == n - 1:
            axes[i][0].set_xlabel("cycles (approx, latency-weighted)" if cycles else "instruction issue order")
    handles = [mpatches.Patch(color=_COLOR[c], label=c) for c in _LANES]
    fig.legend(handles=handles, ncol=6, fontsize=7, loc="lower center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


_SELFTEST_ASM = """\
\t.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
k:
.LBB0_1:
\tv_mfma_f32_32x32x16_bf16 a[0:15], v[0:3], v[4:7], a[0:15]
\tv_mfma_f32_32x32x16_bf16 a[16:31], v[8:11], v[4:7], a[16:31]
\tv_add_u32_e32 v20, 64, v20
\tds_read_b128 v[0:3], v20
\tv_exp_f32_e32 v30, v30
\ts_waitcnt lgkmcnt(0)
\tv_mfma_f32_32x32x16_bf16 a[0:15], v[0:3], v[4:7], a[0:15]
\ts_add_u32 s0, s0, 1
\ts_cbranch_scc0 .LBB0_1
\ts_endpgm
"""


def _selftest():
    """Offline: the hot-loop op extraction always; the render only where matplotlib imports."""
    import tempfile
    fails = []
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "k.s")
        with open(path, "w") as f:
            f.write(_SELFTEST_ASM)
        lab, ops = hot_loop_ops(path, None, 160)
        if lab != ".LBB0_1" or ops.count("mfma") != 3:
            fails.append(f"hot loop {lab!r} ops {ops}")
        if any(o not in _LANES for o in ops):
            fails.append(f"an op class has no lane: {sorted(set(ops) - set(_LANES))}")
        if len(hot_loop_ops(path, None, 2)[1]) != 2:
            fails.append("--max must truncate the op stream")
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("  -- matplotlib not importable: render not exercised")
        else:
            out = os.path.join(td, "sched.png")
            render([(path, "selftest")], out, cycles=True)
            if not os.path.getsize(out):
                fails.append("render wrote an empty png")
    for msg in fails:
        print("FAIL", msg)
    print("[asm_schedule_viz] SELFTEST " + ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


def main():
    if "--selftest" in sys.argv:
        return _selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asm", nargs="?")
    ap.add_argument("--label", default="kernel")
    ap.add_argument("--compare", nargs="+", help="path=LABEL entries for stacked compare")
    ap.add_argument("--loop-label")
    ap.add_argument("--max", type=int, default=160)
    ap.add_argument("--cycles", action="store_true", help="latency-weighted x-axis")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.compare:
        specs = []
        for e in a.compare:
            p, _, l = e.partition("=")
            specs.append((p, l or os.path.basename(p)))
    else:
        specs = [(a.asm, a.label)]
    render(specs, a.out, a.cycles, a.loop_label, a.max)


if __name__ == "__main__":
    sys.exit(main())
