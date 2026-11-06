#!/usr/bin/env python3
"""Real per-wave instruction timeline from rocprofv3 ATT.

Uses a decoded ATT wave file (se*_sm*_sl*_wv0.json) + code.json. Each executed
instruction has a CYCLE timestamp (col0) and a code-row index (col4); we color it
by op-class and place it on that lane at its real cycle, width = gap to the next
instruction (so STALLS show up as the bar's real elapsed width). This is the true
overlap / ping-pong picture (not static issue order, not aggregated PMC).

Requires the ATT decoder (rocprof-trace-decoder) output + matplotlib. Reuses the skill's
*_loop_audit.py classifier.

Usage:
  python3 att_timeline.py <wave.json> <code.json> --start 24000 --span 1600 --out t.png --label NAME
  python3 att_timeline.py --selftest
(GEAK shared kernel tool in kernel_tools/; the pack's scripts/ copy is a shim. Decoder not vendored:
ROCPROF_ATT_LIBRARY_PATH.)
  python3 att_timeline.py --compare A=<wv>,<code>,LABEL B=<wv>,<code>,LABEL --span 1600 --auto --out cmp.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import asm_loop_audit as _A  # gluon / tilelang
except ModuleNotFoundError:
    import isa_loop_audit as _A  # flydsl

_LANES = ["mfma", "valu", "exp", "lds_read", "lds_write", "gld", "gst",
          "waitcnt", "barrier", "nop", "scalar", "other"]
_COLOR = {"mfma": "#d62728", "valu": "#1f77b4", "exp": "#9467bd",
          "lds_read": "#2ca02c", "lds_write": "#98df8a", "gld": "#ff7f0e",
          "gst": "#ffbb78", "waitcnt": "#c7c7c7", "barrier": "#8c564b",
          "nop": "#e377c2", "scalar": "#7f7f7f", "other": "#cccccc"}


def load_wave(wave_json, code_json):
    w = json.load(open(wave_json))["wave"]
    code = json.load(open(code_json))["code"]
    begin = w["begin"]
    ev = []  # (rel_cycle, cost, opclass)
    for e in w["instructions"]:
        idx = e[4]
        if idx >= len(code):
            continue
        asm = (code[idx][0] or "").strip()
        if not asm or asm.startswith(";"):
            continue
        cls, _sym = _A.classify(asm.split()[0])
        ev.append((e[0] - begin, e[2] or 0, cls))
    return ev


def auto_window(ev, span):
    """Pick a steady-state window containing MFMA (the K-loop)."""
    mfma_cyc = [c for c, _, cls in ev if cls == "mfma"]
    if not mfma_cyc:
        return ev[len(ev) // 3][0] if ev else 0
    start = mfma_cyc[len(mfma_cyc) // 2]
    return max(0, start - span // 4)


def plot(ax, ev, start, span, title):
    end = start + span
    win = [(c, cost, cls) for c, cost, cls in ev if start <= c < end]
    trans = 0
    prev_mv = None
    for i, (c, cost, cls) in enumerate(win):
        nxt = win[i + 1][0] if i + 1 < len(win) else c + max(cost, 1)
        w = max(nxt - c, 1)
        y = len(_LANES) - 1 - (_LANES.index(cls) if cls in _LANES else _LANES.index("other"))
        ax.barh(y, w, left=c - start, height=0.82, color=_COLOR.get(cls, "#ccc"),
                edgecolor="none", zorder=3)
        if cls in ("mfma", "valu"):
            if prev_mv is not None and prev_mv != cls:
                trans += 1
            prev_mv = cls
    ax.set_yticks(range(len(_LANES)))
    ax.set_yticklabels(list(reversed(_LANES)), fontsize=7)
    ax.set_xlim(0, span)
    ax.set_title(f"{title}   (window {span} cyc, {len(win)} instr, MFMA<->VALU transitions={trans})",
                 fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)


def render(specs, out, span, start, auto):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    n = len(specs)
    fig, axes = plt.subplots(n, 1, figsize=(15, 2.7 * n + 0.6), squeeze=False)
    for i, (wave_json, code_json, label) in enumerate(specs):
        ev = load_wave(wave_json, code_json)
        s = auto_window(ev, span) if auto else start
        plot(axes[i][0], ev, s, span, f"{label}  [ATT wave, start@{s}cyc]")
        if i == n - 1:
            axes[i][0].set_xlabel("cycles (real, relative to window start; gaps = stalls/bubbles)")
    handles = [mpatches.Patch(color=_COLOR[c], label=c) for c in _LANES]
    fig.legend(handles=handles, ncol=6, fontsize=7, loc="lower center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _selftest():
    import tempfile
    import _att_fixture
    with tempfile.TemporaryDirectory() as td:
        wave, code = _att_fixture.write(td)
        ev = load_wave(wave, code)
        ok = (len(ev) == len(_att_fixture.WAVE_INSTR) and ev[0][0] == 0
              and [c for c, _, k in ev if k == "mfma"] == [412, 432]
              and auto_window(ev, 1600) == 32)
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("  (matplotlib absent: render step skipped -- data path only)")
        else:
            png = os.path.join(td, "t.png")
            render([(wave, code, "fixture")], png, span=600, start=0, auto=False)
            ok = ok and os.path.getsize(png) > 0
    print("[att_timeline] SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(_selftest())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wave", nargs="?")
    ap.add_argument("code", nargs="?")
    ap.add_argument("--label", default="wave")
    ap.add_argument("--compare", nargs="+", help="wave,code,LABEL entries")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--span", type=int, default=1600)
    ap.add_argument("--auto", action="store_true", help="auto-pick a steady-state MFMA window")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.compare:
        specs = [tuple(e.split(",")) for e in a.compare]
    else:
        specs = [(a.wave, a.code, a.label)]
    render(specs, a.out, a.span, a.start, a.auto)


if __name__ == "__main__":
    main()
