#!/usr/bin/env python3
"""Aggregate rocprofv3 ATT code.json per op-class (ATT-measured cycles).

ATT gives measured per-instruction Hit / Latency / Stall / Idle cycles. We classify
each instruction (MFMA/VALU/LDS/...) and sum, giving an ATT-accurate per-op-class
active(latency-stall) vs stall(bubble) breakdown — the real overlap picture that the
static ISA histogram and the whole-kernel PMC aggregate cannot give.

Requires the ATT decoder (rocprof-trace-decoder) so rocprofv3 --att emits code.json.
Reuses asm_loop_audit.py's classifier (same kernel_tools/ dir).
GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/); the Gluon pack's scripts/ copy is a shim.
The ATT decoder (rocprof-trace-decoder) is NOT vendored: point ROCPROF_ATT_LIBRARY_PATH at it when
collecting; without it there is no code.json and this tool has nothing to read (the layer is DEGRADED).

Usage: python3 att_opclass.py <ui_output_agent_*_dispatch_N/code.json> [--png out.png] [--json out.json]
       python3 att_opclass.py --selftest
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import asm_loop_audit as _A  # gluon / tilelang
except ModuleNotFoundError:
    import isa_loop_audit as _A  # flydsl


def analyze(code_json):
    d = json.load(open(code_json))
    hdr = d["header"]
    if isinstance(hdr, str):  # rocprofv3 v3.0.0 emits header as a comma-string
        hdr = [x.strip() for x in hdr.split(",")]
    rows = d.get("code") or []  # can be None if the traced CU caught no waves
    ci = {n: i for i, n in enumerate(hdr)}
    H, L, St, Id = ci["Hit"], ci["Latency"], ci["Stall"], ci["Idle"]
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # hit, lat, stall, idle
    for r in rows:
        asm = (r[0] or "").strip()
        if not asm or asm.startswith(";"):
            continue
        cls, _ = _A.classify(asm.split()[0])
        a = agg[cls]
        a[0] += r[H] or 0
        a[1] += r[L] or 0
        a[2] += r[St] or 0
        a[3] += r[Id] or 0
    return agg


def report(agg, title=""):
    tot_lat = sum(v[1] for v in agg.values()) or 1
    tot_stall = sum(v[2] for v in agg.values()) or 1
    print(f"\n=== {title} ===")
    hdr = ("opclass", "hits", "latency", "lat%", "stall", "stall%", "stall/lat")
    print("{:10s} {:>9s} {:>11s} {:>6s} {:>11s} {:>7s} {:>9s}".format(*hdr))
    for cls, (h, l, st, idl) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        if l == 0 and st == 0:
            continue
        print("{:10s} {:9d} {:11d} {:5.1f}% {:11d} {:6.1f}% {:8.1f}%".format(
            cls, h, l, 100 * l / tot_lat, st, 100 * st / tot_stall, 100 * st / l if l else 0))
    print(f"TOTAL latency={tot_lat}  stall={tot_stall}  "
          f"({100 * tot_stall / tot_lat:.1f}% of latency is stall/bubble)")
    return tot_lat, tot_stall


def render_png(agg, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    items = [(c, v[1], v[2]) for c, v in agg.items() if v[1] > 0]
    items.sort(key=lambda x: -x[1])
    tot = sum(l for _, l, _ in items) or 1
    labels = [c for c, _, _ in items]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(labels) + 1.5))
    for i, (c, l, st) in enumerate(items):
        active = l - st
        ax.barh(i, 100 * active / tot, color="#1f77b4", zorder=3)
        ax.barh(i, 100 * st / tot, left=100 * active / tot, color="#d62728", zorder=3)
        ax.text(100 * l / tot + 0.3, i, f"{100*l/tot:.1f}%  (stall {100*st/l if l else 0:.0f}%)",
                va="center", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of total latency cycles  (blue=active/issuing, red=stall/bubble)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.legend(handles=[mp.Patch(color="#1f77b4", label="active"),
                       mp.Patch(color="#d62728", label="stall/bubble")], fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _selftest():
    import tempfile
    import _att_fixture
    with tempfile.TemporaryDirectory() as td:
        _wave, code = _att_fixture.write(td)
        agg = analyze(code)
        # waitcnt: hit 4, latency 400, stall 380; prologue comment row dropped
        ok = (agg["waitcnt"] == [4, 400, 380, 0] and agg["mfma"][1] == 128
              and "" not in agg and sum(v[1] for v in agg.values()) == 836)
        with open(code, "w") as fh:      # a traced CU that caught no waves: code is null
            json.dump({"header": _att_fixture.HEADER, "code": None}, fh)
        ok = ok and analyze(code) == {}
    print("[att_opclass] SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(_selftest())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code_json")
    ap.add_argument("--json", help="dump per-op-class dict to this path")
    ap.add_argument("--png", help="render a per-op-class active/stall bar chart")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    agg = analyze(a.code_json)
    ttl = a.title or os.path.basename(os.path.dirname(a.code_json))
    report(agg, ttl)
    if a.png:
        render_png(agg, a.png, ttl)
    if a.json:
        out = {c: dict(hit=v[0], latency=v[1], stall=v[2], idle=v[3]) for c, v in agg.items()}
        json.dump(out, open(a.json, "w"), indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
