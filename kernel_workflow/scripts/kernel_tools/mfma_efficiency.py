#!/usr/bin/env python3
"""Observed matrix-instruction intervals from already decoded wave/code JSON.

Reads rocprofv3 ATT artifacts; does not collect traces or install a decoder. Reports
cadence and inter-issue intervals, grouped by the preceding non-matrix instruction.
Those intervals are observations, not isolated latency or proof of a stall cause.
Interpret them alongside SKU/roofline, profile and ISA/IR; the agent chooses and
validates optimizations. No bound-to-tactic mapping is applied.

--ideal-cadence is optional and requires a reference for the actual instruction shape
and architecture. Without it only observed cadence (including p10) is reported.
This is not a roofline calculation or chip-wide MFMA utilization measurement.

Usage: python3 mfma_efficiency.py wave.json code.json [--arch gfx950]
       python3 mfma_efficiency.py --compare wave,code,LABEL wave,code,LABEL
"""
import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_mfma_analysis as _DM

# A dtype alone cannot establish an instruction-shape-specific issue cadence.
_DTYPE_BY_MNEMONIC = (
    # ordered: the fp8/bf8 families must be tested before f16/bf16 so `..._bf8_bf8` is not read
    # as bf16 by a loose substring match.
    ("fp8", ("fp8",)), ("bf8", ("bf8",)),
    ("bf16", ("bf16",)), ("fp16", ("f16",)),
)


def mfma_dtype(mnemonics):
    """The MFMA input dtype, from the mnemonics. None when they disagree or are unrecognised --
    a mixed or unknown stream must not silently pick one arm's cadence."""
    seen = set()
    for m in mnemonics:
        low = m.lower()
        if "fp8" in low and "bf8" in low:
            return None
        for dt, needles in _DTYPE_BY_MNEMONIC:
            if any(n in low for n in needles):
                seen.add(dt)
                break
        else:
            return None
    return seen.pop() if len(seen) == 1 else None


def resolve_ideal_cadence(override, arch, dtype):
    """Only use an explicitly scoped reference; arch/dtype tables omit instruction shape."""
    if override is not None:
        if not math.isfinite(override) or override <= 0:
            raise ValueError("--ideal-cadence must be positive and finite")
        return override, "--ideal-cadence (explicit; caller must establish instruction/arch scope)"
    return None, (f"UNRESOLVED (arch={arch or '?'}, dtype={dtype or '?'}; "
                  "dtype-only hardware constants do not establish a per-instruction cadence)")


def _pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round((p / 100.0) * (len(xs) - 1))))
    return xs[k]


def load(wave_json, code_json):
    """Read a decoded wave; refuse unresolved code indices or malformed timestamps."""
    w = json.loads(Path(wave_json).read_text())["wave"]
    code = json.loads(Path(code_json).read_text())["code"]
    begin = w["begin"]
    if type(begin) not in (int, float) or not math.isfinite(begin):
        raise ValueError("invalid wave begin timestamp")
    ev = []
    for e in w["instructions"]:
        if not isinstance(e, list) or len(e) < 5:
            raise ValueError("invalid decoded instruction record")
        idx, timestamp = e[4], e[0]
        if type(idx) is not int or not 0 <= idx < len(code):
            raise ValueError(f"unresolved code index: {idx}")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < begin:
            raise ValueError("invalid instruction timestamp")
        if not isinstance(code[idx], list) or not code[idx] or not isinstance(code[idx][0], str):
            raise ValueError(f"invalid code record: {idx}")
        asm = code[idx][0].strip()
        if not asm or asm.startswith(";"):
            raise ValueError(f"instruction has no decoded mnemonic: {idx}")
        mnem = asm.split()[0]
        ev.append((timestamp - begin, _DM.sub_class(mnem), mnem))
    ev.sort(key=lambda x: x[0])
    return ev


def analyze(ev):
    indices = [i for i, (_, sc, _) in enumerate(ev) if sc == "mfma"]
    cadence = [ev[j][0] - ev[i][0] for i, j in zip(indices, indices[1:])]
    span = ev[indices[-1]][0] - ev[indices[0]][0] if cadence else None
    cyc_by_cls, cnt_by_cls = Counter(), Counter()
    if cadence:
        for i in range(indices[0] + 1, indices[-1]):
            c, sc, _ = ev[i]
            if sc != "mfma":
                cyc_by_cls[sc] += ev[i + 1][0] - c
                cnt_by_cls[sc] += 1
    return {
        "cadence_mean": statistics.mean(cadence) if cadence else None,
        "cadence_median": statistics.median(cadence) if cadence else None,
        "cadence_p10": _pct(cadence, 10) if cadence else None, "n_mfma": len(indices), "span": span,
        "cyc_by_cls": cyc_by_cls, "cnt_by_cls": cnt_by_cls,
        "mfma_mnemonics": sorted({m for _c, sc, m in ev if sc == "mfma"})}


def report(r, label, ideal, ideal_src="", dtype=None):
    print(f"\n=== {label} ===")
    n, span = r["n_mfma"], r["span"]
    med, mean, p10 = r["cadence_median"], r["cadence_mean"], r["cadence_p10"]
    print(f"  MFMA: {n} in span {span} cyc"
          + (f"  dtype={dtype}" if dtype else "  dtype=? (mnemonics: "
             + ",".join(r.get("mfma_mnemonics") or ["none"])[:60] + ")"))
    if med is None:
        print("  MFMA cadence UNKNOWN: fewer than two matrix instructions; no inter-MFMA interval.")
        return
    print(f"  MFMA cadence (cyc between consecutive MFMA): "
          f"p10={p10} (observed percentile)  median={med:.0f}  mean={mean:.1f}  "
          f"ideal={ideal if ideal is not None else 'UNKNOWN'}"
          + (f" [{ideal_src}]" if ideal_src else ""))
    # reference-free efficiency: median vs the two self/hardware references.
    if med and ideal:
        eff_self = 100.0 * p10 / med if med else 0        # how close to THIS kernel's own best
        eff_ideal = 100.0 * ideal / med if med else 0     # how close to the hardware cadence
        print(f"  cadence ratios: vs-p10-self={eff_self:.0f}%  vs-ideal={eff_ideal:.0f}%")
    elif med:
        print(f"  cadence ratios: vs-p10-self={100.0 * p10 / med:.0f}%  vs-ideal=UNKNOWN "
              f"({ideal_src})")
    print("  Bound classification and optimization choices require agent analysis of "
          "SKU/roofline, profile and ISA/IR evidence, followed by validation.")
    tot = sum(r["cyc_by_cls"].values())
    denom = tot or 1
    print(f"  observed inter-issue intervals grouped by preceding instruction ({tot} cyc total between MFMA):")
    for sc, cyc in r["cyc_by_cls"].most_common():
        cnt = r["cnt_by_cls"][sc]
        print(f"      {sc:16s} {cyc:8g} cyc ({100*cyc/denom:4.1f}%)   {cnt:5d} instr  "
              f"{cyc/cnt if cnt else 0:5.1f} cyc/instr")
    cb = r["cyc_by_cls"]
    print("  These intervals do not establish stall cause, instruction latency or chip utilization.")
    # Display measured instruction groups; their order does not select an optimization.
    roll = [
        ("fp-compute (fp_math+exp+pack)", "fp-compute", cb["valu_fp_math"] + cb["exp"] + cb["valu_fp_pack"]),
        ("LDS instructions",             "lds-feed",   cb["lds"]),
        ("global memory instructions",              "gmem-feed",  cb["gmem"]),
        ("addressing (int)",                      "addressing", cb["valu_addr_int"] + cb["scalar"]),
        ("cvt/mov/other-valu",                    "cvt-mov",    cb["valu_cvt_mov"] + cb["valu_other"]),
        ("memory wait instructions",       "wait",       cb["waitcnt"]),
        ("SYNC (barrier+nop)",                    "sync",       cb["barrier"] + cb["nop"]),
    ]
    print("  rollup of inter-MFMA cycles by instruction group:")
    for group_label, _slug, v in sorted(roll, key=lambda x: -x[2]):
        print(f"      {group_label:40s} {100*v/denom:5.1f}%")
    # Keep LDS and global traffic separate, with the same denominator as the rollup.
    print("  feed-bucket split (LDS and global):")
    print(f"      {'LDS instructions':40s} {100*cb['lds']/denom:5.1f}%")
    print(f"      {'global memory instructions':40s} {100*cb['gmem']/denom:5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wave", nargs="?")
    ap.add_argument("code", nargs="?")
    ap.add_argument("--label", default="wave")
    ap.add_argument("--arch", default=None,
                    help="gfx target for context; dtype-only constants are not a shape-specific cadence")
    ap.add_argument("--ideal-cadence", type=int, default=None,
                    help="OVERRIDE the resolved theoretical cadence (cyc). Only needed when --arch "
                         "cannot be given or the dtype is not inferable; state the source and "
                         "applicable instruction shape for this reference.")
    ap.add_argument("--compare", nargs="+",
                    help="wave,code,LABEL entries (OPTIONAL peer/vendor diagnostic upper-bound)")
    a = ap.parse_args()
    if not a.compare and not (a.wave and a.code):
        ap.error("provide wave.json and code.json, or --compare")
    if a.ideal_cadence is not None and a.ideal_cadence <= 0:
        ap.error("--ideal-cadence must be positive")
    if a.compare and any(len(e.split(",")) != 3 for e in a.compare):
        ap.error("--compare entries must be wave,code,LABEL")
    specs = ([tuple(e.split(",")) for e in a.compare] if a.compare
             else [(a.wave, a.code, a.label)])
    for w, c, l in specs:
        r = analyze(load(w, c))
        # Resolve per stream: --compare can hold arms with different MFMA shapes, and reusing one
        # arm's cadence across them is the same inversion in a different disguise.
        dtype = mfma_dtype(r.get("mfma_mnemonics") or [])
        ideal, src = resolve_ideal_cadence(a.ideal_cadence, a.arch, dtype)
        report(r, l, ideal, src, dtype)


def _selftest():
    """No selftest shipped with this tool before, which is how a hardcoded fp16 cadence survived in
    a file whose verdict depends on it. These assert the resolution, not the formatting."""
    fail = 0

    def chk(cond, msg):
        nonlocal fail
        if not cond:
            print(f"FAIL: {msg}")
            fail = 1

    # dtype off the mnemonic, which is what removes the need to hand-supply a cadence at all.
    chk(mfma_dtype(["v_mfma_f32_32x32x16_fp8_fp8"]) == "fp8", "fp8 mnemonic -> fp8")
    chk(mfma_dtype(["v_mfma_f32_32x32x16_bf8_bf8"]) == "bf8", "bf8 must not read as bf16")
    chk(mfma_dtype(["v_mfma_f32_32x32x8_bf16_1k"]) == "bf16", "bf16_1k -> bf16")
    chk(mfma_dtype(["v_mfma_f32_16x16x16_f16"]) == "fp16", "f16 -> fp16")
    # A mixed stream has no single cadence, so it must refuse rather than pick one arm's.
    chk(mfma_dtype(["v_mfma_f32_16x16x16_f16", "v_mfma_f32_32x32x16_fp8_fp8"]) is None,
        "a mixed dtype stream must resolve to None")
    chk(mfma_dtype(["v_mfma_f32_16x16x32_f8f6f4"]) is None, "an unrecognised family -> None")

    # An explicit override always wins, and says so.
    cyc, src = resolve_ideal_cadence(24, "gfx942", "fp16")
    chk(cyc == 24 and "explicit" in src, f"override should win, got {cyc} {src}")
    # Unknown arch or dtype must yield None -- NOT a fallback constant.
    chk(resolve_ideal_cadence(None, None, "fp8")[0] is None, "no arch -> unresolved, not a default")
    chk(resolve_ideal_cadence(None, "gfx942", None)[0] is None, "no dtype -> unresolved")

    chk(resolve_ideal_cadence(None, "gfx950", "fp16")[0] is None,
        "dtype-only tables cannot establish a shape-specific cadence")
    chk(mfma_dtype(["v_mfma_f32_16x16x16_f16", "v_mfma_f32_16x16x32_f8f6f4"]) is None,
        "unknown instructions must not disappear from a mixed stream")

    print("MFMA_EFFICIENCY SELFTEST " + ("PASS" if not fail else "FAIL"))
    return fail


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()
