#!/usr/bin/env python3
"""Deep MFMA-stream analysis: how clean is the MFMA burst, and what VALU is
interleaved between MFMA instructions.

For a hot-loop .s/.amdgcn, reports:
  1. contiguous MFMA run-length distribution (how many MFMA issue back-to-back)
  2. the instructions that BREAK the MFMA stream (inter-MFMA gaps), bucketed into
     VALU sub-classes: int-addressing / fp-math / pack-convert / exp / lds / other
  3. top interleaved mnemonics

This is the STATIC (issue-order) companion to mfma_efficiency.py (which does the
cycle-accurate ATT version). Reuses the hot-loop finder + classifier from the
skill's *_loop_audit.py (asm_loop_audit for gluon/tilelang, isa_loop_audit for
flydsl). Stdlib only, no GPU.

Usage: python3 deep_mfma_analysis.py <a.s> [--loop-label L]
       python3 deep_mfma_analysis.py --compare a.s=ASM b.s=TRI ...
       python3 deep_mfma_analysis.py --selftest          # offline, synthetic gfx950 loop
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import asm_loop_audit as _A  # gluon / tilelang
except ModuleNotFoundError:
    import isa_loop_audit as _A  # flydsl

# fine-grained VALU sub-class (order matters, first match wins)
_SUB = [
    ("mfma",      re.compile(r"^v_mfma|^v_smfmac|^v_wmma")),
    ("exp",       re.compile(r"^v_(exp|log|rcp|rsq|sqrt)")),
    ("lds",       re.compile(r"^ds_")),
    ("gmem",      re.compile(r"^(global|buffer|flat|scratch)_")),
    # integer / addressing VALU (index math, pointer bumps, masks, shifts)
    ("valu_addr_int", re.compile(r"^v_(add|addc|add_co|add_co_ci|sub|subrev|mad|mul_lo|mul_hi)_(u32|i32|u16|i16)"
                                 r"|^v_(lshl|lshr|ashr|and|or|xor|not|bfe|bfi|alignbit|lshlrev|lshrrev|ashrrev)")),
    # fp compute VALU (softmax / scale / rescale)
    ("valu_fp_math", re.compile(r"^v_(add|sub|subrev|mul|fma|fmac|mad|max|min|med)_f(32|16)")),
    # packed fp
    ("valu_fp_pack", re.compile(r"^v_pk_")),
    # data movement / convert / select
    ("valu_cvt_mov", re.compile(r"^v_(cvt|pack|perm|cndmask|mov|readlane|writelane|readfirstlane|bfrev)")),
    ("valu_other",   re.compile(r"^v_")),
    ("waitcnt",   re.compile(r"^s_waitcnt|^s_wait_")),
    ("barrier",   re.compile(r"^s_barrier|^s_barrier_")),
    ("nop",       re.compile(r"^s_nop")),
    ("scalar",    re.compile(r"^s_")),
]


def sub_class(mn):
    for name, rx in _SUB:
        if rx.match(mn):
            return name
    return "other"


def hot_ops(path, loop_label):
    lines = open(path).read().splitlines()
    loops = _A.find_loops(lines)
    picked = _A.pick_loop(lines, loops, loop_label)
    if picked is None:
        raise SystemExit(f"{path}: no back-edge loop found (pass --loop-label).")
    lab, s, e = picked
    ops = []
    for ln in lines[s:e]:
        mn = _A._mnemonic(ln)
        if mn is None:
            continue
        ops.append((mn, sub_class(mn)))
    return lab, ops


def analyze(ops):
    # 1. MFMA run lengths
    runs = []
    cur = 0
    # 2. inter-MFMA gap content: instructions strictly between two MFMA
    gap_hist = Counter()
    gap_mnem = Counter()
    in_stream = False   # have we seen the first MFMA yet
    gap_len = []
    cur_gap = 0
    for mn, sc in ops:
        if sc == "mfma":
            if cur == 0 and in_stream:
                gap_len.append(cur_gap)
            cur += 1
            in_stream = True
            cur_gap = 0
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
            if in_stream:
                gap_hist[sc] += 1
                gap_mnem[mn] += 1
                cur_gap += 1
    if cur > 0:
        runs.append(cur)
    return runs, gap_hist, gap_mnem, gap_len


def report(lab, ops):
    import statistics
    total = len(ops)
    mfma = sum(1 for _, sc in ops if sc == "mfma")
    runs, gh, gm, gl = analyze(ops)
    print(f"  loop {lab}: {total} instr, {mfma} MFMA")
    if runs:
        print(f"  MFMA contiguous run-length: mean={statistics.mean(runs):.1f} "
              f"max={max(runs)} runs={len(runs)}  dist={dict(Counter(runs).most_common(6))}")
    if gl:
        print(f"  inter-MFMA gap length (instr between MFMA bursts): mean={statistics.mean(gl):.1f} "
              f"max={max(gl)}")
    tot_gap = sum(gh.values()) or 1
    print(f"  what breaks the MFMA stream ({tot_gap} inter-MFMA instr):")
    for sc, n in gh.most_common():
        print(f"      {sc:16s} {n:5d}  ({100*n/tot_gap:4.1f}%)")
    print("  top interleaved mnemonics: "
          + ", ".join(f"{m}x{n}" for m, n in gm.most_common(8)))


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
    """Offline: sub-classing, run lengths and the gap census on a synthetic gfx950 loop."""
    import tempfile
    fails = []
    for mn, want in (("v_mfma_f32_32x32x16_bf16", "mfma"), ("v_wmma_f32_16x16x16_f16", "mfma"),
                     ("v_exp_f32_e32", "exp"), ("ds_read_b128", "lds"),
                     ("v_add_u32_e32", "valu_addr_int"), ("v_fma_f32", "valu_fp_math"),
                     ("v_pk_mul_f32", "valu_fp_pack"), ("v_cvt_pk_bf16_f32", "valu_cvt_mov"),
                     ("s_waitcnt", "waitcnt"), ("s_nop", "nop")):
        if sub_class(mn) != want:
            fails.append(f"sub_class({mn}) = {sub_class(mn)}, want {want}")
    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
        f.write(_SELFTEST_ASM)
        path = f.name
    lab, ops = hot_ops(path, None)
    runs, gh, _gm, gl = analyze(ops)
    if lab != ".LBB0_1":
        fails.append(f"hot loop {lab!r}")
    if runs != [2, 1]:
        fails.append(f"MFMA run lengths {runs}, want [2, 1]")
    if gl != [4] or gh.get("lds") != 1 or gh.get("exp") != 1:
        fails.append(f"gap census wrong: gaps={gl} hist={dict(gh)}")
    for msg in fails:
        print("FAIL", msg)
    print("[deep_mfma_analysis] SELFTEST " + ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


def main():
    if "--selftest" in sys.argv:
        return _selftest()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asm", nargs="?")
    ap.add_argument("--loop-label")
    ap.add_argument("--compare", nargs="+", help="path=LABEL[,loop-label]")
    a = ap.parse_args()
    if a.compare:
        for e in a.compare:
            p, _, rest = e.partition("=")
            label, _, ll = rest.partition(",")
            lab, ops = hot_ops(p, ll or None)
            print(f"\n=== {label or p} ===")
            report(lab, ops)
    else:
        if not a.asm:
            ap.error("provide <a.s> or --compare a.s=LABEL ...")
        lab, ops = hot_ops(a.asm, a.loop_label)
        print(f"\n=== {a.asm} ===")
        report(lab, ops)


if __name__ == "__main__":
    sys.exit(main())
