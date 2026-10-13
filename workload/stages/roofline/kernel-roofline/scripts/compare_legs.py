#!/usr/bin/env python3
"""Side-by-side TraceLens vs rocprof (rocprof-compute): arithmetic intensity, bound type, roofline %.

The two tools can only disagree in three places -- the numerator (FLOPs), the
denominator (bytes), and the peaks -- and AI plus bound type is where those
disagreements become visible *before* they turn into a headroom number. A row
where the two agree on % but sit on different axes is not agreement, it is two
errors cancelling.

Aggregation must match the TL leg's own roll-up exactly (see the docstring in
join_legs.py):

  * weights `w` are each task's own `share_of_op_calls`, normalised WITHIN a
    regime, so decode and prefill are never blended;
  * AI is folded as sum(w*FLOPs) / sum(w*bytes), NOT as a mean of per-shape AIs
    -- AI is a ratio, and averaging ratios weights the small shapes as heavily
    as the ones that move the time;
  * bound type is reported as the share of weighted DEVICE TIME sitting on the
    compute axis, because a seam is not one kernel and "compute-bound" for a
    seam is a mixture, not a label.

rocprof carries no byte column directly, so its bytes are reconstructed per kernel
as (Perf_achieved * t) / AI -- i.e. from its own two counters -- with t from
TraceLens for the reason given in join_legs.py.

    python3 compare_legs.py --rocprof-dir <dir> --tl-rows <rows.json>
"""
import argparse
import json
import re
import sys
from pathlib import Path

from fold import fold_ratio
from join_legs import match, parse_geak

_BOUND_RE = re.compile(r"ridge\(emp\)\s*:\s*([\d.]+) FLOP/byte\s*\((\w+)-bound\)")


def parse_bounds(path):
    """-> {kernel name: (ridge_emp, 'compute'|'memory')}, keyed as parse_geak."""
    out, cur = {}, None
    for line in path.read_text(errors="replace").splitlines():
        if (m := re.match(r"^kernel: (.*)$", line)):
            cur = m.group(1).strip()
        elif cur and (m := _BOUND_RE.search(line)):
            out[cur] = (float(m.group(1)), m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rocprof-dir", required=True, type=Path,
                    help="Directory holding the rocprof leg's <kernel>_<case>_roofline.txt files.")
    ap.add_argument("--tl-rows", required=True, type=Path,
                    help="TraceLens leg's rows.json (same schema as join_legs.py, plus "
                         "flops_g / bytes_mb / bound / peak_tflops / peak_bw_gbps / peak_spec).")
    ap.add_argument("--arm", default=None,
                    help="Arm tag in the rocprof filenames: <kernel>_<case>_<arm>_roofline.txt. "
                         "See join_legs.py --arm.")
    args = ap.parse_args()

    OUT = args.rocprof_dir.resolve()
    if not OUT.is_dir():
        print(f"[compare] --rocprof-dir does not exist: {OUT}", file=sys.stderr)
        return 1

    d = json.loads(args.tl_rows.read_text())
    cases = []
    extra = {}
    absent = []
    suffix = f"_{args.arm}" if args.arm else ""
    for r in d["rows"]:
        f = OUT / f"{r['kernel']}_{r['case']}{suffix}_roofline.txt"
        if not f.exists():
            absent.append(f.name)        # reported below, never silent
            continue
        geak, bounds = parse_geak(f), parse_bounds(f)
        c = {"kernel": r["kernel"], "regime": r["regime"],
             "w": r.get("case_weight"),
             "tl_f": r["flops_g"] * 1e9, "tl_b": r["bytes_mb"] * 1e6,
             "tl_cmp": 1.0 if r["bound"] == "compute" else 0.0, "one": 1.0,
             "g_f": 0.0, "g_b": 0.0, "g_cmp": 0.0, "g_t": 0.0}
        e = extra.setdefault((r["kernel"], r["regime"]),
                             {"ridge_tl": set(), "ridge_g": [], "spec": r["peak_spec"]})
        # TL's ridge is a property of the (peak, mem_bw) pair it roofed against,
        # not of the shape, so it is identical across a kernel/regime's cases.
        e["ridge_tl"].add(round(r["peak_tflops"] * 1e3 / r["peak_bw_gbps"], 1))
        for pk in r["per_kernel"]:
            g = match(pk["kernel"], geak)
            if g is None or g["ai"] in (None, 0) or g["perf"] is None:
                continue
            t_s = pk["ms"] * 1e-3                      # TL device seconds per call
            flops = g["perf"] * 1e9 * t_s
            c["g_f"] += flops
            c["g_b"] += flops / g["ai"]
            c["g_t"] += pk["ms"]
            b = bounds.get(g["name"])
            if b:
                e["ridge_g"].append((pk["ms"], b[0]))
                if b[1] == "compute":
                    c["g_cmp"] += pk["ms"]
        cases.append(c)

    # Every column below is a RATIO folded by the shared module: numerator and
    # denominator summed separately over the weighted cases, divided once. Same
    # arithmetic as join_legs.py and as a single-engine rollup -- see folding.md.
    tl_ai = fold_ratio(cases, "tl_f", "tl_b")
    g_ai = fold_ratio(cases, "g_f", "g_b")
    tl_cmp = fold_ratio(cases, "tl_cmp", "one")   # share of weighted cases
    g_cmp = fold_ratio(cases, "g_cmp", "g_t")     # share of weighted device time
    agg = {k: {**extra[k], "n": tl_ai[k]["n"]} for k in tl_ai}

    print(f"[{OUT.name}] AI and bound type, folded as sum(w*FLOPs)/sum(w*bytes); "
          f"'cmp%' = share of weighted device time on the compute axis\n")
    print(f"{'kernel/regime':<28}{'peak_spec':>20} | {'TL AI':>8}{'ridge':>7}{'cmp%':>6} | "
          f"{'RP AI':>8}{'ridge':>7}{'cmp%':>6} | {'AI ratio':>9}{'n':>4}")
    for k in sorted(agg):
        a = agg[k]
        ai_t = tl_ai[k]["ratio"]
        ai_g = g_ai[k]["ratio"] if g_ai[k]["ratio"] is not None else float("nan")
        # Device-time-weighted ridge: the ridge is a property of the peak pair,
        # so a seam mixing dtypes has no single ridge -- weight it like the time.
        tw = sum(t for t, _ in a["ridge_g"])
        ridge = sum(t * r for t, r in a["ridge_g"]) / tw if tw else float("nan")
        rtl = "/".join(f"{x:g}" for x in sorted(a["ridge_tl"]))
        print(f"{k[0] + '/' + k[1]:<28}{a['spec']:>20} | {ai_t:>8.1f}{rtl:>7}"
              f"{100 * tl_cmp[k]['ratio']:>5.0f}% | {ai_g:>8.1f}{ridge:>7.0f}"
              f"{100 * (g_cmp[k]['ratio'] or 0.0):>5.0f}% | "
              f"{ai_t / ai_g:>8.2f}x{a['n']:>4}")
    if absent:
        covered = sum(a["n"] for a in agg.values())
        print(f"\n[{len(absent)} of {len(d['rows'])} TL rows had no rocprof file -- this "
              f"table covers {covered} case(s), NOT the whole leg]")
        for name in absent[:20]:
            print(f"  missing {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
