#!/usr/bin/env python3
"""Join the rocprof (rocprof-compute) leg onto the TraceLens rollup table.

    python3 join_legs.py --rocprof-dir <dir of *_roofline.txt> \
                         --tl-rows <TL rows.json> [--out <per_case.json>]

The rocprof column is folded EXACTLY the way the TL column is (TL's own
`rollup`): in the TIME domain, not by averaging percentages.

    roof_us(case)  = sum_k  t_k * Eff_k
    pct(kernel,regime) = sum_cases w * roof_us / sum_cases w * measured_us

Folding time is what makes the fold axis-safe. Each kernel's own `attainable`
(the min of its compute peak and AI*BW) is applied BEFORE the sum, so a seam
whose kernels sit on different roof axes never has two different peaks divided
by one common denominator -- which is the cross-axis fold that S3 3a forbids
and that already produced one unquotable number in this report
(mxfp4_fused_moe/prefill).

Weights t_k are TraceLens' per-kernel device ms from the same seam. rocprof's own
durations come from a serialized-dispatch profiling run and are not the deployment
time distribution; rocprof contributes the RATIO (achieved/attainable, from its
hardware counters), TL contributes where the time actually goes.

`--tl-rows` is a JSON file with a `rows` list; each row needs `kernel`, `regime`,
`case`, `measured_us`, `roofline_us`, optional `case_weight`, and a `per_kernel`
list of `{"kernel": <device symbol>, "ms": <device ms>}`. Each row is matched to
`<rocprof-dir>/<kernel>_<case>_roofline.txt`.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from fold import fold_cases, fold_kernels


def parse_geak(path):
    """-> {kernel_name: {"eff": float|None, "eff_ds": float|None, "ai": float, "perf": float}}"""
    out, cur = {}, None
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"^kernel: (.*)$", line)
        if m:
            cur = {"name": m.group(1).strip(), "eff": None, "eff_ds": None,
                   "ai": None, "perf": None, "peak_bw": None, "peak_c": None}
            out[cur["name"]] = cur
            continue
        if cur is None:
            continue
        if (m := re.search(r">> Roofline Eff\.\s*:\s*[\d.]+\s*\(([\d.]+) %\)", line)):
            cur["eff"] = float(m.group(1)) / 100.0
        elif (m := re.search(r">> Roofline Eff\.\(ds\):\s*[\d.]+\s*\(([\d.]+) %\)", line)):
            cur["eff_ds"] = float(m.group(1)) / 100.0
        elif (m := re.search(r"AI \(HBM\)\s*:\s*([\d.]+)", line)):
            cur["ai"] = float(m.group(1))
        elif (m := re.search(r"Perf \(achieved\)\s*:\s*([\d.]+)", line)):
            cur["perf"] = float(m.group(1))
        elif (m := re.search(r"Peak_BW_emp \(HBM\)\s*:\s*([\d.]+)", line)):
            cur["peak_bw"] = float(m.group(1))
        elif (m := re.search(r"Peak_Compute \[(\S+?)\]:\s*([\d.]+)", line)):
            cur["peak_c"] = (m.group(1), float(m.group(2)))
    return out


def match(tl_name, geak):
    """GEAK truncates long mangled names in its per-kernel headers, so an exact
    lookup silently misses the very kernels that matter (every templated ATen or
    rocprim symbol). Fall back to longest-common-prefix containment."""
    if tl_name in geak:
        return geak[tl_name]
    cands = [g for g in geak.values()
             if g["name"].startswith(tl_name[:60]) or tl_name.startswith(g["name"][:60])]
    if len(cands) == 1:
        return cands[0]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rocprof-dir", required=True, type=Path,
                    help="Directory holding the rocprof leg's <kernel>_<case>_roofline.txt files.")
    ap.add_argument("--tl-rows", required=True, type=Path,
                    help="TraceLens leg's rows.json (see module docstring for the schema).")
    ap.add_argument("--out", default=None, type=Path,
                    help="Per-case JSON destination (default: <rocprof-dir>/per_case_join.json).")
    ap.add_argument("--arm", default=None,
                    help="Arm tag in the rocprof filenames, i.e. <kernel>_<case>_<arm>_"
                         "roofline.txt. The two-implementation experiment of combination.md "
                         "section 6 puts both arms in one directory, so without this the "
                         "lookup finds nothing at all. The TL rows must be the matching arm.")
    ap.add_argument("--missing-emp", choices=("drop", "zero"), default="drop",
                    help="What an unscorable kernel does to the EMPIRICAL roof. 'drop' "
                         "(default) takes it out of numerator AND denominator and lets "
                         "the shortfall show up in the cov column -- what folding.md "
                         "prescribes and what the datasheet leg already does. 'zero' "
                         "leaves it in the denominator at 0%% efficiency, which deflates "
                         "the roof and inflates headroom by exactly (1-coverage); it "
                         "reproduces tables published before 2026-10-12.")
    args = ap.parse_args()

    OUT = args.rocprof_dir.resolve()
    if not OUT.is_dir():
        print(f"[join] --rocprof-dir does not exist: {OUT}", file=sys.stderr)
        return 1
    dest = (args.out or OUT / "per_case_join.json").resolve()

    d = json.loads(args.tl_rows.read_text())
    rows = d["rows"]
    per_case = {}
    misses = []
    absent = []
    for r in rows:
        tag = f"{r['kernel']}_{r['case']}"
        suffix = f"_{args.arm}" if args.arm else ""
        f = OUT / f"{tag}{suffix}_roofline.txt"
        if not f.exists():
            # Reported, never silent. A truncated rocprof leg (one lost case, one
            # timeout, a batch loop that ate its own stdin) otherwise yields a table
            # that looks complete because the missing rows are simply not in it.
            absent.append(f.name)
            continue
        if r.get("measured_us") is None:
            absent.append(f"{f.name} (TL row has no measurement)")
            continue
        geak = parse_geak(f)
        unmatched = [pk["kernel"][:50] for pk in r["per_kernel"]
                     if match(pk["kernel"], geak) is None]
        if unmatched:
            misses.append((tag, unmatched))

        def eff_of(pk, which):
            g = match(pk["kernel"], geak)
            return g[which] if g else None

        # Both legs go through fold.fold_kernels, so the within-case arithmetic is
        # the single copy in folding.md. They differ only in their `missing` policy,
        # and that difference is load-bearing rather than cosmetic:
        #
        # The datasheet leg is NOT just a different number, it is sometimes absent:
        # DATASHEET_PEAKS["gfx950"] carries only mfma_* entries, so a VALU kernel
        # (e.g. _sparse_mla_reduce_kernel -> valu_f32) has no datasheet peak at all.
        # Scoring it 0 would silently deflate the whole seam -> "drop".
        #
        # Both legs now drop an unscorable kernel from numerator AND denominator and
        # report the shortfall as coverage. Scoring it 0 instead (--missing-emp zero,
        # what this leg did before 2026-10-12) leaves it in the denominator, which
        # deflates the roof and inflates headroom by exactly 1-coverage. See
        # folding.md; the cov column is what makes the choice visible either way.
        roof_e, cov, no_eff = fold_kernels(
            r["per_kernel"], lambda pk: eff_of(pk, "eff"), r["measured_us"],
            missing=args.missing_emp)
        roof_d, cov_d, no_eff_ds = fold_kernels(
            r["per_kernel"], lambda pk: eff_of(pk, "eff_ds"), r["measured_us"],
            missing="drop")
        per_case[tag] = {
            "kernel": r["kernel"], "regime": r["regime"], "case": r["case"],
            "w": r.get("case_weight"),
            "measured_us": r["measured_us"],
            "tl_roof_us": r["roofline_us"],
            "rp_roof_us_emp": roof_e,
            "rp_roof_us_ds": roof_d,
            "coverage": cov, "coverage_ds": cov_d,
            "no_eff": no_eff, "no_eff_ds": no_eff_ds,
        }

    # Fold per (kernel, regime) through the shared module -- the SAME call the
    # tl-only leg makes from fold.py's CLI, which is what makes the three columns
    # below comparable to each other and to a single-engine table.
    #
    # All three columns fold over `per_case`, i.e. over THE SAME case set the rocprof
    # leg covered. The TL rollup in rows.json spans every shape, including the ones
    # rocprof never profiled; re-folding it here is what puts the two columns over a
    # common denominator, and is why a join table's TL% is NOT the standalone
    # tl-only TL%. (folding.md, "Comparing two tables".)
    cases = list(per_case.values())
    agg_tl = fold_cases(cases, roof_key="tl_roof_us")
    agg_e = fold_cases(cases, roof_key="rp_roof_us_emp")
    agg_d = fold_cases(cases, roof_key="rp_roof_us_ds", cov_key="coverage_ds")

    print(f"{'kernel/regime':<32}{'ax':>8} | {'TL%':>7}{'TLsp':>7} | "
          f"{'RPemp%':>9}{'sp':>6} | {'RPds%':>8}{'sp':>6} | {'n':>3}{'cov':>6}{'covds':>6}")
    for k in sorted(agg_tl, key=lambda x: (x[0], x[1])):
        a = agg_tl[k]
        pe = (agg_e[k]["pct"] or 0.0) if k in agg_e else 0.0
        pd = (agg_d[k]["pct"] or 0.0) if k in agg_d else 0.0
        cov = agg_e[k]["coverage"] if k in agg_e and agg_e[k]["coverage"] is not None else 0.0
        cov_d = agg_d[k]["coverage"] if k in agg_d and agg_d[k]["coverage"] is not None else 0.0
        tlp = a["pct"] or 0.0
        tls = 1 / tlp if tlp else 0.0
        # S3 6 L3: roofline_pct outside (0,1] is not a result, it is a disproof of
        # the denominator. Clamp for display, mark suspect, and publish NO
        # speedup -- a clamped 100% must never read as "saturated". Every hit
        # here is a bf16 MFMA GEMM against rocprof's empirical mfma_bf16 peak of
        # 614663 GFLOP/s, which is half its own mfma_f16 peak (1227048) and so
        # already fails S3 8.2's bf16==fp16 rule; a real 8192^3 torch.mm on this
        # same GPU read ~1450000 GFLOP/s. Use the datasheet column for bf16.
        def cell(p, w=8):
            if p <= 0:
                return f"{'--':>{w}}{'':>6}"
            if p > 1.0:
                return f"{min(p, 1.0):>{w - 1}.0%}!{'suspect':>6}"
            return f"{p:>{w}.1%}{1 / p:>5.2f}x"

        print(f"{k[0] + '/' + k[1]:<32}{'':>8} | {tlp:>6.1%}{tls:>6.2f}x | "
              f"{cell(pe)} | {cell(pd, 7)} | {a['n']:>3}{cov:>6.0%}{cov_d:>6.0%}")
    if absent:
        print(f"\n[{len(absent)} of {len(rows)} TL rows had no rocprof file -- this table "
              f"covers {len(per_case)} case(s), NOT the whole leg]")
        for name in absent[:20]:
            print(f"  missing {name}")
    if misses:
        print("\n[unmatched TL kernels -> no rocprof block]")
        for tag, u in misses[:20]:
            print(f"  {tag}: {u}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(per_case, open(dest, "w"), indent=1)
    print(f"\nper-case -> {dest}  ({len(per_case)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
