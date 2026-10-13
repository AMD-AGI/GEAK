#!/usr/bin/env python3
"""Re-derive Roofline Eff. from an ALREADY-COLLECTED profile, against a new empirical-peak CSV.

The rocprof-compute 3.6.0 fix ("Fixed roofline benchmark MFMA FP16/BF16/INT8 peaks for MI 350")
changes only the MICROBENCHMARK -- the ceilings. The counters are untouched: `analyze -b 4`
prints the achieved rate and the empirical peak as two separate columns of the same table, and
only the second column moves. So every one of the 40 cases can be rescored offline from the
saved `*_roofline_raw.txt`, without putting the GPU back under rocprof. Re-profiling would in
fact be WORSE than useless here: it would re-run 40 workloads on a machine whose neighbours
have changed since, mixing a denominator fix with a fresh draw of measurement noise, and the
two would be inseparable in the diff.

Two scopes, because the new CSV moves two independent things and they deserve separate blame:

  --scope compute   Substitute only the compute ceilings (MFMA/VALU/IOPs). This ISOLATES the
                    3.6.0 fix. gfx950 bf16 goes 614 663 -> 2 435 517 GFLOP/s (3.96x), fp16
                    1 227 048 -> 2 162 758 (1.76x), int8 1 222 436 -> 4 854 023 (3.97x); every
                    dtype the fix did not touch (F8/F4/F6/F32/F64) moves by <1%, which is the
                    control that says this is a fix and not a re-benchmark drifting.

  --scope all       Also substitute the memory ceilings. HBMBw reads 6246 -> 7080 GB/s (1.13x).
                    That one is NOT covered by any changelog entry, and rocprof's HBM bench is
                    the same one already documented as drifting 5326 -> 6181 -> 6246 on this
                    GPU, so 1.13x is inside its own known scatter and must not be sold as a fix.
                    It is offered separately so a memory-bound row's move can be attributed.

Kernel names come from the previously written `*_roofline.txt` rather than from the raw text,
because `skip_filtered_rows` un-truncates them against `pmc_kernel_top.csv` in the profile
directory -- which run_roofline deletes in its `finally`. Taking the names from our own prior
output keeps the join keys byte-identical across the two scorings; the
alternative would silently rename every long `Cijk_...` GEMM and look like missing kernels.
"""
import argparse
import csv
import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RR = HERE / "run_roofline.py"

# rocprof's 4.1 table row label -> column name in roofline.csv. Verified row by row against
# the old CSV: every `Peak (Empirical)` cell in a saved raw file equals the mapped column.
PEAK_COLS = {
    "VALU FLOPs (F16)": "FP16Flops",
    "VALU FLOPs (F32)": "FP32Flops",
    "VALU FLOPs (F64)": "FP64Flops",
    "MFMA FLOPs (F64)": "MFMAF64Flops",
    "MFMA FLOPs (F32)": "MFMAF32Flops",
    "MFMA FLOPs (F16)": "MFMAF16Flops",
    "MFMA FLOPs (BF16)": "MFMABF16Flops",
    "MFMA FLOPs (F8)": "MFMAF8Flops",
    "MFMA FLOPs (F6F4)": "MFMAF6F4Flops",
    "MFMA IOPs (Int8)": "MFMAI8Ops",
}
MEM_COLS = {
    "HBM Bandwidth": "HBMBw",
    "L2 Cache Bandwidth": "L2Bw",
    "L1 Cache Bandwidth": "L1Bw",
    "LDS Bandwidth": "LDSBw",
}

_NAME_RE = re.compile(r"^kernel: (.*)$")


def load_rr():
    spec = importlib.util.spec_from_file_location("run_roofline", RR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_roofline"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_peaks(csv_path: Path) -> dict:
    row = next(iter(csv.DictReader(open(csv_path))))
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--peaks-csv", required=True)
    ap.add_argument("--scope", choices=["compute", "all"], default="compute")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--hbm-peak-const", type=float, default=None,
                    help="MUST match the value the leg was originally rendered with, or the "
                         "re-score moves two variables at once.")
    a = ap.parse_args()

    if Path(a.in_dir).resolve() == Path(a.out_dir).resolve():
        print("[repeak] --out-dir must differ from --in-dir", file=sys.stderr)
        return 1

    rr = load_rr()
    newp = load_peaks(Path(a.peaks_csv))
    cols = dict(PEAK_COLS)
    if a.scope == "all":
        cols.update(MEM_COLS)

    in_dir, out_dir = Path(a.in_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_files = n_sub = n_miss = 0
    for raw in sorted(in_dir.glob("*_roofline_raw.txt")):
        stem = raw.name[: -len("_roofline_raw.txt")]
        old_txt = in_dir / f"{stem}_roofline.txt"
        if not old_txt.is_file():
            print(f"[skip] {stem}: no prior _roofline.txt to take names from")
            continue

        content = raw.read_text()
        blocks = rr.parse_kernel_blocks(content)
        names = [m.group(1) for m in (_NAME_RE.match(l) for l in
                                      old_txt.read_text().split("\n")) if m]
        if len(names) != len(blocks):
            print(f"[skip] {stem}: {len(names)} names vs {len(blocks)} parsed kernels")
            continue

        rates_list, ai_list = [], []
        for _, rates, ai in blocks:
            new_rates = {}
            for metric, (val, peak, unit) in rates.items():
                col = cols.get(metric.strip())
                if col is None:
                    new_rates[metric] = (val, peak, unit)      # not in scope: keep
                elif col in newp:
                    new_rates[metric] = (val, newp[col], unit)
                    n_sub += 1
                else:
                    # New CSV lacks the column (e.g. the VALU BF16/FP8 rows upstream removed
                    # as erroneously added). Leaving the old peak would silently mix two
                    # microbenches in one table, so drop the ceiling to N/A instead.
                    new_rates[metric] = (val, None, unit)
                    n_miss += 1
            rates_list.append(new_rates)
            ai_list.append(ai)

        report = (rr.roofline_summary(names, rates_list, ai_list) + "\n"
                  + rr.compute_roofline_efficiency(names, rates_list, ai_list, "auto",
                                                   a.hbm_peak_const, None, "both", a.arch))
        (out_dir / f"{stem}_roofline.txt").write_text(report)
        n_files += 1

    print(f"[repeak] scope={a.scope}  files={n_files}  peaks substituted={n_sub}  "
          f"dropped-to-N/A={n_miss}  -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
