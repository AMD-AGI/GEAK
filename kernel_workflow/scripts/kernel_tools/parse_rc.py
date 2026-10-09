#!/usr/bin/env python3
"""Parse a `rocprof-compute analyze` text report into JSON.

Extracted from round_report.py so the OPTIONAL rocprof-compute evidence path does not drag the
per-round ledger/figure machinery in with it. `rocprof_compute_probe.sh` is the only caller;
round_report.py re-exports these names so the DSLs that still run the gated pipeline are
unaffected.

    python3 parse_rc.py <analyze.txt> [--out-json rc_metrics.json]
    python3 parse_rc.py --selftest

GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/); the Gluon pack's scripts/parse_rc.py is
a shim. Interpretation follows GEAK's profiling_guide.md: warp-state buckets are read as ratios over
Wave Cycles (dependency wait -> C1 latency, NOT memory-bound by itself; issue wait -> C2 occupancy).

Two rules the callers depend on:

  * A zero-row parse is NOT written. Every field would be null, and a metrics file full of nulls
    is indistinguishable downstream from a kernel that genuinely reads zero -- it would let a
    crashed collection satisfy the preflight EVIDENCE gate that exists to catch it.
  * If the named file yields nothing, the `rc_analyze*` siblings beside it are tried. A by-hand
    recovery of a failed analyze lands on a new name; reading only the canonical one is how a
    campaign ended up with the SOL text on disk and every machine reader still seeing null.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

# ------------------------------------------------------------------- rocprof-compute parse
# rocprof-compute analyze tables use Unicode box-drawing '│' (U+2502) as the delimiter.
_B = r"[|\u2502]"
_RC_ROW = re.compile(
    _B + r"\s*(\d+\.\d+(?:\.\d+)?)\s*" + _B + r"\s*([^|\u2502]+?)\s*" + _B
    + r"\s*([^|\u2502]*?)\s*" + _B + r"\s*([^|\u2502]*?)\s*" + _B
    + r"\s*([^|\u2502]*?)\s*" + _B + r"\s*([^|\u2502]*?)\s*" + _B)
# 4-column SOL-style rows (id | name | value | unit) used by §16.1/§17.1/§12.1 (Coalescing etc.)
_RC_ROW4 = re.compile(
    _B + r"\s*(\d+\.\d+(?:\.\d+)?)\s*" + _B + r"\s*([^|\u2502]+?)\s*" + _B
    + r"\s*([^|\u2502]*?)\s*" + _B + r"\s*([^|\u2502]*?)\s*" + _B + r"\s*$")


def parse_rc_analyze(path: str) -> dict:
    """Parse `rocprof-compute analyze --block 2 3 6` text -> structured SOL/memory/LDS/SPI."""
    rows = {}
    by_id = {}

    def f(x):
        try:
            return float(x.replace(",", ""))
        except ValueError:
            return None
    for ln in open(path, errors="ignore"):
        m = _RC_ROW.search(ln)
        if m:
            mid, name, avg, unit, peak, pct = (x.strip() for x in m.groups())
            rows[name] = {"id": mid, "avg": f(avg), "unit": unit, "peak": f(peak), "pct": f(pct)}
            by_id[mid] = rows[name]
            continue
        m4 = _RC_ROW4.search(ln)
        if m4:
            mid, name, val, unit = (x.strip() for x in m4.groups())
            # store the single value under avg AND pct so name-based accessors find it
            rows.setdefault(name, {"id": mid, "avg": f(val), "unit": unit,
                                   "peak": None, "pct": f(val)})
            by_id.setdefault(mid, rows[name])

    def pct(name):
        return rows.get(name, {}).get("pct")
    def avg(name):
        return rows.get(name, {}).get("avg")
    def avg_id(mid):
        return by_id.get(mid, {}).get("avg")
    # unified mfma_pct = the active dtype's %-of-peak (a kernel uses ONE MFMA dtype; the others
    # read 0.0, so take the max of the non-zero ones - fixes bf16-vs-f16 mis-selection).
    _mfma_dtypes = {
        "bf16": pct("MFMA FLOPs (BF16)"), "f16": pct("MFMA FLOPs (F16)"),
        "f8": pct("MFMA FLOPs (F8)"), "f32": pct("MFMA FLOPs (F32)"),
        "f64": pct("MFMA FLOPs (F64)"), "int8": pct("MFMA IOPs (Int8)"),
    }
    _mfma_nonzero = [v for v in _mfma_dtypes.values() if v]
    sol = {
        "mfma_pct": max(_mfma_nonzero) if _mfma_nonzero else None,
        "mfma_by_dtype_pct": {k: v for k, v in _mfma_dtypes.items() if v},
        "mfma_bf16_pct": _mfma_dtypes["bf16"],
        "mfma_f16_pct": _mfma_dtypes["f16"],
        "valu_flops_pct": pct("VALU FLOPs"),
        "valu_iops_pct": pct("VALU IOPs"),
        "ipc": avg("IPC"),
        "valu_active_threads_pct": pct("VALU Active Threads"),
    }
    memory = {
        "vL1D_hit_pct": avg("vL1D Cache Hit Rate") or avg("Hit rate") or avg_id("16.1.0"),
        "L2_hit_pct": avg("L2 Cache Hit Rate") or avg_id("17.1.2"),
        "sL1D_hit_pct": avg("sL1D Cache Hit Rate"),
        "vL1D_bw_pct": pct("vL1D Cache BW") or avg_id("16.1.1"),
        "L2_bw_pct": pct("L2 Cache BW") or avg_id("17.1.1"),
        "coalescing_pct": avg("Coalescing") or avg_id("16.1.3"),
        "stalled_on_l2_data_pct": avg("Stalled on L2 Data") or avg_id("16.2.0"),
        "l2fabric_rd_bw": avg("L2-Fabric Read BW"),
        "l2fabric_wr_bw": avg("L2-Fabric Write BW") or avg("L2-Fabric Write and Atomic BW"),
        "l2fabric_rd_latency_cyc": avg("L2-Fabric Read Latency") or avg("Read Latency") or avg_id("17.2.9"),
        "l2fabric_wr_latency_cyc": avg("L2-Fabric Write Latency")
            or avg("Write and Atomic Latency") or avg_id("17.2.10"),
    }
    # warp-state aggregate buckets (block 7.2 Wavefront Runtime Stats) - cycle counts
    warp_state = {
        "dependency_wait_cyc": avg("Dependency Wait Cycles") or avg_id("7.2.4"),
        "issue_wait_cyc": avg("Issue Wait Cycles") or avg_id("7.2.5"),
        "active_cyc": avg("Active Cycles") or avg_id("7.2.6"),
        "wave_cyc": avg("Wave Cycles") or avg_id("7.2.3"),
    }
    # instruction mix (block 10.2 VALU arith / 10.3 spill-stack / 10.4 MFMA by dtype)
    instr_mix = {
        "int32": avg("INT32") or avg_id("10.2.0"),
        "int64": avg("INT64") or avg_id("10.2.1"),
        "f16_fma": avg("F16-FMA"), "f32_add": avg("F32-ADD"), "f32_mul": avg("F32-MUL"),
        "f32_fma": avg("F32-FMA"), "f16_trans": avg("F16-Trans"), "f32_trans": avg("F32-Trans"),
        "conversion": avg("Conversion"),
        "spill_stack": avg("Spill/Stack Instr"),
        "mfma_i8": avg("MFMA-I8"), "mfma_f8": avg("MFMA-F8"), "mfma_f16": avg("MFMA-F16"),
        "mfma_bf16": avg("MFMA-BF16"), "mfma_f32": avg("MFMA-F32"),
    }
    lds = {
        "theoretical_bw_pct": pct("Theoretical LDS Bandwidth"),
        "bank_conflict_per_access": avg("LDS Bank Conflicts/Access"),
    }
    # SPI / Workgroup Manager occupancy limiters (block 6); often empty = not resource-capped
    spi = {name: v.get("avg") for name, v in rows.items() if "Insufficient" in name}
    occ = {"wavefront_occupancy": avg("Wavefront Occupancy"),
           "dispatched_workgroups": avg("Dispatched Workgroups")}
    bubble = (100.0 - sol["mfma_pct"]) if sol["mfma_pct"] is not None else None
    return {"sol": sol, "memory": memory, "lds": lds, "spi_insufficient": spi,
            "occupancy": occ, "warp_state": warp_state, "instr_mix": instr_mix,
            "bubble_mfma_idle_pct": bubble, "_n_rows": len(rows)}


def _candidates(src: str) -> list[str]:
    """`src` first, then its `rc_analyze*` siblings, newest last-modified first."""
    sibs = sorted((p for p in glob.glob(os.path.join(os.path.dirname(src) or ".", "rc_analyze*"))
                   if os.path.isfile(p) and os.path.abspath(p) != os.path.abspath(src)),
                  key=os.path.getmtime, reverse=True)
    return [src] + sibs


def _diagnose(path: str) -> str:
    try:
        with open(path, errors="ignore") as f:
            txt = f.read()
    except OSError as e:
        return f"unreadable ({e})"
    if not txt.strip():
        return "empty"
    if "PermissionError" in txt or "Permission denied" in txt:
        return "holds a PermissionError, not a report -- analyze never ran to completion"
    if "Traceback (most recent call last)" in txt:
        return "holds a python traceback, not a report"
    return "no parseable rocprof-compute table rows"


def _selftest() -> int:
    import tempfile
    fails = []
    table = "\n".join([
        "2. System Speed-of-Light",
        "│ 2.1.0  │ VALU FLOPs          │ 120.0 │ GFLOP │ 1000.0 │ 12.0 │",
        "│ 2.1.6  │ MFMA FLOPs (BF16)   │ 900.0 │ GFLOP │ 1000.0 │ 90.0 │",
        "│ 2.1.7  │ MFMA FLOPs (F16)    │ 0.0   │ GFLOP │ 1000.0 │ 0.0  │",
        "│ 7.2.3  │ Wave Cycles         │ 1000  │ cyc   │        │      │",
        "│ 7.2.4  │ Dependency Wait Cycles │ 450 │ cyc   │        │      │",
        "│ 7.2.5  │ Issue Wait Cycles   │ 150   │ cyc   │        │      │",
        "│ 16.1.3 │ Coalescing          │ 87.5  │ pct   │",
    ])
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "rc_analyze.txt")
        with open(src, "w") as fh:
            fh.write(table + "\n")
        rc = parse_rc_analyze(src)
        if rc["sol"]["mfma_pct"] != 90.0 or rc["sol"]["mfma_by_dtype_pct"] != {"bf16": 90.0}:
            fails.append(f"mfma_pct must be the active dtype's (bf16=90): {rc['sol']}")
        if rc["warp_state"]["dependency_wait_cyc"] != 450 or rc["warp_state"]["wave_cyc"] != 1000:
            fails.append(f"warp-state buckets misparsed: {rc['warp_state']}")
        if rc["memory"]["coalescing_pct"] != 87.5:
            fails.append("4-column row (Coalescing) misparsed")
        if rc["bubble_mfma_idle_pct"] != 10.0:
            fails.append("bubble = 100 - mfma_pct")
        # a crashed analyze next to a recovered sibling: the sibling is found, the crash diagnosed
        crash = os.path.join(td, "rc_analyze_crash.txt")
        with open(crash, "w") as fh:
            fh.write("Traceback (most recent call last):\nPermissionError: [Errno 13]\n")
        if parse_rc_analyze(crash)["_n_rows"] != 0 or "PermissionError" not in _diagnose(crash):
            fails.append("a crashed analyze must parse to zero rows and be diagnosed")
        if src not in _candidates(crash):
            fails.append("rc_analyze* siblings must be candidates")
    for f in fails:
        print("FAIL", f)
    print("[parse_rc] SELFTEST " + ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


def main() -> None:
    if "--selftest" in sys.argv[1:]:
        sys.exit(_selftest())
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    src = sys.argv[1]
    out = sys.argv[sys.argv.index("--out-json") + 1] if "--out-json" in sys.argv else src + ".json"
    tried = []
    for cand in _candidates(src):
        rc = parse_rc_analyze(cand)
        if rc.get("_n_rows"):
            rc["_parsed_from"] = cand
            with open(out, "w") as f:
                json.dump(rc, f, indent=2)
            note = "" if os.path.samefile(cand, src) else f"  [recovered from a sibling of {src}]"
            print(f"wrote {out}  (rows parsed: {rc['_n_rows']}, source: {cand}){note}")
            return
        tried.append(f"{os.path.basename(cand)}: {_diagnose(cand)}")
    # Refusing to write is the point: an all-null metrics file reads downstream as a collected SOL
    # layer, which is the exact confusion the preflight EVIDENCE gate exists to catch.
    print(f"NO metrics written: nothing under {os.path.dirname(src) or '.'} parsed into rows.",
          file=sys.stderr)
    for t in tried:
        print(f"  - {t}", file=sys.stderr)
    sys.exit(4)


if __name__ == "__main__":
    main()
