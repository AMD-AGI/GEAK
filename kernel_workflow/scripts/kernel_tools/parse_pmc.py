#!/usr/bin/env python3
"""Parse rocprofv3 CSV counter_collection across pmc passes, filter to one kernel
substring, average each counter over its dispatches, and expose derived per-unit busy
metrics + the occupancy meta (VGPR/AGPR/SGPR/LDS/Grid).

The DYNAMIC half of the kernel breakdown (the STATIC half is asm_loop_audit.py). Import
`parse_pmc(dir, kfilter)` from kernel_breakdown.py, or run standalone:

    python3 parse_pmc.py <rocprofv3_out_dir> [kernel_substr]

    python3 parse_pmc.py <dir> [kernel_substr] [--arch gfx950]
    python3 parse_pmc.py --print-groups        # the counter groups (one "name: C1 C2" per line)
    python3 parse_pmc.py --top-kernel <trace_dir>   # dominant non-helper kernel of a kernel trace
    python3 parse_pmc.py --selftest

GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/); the Gluon pack's
scripts/parse_pmc.py is a shim. PMC_GROUPS below is the SINGLE SOURCE of the counter set:
GEAK's kernel_workflow/scripts/profile_kernel.sh --pmc collects exactly these groups (one rocprofv3
pass per group, through rocprofv3_safe.sh) into <out>/pmc/pmc_<group>/, which this parser reads.
Handles that layout (pmc_*/), the older derived/ / counters/ layout, and rocprof-compute pmc_perf.csv.
Stdlib only, no GPU. rocprof-compute derived-metric CSVs can also be fed by pointing at a
dir containing *counter_collection.csv or a rocprof-compute analyze CSV export.

Interpretation (GEAK kernel_workflow/knowledge/profiling_guide.md): read the BUSY counters
VALUBusy / MfmaUtil for throughput; VALUUtilization is lane occupancy during VALU-active cycles
(a duty-cycle), not throughput, and must not be read as "VALU-bound".
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict

# --------------------------------------------------------------------------------------------
# VERIFIED COLLECTION GROUPS (gfx942, ROCm 7.1.0, live collection; list what THIS build exposes
# with `rocprofv3 -L` or `rocprofv3-avail list --pmc` before trusting a name on another arch/ROCm).
#
# rocprofv3 ALONE reaches the whole memory block and most of the SOL block -- rocprof-compute is
# not required for either. What IS required is GROUPING: a derived metric expands into several
# raw hardware counters, so too many in one --pmc list exceeds the per-pass counter slots. What
# happens then is VERSION-DEPENDENT: some rocprofv3 releases REPLAY the app over several passes,
# some ABORT it (SIGABRT, with a traceback that points at the workload and reads like a workload
# bug), and some HANG while holding the GPU lock. rocprofv3_safe.sh's timeout turns the hang into
# exit 124 and the abort shows as rc 134 / a signal; both are a DEGRADED layer, never zeros.
#
#   observed range: 4 derived OK on some boxes | 4 derived SIGABRT on others (the counter-slot
#   budget is BOX- and VERSION-dependent, not a fixed 4). So <=4 is a STARTING budget, not a
#   guarantee -- the collector must BISECT the counter list on an abort/timeout and drop only the
#   counters that individually fail, instead of losing the whole metric (collect_pmc_group below).
PMC_GROUPS = {
    # C1 -- the one bound that cannot be inferred from the ISA
    "memory":   ["FETCH_SIZE", "WRITE_SIZE", "TCC_HIT_sum", "TCC_MISS_sum"],
    # C1 cross-check: an INDEPENDENT route to the same DRAM byte count (TCC->EA requests). Its own
    # pass because it does not fit the memory group's counter budget. Two routes that agree inside
    # 1% is what makes an achieved bandwidth citable; a single route can be silently mis-scaled.
    "memory_ea": ["TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"],
    # SOL-equivalent issue-side rates (redundant with ATT + static when those are available)
    "sol":      ["VALUBusy", "MfmaUtil", "OccupancyPercent"],
    "stall":    ["MemUnitStalled", "LDSBankConflict"],
    # metrics.warp_state.wait_over_busy = SQ_WAIT_ANY / SQ_BUSY_CYCLES -- the primary stall
    # discriminator the reference tables advertise for the ATT-down path.
    "waitbusy": ["SQ_WAIT_ANY", "SQ_BUSY_CYCLES"],
    "lds_raw":  ["SQ_LDS_BANK_CONFLICT", "SQ_LDS_IDX_ACTIVE"],
}
# ABSENT on gfx942 despite appearing in older docs -- never emit these:
#   TCC_EA_RDREQ_sum (it is TCC_EA0_*), MemUnitBusy, L2CacheHit
# NOT A SYNONYM: TCC_EA0_RDREQ_DRAM_sum (DRAM-destined read requests only) is a different counter
# from TCC_EA0_RDREQ_sum (all L2->EA read requests). The pack's former profile_kernel.sh collected
# the _DRAM_ variant while this parser reads TCC_EA0_RDREQ_sum / TCC_EA0_WRREQ_sum, so its numbers
# were never used. GEAK's profile_kernel.sh --pmc now collects memory_ea exactly as named here.
#
# gfx950 CAVEAT (CDNA4, the main line): FETCH_SIZE and the TCC_BUBBLE-derived read bytes UNDER-count
# on gfx950 (rocprof-compute reports the same), so the fetch_write route reads LOW there; prefer the
# tcc_miss / tcc_ea0 routes and carry the range. `--arch gfx950` stamps the caveat into the output.
# --------------------------------------------------------------------------------------------

# WHERE THESE COUNTERS SIT, which bounds what they can possibly tell you: TCC *is* the L2, and the
# L2 is the gateway from an XCD onto the Infinity Fabric. Every counter here -- TCC_MISS, FETCH_SIZE,
# WRITE_SIZE, TCC_EA0_* -- is therefore measured at the L2/fabric boundary. Behind that boundary sits
# a MEMORY-SIDE last-level cache (Infinity Cache, an order of magnitude larger than the L2), and only
# behind THAT is DRAM. So these counters measure FABRIC traffic, and whether a byte was served by the
# memory-side cache or by HBM is invisible to all of them. Consequences, both load-bearing:
#   * a "DRAM bandwidth" derived here is an upper bound on DRAM traffic and an exact fabric traffic;
#     the two coincide only once the footprint clears the memory-side cache (sku.json mall_mb).
#   * comparing it against an HBM ceiling is a category error below that footprint -- it reads as a
#     high fraction of the HBM ceiling while nothing went to HBM at all.
#
# Byte scaling for the C1 conversion. BOTH are box/arch facts, not universal constants:
#   * FETCH_SIZE / WRITE_SIZE are documented in KILOBYTES (1024 B) -- verify on the target ROCm
#     version, because a KB-vs-B slip is a 1024x error and a 1000-vs-1024 slip is a silent 2.4%.
#   * a TCC (L2) miss moves one 128 B line on CDNA. Verify per arch before trusting the number.
# The two routes exist to CATCH exactly these slips: they are independent, so a disagreement is
# reported rather than resolved by preference.
FETCH_SIZE_UNIT_BYTES = 1024
TCC_LINE_BYTES = 128
BW_CROSSCHECK_TOL = 0.01        # two routes agreeing inside 1% is the bar for citing a bandwidth


def collect_pmc_group(counters, run_pass, is_abort=None):
    """Collect a PMC group, BISECTING the counter list on an abort so one over-budget counter does
    not lose the whole metric.

    `run_pass(list_of_counters) -> rc` runs ONE rocprofv3 --pmc pass (the caller owns the actual
    command / locus; rocprofv3_safe.sh is the intended runner) and returns its exit status. A pass
    "aborted" when the counter-slot overflow killed or wedged it: by default rc < 0 (signal, e.g.
    -6 SIGABRT), rc == 134 (128+6), or rc == 124 (rocprofv3_safe.sh's timeout on a HANG). The
    counter budget is box-dependent (some gfx942 boxes SIGABRT even at 4 derived), so we do not
    assume a fixed group size: try the whole list, and on abort split in half and recurse, dropping
    only the singletons that individually abort.

    Returns {"passes": [[counters...], ...], "dropped": [counter, ...]} -- the counter groups that
    collected OK (each a real profiling pass the caller already ran) and the counters that abort
    alone (unavailable on this box). Deterministic; no GPU needed for the bisection logic itself."""
    if is_abort is None:
        is_abort = lambda rc: rc is not None and (rc < 0 or rc in (124, 134))
    passes, dropped = [], []

    def _try(group):
        if not group:
            return
        rc = run_pass(list(group))
        if not is_abort(rc):
            passes.append(list(group))      # collected OK (rc may be 0 or a benign non-abort)
            return
        if len(group) == 1:
            dropped.append(group[0])         # this single counter overflows the box -> unavailable
            return
        mid = len(group) // 2
        _try(group[:mid])
        _try(group[mid:])

    _try(list(counters))
    return {"passes": passes, "dropped": dropped}


def _dispatch_duration(row, meta, seen_dispatch):
    """Collect the per-dispatch duration (ns) -- the DENOMINATOR of every achieved rate.

    Without it the memory counters are a byte count with no time, which is why C1 could be
    collected and never converted. The LONG format repeats a dispatch once per counter, so
    de-dup on the dispatch identity before the median."""
    start, end = row.get("Start_Timestamp"), row.get("End_Timestamp")
    if not start or not end:
        return
    key = row.get("Dispatch_ID") or row.get("Correlation_ID") or row.get("Dispatch_Index") or start
    if key in seen_dispatch:
        return
    try:
        dur = float(end) - float(start)
    except (TypeError, ValueError):
        return
    if dur > 0:
        seen_dispatch.add(key)
        meta["durations_ns"].append(dur)


def _ingest_long(reader, kfilter, vals, meta, seen_dispatch):
    """rocprofv3 LONG format: one row per (kernel, counter) with Counter_Name/Counter_Value."""
    n = 0
    for row in reader:
        kn = row.get("Kernel_Name", "")
        if kfilter and kfilter not in kn:
            continue
        try:
            vals[row["Counter_Name"]].append(float(row["Counter_Value"]))
        except (KeyError, ValueError):
            continue
        n += 1
        _dispatch_duration(row, meta, seen_dispatch)
        meta["VGPR"] = row.get("VGPR_Count", meta["VGPR"])
        meta["AGPR"] = row.get("Accum_VGPR_Count", meta["AGPR"])
        meta["SGPR"] = row.get("SGPR_Count", meta["SGPR"])
        meta["LDS"] = row.get("LDS_Block_Size", meta["LDS"])
        meta["Grid"] = row.get("Grid_Size", meta["Grid"])
        meta["WG"] = row.get("Workgroup_Size", meta["WG"])
    return n


# non-counter identity/metadata columns in the rocprof-compute WIDE format (everything else is a counter)
_WIDE_META_COLS = {
    "Dispatch_ID", "GPU_ID", "Queue_ID", "Queue_Index", "PID", "TID", "SIG", "OBJ",
    "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup", "Scratch_Per_Workitem",
    "Arch_VGPR", "Accum_VGPR", "SGPR", "Kernel_Name", "Start_Timestamp", "End_Timestamp",
    "Correlation_ID", "Node_ID", "Dispatch_Index",
}


def _ingest_wide(reader, kfilter, vals, meta, seen_dispatch):
    """rocprof-compute WIDE format (pmc_perf.csv): Kernel_Name column + one column per counter,
    one row per dispatch. Every non-meta column whose value parses as a float is a counter."""
    n = 0
    for row in reader:
        kn = row.get("Kernel_Name", "")
        if kfilter and kfilter not in kn:
            continue
        n += 1
        _dispatch_duration(row, meta, seen_dispatch)
        for col, raw in row.items():
            base = col.split(".")[0]  # tolerate pandas de-dup suffixes like Kernel_ID.1
            if base in _WIDE_META_COLS or base.startswith("Kernel_ID"):
                continue
            try:
                vals[base].append(float(raw))
            except (TypeError, ValueError):
                continue
        meta["VGPR"] = row.get("Arch_VGPR", meta["VGPR"])
        meta["AGPR"] = row.get("Accum_VGPR", meta["AGPR"])
        meta["SGPR"] = row.get("SGPR", meta["SGPR"])
        meta["LDS"] = row.get("LDS_Per_Workgroup", meta["LDS"])
        meta["Grid"] = row.get("Grid_Size", meta["Grid"])
        meta["WG"] = row.get("Workgroup_Size", meta["WG"])
    return n


def parse_pmc(prof_dir: str, kfilter: str = "_attn_fwd") -> dict:
    """Return {"vals": {counter: [values]}, "meta": {...}} filtered to kfilter.

    Handles BOTH profiler CSV shapes automatically:
      * rocprofv3 LONG  -> *counter_collection.csv  (Counter_Name/Counter_Value rows)
      * rocprof-compute WIDE -> pmc_perf.csv         (one column per counter, one row/dispatch)
    A file is treated as WIDE when it has a Kernel_Name column but no Counter_Name column
    (the rocprof-compute 3.x layout under workloads/<name>/<sku>/pmc_perf.csv)."""
    vals: dict[str, list[float]] = defaultdict(list)
    meta = {"VGPR": None, "AGPR": None, "SGPR": None, "LDS": None, "Grid": None, "WG": None,
            "durations_ns": []}
    seen_dispatch: set = set()
    patterns = [
        # rocprofv3 long format
        os.path.join(prof_dir, "pmc_*", "*", "*counter_collection.csv"),
        os.path.join(prof_dir, "**", "*counter_collection.csv"),
        os.path.join(prof_dir, "*counter_collection.csv"),
        # rocprof-compute wide format (pmc_perf.csv, possibly under workloads/<name>/<sku>/)
        os.path.join(prof_dir, "**", "pmc_perf.csv"),
        os.path.join(prof_dir, "pmc_perf.csv"),
    ]
    seen = set()
    for pat in patterns:
        for path in sorted(glob.glob(pat, recursive=True)):
            if path in seen:
                continue
            seen.add(path)
            try:
                with open(path) as f:
                    reader = csv.DictReader(f)
                    cols = set(reader.fieldnames or [])
                    if "Counter_Name" in cols:
                        _ingest_long(reader, kfilter, vals, meta, seen_dispatch)
                    elif "Kernel_Name" in cols:
                        _ingest_wide(reader, kfilter, vals, meta, seen_dispatch)
                    # else: unrecognized CSV; skip
            except OSError:
                continue
    meta["duration_ns"] = _median(meta["durations_ns"])
    meta["n_dispatches"] = len(meta["durations_ns"])
    return {"vals": dict(vals), "meta": meta}


_HELPER_TOKENS = ("at::", "aten", "elementwise_kernel", "vectorized_elementwise", "reduce_kernel",
                  "cutlass", "gemm_universal", "cublas", "hipblas", "rocblas", "_copy_kernel",
                  "fill_kernel", "cast_kernel")


def _is_helper_kernel(name: str) -> bool:
    """torch/aten/library helper kernels that pollute a trace (same rule as capture.sh)."""
    import re as _re
    nl = name.lower()
    if any(t in nl for t in _HELPER_TOKENS):
        return True
    return _re.search(r"(?:^|[^a-z0-9])ck_", nl) is not None   # CK only as a word-prefix


def top_kernel_from_trace(trace_dir: str):
    """(name, total_ns, dispatches) of the dominant NON-helper kernel in a rocprofv3 kernel trace
    (`*kernel_trace.csv` anywhere under trace_dir), or None. Used to pick the --kernel filter
    when the caller did not name one: every counter pass must be filtered (rocprofv3_safe guard 3)."""
    tot: dict = defaultdict(float)
    cnt: dict = defaultdict(int)
    for path in glob.glob(os.path.join(trace_dir, "**", "*kernel_trace.csv"), recursive=True):
        try:
            with open(path) as f:
                for row in csv.DictReader(f):
                    n = (row.get("Kernel_Name") or row.get("kernel_name") or "").strip()
                    if not n or _is_helper_kernel(n):
                        continue
                    try:
                        dur = float(row.get("End_Timestamp", 0)) - float(row.get("Start_Timestamp", 0))
                    except (TypeError, ValueError):
                        dur = 0.0
                    tot[n] += max(dur, 0.0)
                    cnt[n] += 1
        except OSError:
            continue
    if not tot:
        return None
    name = max(tot, key=lambda k: (tot[k], cnt[k]))
    return name, tot[name], cnt[name]


def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def avg(vals: dict, name: str):
    xs = vals.get(name, [])
    return sum(xs) / len(xs) if xs else None


def memory_derived(parsed: dict, line_bytes: int = TCC_LINE_BYTES, arch: str | None = None) -> dict:
    """C1: turn the memory counters into an ACHIEVED DRAM bandwidth + L2 hit rate.

    This is the one bound that cannot be inferred from the ISA, and for a long time it was
    collected and never converted -- the counters sat in the CSV while every downstream read
    compared the kernel against a DATASHEET peak instead of its own measured rate. A byte count
    with no denominator is not a bandwidth.

    TWO INDEPENDENT ROUTES, deliberately not reconciled:
      * TCC_MISS_sum * 128 B  -- L2 misses, i.e. lines that actually went to DRAM
      * (FETCH_SIZE + WRITE_SIZE) * 1 KB -- the profiler's own video-memory traffic counters
    plus, when the `memory_ea` group was collected, TCC_EA0_{RD,WR}REQ_sum * 128 B as a third.
    They measure the same physical traffic by different paths, so agreement inside
    BW_CROSSCHECK_TOL is what makes the number citable and a disagreement is REPORTED, never
    silently resolved by preferring one -- a disagreement usually means a unit/line-size
    assumption above is wrong on this box, and picking a favourite hides that.

    Returns {} when the memory group was never collected, and a `_missing` note when the bytes
    are there but the duration is not (a missing number is reported as missing, never as zero)."""
    v, m = parsed.get("vals", {}), parsed.get("meta", {})
    fetch, write = avg(v, "FETCH_SIZE"), avg(v, "WRITE_SIZE")
    miss, hit = avg(v, "TCC_MISS_sum"), avg(v, "TCC_HIT_sum")
    ea_rd, ea_wr = avg(v, "TCC_EA0_RDREQ_sum"), avg(v, "TCC_EA0_WRREQ_sum")
    if all(x is None for x in (fetch, write, miss, hit, ea_rd, ea_wr)):
        return {}

    out: dict = {}
    if arch and str(arch).startswith("gfx95") and (fetch is not None or write is not None):
        out["_fetch_size_caveat"] = (
            f"{arch}: FETCH_SIZE (and TCC_BUBBLE-derived read bytes) under-count on gfx95x, so the "
            f"fetch_write route is a LOWER bound here; read the tcc_miss / tcc_ea0 routes and carry "
            f"dram_mb_range rather than citing fetch_write.")
    if miss is not None and hit is not None and (miss + hit) > 0:
        out["l2_hit_pct"] = round(100.0 * hit / (hit + miss), 2)
    # DRAM byte routes (bytes, not rates -- the rate needs the duration below)
    routes: dict = {}
    if miss is not None:
        routes["tcc_miss"] = miss * line_bytes
    if fetch is not None or write is not None:
        routes["fetch_write"] = ((fetch or 0) + (write or 0)) * FETCH_SIZE_UNIT_BYTES
    if ea_rd is not None or ea_wr is not None:
        routes["tcc_ea0"] = ((ea_rd or 0) + (ea_wr or 0)) * line_bytes
    out["dram_bytes_by_route"] = {k: round(b) for k, b in routes.items()}
    # The miss route is the NOMINAL one -- it is what the byte models are written against and what a
    # streaming ceiling probe reproduces exactly. It is NOT automatically the true DRAM traffic: it
    # counts whole-line misses, so everything that moves bytes without producing one (partial-line
    # stores, write-allocate refills, atomic round-trips) is invisible to it, and on such a kernel it
    # reads LOW -- which is the dangerous direction, because low bytes look like "memory does not
    # bind" and send the next round at the wrong axis. So when the routes disagree the honest answer
    # is the RANGE, and `dram_mb` is only citable while `_crosscheck_ok`.
    primary = "tcc_miss" if "tcc_miss" in routes else next(iter(routes), None)
    if primary:
        out["dram_bytes"] = round(routes[primary])
        out["dram_mb"] = round(routes[primary] / 1e6, 2)
        out["_bytes_route"] = primary
    if len(routes) > 1:
        lo, hi = min(routes.values()), max(routes.values())
        delta = (hi - lo) / hi if hi else 0.0
        out["dram_mb_range"] = [round(lo / 1e6, 2), round(hi / 1e6, 2)]
        out["_crosscheck_delta"] = round(delta, 4)
        out["_crosscheck_ok"] = bool(delta <= BW_CROSSCHECK_TOL)
        if delta > BW_CROSSCHECK_TOL:
            out["_crosscheck_warning"] = (
                f"the DRAM byte routes disagree by {delta:.1%} (> {BW_CROSSCHECK_TOL:.0%}): "
                + ", ".join(f"{k}={b/1e6:.1f} MB" for k, b in sorted(routes.items()))
                + f". Two causes, and they need OPPOSITE responses. (1) A unit slip, which shows up "
                  f"on a pure stream too -- rerun the known-bytes probe (mem_bw_probe.py) and check "
                  f"FETCH_SIZE's unit ({FETCH_SIZE_UNIT_BYTES} B assumed) and the TCC line size "
                  f"({line_bytes} B assumed) on THIS box. (2) A real access-shape effect, which "
                  f"appears ONLY on the kernel while the same probe stays inside tolerance -- then "
                  f"no route is wrong, they measure different things, and the answer is the range "
                  f"dram_mb_range. Do not cite `dram_mb` as an achieved bandwidth either way; pass "
                  f"every route to hw_budget.py --measured-dram-mb and carry the width.")

    dur_ns = m.get("duration_ns")
    if not dur_ns:
        out["_missing"] = ("no dispatch duration in the CSV (Start_Timestamp/End_Timestamp) -> "
                           "bytes only, NO achieved bandwidth. Re-collect with a profiler run that "
                           "emits timestamps; do not read the byte count as a rate.")
        return out
    dur_s = dur_ns * 1e-9
    out["duration_us"] = round(dur_ns / 1e3, 3)
    out["n_dispatches"] = m.get("n_dispatches")
    for k, b in routes.items():
        out[f"achieved_tb_s_{k}"] = round(b / dur_s / 1e12, 3)
    if len(routes) > 1:
        rates = [routes[k] / dur_s / 1e12 for k in routes]
        out["achieved_dram_tb_s_range"] = [round(min(rates), 3), round(max(rates), 3)]
    if primary:
        out["achieved_dram_tb_s"] = out[f"achieved_tb_s_{primary}"]
        out["_ceiling_note"] = ("compare this against the IN-SHAPE ceiling from mem_bw_probe.py, "
                                "NOT the datasheet peak -- the datasheet peak is unreachable at any "
                                "real access shape, so %-of-peak understates and the gap inflates.")
        # ...and before comparing at all, check that HBM was in the path. These counters live at the
        # L2/fabric boundary, in front of the memory-side cache, so a kernel whose footprint fits in
        # that cache produces a large "DRAM" rate with no DRAM traffic behind it.
        out["_residency_note"] = ("this is FABRIC traffic. It equals DRAM traffic only once the "
                                  "kernel's footprint clears the memory-side LLC (sku.json "
                                  "mall_mb); below that, a high %-of-HBM-ceiling can be reached "
                                  "with nothing going to HBM. Pass the footprint to hw_budget.py "
                                  "--footprint-mb so the residency check runs.")
    return out


def _relative_busy_from_raw(v: dict) -> dict:
    """RELATIVE per-unit busy shares from RAW rocprof-compute counters, for when the
    analyze-derived names (MfmaUtil/VALUBusy/...) are absent (raw pmc_perf.csv, no analyze).

    classify.py is relative-first: it needs which unit dominates, not absolute %-of-peak. Raw
    SQ_/TA_/TD_/GRBM counters are cross-CU sums, so we normalize each active-cycle counter to the
    busiest one (share of the max) — a robust relative ranking that does NOT need the CU-count
    denominator that a broken `analyze` would have supplied. Returns {} if the raw counters are
    absent too (genuinely PMC-blind)."""
    busy = {
        "mfma": avg(v, "SQ_VALU_MFMA_BUSY_CYCLES"),
        "valu": avg(v, "SQ_ACTIVE_INST_VALU"),
        "vmem": avg(v, "TA_TA_BUSY_sum") or avg(v, "TD_TD_BUSY_sum"),
        "lds": avg(v, "SQ_LDS_IDX_ACTIVE") or avg(v, "SQ_ACTIVE_INST_LDS"),
        "salu": avg(v, "SQ_ACTIVE_INST_SCA"),
    }
    present = {k: x for k, x in busy.items() if x is not None}
    if not present:
        return {}
    denom = max(present.values()) or 1.0
    return {k: 100.0 * x / denom for k, x in present.items()}


def derived(parsed: dict, arch: str | None = None) -> dict:
    """Named per-unit busy metrics + bubble. Prefer rocprof-compute analyze-derived names; when
    those are absent (raw pmc_perf.csv only), fall back to RELATIVE busy shares from raw counters
    so a live profile is NEVER mis-reported as PMC-blind just because `analyze` didn't run."""
    v = parsed["vals"]
    mfma = avg(v, "MfmaUtil")
    rel = _relative_busy_from_raw(v) if mfma is None else {}
    if rel and "mfma" in rel:
        mfma = rel["mfma"]
    d = {
        "MfmaUtil": mfma,
        "VALUBusy": avg(v, "VALUBusy") or rel.get("valu"),
        "VALUUtilization": avg(v, "VALUUtilization") or rel.get("valu"),
        "MemUnitStalled": avg(v, "MemUnitStalled"),
        "MemUnitBusy": avg(v, "MemUnitBusy") or rel.get("vmem"),
        "LDSBankConflict": avg(v, "LDSBankConflict") or rel.get("lds"),
        "SALUBusy": avg(v, "SALUBusy") or rel.get("salu"),
        "occupancy": avg(v, "MeanOccupancyPerActiveCU") or avg(v, "OccupancyPercent"),
        # bubble = fraction of time the matrix engine is idle (100 - MFMA busy).
        "bubble_mfma_idle": (100.0 - mfma) if mfma is not None else None,
        # relative-busy provenance so consumers know these are shares-of-max, not %-of-peak
        "_relative_busy": rel or None,
        # C1: achieved DRAM bandwidth + L2 hit, converted from the memory group (was collected and
        # never converted, so every downstream read fell back to the datasheet peak)
        "memory": memory_derived(parsed, arch=arch) or None,
    }
    return d


def main() -> None:
    argv = list(sys.argv[1:])
    if "--top-kernel" in argv:
        i = argv.index("--top-kernel")
        top = top_kernel_from_trace(argv[i + 1] if i + 1 < len(argv) else ".")
        if top is None:
            sys.exit(1)
        print(top[0])
        return
    if "--print-groups" in argv:
        for name, counters in PMC_GROUPS.items():
            print(f"{name}: {' '.join(counters)}")
        return
    arch = None
    if "--arch" in argv:
        i = argv.index("--arch")
        arch = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
    if not argv:
        sys.exit("usage: parse_pmc.py <rocprofv3_out_dir> [kernel_substr] [--arch gfxNNN] | --print-groups")
    prof_dir = argv[0]
    kfilter = argv[1] if len(argv) > 1 else "_attn_fwd"
    parsed = parse_pmc(prof_dir, kfilter)
    vals, meta = parsed["vals"], parsed["meta"]
    print(f"=== {prof_dir}  (kernel filter: {kfilter!r}) ===")
    if not vals:
        print("no matching counters (PMC blind on this kernel? -> use static ISA path).")
        return
    print("dispatches matched per counter: "
          + ", ".join(f"{k}={len(x)}" for k, x in sorted(vals.items())))
    print(f"VGPR={meta['VGPR']} AGPR={meta['AGPR']} SGPR={meta['SGPR']} "
          f"LDS={meta['LDS']}B  Grid={meta['Grid']} WG={meta['WG']}")
    print("--- derived (busy/throughput + bubble) ---")
    d = derived(parsed, arch=arch)
    mem = d.pop("memory", None)
    for k, val in d.items():
        if isinstance(val, dict):
            print(f"  {k:24s} " + ", ".join(f"{kk}={_fmt(vv)}" for kk, vv in val.items()))
        elif isinstance(val, (int, float)):
            print(f"  {k:24s} {val:.2f}")
        else:
            print(f"  {k:24s} n/a")
    print("--- C1: achieved DRAM bandwidth + L2 hit (memory group) ---")
    if not mem:
        print("  memory group not collected -> you do not know whether this kernel is "
              "bandwidth-bound.\n  Collect: rocprofv3 --pmc " + " ".join(PMC_GROUPS["memory"]))
    else:
        for k, val in mem.items():
            if isinstance(val, dict):
                print(f"  {k:24s} " + ", ".join(f"{kk}={_fmt(vv)}" for kk, vv in val.items()))
            else:
                print(f"  {k:24s} {_fmt(val)}")


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1e6 else f"{v:.3e}"
    return str(v)


def _selftest() -> int:
    # collect_pmc_group bisects on abort and drops only the individually-overflowing counters.
    # Fake box: any pass CONTAINING "BAD" aborts (SIGABRT -> rc -6); everything else collects.
    def run(group):
        return -6 if "BAD" in group else 0
    r = collect_pmc_group(["FETCH_SIZE", "WRITE_SIZE", "TCC_HIT_sum", "TCC_MISS_sum"], run)
    assert r["dropped"] == [] and r["passes"] == [["FETCH_SIZE", "WRITE_SIZE", "TCC_HIT_sum", "TCC_MISS_sum"]], r
    r2 = collect_pmc_group(["A", "BAD", "C", "D"], run)
    # survivors collected (in some split grouping), BAD dropped, and BAD never appears in a pass
    survivors = [c for grp in r2["passes"] for c in grp]
    assert set(survivors) == {"A", "C", "D"} and r2["dropped"] == ["BAD"], r2
    assert all("BAD" not in grp for grp in r2["passes"]), r2
    # a box that aborts on ANY multi-counter pass but is fine one-at-a-time -> all singletons
    def run_strict(group):
        return -6 if len(group) > 1 else 0
    r3 = collect_pmc_group(["W", "X", "Y"], run_strict)
    assert sorted(c for g in r3["passes"] for c in g) == ["W", "X", "Y"] and r3["dropped"] == [], r3
    assert all(len(g) == 1 for g in r3["passes"]), r3
    # a box that aborts on EVERYTHING, even singletons -> all dropped, no passes
    r4 = collect_pmc_group(["P", "Q"], lambda g: -6)
    assert r4["passes"] == [] and sorted(r4["dropped"]) == ["P", "Q"], r4
    # a HANG (rocprofv3_safe.sh timeout, rc 124) bisects like an abort
    r5 = collect_pmc_group(["A", "HANG", "C"], lambda g: 124 if "HANG" in g else 0)
    assert r5["dropped"] == ["HANG"] and sorted(c for g in r5["passes"] for c in g) == ["A", "C"], r5
    # the counter set profile_kernel.sh --pmc collects: memory_ea is the TCC_EA0_{RD,WR}REQ_sum pair
    # this parser reads -- never the DRAM-only _DRAM_ variant -- and no listed-absent name appears
    assert PMC_GROUPS["memory_ea"] == ["TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"], PMC_GROUPS
    _all = {c for g in PMC_GROUPS.values() for c in g}
    assert not _all & {"TCC_EA0_RDREQ_DRAM_sum", "TCC_EA_RDREQ_sum", "MemUnitBusy", "L2CacheHit"}, _all

    # ---- C1 conversion: counters -> achieved DRAM bandwidth ---------------------------------
    # Reference point is a measured fused-MoE gemm1: 246.35 MB of DRAM in 53.52 us = 4.60 TB/s.
    # 246.35 MB / 128 B = 1_924_609 misses; the same traffic as FETCH_SIZE KB = 240_576.
    misses = 246.35e6 / TCC_LINE_BYTES
    parsed = {"vals": {"TCC_MISS_sum": [misses], "TCC_HIT_sum": [misses * 0.0684],
                       "FETCH_SIZE": [246.35e6 / FETCH_SIZE_UNIT_BYTES], "WRITE_SIZE": [0.0]},
              "meta": {"duration_ns": 53520.0, "n_dispatches": 9}}
    md = memory_derived(parsed)
    assert abs(md["achieved_dram_tb_s"] - 4.60) < 0.01, md
    assert abs(md["dram_mb"] - 246.35) < 0.01, md
    assert md["_crosscheck_ok"] and md["_crosscheck_delta"] < 0.01, md
    assert 6.3 < md["l2_hit_pct"] < 6.5, md
    assert "_missing" not in md, md
    # a wrong unit assumption on one route must SURFACE, not be silently resolved
    bad = memory_derived({"vals": {**parsed["vals"], "FETCH_SIZE": [246.35e6]},
                          "meta": parsed["meta"]})
    assert not bad["_crosscheck_ok"] and "_crosscheck_warning" in bad, bad
    assert bad["achieved_dram_tb_s"] == md["achieved_dram_tb_s"], "primary route must not move"
    # ...and the disagreement must be usable, not just announced: a RANGE is emitted so a caller can
    # carry the width instead of inheriting the nominal route's answer. The nominal route counts
    # whole-line misses only, so on a kernel that moves bytes without producing them it reads LOW --
    # the direction that fakes "memory does not bind" -- which is why the low end must stay visible.
    b_lo, b_hi = bad["dram_mb_range"]
    assert b_lo < b_hi and b_lo <= bad["dram_mb"] <= b_hi, bad
    r_lo, r_hi = bad["achieved_dram_tb_s_range"]
    assert r_lo <= bad["achieved_dram_tb_s"] <= r_hi and r_lo < r_hi, bad
    # the agreeing case must NOT invent a width
    assert md["dram_mb_range"][0] == md["dram_mb_range"][1] or md["_crosscheck_ok"], md
    assert "_crosscheck_warning" not in md, md
    # gfx95x: FETCH_SIZE under-counts -> the caveat is stamped, the primary (miss) route is unchanged
    md950 = memory_derived(parsed, arch="gfx950")
    assert "_fetch_size_caveat" in md950 and md950["achieved_dram_tb_s"] == md["achieved_dram_tb_s"], md950
    assert "_fetch_size_caveat" not in memory_derived(parsed, arch="gfx942")
    # bytes without a duration = no bandwidth, reported as missing (never as zero)
    nodur = memory_derived({"vals": parsed["vals"], "meta": {}})
    assert "_missing" in nodur and "achieved_dram_tb_s" not in nodur, nodur
    assert nodur["dram_mb"] > 0, nodur
    # memory group never collected -> {} (and derived() carries None, not a fabricated 0)
    assert memory_derived({"vals": {"MfmaUtil": [50.0]}, "meta": {}}) == {}
    assert derived({"vals": {"MfmaUtil": [50.0]}, "meta": {}})["memory"] is None

    # ---- end-to-end through a rocprofv3 LONG CSV: duration comes from the timestamps ---------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "x_counter_collection.csv"), "w") as f:
            f.write("Dispatch_ID,Kernel_Name,Counter_Name,Counter_Value,"
                    "Start_Timestamp,End_Timestamp,VGPR_Count\n")
            for did, (start, end) in enumerate([(1000, 1000 + 53520), (200000, 200000 + 53520),
                                                (400000, 400000 + 60000)], start=1):
                for cn, cv in (("TCC_MISS_sum", misses), ("TCC_HIT_sum", 0.0)):
                    f.write(f"{did},my_moe_gemm1_kernel,{cn},{cv},{start},{end},128\n")
        p = parse_pmc(td, "moe_gemm1")
        assert p["meta"]["n_dispatches"] == 3, p["meta"]        # de-duped across counters
        assert p["meta"]["duration_ns"] == 53520.0, p["meta"]   # MEDIAN, not the 60000 outlier
        e2e = memory_derived(p)
        assert abs(e2e["achieved_dram_tb_s"] - 4.60) < 0.01, e2e
        assert parse_pmc(td, "no_such_kernel")["meta"]["duration_ns"] is None
        # dominant non-helper kernel of a kernel trace (the default --kernel for profile_kernel --pmc)
        with open(os.path.join(td, "r_kernel_trace.csv"), "w") as f:
            f.write("Kernel_Name,Start_Timestamp,End_Timestamp\n"
                    "at::native::elementwise_kernel,0,900000\n"
                    "_gemm_kernel_BLOCK_M,0,500\n_gemm_kernel_BLOCK_M,1000,1500\n"
                    "ck_gemm_helper,0,99999\nsmall_kernel,0,10\n")
        assert top_kernel_from_trace(td)[0] == "_gemm_kernel_BLOCK_M", top_kernel_from_trace(td)
    print("[parse_pmc] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()
