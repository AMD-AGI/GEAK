#!/usr/bin/env python3
"""Measure the HBM bandwidth this GPU reaches at a kernel's access shape (run length, stride,
read:write mix, program count), as a lo-hi range to pass to hw_budget.py --measured-hbm-tb-s.

Design notes (kept here, not in --help):
- The datasheet HBM peak is not reached by any real access, so a roofline that divides by it
  overstates the room left. This probe measures the ceiling at the requested shape instead.
- Each knob moves the ceiling independently: --rw-mix (a mixed stream reaches neither the pure-read
  nor the pure-write rate; which one is higher is per part), --runlen/--stride (runs of >= one
  128 B line cost a few percent against a linear stream on MI350X; below one line the waste is
  about (128/runlen)x), --nprog (the ceiling rises with parallelism at a fixed footprint).
- The output is a range: repeat readings drift a few percent between sessions.
- A too-high ceiling shrinks every gap computed from it, so three gates run before a number is
  returned: (1) byte accounting -- every claimed byte is distinct and inside its buffer
  (`verify_plan`); (2) cache residency -- the footprint must clear the memory-side LLC (MALL), not
  just the per-XCD L2; (3) no shaped access may read above a LINEAR stream measured in the same
  process at the same mix and program count (tolerance: REF_TOL, wider for masked runs and for the
  readings' own spread, see reference_tolerance).
- Kernel shape (F5 of the tool audit): each tile iteration keeps >= 16 KiB of loads in flight per
  program ([ROWS, run] tile), reads and writes go to separate contiguous buffers, and element
  offsets are 64-bit; each sample is preceded by a read-evict, not a write/zero flush. A
  one-run-per-iteration loop timed after a zero-fill flush read ~10% (linear) to ~13% (2:1 mix) below an
  independent tiled stream on MI350X, and 32-bit offsets wrapped at runlen <= 256 with a 4 KiB
  stride (span > 8 GiB).

Usage (the part comes from --hardware-context, written once per run by
`python3 $KT/hw_context.py --output $EVAL_DIR/hardware_context.json`; --sku names a reference part
used as given, not checked against this GPU -- do not guess it):
  mem_bw_probe.py --hardware-context $EVAL_DIR/hardware_context.json --footprint-mb 2048
  mem_bw_probe.py --sku MI355X --footprint-mb 2048                       # linear read
  mem_bw_probe.py --sku MI355X --runlen 2048 --stride 32768 --rw-mix 40:60
  mem_bw_probe.py --sku MI355X --nprog-sweep 1000,2000,4000,8000
  hw_budget.py --sku MI355X ... --measured-hbm-tb-s <lo>,<hi> --measured-from <probe.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import _quickhelp  # noqa: E402
import _hwdata  # noqa: E402  (perf_knowledge/hardware/data/sku.json -- the single SKU source)

LINE_BYTES = 128
MIB = 1 << 20      # capacities and footprints are binary; rates (TB/s) stay decimal


# --------------------------------------------------------------------------- pure planning math
def _find_sku_json():
    p = _hwdata.find("sku.json")
    return str(p) if p else None


def load_sku(name):
    p = _find_sku_json()
    if not p:
        sys.exit("[mem_bw_probe] cannot find sku.json ($GEAK_HW_DATA_DIR or "
                 "perf_knowledge/hardware/data)")
    skus = json.load(open(p))["skus"]
    if name not in skus:
        sys.exit(f"[mem_bw_probe] unknown SKU {name!r}; known: {', '.join(skus)}")
    return skus[name]


MAX_RUNS = 8   # runs per read/write group: the widest probed ratio is 7:1 (or 1:7)


def _parse_rw_mix(mix: str):
    try:
        r_s, w_s = mix.split(":")
        r, w = float(r_s), float(w_s)
    except ValueError:
        raise ValueError(f"--rw-mix wants R:W (e.g. 40:60 or 1:0), got {mix!r}") from None
    if not (math.isfinite(r) and math.isfinite(w)):
        raise ValueError(f"--rw-mix must be finite numbers, got {mix!r}")
    if r < 0 or w < 0 or r + w <= 0:
        raise ValueError(f"--rw-mix must be non-negative and non-zero, got {mix!r}")
    return r, w


def rw_mix_to_runs(mix: str, max_runs: int = MAX_RUNS):
    """"40:60" -> (2, 3) read/write run counts per group. The ratio is what the ceiling responds
    to, so reduce it to the smallest pair that expresses it (a group of 100 runs would work and
    would also bloat the unrolled kernel for no measurement gain). A ratio no group of at most
    max_runs runs expresses is rounded to the nearest one (9:1 -> 7:1); rw_mix_resolve says so."""
    r, w = _parse_rw_mix(mix)
    if w == 0:
        return 1, 0
    if r == 0:
        return 0, 1
    best = None
    for total in range(2, max_runs + 1):
        for nr in range(1, total):
            nw = total - nr
            err = abs(nr / (nr + nw) - r / (r + w))
            if best is None or err < best[0] - 1e-12:
                best = (err, nr, nw)
    _, nr, nw = best
    return nr, nw


def rw_mix_resolve(mix: str):
    """(nr, nw, warning): the run pair the probe times, and a warning when its ratio is not the
    requested one. The record is labelled with the ratio measured (nr:nw), never the request."""
    r, w = _parse_rw_mix(mix)
    nr, nw = rw_mix_to_runs(mix)
    if abs(nr / (nr + nw) - r / (r + w)) <= 1e-9:
        return nr, nw, None
    return nr, nw, (f"--rw-mix {mix} is run as {nr}:{nw}: a group holds at most {MAX_RUNS} runs, "
                    f"so the widest ratio the probe runs is {MAX_RUNS - 1}:1 (or 1:{MAX_RUNS - 1}) "
                    f"and others round to the nearest pair. The record is labelled {nr}:{nw}; "
                    f"quote the ceiling at that ratio, not at {mix}.")


TILE_BYTES = 16384      # loads in flight per program per tile iteration (fp32 tile [rows, run_p2])
TILE_ITERS = 4          # default tile iterations per program -> default program count
LINEAR_REF_RUNLEN = 4096


def plan(footprint_mb, runlen, stride, nr, nw, nprog=None):
    """Turn the requested access shape into an exact, checkable launch plan.

    Every byte this plan claims to move must be a DISTINCT byte, so the slot maps are injective by
    construction (read slot = group*nr + i in SRC, write slot = group*nw + j in DST) and
    `verify_plan` re-checks them. Reads and writes use separate buffers, so a linear (stride ==
    runlen) mix is two contiguous streams. nprog=None picks TILE_ITERS tile iterations per program."""
    if stride < runlen:
        raise ValueError(f"--stride ({stride}) < --runlen ({runlen}) would make runs OVERLAP, so "
                         f"the distinct-byte count would be smaller than the claimed footprint")
    if runlen <= 0 or runlen % 4:
        raise ValueError(f"--runlen must be a positive multiple of 4 B (the probe streams fp32); "
                         f"round a row of {runlen} B up to {max(4, -(-runlen // 4) * 4)}")
    run_elems = runlen // 4                       # fp32 stream
    # any run length: the vector path is the next power of two, masked to the run (the tail lanes
    # issue no memory traffic), so a real row size (e.g. a 656 B fp8 row) is probed as-is
    run_p2 = 1 << (run_elems - 1).bit_length()
    rows = max(1, TILE_BYTES // (run_p2 * 4))     # groups per tile iteration (power of two)
    runs_per_group = nr + nw
    # MiB, not decimal MB: the footprint is compared against a cache CAPACITY (powers of two);
    # sizing it in decimal MB left the residency bar ~5% short. Rates stay decimal (TB/s).
    target_bytes = int(footprint_mb * MIB)
    n_groups = max(1, target_bytes // (runs_per_group * runlen))
    if nprog is None:
        groups_per_prog = rows * TILE_ITERS
        nprog = -(-n_groups // groups_per_prog)
    else:
        groups_per_prog = -(-max(1, math.ceil(n_groups / nprog)) // rows) * rows
    n_groups = groups_per_prog * nprog            # exact, so the byte count is exact (no row mask)
    return {
        "run_bytes": runlen, "run_elems": run_elems, "run_p2": run_p2, "stride_bytes": stride,
        "partial_lines": bool(runlen % LINE_BYTES or stride % LINE_BYTES),
        "stride_elems": stride // 4, "nr": nr, "nw": nw, "rows": rows,
        "n_groups": n_groups, "groups_per_prog": groups_per_prog, "nprog": nprog,
        "read_bytes": n_groups * nr * runlen,
        "write_bytes": n_groups * nw * runlen,
        "moved_bytes": n_groups * runs_per_group * runlen,
        "read_span_bytes": n_groups * nr * stride,
        "write_span_bytes": n_groups * nw * stride,
        "span_bytes": n_groups * runs_per_group * stride,
    }


def reference_plan(footprint_mb, runlen, nr, nw, nprog=None):
    """Gate 3's bound: the LINEAR stream at the same mix and program count. Any linear stream is the
    same contiguous traffic, so it is planned at LINEAR_REF_RUNLEN whatever the shaped run is (a
    reference at the shaped run length was itself issue-bound at short runs)."""
    del runlen  # the reference does not depend on it, on purpose
    return plan(footprint_mb, LINEAR_REF_RUNLEN, LINEAR_REF_RUNLEN, nr, nw, nprog)


def run_offsets(p, gid):
    """fp32 element offsets of group `gid`'s read runs (in SRC) and write runs (in DST): the same
    index map the kernel uses, for the byte audit and the tests."""
    s = p["stride_elems"]
    return ([(gid * p["nr"] + i) * s for i in range(p["nr"])],
            [(gid * p["nw"] + j) * s for j in range(p["nw"])])


def verify_plan(p, buf_bytes):
    """Distinct-byte audit. `buf_bytes` is the total allocation (SRC + DST). Returns a list of
    problems (empty = the plan is honest)."""
    bad = []
    if p["span_bytes"] > buf_bytes:
        bad.append(f"the address span {p['span_bytes']/MIB:.1f} MiB exceeds the allocated "
                   f"{buf_bytes/MIB:.1f} MiB -> the kernel would wrap and re-touch bytes it has "
                   f"already counted")
    last = p["n_groups"] - 1
    reads, writes = run_offsets(p, last)
    for offs, span, name in ((reads, p["read_span_bytes"], "read"),
                             (writes, p["write_span_bytes"], "write")):
        if offs and (max(offs) + p["run_elems"]) * 4 > span:
            bad.append(f"the last {name} run ends past its {span/MIB:.1f} MiB buffer")
    slots = p["n_groups"] * (p["nr"] + p["nw"])
    if slots * p["run_bytes"] != p["moved_bytes"]:
        bad.append("slot count and moved bytes disagree -> the byte accounting is inconsistent")
    if p["stride_bytes"] < p["run_bytes"]:
        bad.append("stride < runlen -> runs overlap and the moved bytes are double-counted")
    return bad


REF_TOL = 0.02          # gate 3 tolerance for a whole-vector (power-of-two) run
NONPOW2_REF_TOL = 0.05  # ... and its floor for a masked (non-power-of-two) run, see reference_tolerance


def reference_tolerance(p, ref_readings=None, readings=None):
    """How far above the linear reference a shaped reading may land before gate 3 refuses it.

    Gate 3 compares the MAX shaped reading with the MAX linear one, so the tolerance can be no
    tighter than the readings' own spread (either set's max can be its high outlier). A run that is
    not a power of two goes down the masked vector path (e.g. 640 B = 160 of 256 lanes: one warp
    idle, one half-masked) in BOTH probes; neither stream is bandwidth-bound there, and where its
    runs land on the DRAM channels -- packed (linear) or spread (strided) -- can move the rate a
    few percent either way, so a strided read can land just above its linear reference with no
    cache reuse. A cache-reuse or double-counted-bytes error overshoots by far more."""
    tol = REF_TOL
    if p.get("run_p2") and p.get("run_elems") != p["run_p2"]:
        tol = max(tol, NONPOW2_REF_TOL)
    for rs in (ref_readings, readings):
        if rs and min(rs) > 0:
            tol = max(tol, (max(rs) - min(rs)) / min(rs))
    return tol


def verdict(bw_tb_s, sku, footprint_mb, reference_tb_s=None, l2_margin=4.0, ref_tol=REF_TOL, rmw=False):
    """Accept or REFUSE a bandwidth reading. A refusal is the tool working, not the tool failing.

    Returns {"ok": bool, "reasons": [...]}. `reasons` is never a preference between numbers: a
    reading above what the device can stream is not "optimistic", it is not DRAM traffic at all.
    `reference_tb_s` is the LINEAR same-mix same-parallelism stream measured in this process -- the
    tight bound. The datasheet peak sits well above anything reachable, so on its own it only
    catches gross errors: a false ceiling that is still under the peak passes it untouched."""
    reasons = []
    # The physical limit is the PIN RATE. A row whose own ceiling is a measurement (sku.json
    # peak_hbm_basis=measured, e.g. gfx1151's 0.212 TB/s vs the 0.256 pin rate) must not be used
    # as the impossibility bound: a better probe legitimately lands between the two.
    memory = _hwdata.memory_peak(sku)
    peak = memory["value_tb_s"]
    if peak is None:
        reasons.append("no specification bandwidth: a historical measurement is not a physical limit")
    # The cache the footprint has to CLEAR is the last one in front of DRAM, and on these parts that
    # is the MEMORY-SIDE cache behind the fabric (AMD Infinity Cache), not the per-XCD L2. They
    # differ by ~8x, so barring on the L2 admits a footprint that clears the L2 and still lives
    # entirely in the memory-side cache -- which is a cache-bandwidth reading wearing a DRAM label,
    # in the direction that shrinks every gap derived from it.
    llc_mb, llc_name = sku.get("mall_mb"), "memory-side LLC (Infinity Cache)"
    if not llc_mb:
        llc_mb, llc_name = sku.get("l2_mb"), "L2 (memory-side LLC capacity UNKNOWN for this SKU)"
    if peak and bw_tb_s > peak:
        reasons.append(
            f"{bw_tb_s:.3f} TB/s EXCEEDS this SKU's {memory['basis']} peak {peak} TB/s -- no DRAM stream "
            f"can do that, so some of the footprint was served from cache (or the byte accounting "
            f"double-counts addresses). REJECTED: a too-high ceiling shrinks every gap computed "
            f"from it.")
    if rmw:
        # In-place RMW of the kernel's OWN working set: a sub-LLC footprint is the point (the kernel
        # gets the same partial cache service), so the residency bar becomes a note. Below the L2
        # the reading is an L2 number for any kernel, and stays refused.
        l2 = sku.get("l2_mb")
        if l2 and footprint_mb < RMW_MIN_L2_MULT * l2:
            reasons.append(
                f"in-place RMW footprint {footprint_mb:.0f} MiB is under {RMW_MIN_L2_MULT:g}x the "
                f"{l2} MiB L2 -- that is an L2 measurement for any kernel. Use the kernel's real "
                f"working set, >= {RMW_MIN_L2_MULT * l2:.0f} MiB.")
    elif llc_mb and footprint_mb < l2_margin * llc_mb:
        reasons.append(
            f"footprint {footprint_mb:.0f} MiB is under {l2_margin:g}x the {llc_mb} MiB "
            f"{llc_name} -- this measures cache, not DRAM. Raise --footprint-mb to "
            f">= {l2_margin*llc_mb:.0f}.")
    if not sku.get("mall_mb") and not rmw:
        reasons.append(
            "this SKU has no established memory-side LLC capacity (sku.json mall_mb is null), so "
            "the residency bar above fell back to the L2 and is LOWER than it should be. Establish "
            "the capacity, or clear it empirically: sweep --footprint-mb upward and take the "
            "ceiling only from the plateau where it stops falling.")
    if reference_tb_s and bw_tb_s > reference_tb_s * (1 + ref_tol):
        reasons.append(
            f"{bw_tb_s:.3f} TB/s is above the LINEAR reference {reference_tb_s:.3f} TB/s measured "
            f"at the same mix and program count (+{ref_tol:.1%} tolerance). A strided / short-run "
            f"shape cannot beat a contiguous one, so this is cache reuse or a byte-accounting "
            f"error, not a higher ceiling.")
    return {"ok": not reasons, "reasons": reasons}


RMW_MIN_L2_MULT = 2.0
RMW_SLAB_ELEMS = 16384         # fp32 elements per program (64 KiB contiguous)
RMW_BLOCK = 4096


def plan_rmw(footprint_mb):
    """In-place read-modify-write of ONE buffer: every element is read once and written back once,
    in contiguous 64 KiB slabs (one per program). The footprint is the buffer, the traffic 2x it."""
    target = int(footprint_mb * MIB)
    nprog = max(1, target // (RMW_SLAB_ELEMS * 4))
    n = nprog * RMW_SLAB_ELEMS
    return {"mode": "rmw-inplace", "nprog": nprog, "n_elems": n,
            "run_bytes": RMW_SLAB_ELEMS * 4, "stride_bytes": RMW_SLAB_ELEMS * 4, "nr": 1, "nw": 1,
            "footprint_bytes": n * 4, "read_bytes": n * 4, "write_bytes": n * 4,
            "moved_bytes": 2 * n * 4, "span_bytes": n * 4}


def _build_rmw_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _rmw(BUF, BLOCK: tl.constexpr, SLAB: tl.constexpr):
        base = tl.program_id(0).to(tl.int64) * SLAB
        for i in tl.static_range(0, SLAB, BLOCK):
            off = base + i + tl.arange(0, BLOCK)
            x = tl.load(BUF + off, cache_modifier=".cg")
            tl.store(BUF + off, x * 0.999 + 1e-3, cache_modifier=".wt")
    return _rmw


def measure_rmw(p, repeats, iters, warmup, graph=True):
    """Timed exactly like a task harness: GEAK harness_lib.time_op (read-evict flush before every
    sample, CUDA events, median), as a captured-graph replay unless graph=False."""
    import torch
    sys.path.insert(0, HERE)
    import ab_bench  # noqa: E402 -- reuse its harness_lib locator (GEAK_HARNESS_LIB or the checkout)
    hl = ab_bench.load_harness_lib()
    kernel = _build_rmw_kernel()
    buf = (torch.randn(p["n_elems"], device="cuda") * 0.1)

    def run():
        kernel[(p["nprog"],)](buf, BLOCK=RMW_BLOCK, SLAB=RMW_SLAB_ELEMS, num_warps=4)
    run()
    torch.cuda.synchronize()
    readings, timers = [], set()
    for _ in range(repeats):
        d = hl.time_op(run, warmup=warmup, repeats=iters, graph=graph, detail=True)
        if d is None:
            sys.exit("[mem_bw_probe] harness_lib.time_op returned None (the RMW kernel raised)")
        ms = float(d["ms"]) if isinstance(d, dict) else float(d)
        timers.add(d.get("timer", "?") if isinstance(d, dict) else "?")
        readings.append(p["moved_bytes"] / (ms * 1e-3) / 1e12)
    p["timer"] = sorted(timers)
    return readings


def ceiling_range(readings):
    """A ceiling is an interval. Repeat readings of ONE configuration differ by 3-4% across
    sessions, so the deliverable is [min, max] with the median for reference -- and the spread is
    the threshold a claimed gap has to clear to be worth a round."""
    if not readings:
        return None
    lo, hi = min(readings), max(readings)
    return {"lo_tb_s": round(lo, 3), "hi_tb_s": round(hi, 3),
            "median_tb_s": round(statistics.median(readings), 3),
            "spread_pct": round(100 * (hi - lo) / hi, 2) if hi else 0.0,
            "n": len(readings)}


# --------------------------------------------------------------------------- GPU half
def _build_kernel():
    """Import triton lazily so --selftest and the planning math work on a box with no GPU."""
    import triton
    import triton.language as tl

    @triton.jit
    def _stream(SRC, DST, OUT, groups_per_prog,
                RUN_ELEMS: tl.constexpr, STRIDE_ELEMS: tl.constexpr,
                NR: tl.constexpr, NW: tl.constexpr, RUN_P2: tl.constexpr, ROWS: tl.constexpr):
        # one [ROWS, RUN_P2] tile per iteration keeps >= TILE_BYTES in flight; offsets are int64
        # because a short run at a large stride spans more than 2**31 elements
        pid = tl.program_id(0)
        r = tl.arange(0, ROWS)[:, None]
        c = tl.arange(0, RUN_P2)[None, :]
        m = (c < RUN_ELEMS) & (r < ROWS)
        acc = tl.zeros([ROWS, RUN_P2], dtype=tl.float32)
        g0 = pid.to(tl.int64) * groups_per_prog
        for it in range(0, groups_per_prog, ROWS):
            gid = (g0 + it + r).to(tl.int64)
            for i in tl.static_range(NR):
                acc += tl.load(SRC + (gid * NR + i) * STRIDE_ELEMS + c, mask=m, other=0.0)
            for j in tl.static_range(NW):
                tl.store(DST + (gid * NW + j) * STRIDE_ELEMS + c, acc, mask=m)
        if NW == 0:
            # keep the reads live: without a consumer the loads are dead code
            tl.store(OUT + pid, tl.sum(tl.sum(acc, axis=1), axis=0))
    return _stream


def measure(p, repeats, iters, warmup):
    import torch
    kernel = _build_kernel()
    rd, wr = p["read_span_bytes"], p["write_span_bytes"]
    free, _total = torch.cuda.mem_get_info()
    need = rd + wr + p["nprog"] * 4 + EVICT_BYTES
    if need > 0.8 * free:
        sys.exit(f"[mem_bw_probe] this shape needs {need/1e9:.1f} GB (read span {rd/1e9:.1f} + "
                 f"write span {wr/1e9:.1f} GB) but only {free/1e9:.1f} GB is free. A stride/runlen "
                 f"ratio of {p['stride_bytes']//p['run_bytes']}x multiplies the span by that ratio: "
                 f"lower --footprint-mb (keep it several x the MALL) or shorten the stride.")
    # content is irrelevant to a bandwidth probe, and randn on tens of GB costs minutes
    src = torch.empty(max(rd // 4, 1), dtype=torch.float32, device="cuda").fill_(1.0)
    dst = torch.empty(max(wr // 4, 1), dtype=torch.float32, device="cuda")
    out = torch.empty(p["nprog"], dtype=torch.float32, device="cuda")
    alloc = (src.numel() if rd else 0) * 4 + (dst.numel() if wr else 0) * 4
    bad = verify_plan(p, alloc)
    if bad:
        sys.exit("[mem_bw_probe] plan is not self-consistent:\n  - " + "\n  - ".join(bad))

    def run():
        kernel[(p["nprog"],)](src, dst, out, p["groups_per_prog"],
                              RUN_ELEMS=p["run_elems"], STRIDE_ELEMS=p["stride_elems"],
                              NR=p["nr"], NW=p["nw"], RUN_P2=p["run_p2"], ROWS=p["rows"],
                              num_warps=4)
    evict = torch.empty(EVICT_BYTES // 4, dtype=torch.float32, device="cuda").fill_(0.0)
    readings = []
    for _ in range(repeats):
        ms = _time_read_evict(run, evict, iters, warmup)
        readings.append(p["moved_bytes"] / (ms * 1e-3) / 1e12)
    return readings


EVICT_BYTES = 1 << 30


def _time_read_evict(run, evict, iters, warmup):
    """Median ms of one launch, each sample preceded by a READ of a 1 GiB buffer. A write/zero
    flush (triton do_bench) leaves dirty lines that are written back inside the timed kernel and
    read ~10% low on MI350X; a read flush evicts without that cost."""
    import torch
    for _ in range(max(1, warmup // 10)):
        run()
    torch.cuda.synchronize()
    times = []
    for _ in range(max(5, iters // 3)):
        evict.sum()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        run()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


# --------------------------------------------------------------------------- CLI
def _tool_versions():
    out = {}
    for mod in ("torch", "triton"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- a missing module is recorded, not fatal
            out[mod] = None
    return out


UNDER_SATURATED_FRAC = 0.45


def _emit(sku_name, sku, p, readings, requested_mb, label, out_json, reference_tb_s=None,
          ref_tol=REF_TOL, extra=None):
    rng = ceiling_range(readings)
    rmw = p.get("mode") == "rmw-inplace"
    v = verdict(rng["hi_tb_s"], sku, (p["footprint_bytes"] if rmw else p["moved_bytes"]) / MIB,
                reference_tb_s, ref_tol=ref_tol, rmw=rmw)
    print(f"\n=== in-shape HBM ceiling: {sku_name} / {label} ===")
    print(f"  shape      runlen {p['run_bytes']} B, stride {p['stride_bytes']} B, "
          f"mix {p['nr']}r:{p['nw']}w, {p['nprog']} programs")
    print(f"  footprint  {p['moved_bytes']/MIB:.1f} MiB moved "
          f"({p['read_bytes']/MIB:.1f} read + {p['write_bytes']/MIB:.1f} write), "
          f"span {p['span_bytes']/MIB:.1f} MiB")
    print("  readings   " + ", ".join(f"{r:.3f}" for r in readings) + " TB/s")
    if reference_tb_s:
        print(f"  vs linear  {100 * statistics.median(readings) / reference_tb_s:.1f}% of the linear "
              f"stream ({reference_tb_s:.3f} TB/s, same mix and program count)")
    if p.get("partial_lines"):
        print(f"  note       runs/stride are not whole {LINE_BYTES} B lines: the rate counts the bytes the "
              f"kernel asks for, while DRAM moves whole lines -- the same accounting your kernel's "
              f"useful-byte count uses, so compare like with like.")
    if rmw:
        llc = sku.get("mall_mb")
        print(f"  mode       IN-PLACE RMW: one {p['footprint_bytes']/MIB:.1f} MiB buffer read then written "
              f"back, contiguous {p['run_bytes'] // 1024} KiB runs; timed like the task harness "
              f"(harness_lib.time_op, read-evict before each sample, timer {','.join(p.get('timer', []))})")
        if llc and p["footprint_bytes"] / MIB < 4 * llc:
            print(f"  note       footprint is inside 4x the {llc} MiB memory-side LLC: allowed for RMW "
                  f"because it is the kernel's OWN working set size, so the cache serves it the same "
                  f"share it serves the kernel. This is a same-shape ceiling for THAT footprint, not a "
                  f"DRAM ceiling -- quote it only for a kernel of this working set.")
    if not v["ok"]:
        print("  *** REJECTED ***")
        for r in v["reasons"]:
            print(f"    - {r}")
        print("  no ceiling returned. Fix the probe, do not use the number.")
    else:
        print(f"  IN-SHAPE CEILING = {rng['lo_tb_s']}-{rng['hi_tb_s']} TB/s "
              f"(median {rng['median_tb_s']}, spread {rng['spread_pct']}% over {rng['n']} reads)")
        _memory = _hwdata.memory_peak(sku)
        _pin = _memory["value_tb_s"]
        print(f"  specification peak = {_pin} TB/s [{_memory['basis']}] "
              f"-> reachable fraction {100*rng['median_tb_s']/_pin:.0f}%")
        print(f"  feed the budget: --measured-hbm-tb-s {rng['lo_tb_s']},{rng['hi_tb_s']}")
        print(f"  a gap smaller than the {rng['spread_pct']}% spread is inside the error bar and "
              f"is not a reason to open a round.")
        # A too-LOW ceiling shrinks the computed gap just like a too-high one. The bar is set below
        # what a correct linear stream reaches on the parts measured (MI350X: ~59-66% of the 8 TB/s
        # datasheet value), and only linear whole-line shapes are checked: a short-run or strided
        # shape can be legitimately slower.
        frac = rng["median_tb_s"] / _pin if _pin else None
        linear = p.get("stride_bytes") == p.get("run_bytes") and p.get("run_bytes", 0) >= LINE_BYTES
        if frac is not None and linear and frac < UNDER_SATURATED_FRAC:
            print(f"  WARNING: {100*frac:.0f}% of the specification peak for a linear stream is low; "
                  f"the probe may be under-saturating\n    (too few programs). Cross-check with "
                  f"--nprog-sweep or an independent stream benchmark.")
        _fp = p["footprint_bytes"] if rmw else p["moved_bytes"]
        if abs(_fp / MIB - requested_mb) / requested_mb > 0.02:
            print(f"  note: footprint rounded to {_fp/MIB:.1f} MiB from the requested "
                  f"{requested_mb:.1f} MiB (groups must divide across programs). Compare sweep "
                  f"points on the printed footprint, not the requested one.")
    rec = {"sku": sku_name, "label": label, "shape": p, "readings_tb_s": readings,
           "ceiling": rng, "accepted": v["ok"], "refusals": v["reasons"],
           "mode": p.get("mode", "stream"),
           "hbm_ceiling_source": "in-shape-probe",
           # the roofline basis pair: a ceiling from this record is an in-shape probe denominator
           # (it may gate/close together with a counters numerator); name the tool and versions.
           "denominator_basis": "in-shape probe",
           "tool": "mem_bw_probe.py", "tool_versions": _tool_versions(),
           "linear_reference_tb_s": reference_tb_s,
           "linear_reference_tol": ref_tol if reference_tb_s else None,
           "datasheet_peak_tb_s": (_hwdata.memory_peak(sku)["value_tb_s"]
                                   if _hwdata.memory_peak(sku)["basis"] == "datasheet" else None),
           "specification_memory_peak": _hwdata.memory_peak(sku),
           "sku_peak_hbm_basis": sku.get("peak_hbm_basis"), **(extra or {})}
    if out_json:
        with open(out_json, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  wrote {out_json}")
    return rec


QUICK_HELP = """
mem_bw_probe.py -- HBM bandwidth this GPU reaches at a kernel's access shape  (GPU, ~1 min)

Useful when hw_budget.py reports `bound: memory` and the stop/continue call rests on that axis:
the datasheet HBM peak is an upper bound no real stream reaches.

  python3 $KT/hw_context.py --output $EVAL_DIR/hardware_context.json    # once per run, no GPU work
  bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID python3 $KT/mem_bw_probe.py \\
      --hardware-context $EVAL_DIR/hardware_context.json --footprint-mb 2048 --json $EVAL_DIR/mem_bw.json
  shape options:
   --rw-mix R:W          bytes read : bytes written (e.g. 2:1 for two inputs, one output)
   --runlen B --stride S contiguous bytes per run and bytes between run starts. Runs of >= 128 B
                         (one L2 line) cost a few percent against a linear stream; below one 128 B
                         line the waste is about (128/runlen)x
   --rmw --footprint-mb <working set MiB>   in-place update of one buffer (state, KV, accumulator)

READ: IN-SHAPE CEILING lo-hi TB/s, then `feed the budget: --measured-hbm-tb-s lo,hi` (pass with
--measured-from <json>). A gap inside the printed spread is noise. A refusal names its gate:
"fits in cache" -> raise --footprint-mb; "above the linear reference" -> re-check --runlen/--stride.
"""


def _check_args(ap, a):
    """Refuse bad values with a usage error before any SKU, context or GPU work. Parses the sweep
    lists in place (comma strings -> lists of int)."""
    def int_list(flag, text):
        try:
            vals = [int(x) for x in text.split(",")]
        except ValueError:
            ap.error(f"{flag} wants a comma list of integers, got {text!r}")
        return vals

    def runlen_ok(flag, v):
        if v <= 0 or v % 4:
            ap.error(f"{flag} must be a positive multiple of 4 B (the probe streams fp32), got {v}")

    if not (math.isfinite(a.footprint_mb) and a.footprint_mb > 0):
        ap.error(f"--footprint-mb must be a positive number, got {a.footprint_mb:g}")
    if a.runlen is not None:
        runlen_ok("--runlen", a.runlen)
    if a.stride is not None:
        runlen_ok("--stride", a.stride)
        if a.stride < (a.runlen or 4096) and not a.runlen_sweep and not a.rmw:
            ap.error(f"--stride ({a.stride}) < --runlen ({a.runlen or 4096}) would make runs overlap")
    try:
        rw_mix_resolve(a.rw_mix)
    except ValueError as e:
        ap.error(str(e))
    if a.nprog is not None and a.nprog <= 0:
        ap.error(f"--nprog must be a positive program count, got {a.nprog}")
    if a.nprog_sweep is not None:
        a.nprog_sweep = int_list("--nprog-sweep", a.nprog_sweep)
        if any(n <= 0 for n in a.nprog_sweep):
            ap.error(f"--nprog-sweep entries must be positive program counts, got {a.nprog_sweep}")
    if a.runlen_sweep is not None:
        a.runlen_sweep = int_list("--runlen-sweep", a.runlen_sweep)
        for rl in a.runlen_sweep:
            runlen_ok("--runlen-sweep", rl)
    for flag, v, lo in (("--repeats", a.repeats, 1), ("--iters", a.iters, 1), ("--warmup", a.warmup, 0)):
        if v < lo:
            ap.error(f"{flag} must be >= {lo}, got {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _quickhelp.add_flag(ap)
    ap.add_argument("--sku", help="SKU key from perf_knowledge/hardware/data/sku.json (for the "
                                  "refusals), e.g. MI355X")
    ap.add_argument("--hardware-context", help="validated run context; defaults to GEAK_HARDWARE_CONTEXT")
    ap.add_argument("--footprint-mb", type=float, default=2048.0,
                    help="unique MiB MOVED per pass (binary, to match how cache capacity is "
                         "quoted); must be several x the MEMORY-SIDE LLC (sku.json mall_mb), not "
                         "the L2 -- see gate 2 (default 2048)")
    ap.add_argument("--runlen", type=int, default=None,
                    help="contiguous bytes per run: any multiple of 4 B (a real row size, e.g. 656; "
                         "the vector path is masked to it) (default 4096; not with --rmw)")
    ap.add_argument("--stride", type=int,
                    help="bytes between run starts (default = --runlen, i.e. a linear stream)")
    ap.add_argument("--rw-mix", default="1:0",
                    help="read:write ratio, e.g. 1:0, 50:50, 40:60. Run as the nearest pair in a "
                         "group of at most 8 runs (7:1 down to 1:7); the record carries the ratio run")
    ap.add_argument("--nprog", type=int, default=None,
                    help="programs over the footprint (default: 4 tile iterations of 16 KiB per "
                         "program; not with --rmw)")
    ap.add_argument("--nprog-sweep", help="comma list of program counts over the SAME footprint "
                                          "(shows whether the ceiling is parallelism-limited)")
    ap.add_argument("--runlen-sweep", help="comma list of run lengths in bytes")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the linear-stream reference for a shaped probe. Only do this if you "
                         "are supplying the plausibility bound another way -- without it, a shaped "
                         "reading has no tight upper bound and only the (loose) datasheet peak "
                         "guards it")
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent readings per configuration -> the range (default 3)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rmw", action="store_true",
                    help="IN-PLACE read-modify-write: one buffer of --footprint-mb read and written "
                         "back in contiguous runs (a state / KV / accumulator update). A separate "
                         "read-A/write-B --rw-mix 1:1 stream does NOT model it. Allowed below the "
                         "memory-side LLC bar when --footprint-mb is the kernel's own working set "
                         "(it must still clear 2x the L2). Timed like a task harness: "
                         "harness_lib.time_op with a read-evict flush per sample, graph replay "
                         "unless --eager")
    ap.add_argument("--eager", action="store_true",
                    help="with --rmw: time eager launches instead of a CUDA-graph replay (match the "
                         "timer your --measured-ms came from)")
    ap.add_argument("--json", help="write the record here")
    ap.add_argument("--selftest", action="store_true")
    _quickhelp.handle(ap, QUICK_HELP)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    _check_args(ap, a)
    try:
        a.sku, context = _hwdata.run_sku(a.sku, a.hardware_context)
        if context:
            import hardware_context as hc
            if not hc.same_device_scope(context, hc.probe()):
                ap.error("live GPU differs from the run hardware context")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
        ap.error(str(e))
    if not a.sku:
        ap.error("no SKU: the physical-limit and cache-residency refusals need the SKU's peak/L2. Pass "
                 "--hardware-context $EVAL_DIR/hardware_context.json, created once per run with: "
                 f"{_hwdata.context_command()}  (or --selftest). Do not guess --sku: MI350X and MI355X "
                 "are both gfx950 with different clocks and peaks.")
    if context is None:
        print(_hwdata.sku_reference_warning(a.sku), file=sys.stderr)
    sku = load_sku(a.sku)
    if a.eager and not a.rmw:
        ap.error("--eager applies to --rmw only")
    if a.rmw:
        # --rmw has a fixed shape: one program per contiguous 64 KiB slab of the buffer, so the
        # footprint alone sets both the run length and the program count. Accepting --runlen /
        # --nprog and not using them would label the record with a shape that was never timed.
        bad = [f for f, v in (("--runlen", a.runlen), ("--nprog", a.nprog), ("--stride", a.stride),
                              ("--nprog-sweep", a.nprog_sweep), ("--runlen-sweep", a.runlen_sweep))
               if v is not None] + (["--rw-mix"] if a.rw_mix != "1:0" else [])
        if bad:
            ap.error(f"--rmw is its own shape (in place, 1 read : 1 write, one program per "
                     f"contiguous {RMW_SLAB_ELEMS * 4 // 1024} KiB slab; --footprint-mb sets the "
                     f"program count): drop {' / '.join(bad)}")
    if a.runlen is None:
        a.runlen = 4096
    nr, nw, mix_warning = rw_mix_resolve(a.rw_mix)
    if mix_warning:
        print(f"[mem_bw_probe] WARNING: {mix_warning}", file=sys.stderr)
    mix_fields = {"rw_mix_requested": a.rw_mix, "rw_mix_measured": f"{nr}:{nw}"}
    stride = a.stride or a.runlen

    def one(runlen, strd, nprog, label, out_json=None):
        """Measure one configuration, preceded by its LINEAR reference when it is a shaped access
        (gate 3: nothing shaped may read above a contiguous stream at the same mix/parallelism)."""
        ref, ref_readings = None, None
        if strd != runlen and not a.no_reference:
            rp = reference_plan(a.footprint_mb, runlen, nr, nw, nprog)
            ref_readings = measure(rp, a.repeats, a.iters, a.warmup)
            ref = max(ref_readings)
            print(f"\n  [linear reference for {label}] "
                  + ", ".join(f"{r:.3f}" for r in ref_readings) + " TB/s")
        p = plan(a.footprint_mb, runlen, strd, nr, nw, nprog)
        readings = measure(p, a.repeats, a.iters, a.warmup)
        return _emit(a.sku, sku, p, readings, a.footprint_mb, label, out_json, ref,
                     ref_tol=reference_tolerance(p, ref_readings, readings), extra=mix_fields)

    if a.rmw:
        p = plan_rmw(a.footprint_mb)
        rec = _emit(a.sku, sku, p, measure_rmw(p, a.repeats, a.iters, a.warmup, graph=not a.eager),
                    a.footprint_mb, f"in-place RMW {a.footprint_mb:g} MiB", a.json)
        return 0 if rec["accepted"] else 1
    recs = []
    if a.nprog_sweep:
        for n in a.nprog_sweep:
            recs.append(one(a.runlen, stride, n, f"nprog={n}"))
        print("\n  the ceiling is a function of parallelism: quote it WITH the program count.")
    elif a.runlen_sweep:
        for rl in a.runlen_sweep:
            # with no explicit --stride each entry stays LINEAR at its own run length. Carrying one
            # stride across the sweep would silently turn the short entries into strided shapes and
            # measure two variables at once.
            recs.append(one(rl, max(a.stride, rl) if a.stride else rl, a.nprog, f"runlen={rl}B"))
    else:
        recs.append(one(a.runlen, stride, a.nprog, f"{nr}:{nw} mix", a.json))
    if a.json and len(recs) > 1:
        with open(a.json, "w") as f:
            json.dump(recs, f, indent=2)
    return 1 if any(not r["accepted"] for r in recs) else 0


def _selftest():
    # in-place RMW: traffic is 2x the buffer, sub-LLC footprints pass, sub-2xL2 ones do not
    sku_r = {"peak_hbm_tb_s": 8.0, "peak_hbm_basis": "datasheet", "l2_mb": 32, "mall_mb": 256}
    pr = plan_rmw(200.0)
    assert pr["moved_bytes"] == 2 * pr["footprint_bytes"] and pr["read_bytes"] == pr["write_bytes"]
    assert pr["footprint_bytes"] == pr["nprog"] * RMW_SLAB_ELEMS * 4
    assert abs(pr["footprint_bytes"] / MIB - 200.0) < 0.1, pr
    assert verdict(5.6, sku_r, 200.0, rmw=True)["ok"], "a sub-MALL RMW working set is allowed"
    assert not verdict(5.6, sku_r, 200.0)["ok"], "the same footprint as a stream is still refused"
    assert not verdict(5.6, sku_r, 40.0, rmw=True)["ok"], "under 2x L2 is an L2 number"
    assert not verdict(9.0, sku_r, 200.0, rmw=True)["ok"], "above the pin rate is never a ceiling"
    # rw-mix reduction: the RATIO is what matters, expressed in the smallest run pair
    assert rw_mix_to_runs("1:0") == (1, 0) and rw_mix_to_runs("0:1") == (0, 1)
    assert rw_mix_to_runs("50:50") == (1, 1)
    nr, nw = rw_mix_to_runs("40:60")
    assert (nr, nw) == (2, 3), (nr, nw)
    assert abs(nr / (nr + nw) - 0.4) < 1e-9
    assert rw_mix_resolve("40:60") == (2, 3, None)
    nr9, nw9, w9 = rw_mix_resolve("9:1")
    assert (nr9, nw9) == (7, 1) and w9 and "run as 7:1" in w9, (nr9, nw9, w9)
    for bad in ("1", "-1:2", "0:0", "nan:1", "inf:1", "1:inf", "abc"):
        try:
            rw_mix_to_runs(bad)
            raise AssertionError(f"rw_mix accepted {bad!r}")
        except ValueError:
            pass

    # plan: byte accounting is exact and the slot map is injective
    p = plan(1024.0, 4096, 4096, 1, 0, 1000)
    assert p["moved_bytes"] == p["read_bytes"] and p["write_bytes"] == 0
    assert p["moved_bytes"] % (p["nprog"] * p["run_bytes"]) == 0
    assert verify_plan(p, p["span_bytes"]) == []
    pm = plan(1000.0, 2048, 32768, 2, 3, 990)
    assert pm["read_bytes"] / pm["moved_bytes"] == 0.4, pm
    assert pm["span_bytes"] == pm["moved_bytes"] * 16, pm     # stride 16x runlen
    # a plan whose span outruns its buffer would silently re-touch counted bytes
    assert verify_plan(pm, pm["span_bytes"] - 1), "must flag a span past the allocation"
    # overlapping runs are rejected at plan time, not discovered in the number
    for kwargs in (dict(runlen=4096, stride=2048), dict(runlen=4002, stride=4002),
                   dict(runlen=0, stride=128)):
        try:
            plan(1024.0, kwargs["runlen"], kwargs["stride"], 1, 0, 100)
            raise AssertionError(f"plan accepted {kwargs}")
        except ValueError:
            pass

    # a real (non power of two) row is probed as-is: 656 B = 164 fp32 lanes on a 256-lane masked path
    pr656 = plan(512.0, 656, 656, 1, 0, 1000)
    assert pr656["run_elems"] == 164 and pr656["run_p2"] == 256 and pr656["partial_lines"], pr656
    assert pr656["moved_bytes"] == pr656["n_groups"] * 656 and verify_plan(pr656, pr656["span_bytes"]) == []
    assert not plan(1024.0, 4096, 4096, 1, 0, 10)["partial_lines"]
    # the three gates, and what each one actually catches
    sku = {"peak_hbm_tb_s": 8.0, "peak_hbm_basis": "datasheet", "l2_mb": 32, "mall_mb": 256}
    assert verdict(5.31, sku, 2048)["ok"]
    # gate 1 is the plan audit above -- and it is the ONLY gate that catches an over-read which
    # stays UNDER the datasheet peak: the peak bound passes it, and so does the footprint bound.
    assert verdict(7.31, sku, 2048)["ok"], \
        "the datasheet peak is too loose to catch a plausible-looking over-read; that is gate 1's job"
    bad_plan = plan(227.0, 4096, 4096, 1, 0, 990)
    assert verify_plan(bad_plan, bad_plan["span_bytes"] // 8), \
        "a plan claiming 8x the bytes its allocation holds must be rejected before timing"
    # gate 2: cache residency -- and the bar is the MEMORY-SIDE cache behind the fabric, not the
    # per-XCD L2. A footprint that clears 4x the L2 while still fitting inside the memory-side cache
    # is the exact reading this gate exists to stop, and barring on the L2 would wave it through.
    assert not verdict(5.0, sku, 100)["ok"], "a footprint under the L2 bar is a cache probe"
    v_mall = verdict(5.0, sku, 4 * sku["l2_mb"] + 1)
    assert not v_mall["ok"] and "memory-side LLC" in v_mall["reasons"][0], \
        "clearing 4x the L2 while inside the memory-side cache must still be refused"
    assert verdict(5.0, sku, 4 * sku["mall_mb"])["ok"], "clearing 4x the memory-side cache passes"
    # a SKU whose memory-side capacity was never established cannot certify a ceiling at all: the
    # bar silently degrades to the L2, so say so instead of returning a clean ok
    v_unk = verdict(5.31, {"peak_hbm_tb_s": 8.0, "peak_hbm_basis": "datasheet", "l2_mb": 32}, 2048)
    assert not v_unk["ok"] and any("mall_mb is null" in r for r in v_unk["reasons"]), v_unk
    # gate 3: no shaped access beats a linear stream at the same mix/parallelism
    v_ref = verdict(5.90, sku, 2048, reference_tb_s=5.50)
    assert not v_ref["ok"] and "LINEAR reference" in v_ref["reasons"][0], v_ref
    assert verdict(5.55, sku, 2048, reference_tb_s=5.50)["ok"], "2% tolerance absorbs noise"
    # a masked (non power of two) run may land a few % above its linear reference without cache
    # reuse; a gross over-read is still refused. Pow2 runs keep the 2% bar, and the tolerance
    # never drops below the readings' own spread.
    p640 = plan(2048.0, 640, 1024, 1, 0, 1980)
    assert p640["run_elems"] == 160 and p640["run_p2"] == 256
    assert verdict(2.36, sku, 2048, reference_tb_s=2.26,
                   ref_tol=reference_tolerance(p640, [2.24, 2.26], [2.35, 2.36]))["ok"]
    assert not verdict(2.60, sku, 2048, reference_tb_s=2.26,
                       ref_tol=reference_tolerance(p640, [2.24, 2.26], [2.58, 2.60]))["ok"], \
        "a +15% over-read is still refused on a masked run"
    p4k = plan(2048.0, 4096, 8192, 1, 0, 1980)
    assert reference_tolerance(p4k, [5.50, 5.51], [5.40, 5.41]) == REF_TOL
    assert abs(reference_tolerance(p4k, [5.30, 5.50], [5.5]) - 0.2 / 5.3) < 1e-12, "spread widens it"
    # the loose gate still catches gross errors
    assert not verdict(8.01, sku, 4096)["ok"], "above peak must always be refused"
    v_both = verdict(9.0, sku, 100, reference_tb_s=5.5)
    assert not v_both["ok"] and len(v_both["reasons"]) == 3, v_both

    # the ceiling is a range, and the spread is the threshold a gap must clear
    # a MEASURED table ceiling is not the physical bound: gfx1151's row stores 0.212 (measured)
    # with a 0.256 pin rate, and a better probe reading in between must pass gate 1's peak check
    apu = {"peak_hbm_tb_s": 0.212, "datasheet_hbm_tb_s": 0.256, "l2_mb": 2, "mall_mb": 32}
    assert verdict(0.230, apu, 2048)["ok"], verdict(0.230, apu, 2048)
    assert not verdict(0.260, apu, 2048)["ok"], "above the pin rate must be refused"
    # the real table resolves through _hwdata and carries the decided gfx950/gfx942 bandwidths
    assert load_sku("MI355X")["peak_hbm_tb_s"] == 8.0 and load_sku("MI325X")["peak_hbm_tb_s"] == 6.0

    r = ceiling_range([5.310, 5.501, 5.455])
    assert r["lo_tb_s"] == 5.31 and r["hi_tb_s"] == 5.501 and r["n"] == 3
    assert abs(r["spread_pct"] - 3.47) < 0.05, r
    assert ceiling_range([]) is None
    # --rmw fixes its own shape: --runlen / --nprog are refused, not silently dropped (argparse
    # exits before any GPU work, so this runs on a CPU-only box)
    import subprocess
    for extra in (["--runlen", "640"], ["--nprog", "100"]):
        cp = subprocess.run([sys.executable, os.path.abspath(__file__), "--sku", "MI355X", "--rmw",
                             "--footprint-mb", "200", *extra], capture_output=True, text=True,
                            env={k: v for k, v in os.environ.items() if k != "GEAK_HARDWARE_CONTEXT"})
        assert cp.returncode == 2 and f"drop {extra[0]}" in cp.stderr, (extra, cp.returncode, cp.stderr)
    print("[mem_bw_probe] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
