#!/usr/bin/env python3
"""Probe the IN-SHAPE HBM ceiling: the bandwidth this box actually reaches at YOUR access shape.

Why this exists. A roofline that divides by the datasheet peak from `sku.json` produces a gap that
is part measurement and part fiction -- and the fiction points one way only: it always claims more
headroom than exists, because that peak is not reachable at any real access shape. The denominator
then decides the verdict: the same kernel and the same bytes read as "big prize, keep going"
against the datasheet peak, as "worth continuing" against a read-only peak, and as "close it out"
against the ceiling its own access shape can actually reach. This tool measures the third one: the
ceiling at YOUR run length, stride, read/write mix and parallelism.

Three knobs, because each one moves the ceiling independently, and by more than the gaps these
campaigns argue about:
  --rw-mix     a mixed stream does NOT reach the pure-read or pure-write peak, and which of those
               two is higher is a per-part property -- do not carry a read/write asymmetry over
               from another SKU, probe YOUR mix on THIS box
  --runlen/--stride  a k-major or blocked weight read walks short runs with a large stride; that
               costs a few percent against a linear stream, and a few percent is decision-sized
  --nprog-sweep  the ceiling RISES with parallelism over a fixed footprint, so a ceiling quoted
               without its program count is under-specified

The output is a RANGE, never a single number: repeat readings of one configuration drift by a few
percent between sessions, and that spread is the honest error bar. A gap narrower than the range is
not a reason to open another round.

A too-HIGH ceiling is the dangerous direction: it makes every gap computed from it look smaller
than it is, which closes a campaign early and leaves the win on the table. So three independent
gates must pass before a number is returned, and they catch different failures:

  1. BYTE ACCOUNTING (`verify_plan`, before the GPU runs). Every byte the plan claims must be a
     distinct byte. A plan that layers translated copies over an already-tiled base can claim
     several times its own footprint as "useful" traffic and report a ceiling that is under the
     datasheet peak -- so no peak check would catch it -- while actually measuring cache. The
     span-vs-allocation and injective-slot audits reject such a plan before it is ever timed.
  2. CACHE RESIDENCY. The cache the footprint must clear is the LAST one in front of DRAM, which on
     these parts is the MEMORY-SIDE cache behind the Infinity Fabric -- not the per-XCD L2, which is
     an order of magnitude smaller and is merely the gateway ONTO the fabric. Clearing a few times
     the L2 while still fitting inside the memory-side cache yields a cache-bandwidth number wearing
     a DRAM label, and it fails no other gate because it sits under the datasheet peak.
  3. NO SHAPE BEATS A LINEAR STREAM. At the same mix and the same program count, a strided /
     short-run access cannot exceed a contiguous one -- so the tool measures that linear reference
     in the same process and refuses any shaped reading above it. This is the tight bound the
     datasheet peak is not: the peak sits well above anything reachable, so it only catches gross
     errors, and it is kept only for those.

Usage:
  # linear read stream (the generic ceiling)
  mem_bw_probe.py --sku MI355X --footprint-mb 2048

  # a real weight-read shape: short runs, large stride, read-heavy mix
  mem_bw_probe.py --sku MI355X --runlen 2048 --stride 32768 --rw-mix 40:60

  # is the ceiling parallelism-limited at my program count?
  mem_bw_probe.py --sku MI355X --nprog-sweep 1000,2000,4000,8000

  # then feed it to the budget, which prints [calibrated] instead of [datasheet]
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


def rw_mix_to_runs(mix: str, max_runs: int = 8):
    """"40:60" -> (2, 3) read/write run counts per group. The ratio is what the ceiling responds
    to, so reduce it to the smallest pair that expresses it (a group of 100 runs would work and
    would also bloat the unrolled kernel for no measurement gain)."""
    try:
        r_s, w_s = mix.split(":")
        r, w = float(r_s), float(w_s)
    except ValueError:
        raise ValueError(f"--rw-mix wants R:W (e.g. 40:60 or 1:0), got {mix!r}") from None
    if r < 0 or w < 0 or r + w <= 0:
        raise ValueError(f"--rw-mix must be non-negative and non-zero, got {mix!r}")
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


def plan(footprint_mb, runlen, stride, nr, nw, nprog):
    """Turn the requested access shape into an exact, checkable launch plan.

    Every byte this plan claims to move must be a DISTINCT byte: a footprint assembled from
    overlapping index math reports more traffic than it causes and reads back as super-physical
    bandwidth. So the slot map here is injective by construction (slot = group*(nr+nw) + i) and
    `verify_plan` re-checks it instead of trusting the formula."""
    if stride < runlen:
        raise ValueError(f"--stride ({stride}) < --runlen ({runlen}) would make runs OVERLAP, so "
                         f"the distinct-byte count would be smaller than the claimed footprint")
    if runlen % LINE_BYTES:
        raise ValueError(f"--runlen must be a multiple of the {LINE_BYTES} B cacheline")
    run_elems = runlen // 4                       # fp32 stream
    if run_elems & (run_elems - 1):
        raise ValueError(f"--runlen/4 must be a power of two (got {run_elems} elements)")
    runs_per_group = nr + nw
    # MiB, not decimal MB. This number's whole job is to be compared against a cache CAPACITY, and
    # capacities are quoted in powers of two (the 256 "MB" Infinity Cache is 256 MiB). Sizing the
    # footprint in decimal MB left the residency bar 4.86% off in the direction that admits a
    # too-small footprint -- i.e. a cache reading wearing a DRAM label. Rates stay decimal (TB/s),
    # because that is how bandwidth is quoted; the two conventions are kept apart on purpose.
    target_bytes = int(footprint_mb * MIB)
    n_groups = max(1, target_bytes // (runs_per_group * runlen))
    groups_per_prog = max(1, math.ceil(n_groups / nprog))
    n_groups = groups_per_prog * nprog            # exact, so the byte count is exact
    return {
        "run_bytes": runlen, "run_elems": run_elems, "stride_bytes": stride,
        "stride_elems": stride // 4, "nr": nr, "nw": nw,
        "n_groups": n_groups, "groups_per_prog": groups_per_prog, "nprog": nprog,
        "read_bytes": n_groups * nr * runlen,
        "write_bytes": n_groups * nw * runlen,
        "moved_bytes": n_groups * runs_per_group * runlen,
        "span_bytes": n_groups * runs_per_group * stride,
    }


def verify_plan(p, buf_bytes):
    """Distinct-byte audit. Returns a list of problems (empty = the plan is honest)."""
    bad = []
    if p["span_bytes"] > buf_bytes:
        bad.append(f"the address span {p['span_bytes']/MIB:.1f} MiB exceeds the allocated "
                   f"{buf_bytes/MIB:.1f} MiB -> the kernel would wrap and re-touch bytes it has "
                   f"already counted")
    slots = p["n_groups"] * (p["nr"] + p["nw"])
    if slots * p["run_bytes"] != p["moved_bytes"]:
        bad.append("slot count and moved bytes disagree -> the byte accounting is inconsistent")
    if p["stride_bytes"] < p["run_bytes"]:
        bad.append("stride < runlen -> runs overlap and the moved bytes are double-counted")
    return bad


def verdict(bw_tb_s, sku, footprint_mb, reference_tb_s=None, l2_margin=4.0, ref_tol=0.02):
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
    peak = sku.get("datasheet_hbm_tb_s") or sku.get("peak_hbm_tb_s")
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
            f"{bw_tb_s:.3f} TB/s EXCEEDS this SKU's datasheet peak {peak} TB/s -- no DRAM stream "
            f"can do that, so some of the footprint was served from cache (or the byte accounting "
            f"double-counts addresses). REJECTED: a too-high ceiling shrinks every gap computed "
            f"from it.")
    if llc_mb and footprint_mb < l2_margin * llc_mb:
        reasons.append(
            f"footprint {footprint_mb:.0f} MiB is under {l2_margin:g}x the {llc_mb} MiB "
            f"{llc_name} -- this measures cache, not DRAM. Raise --footprint-mb to "
            f">= {l2_margin*llc_mb:.0f}.")
    if not sku.get("mall_mb"):
        reasons.append(
            "this SKU has no established memory-side LLC capacity (sku.json mall_mb is null), so "
            "the residency bar above fell back to the L2 and is LOWER than it should be. Establish "
            "the capacity, or clear it empirically: sweep --footprint-mb upward and take the "
            "ceiling only from the plateau where it stops falling.")
    if reference_tb_s and bw_tb_s > reference_tb_s * (1 + ref_tol):
        reasons.append(
            f"{bw_tb_s:.3f} TB/s is above the LINEAR reference {reference_tb_s:.3f} TB/s measured "
            f"at the same mix and program count (+{ref_tol:.0%} tolerance). A strided / short-run "
            f"shape cannot beat a contiguous one, so this is cache reuse or a byte-accounting "
            f"error, not a higher ceiling.")
    return {"ok": not reasons, "reasons": reasons}


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
    def _stream(SRC, DST, groups_per_prog, n_groups,
                RUN_ELEMS: tl.constexpr, STRIDE_ELEMS: tl.constexpr,
                NR: tl.constexpr, NW: tl.constexpr):
        pid = tl.program_id(0)
        idx = tl.arange(0, RUN_ELEMS)
        acc = tl.zeros([RUN_ELEMS], dtype=tl.float32)
        for g in range(groups_per_prog):
            gid = pid * groups_per_prog + g
            if gid < n_groups:
                slot = gid * (NR + NW)
                for i in tl.static_range(NR):
                    acc += tl.load(SRC + (slot + i) * STRIDE_ELEMS + idx)
                for j in tl.static_range(NW):
                    tl.store(DST + (slot + NR + j) * STRIDE_ELEMS + idx, acc)
        if NW == 0:
            # keep the reads live: without a consumer the loads are dead code and the "read
            # bandwidth" would be the bandwidth of an empty kernel
            tl.store(DST + pid, tl.sum(acc, axis=0))
    return _stream


def measure(p, repeats, iters, warmup):
    import torch
    import triton
    kernel = _build_kernel()
    buf_bytes = p["span_bytes"]
    dev = "cuda"
    # A strided shape spans stride/runlen times its footprint, and BOTH buffers must cover the span.
    # Refuse before the allocator does: an OOM here reads as a tool bug, and a swapped/committed
    # allocation would silently change what is being timed.
    free, total = torch.cuda.mem_get_info()
    need = 2 * max(buf_bytes, p["nprog"] * 4)
    if need > 0.8 * free:
        sys.exit(f"[mem_bw_probe] this shape needs {need/1e9:.1f} GB (span {buf_bytes/1e9:.1f} GB x2 "
                 f"buffers) but only {free/1e9:.1f} GB is free. A stride/runlen ratio of "
                 f"{p['stride_bytes']//p['run_bytes']}x multiplies the span by that ratio -- lower "
                 f"--footprint-mb (keep it >= a few x L2) or probe the stride with a shorter run.")
    # content is irrelevant to a bandwidth probe, and randn on tens of GB costs minutes
    src = torch.empty(max(buf_bytes // 4, p["nprog"]), dtype=torch.float32, device=dev).fill_(1.0)
    dst = torch.empty_like(src)
    bad = verify_plan(p, src.numel() * 4)
    if bad:
        sys.exit("[mem_bw_probe] plan is not self-consistent:\n  - " + "\n  - ".join(bad))

    def run():
        kernel[(p["nprog"],)](src, dst, p["groups_per_prog"], p["n_groups"],
                              RUN_ELEMS=p["run_elems"], STRIDE_ELEMS=p["stride_elems"],
                              NR=p["nr"], NW=p["nw"])
    readings = []
    for _ in range(repeats):
        # do_bench flushes L2 between iterations; without that the first cell reads a warm cache
        ms = triton.testing.do_bench(run, warmup=warmup, rep=iters, return_mode="median")
        readings.append(p["moved_bytes"] / (ms * 1e-3) / 1e12)
    return readings


# --------------------------------------------------------------------------- CLI
def _tool_versions():
    out = {}
    for mod in ("torch", "triton"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- a missing module is recorded, not fatal
            out[mod] = None
    return out


def _emit(sku_name, sku, p, readings, requested_mb, label, out_json, reference_tb_s=None):
    rng = ceiling_range(readings)
    v = verdict(rng["hi_tb_s"], sku, p["moved_bytes"] / MIB, reference_tb_s)
    print(f"\n=== in-shape HBM ceiling: {sku_name} / {label} ===")
    print(f"  shape      runlen {p['run_bytes']} B, stride {p['stride_bytes']} B, "
          f"mix {p['nr']}r:{p['nw']}w, {p['nprog']} programs")
    print(f"  footprint  {p['moved_bytes']/MIB:.1f} MiB moved "
          f"({p['read_bytes']/MIB:.1f} read + {p['write_bytes']/MIB:.1f} write), "
          f"span {p['span_bytes']/MIB:.1f} MiB")
    print(f"  readings   " + ", ".join(f"{r:.3f}" for r in readings) + " TB/s")
    if not v["ok"]:
        print("  *** REJECTED ***")
        for r in v["reasons"]:
            print(f"    - {r}")
        print("  no ceiling returned. Fix the probe, do not use the number.")
    else:
        print(f"  IN-SHAPE CEILING = {rng['lo_tb_s']}-{rng['hi_tb_s']} TB/s "
              f"(median {rng['median_tb_s']}, spread {rng['spread_pct']}% over {rng['n']} reads)")
        _pin = sku.get("datasheet_hbm_tb_s") or sku.get("peak_hbm_tb_s")
        print(f"  datasheet peak   = {_pin} TB/s "
              f"-> reachable fraction {100*rng['median_tb_s']/_pin:.0f}%")
        print(f"  feed the budget: --measured-hbm-tb-s {rng['lo_tb_s']},{rng['hi_tb_s']}")
        print(f"  a gap smaller than the {rng['spread_pct']}% spread is inside the error bar and "
              f"is not a reason to open a round.")
        # An UNDER-saturating probe is dangerous in the same direction as an over-reading one:
        # a too-low ceiling also shrinks the computed gap and closes the campaign early.
        _pin = sku.get("datasheet_hbm_tb_s") or sku.get("peak_hbm_tb_s")
        frac = rng["median_tb_s"] / _pin if _pin else None
        if frac is not None and frac < 0.6:
            print(f"  WARNING: {100*frac:.0f}% of peak is low even for a shaped access. Either this "
                  f"shape really is\n    that slow, or THIS PROBE is under-saturating (too few "
                  f"programs, runs too short, not enough\n    independent streams in flight). Raise "
                  f"--nprog / --runlen and cross-check against a known-good\n    stream benchmark "
                  f"before citing it -- a ceiling measured too LOW shrinks the gap just as badly "
                  f"as one measured too high.")
        if abs(p["moved_bytes"] / MIB - requested_mb) / requested_mb > 0.02:
            print(f"  note: footprint rounded to {p['moved_bytes']/MIB:.1f} MiB from the requested "
                  f"{requested_mb:.1f} MiB (groups must divide across programs). Compare sweep "
                  f"points on the printed footprint, not the requested one.")
    rec = {"sku": sku_name, "label": label, "shape": p, "readings_tb_s": readings,
           "ceiling": rng, "accepted": v["ok"], "refusals": v["reasons"],
           "hbm_ceiling_source": "in-shape-probe",
           # the roofline basis pair: a ceiling from this record is an in-shape probe denominator
           # (it may gate/close together with a counters numerator); name the tool and versions.
           "denominator_basis": "in-shape probe",
           "tool": "mem_bw_probe.py", "tool_versions": _tool_versions(),
           "linear_reference_tb_s": reference_tb_s,
           "datasheet_peak_tb_s": sku.get("datasheet_hbm_tb_s") or sku.get("peak_hbm_tb_s"),
           "sku_peak_hbm_basis": sku.get("peak_hbm_basis")}
    if out_json:
        with open(out_json, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  wrote {out_json}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sku", help="SKU key from perf_knowledge/hardware/data/sku.json (for the "
                                  "refusals), e.g. MI355X")
    ap.add_argument("--footprint-mb", type=float, default=2048.0,
                    help="unique MiB MOVED per pass (binary, to match how cache capacity is "
                         "quoted); must be several x the MEMORY-SIDE LLC (sku.json mall_mb), not "
                         "the L2 -- see gate 2 (default 2048)")
    ap.add_argument("--runlen", type=int, default=4096,
                    help="contiguous bytes per run (multiple of the 128 B line; /4 a power of two)")
    ap.add_argument("--stride", type=int,
                    help="bytes between run starts (default = --runlen, i.e. a linear stream)")
    ap.add_argument("--rw-mix", default="1:0", help="read:write ratio, e.g. 1:0, 50:50, 40:60")
    ap.add_argument("--nprog", type=int, default=1980, help="programs over the footprint")
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
    ap.add_argument("--json", help="write the record here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if not a.sku:
        ap.error("--sku is required: the physical-limit and cache-residency refusals are not "
                 "optional, and both need the SKU's peak/L2 (or --selftest)")
    sku = load_sku(a.sku)
    nr, nw = rw_mix_to_runs(a.rw_mix)
    stride = a.stride or a.runlen

    def one(runlen, strd, nprog, label, out_json=None):
        """Measure one configuration, preceded by its LINEAR reference when it is a shaped access
        (gate 3: nothing shaped may read above a contiguous stream at the same mix/parallelism)."""
        ref = None
        if strd != runlen and not a.no_reference:
            rp = plan(a.footprint_mb, runlen, runlen, nr, nw, nprog)
            ref_readings = measure(rp, a.repeats, a.iters, a.warmup)
            ref = max(ref_readings)
            print(f"\n  [linear reference for {label}] "
                  + ", ".join(f"{r:.3f}" for r in ref_readings) + " TB/s")
        p = plan(a.footprint_mb, runlen, strd, nr, nw, nprog)
        return _emit(a.sku, sku, p, measure(p, a.repeats, a.iters, a.warmup),
                     a.footprint_mb, label, out_json, ref)

    recs = []
    if a.nprog_sweep:
        for n in [int(x) for x in a.nprog_sweep.split(",")]:
            recs.append(one(a.runlen, stride, n, f"nprog={n}"))
        print("\n  the ceiling is a function of parallelism: quote it WITH the program count.")
    elif a.runlen_sweep:
        for rl in [int(x) for x in a.runlen_sweep.split(",")]:
            # with no explicit --stride each entry stays LINEAR at its own run length. Carrying one
            # stride across the sweep would silently turn the short entries into strided shapes and
            # measure two variables at once.
            recs.append(one(rl, max(a.stride, rl) if a.stride else rl, a.nprog, f"runlen={rl}B"))
    else:
        recs.append(one(a.runlen, stride, a.nprog, f"{a.rw_mix} mix", a.json))
    if a.json and len(recs) > 1:
        with open(a.json, "w") as f:
            json.dump(recs, f, indent=2)
    return 1 if any(not r["accepted"] for r in recs) else 0


def _selftest():
    # rw-mix reduction: the RATIO is what matters, expressed in the smallest run pair
    assert rw_mix_to_runs("1:0") == (1, 0) and rw_mix_to_runs("0:1") == (0, 1)
    assert rw_mix_to_runs("50:50") == (1, 1)
    nr, nw = rw_mix_to_runs("40:60")
    assert (nr, nw) == (2, 3), (nr, nw)
    assert abs(nr / (nr + nw) - 0.4) < 1e-9
    for bad in ("1", "-1:2", "0:0"):
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
    for kwargs in (dict(runlen=4096, stride=2048), dict(runlen=4000, stride=4000),
                   dict(runlen=3 * 128, stride=3 * 128)):
        try:
            plan(1024.0, kwargs["runlen"], kwargs["stride"], 1, 0, 100)
            raise AssertionError(f"plan accepted {kwargs}")
        except ValueError:
            pass

    # the three gates, and what each one actually catches
    sku = {"peak_hbm_tb_s": 8.0, "l2_mb": 32, "mall_mb": 256}
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
    v_unk = verdict(5.31, {"peak_hbm_tb_s": 8.0, "l2_mb": 32}, 2048)
    assert not v_unk["ok"] and any("mall_mb is null" in r for r in v_unk["reasons"]), v_unk
    # gate 3: no shaped access beats a linear stream at the same mix/parallelism
    v_ref = verdict(5.90, sku, 2048, reference_tb_s=5.50)
    assert not v_ref["ok"] and "LINEAR reference" in v_ref["reasons"][0], v_ref
    assert verdict(5.55, sku, 2048, reference_tb_s=5.50)["ok"], "2% tolerance absorbs noise"
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
    print("[mem_bw_probe] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
