#!/usr/bin/env python3
"""Mechanical primitives for the `roofline` analysis skill.

OPTIONAL helpers only. The skill is defined by SKILL.md and an agent can execute it by hand; this
module exists so the boring parts (peak-table parsing, counter parsing, unit math) are not retyped.

Design rules, per knowledge/analysis_skills/INDEX.md:
  * no hard third-party dependency (stdlib only; torch is imported lazily and optionally)
  * every helper is defensive and returns a value or None -- it never raises at the caller
  * nothing here decides anything; the routing rules live in SKILL.md

Self-test:  python3 roofline_tools.py --selftest
"""
from __future__ import annotations

import json
import math
import os
import re

SKILL_NAME = "roofline"
SKILL_VERSION = "1"

# element size in bytes, keyed by the dtype spellings that show up in profiles/configs
_DTYPE_BYTES = {
    "fp4": 0.5, "float4": 0.5, "mxfp4": 0.5, "e2m1": 0.5,
    "fp8": 1, "float8": 1, "e4m3": 1, "e5m2": 1, "mxfp8": 1, "fp8_e4m3": 1, "int8": 1, "uint8": 1,
    "bf16": 2, "bfloat16": 2, "c10::bfloat16": 2, "fp16": 2, "float16": 2, "half": 2, "int16": 2,
    "fp32": 4, "float32": 4, "float": 4, "int32": 4,
    "fp64": 8, "float64": 8, "double": 8, "int64": 8, "long": 8,
}

# target_eff priors -- SKILL.md section 7 is the source of truth; keep the two in sync.
TARGET_EFF = {
    "gemm": 0.90,
    "moe": 0.90,
    "elementwise": 0.875,
    "attn": 0.50,
}

#: Only kernels big enough for a headroom estimate to change a decision are worth analysing.
#: Below this the Amdahl ceiling is under the noise band anyway, so modelling them adds failure
#: modes without adding information. Callers should pass the run's HEAD_THRESHOLD_PCT.
DEFAULT_MIN_PCT_GPU = 5.0
DEFAULT_TOP_N = 8

#: Kernel launch + scheduling overhead. A launch whose duration is within a small multiple of this
#: is timed by dispatch, not by its transfer or its math, so a roofline ratio is meaningless for it
#: (its lever is fusion / graph capture, not kernel tuning). Order-of-magnitude runtime constant,
#: not a per-model tuning knob -- override per box if measured.
LAUNCH_OVERHEAD_S = 5e-6
LATENCY_BOUND_FACTOR = 2.0

#: A roof (memory or compute) counts as the binding limiter only when the kernel actually gets near
#: it. Below this utilization on BOTH axes -- and above the dispatch floor -- the kernel is
#: latency/occupancy-bound, NOT bandwidth- or compute-bound. This is the single most common mislabel:
#: a small arithmetic intensity does not by itself make a kernel memory-bound.
UTIL_BOUND_THRESHOLD = 0.60

#: The only bound types this skill may emit. Anything else is a modelling escape and must degrade
#: to "unknown" rather than inventing a category the consumer has no routing rule for.
BOUND_TYPES = ("memory", "compute", "latency", "unknown")

#: How the ACHIEVED side of the ratio was obtained. Only hardware counters are a measurement:
#: `2*M*N*K` is what the op would cost if the kernel did exactly the arithmetic we assume, and
#: `experts_hit` is an expected value over a routing distribution nobody observed. Both are useful
#: priors and neither is a result. `mixed` = one axis counted, the other modelled.
MEASUREMENT_BASES = ("counters", "mixed", "model")


def select_entries(entries, min_pct_gpu=DEFAULT_MIN_PCT_GPU, top_n=DEFAULT_TOP_N):
    """Head-scoped selection: the entries worth a roofline estimate, biggest first.

    Everything below the bar is SKIPPED (absent from the artifact), not "degraded" -- a kernel too
    small to matter is not a modelling failure and should not read as one.
    """
    try:
        sel = [e for e in (entries or []) if float(e.get("pct_gpu_time") or 0) >= float(min_pct_gpu)]
    except (TypeError, ValueError, AttributeError):
        return []
    sel.sort(key=lambda e: -float(e.get("pct_gpu_time") or 0))
    return sel[:int(top_n)] if top_n else sel


def dtype_bytes(name, default=2):
    """Element size in bytes for a dtype spelling. Unknown -> `default` (never raises)."""
    if name is None:
        return default
    if isinstance(name, (int, float)):
        return float(name)
    key = str(name).strip().lower().lstrip("torch.")
    if key in _DTYPE_BYTES:
        return _DTYPE_BYTES[key]
    for k, v in _DTYPE_BYTES.items():          # substring fallback: "c10::BFloat16", "torch.float8_e4m3fnuz"
        if k in key:
            return v
    return default


# ---------------------------------------------------------------- peaks

def load_peaks(peaks_md_path, gfx):
    """Parse the ```yaml blocks in peaks.md and return the one matching `gfx`.

    Returns {"hbm_bw_bytes_s": float, "flops": {dtype: float}, "cu": int, "source": "table",
             "confidence": "high"} or None when the file or the section is absent.
    """
    try:
        with open(peaks_md_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError):
        return None

    for block in re.findall(r"```yaml\s*\n(.*?)```", text, re.S):
        if not re.search(r"^\s*gfx:\s*%s\s*$" % re.escape(str(gfx)), block, re.M):
            continue
        out = {"flops": {}, "source": "table", "confidence": "high"}
        in_flops = False
        for line in block.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            body = line.split("#", 1)[0].rstrip()
            if re.match(r"^\s*flops:\s*$", body):
                in_flops = True
                continue
            m = re.match(r"^(\s*)([A-Za-z0-9_]+):\s*(.*)$", body)
            if not m:
                continue
            indent, key, val = m.group(1), m.group(2), m.group(3).strip()
            if in_flops and len(indent) >= 2 and val:
                try:
                    out["flops"][key] = float(val)
                except ValueError:
                    pass
                continue
            in_flops = False
            if not val:
                continue
            try:
                out[key] = float(val) if re.match(r"^[-+0-9.eE]+$", val) else val
            except ValueError:
                out[key] = val
        if out.get("hbm_bw_bytes_s"):
            return out
    return None


def derive_peaks_from_props(device=0):
    """Fallback peaks from torch device properties. confidence='low' -- see peaks.md.

    The derived HBM figure is frequently wrong for HBM3/3E (understates the pin rate), which is
    exactly why the caller must treat this as display-only.
    """
    try:
        import torch  # noqa: PLC0415 - optional, lazily imported on purpose
        p = torch.cuda.get_device_properties(device)
    except Exception:
        return None
    try:
        bw = float(getattr(p, "memory_clock_rate", 0)) * 1e3 * (float(getattr(p, "memory_bus_width", 0)) / 8.0) * 2.0
    except Exception:
        bw = 0.0
    if bw <= 0:
        return None
    return {
        "hbm_bw_bytes_s": bw,
        "flops": {},
        "cu": int(getattr(p, "multi_processor_count", 0) or 0),
        "source": "derived",
        "confidence": "low",
    }


def peak_flops_for(peaks, dtype_name):
    """Peak FLOP/s for a dtype, falling back to the largest tabulated value. None if unknown."""
    if not peaks:
        return None
    flops = peaks.get("flops") or {}
    if not flops:
        return None
    key = str(dtype_name or "").strip().lower()
    for k, v in flops.items():
        if k and k in key:
            return float(v)
    return float(max(flops.values()))


# ---------------------------------------------------------------- op models

def experts_hit(num_experts, pairs):
    """Expected number of DISTINCT experts touched by `pairs` = M*top_k routed token-expert pairs.

    E*(1-(1-1/E)^pairs). At decode this is what decides MoE weight traffic.

    This is an EXPECTED VALUE over an assumed-uniform router, not an observation: a router with any
    real skew touches fewer experts, and one batch is not its mean. Bytes derived from it therefore
    go into `roofline_metrics` with `bytes_measured=False`, which is the default -- the resulting
    row is an annotation, and only `roofline_metrics_from_counters` can make it rankable.
    """
    try:
        E, n = float(num_experts), float(pairs)
        if E <= 0 or n <= 0:
            return 0.0
        return E * (1.0 - math.pow(1.0 - 1.0 / E, n))
    except (TypeError, ValueError, OverflowError):
        return 0.0


# ---------------------------------------------------------------- the metric

def _tag_basis(out, roof_axis, bytes_measured, flops_measured):
    """Record how the achieved side was obtained, and whether that makes the row rankable.

    Rankability is decided per AXIS, not per row: `roofline_pct` is achieved/peak on ONE roof, so
    what has to be counted is the quantity that roof is made of -- bytes for a memory-side verdict,
    FLOPs for a compute-side one. A memory-bound row whose bytes came from FETCH_SIZE/WRITE_SIZE is
    a measurement even if nobody counted its FLOPs; a compute-side row resting on `2*M*N*K` is not,
    however plausible the arithmetic looks.
    """
    out["bytes_measured"] = bool(bytes_measured)
    out["flops_measured"] = bool(flops_measured)
    if bytes_measured and flops_measured:
        out["measurement_basis"] = "counters"
    elif bytes_measured or flops_measured:
        out["measurement_basis"] = "mixed"
    else:
        out["measurement_basis"] = "model"
    axis_measured = flops_measured if roof_axis == "compute" else bytes_measured
    out["confidence"] = (
        "high" if out["measurement_basis"] == "counters"
        else "medium" if axis_measured else "low"
    )
    out["rankable"] = bool(
        axis_measured and not out.get("suspect") and out.get("headroom_class") != "unknown"
    )
    if not axis_measured:
        out["basis_note"] = (
            "achieved %s is an ESTIMATE, not a measurement -- display and annotate only, do not "
            "rank on it (SKILL.md section 5). Re-run with rocprofv3 counters via "
            "roofline_metrics_from_counters() to make this row rankable."
            % ("FLOPs" if roof_axis == "compute" else "bytes")
        )
    return out


def roofline_metrics(bytes_moved, flops, t_seconds, peak_bw, peak_flops, target_eff,
                     pct_gpu_time=0.0, launch_overhead_s=LAUNCH_OVERHEAD_S,
                     bytes_measured=False, flops_measured=False):
    """Core arithmetic of SKILL.md section 3 step 5. Returns a dict, or None on unusable input.

    `bytes_measured` / `flops_measured` state where the ACHIEVED side came from, and they default
    to False because an un-migrated caller is telling us nothing -- and "nothing" must read as an
    estimate, not as a measurement. They set `measurement_basis`, `confidence` and `rankable`; see
    `_tag_basis`. Prefer `roofline_metrics_from_counters`, which sets them from the counters.

    Two outcomes deliberately produce NO verdict (`headroom_class="unknown"`), because in both the
    ratio is not evidence about the kernel:
      * dispatch-bound -- the launch is timed by overhead, not by its transfer or its math;
      * infeasible (`raw_pct` outside (0,1]) -- the byte/FLOP model is wrong. A clamped 100% must
        never be read as "saturated"; that would turn a modelling failure into a routing decision.
    """
    try:
        b, f, t = float(bytes_moved or 0), float(flops or 0), float(t_seconds or 0)
        pbw, pfl, tgt = float(peak_bw or 0), float(peak_flops or 0), float(target_eff or 0)
    except (TypeError, ValueError):
        return None
    if t <= 0 or pbw <= 0 or tgt <= 0 or (b <= 0 and f <= 0):
        return None

    achieved_bw = b / t
    achieved_flops = (f / t) if f > 0 else 0.0
    ai = (f / b) if b > 0 else float("inf")
    ridge = (pfl / pbw) if (pfl > 0 and pbw > 0) else None

    # AI picks which roof the kernel is walking TOWARD (the ceiling it would hit if it stopped
    # stalling); utilization tells whether it is actually near that roof. Both are needed -- a small
    # AI does NOT by itself mean memory-bound. roofline_pct is measured on the AI-selected roof.
    compute_bound = bool(ridge is not None and pfl > 0 and ai > ridge)
    hbm_util = achieved_bw / pbw
    compute_util = (achieved_flops / pfl) if pfl > 0 else 0.0
    raw_pct = compute_util if compute_bound else hbm_util
    roof_axis = "compute" if compute_bound else "memory"

    out = {
        "bytes_est": b, "flops_est": f, "t_ms": t * 1e3,
        "achieved_bw_bytes_s": achieved_bw, "achieved_flops": achieved_flops,
        "hbm_util": hbm_util, "compute_util": compute_util,
        "arithmetic_intensity": ai, "ridge_point": ridge,
        "roofline_pct_raw": raw_pct, "target_eff": tgt, "suspect": False,
        # Which roof `roofline_pct` is measured against. `bound_type` is NOT a substitute: it can
        # read "latency" while the ratio is still taken on the memory or compute roof, and
        # `fold_cases` must not average two ratios that have different denominators.
        "roof_axis": roof_axis,
    }

    # (1) Dispatch-bound by time: the launch is timed by scheduling overhead, not by its own transfer
    # or math, so no roofline ratio applies. No verdict; the lever is fusion / graph capture.
    if t <= float(launch_overhead_s or 0) * LATENCY_BOUND_FACTOR:
        out.update(bound_type="latency", roofline_pct=min(max(raw_pct, 0.0), 1.0),
                   attainable_speedup=1.0, expected_e2e_gain_pct=0.0,
                   headroom_class="unknown",
                   note="per-launch time is within launch-overhead scale -> dispatch-bound; "
                        "roofline not applicable, lever is fusion / graph capture")
        return _tag_basis(out, roof_axis, bytes_measured, flops_measured)

    # (2) Infeasible: raw_pct outside (0,1] means the byte/FLOP model is wrong, NOT that the kernel is
    # at the wall -- so it must not yield a verdict. (A compute-axis >100% is usually an unvalidated
    # peak, e.g. the BF16 MFMA microbench reading ~2x low.) Clamp for display, refuse to classify, and
    # hand back the feasibility bound the model violated so stage C knows what to measure.
    if not (0.001 <= raw_pct <= 1.0):
        out.update(bound_type=roof_axis, roofline_pct=min(max(raw_pct, 0.0), 1.0),
                   attainable_speedup=1.0, expected_e2e_gain_pct=0.0,
                   headroom_class="unknown", suspect=True,
                   bytes_upper_bound=pbw * t, flops_upper_bound=(pfl * t) if pfl > 0 else None,
                   note="model infeasible (raw %.3f outside (0,1]) -> NOT a saturation verdict; "
                        "re-estimate with a tighter model or measure with counters (stage C)"
                        % raw_pct)
        return _tag_basis(out, roof_axis, bytes_measured, flops_measured)

    # (3) True limiter by utilization. If NEITHER roof is near its ceiling (and we already ruled out
    # the dispatch floor), the kernel is latency/occupancy-bound. It still has recoverable headroom --
    # keep the verdict, so a low-utilization head like paged attention still ranks by its headroom --
    # but the lever is occupancy / shorter dependency chains / fusion, NOT byte reduction (which only
    # helps a genuinely bandwidth-bound, high-util head).
    if compute_util < UTIL_BOUND_THRESHOLD and hbm_util < UTIL_BOUND_THRESHOLD:
        bound = "latency"
    else:
        bound = roof_axis

    attainable = max(1.0, tgt / raw_pct)
    out.update(bound_type=bound, roofline_pct=raw_pct, attainable_speedup=attainable,
               expected_e2e_gain_pct=float(pct_gpu_time or 0.0) * (1.0 - 1.0 / attainable),
               headroom_class=classify_headroom(raw_pct, tgt))
    return _tag_basis(out, roof_axis, bytes_measured, flops_measured)


def classify_headroom(roofline_pct, target_eff):
    """SKILL.md section 3 step 6.

    Banded against `target_eff`, NOT against the raw roofline: what matters is the distance to what
    a good implementation of this class can realistically reach. Within 10% of that target means
    tuning has nothing left to give (88% vs a 90% target is saturated, not "nearly there").
    """
    try:
        p, t = float(roofline_pct), float(target_eff)
    except (TypeError, ValueError):
        return "unknown"
    if p <= 0 or t <= 0:
        return "unknown"
    if p >= 0.9 * t:
        return "saturated"
    if p >= 0.6 * t:
        return "moderate"
    return "underperforming"


# ---------------------------------------------------------------- per-shape fold

def fold_cases(cases, target_eff, pct_gpu_time=0.0, min_axis_share=0.0):
    """Fold per-SHAPE rows of one kernel into per-axis summaries, weighted by deployment time.

    Why this exists
    ---------------
    A kernel name is not an operating point. `roofline_metrics` answers a question about ONE
    launch at ONE shape, and section 3 step 5 says so: "Compare bytes for ONE launch against ONE
    launch's `base_latency_ms`." But the input the skill was given violates that -- the profile's
    `base_latency_ms` is `total_us / count` over a whole phase (`parse_profile.py:199`), i.e. the
    mean over every shape the kernel ran, while the byte side is modelled from ONE representative
    shape (`_est_shape`, `parse_profile.py:203`). Numerator and denominator come from different
    operating points.

    The error has a FIXED SIGN, which is what makes it dangerous rather than merely imprecise. A
    prefill phase is typically a few very large chunks plus many small remainders: the modal shape
    is small, the mean time is dragged up by the large ones. Bytes(small) / mean_time(dragged up)
    understates achieved bandwidth, so `roofline_pct` reads low, so
    `attainable_speedup = target_eff / roofline_pct` reads HIGH. The skill systematically reports
    phantom headroom on exactly the kernels whose shape distribution is widest -- and
    `expected_e2e_gain_pct` then ranks optimisation budget by it. That is section 9.1's own
    complaint ("a modelling failure turning into a plan") arriving by a second route.

    So: compute one row per shape, each with its own bytes AND its own time, then fold here.

    The weighting is exact, not a heuristic
    ---------------------------------------
    With `pct_i = bytes_i / (t_i * peak)` and weight `w_i = calls_i * t_i`:

        sum(w_i * pct_i) / sum(w_i) = sum(calls_i * bytes_i) / (peak * sum(calls_i * t_i))

    The right-hand side is total bytes over total time over peak -- the true aggregate achieved
    ratio. Time-weighting the percentages IS the aggregate; it is not an approximation of it.

    That identity holds only while `peak` is common to every term, which is why rows are grouped by
    `roof_axis` and never folded across it. A memory-side ratio and a compute-side ratio have
    different denominators; averaging them yields a number with no physical meaning. Mixed input
    therefore returns SEVERAL summaries, one per axis, and the caller reports them side by side --
    the same shape as the skill's existing per-axis `rankable`.

    `calls` must come from the deployment trace
    -------------------------------------------
    It is the number of times the SERVER runs that shape, never the number of replays a unit test
    happened to do. Weighting by a test's own loop count would let the harness's configuration
    decide the production verdict. Rows without a positive `calls` are dropped and counted in
    `excluded`, because a silent default of 1 would quietly flatten the distribution this function
    exists to respect.

    Only rows with a verdict may be folded
    --------------------------------------
    Dispatch-bound rows and L3-infeasible rows carry `headroom_class="unknown"` precisely because
    their ratio is not evidence about the kernel. Folding them in at their clamped value would
    manufacture saturation out of a measurement failure. They are excluded and their share of the
    weight is reported, so a summary resting on 40% of the traffic cannot be mistaken for one
    resting on all of it.

    Parameters
    ----------
    cases : iterable of dicts, each `{"row": <roofline_metrics output>, "calls": <int>, ...}`.
        Any other keys (e.g. "name", "m") are carried through to the summary's `members`.
    target_eff : the class target, as passed to `roofline_metrics`.
    pct_gpu_time : the kernel's share of GPU time. Split across axes in proportion to the weight
        each axis carries, so the per-axis `expected_e2e_gain_pct` sum stays within the kernel's
        actual budget instead of each axis claiming all of it.
    min_axis_share : drop axis groups holding less than this fraction of the total weight (0..1).

    Returns `{"axes": [...], "excluded": {...}, "total_weight_ms": float}`, or None on bad input.
    """
    try:
        tgt = float(target_eff or 0)
        pct = float(pct_gpu_time or 0)
    except (TypeError, ValueError):
        return None
    if tgt <= 0:
        return None

    kept, excluded = [], []
    total_w = 0.0
    for c in (cases or []):
        if not isinstance(c, dict):
            continue
        row = c.get("row")
        try:
            calls = float(c.get("calls") or 0)
            t_ms = float((row or {}).get("t_ms") or 0)
        except (TypeError, ValueError, AttributeError):
            calls, t_ms = 0.0, 0.0
        w = calls * t_ms
        if not isinstance(row, dict) or calls <= 0 or t_ms <= 0:
            excluded.append({"case": {k: v for k, v in c.items() if k != "row"},
                             "weight_ms": max(w, 0.0),
                             "reason": "no deployment calls" if calls <= 0 else "no per-shape time"})
            total_w += max(w, 0.0)
            continue
        total_w += w
        if row.get("headroom_class") == "unknown" or row.get("suspect"):
            excluded.append({"case": {k: v for k, v in c.items() if k != "row"}, "weight_ms": w,
                             "reason": row.get("note") or "no verdict (%s)" % row.get("headroom_class")})
            continue
        kept.append((c, row, w))

    if total_w <= 0:
        return None

    axes = {}
    for c, row, w in kept:
        ax = row.get("roof_axis") or ("compute" if row.get("bound_type") == "compute" else "memory")
        g = axes.setdefault(ax, {"w": 0.0, "wp": 0.0, "bytes": 0.0, "flops": 0.0,
                                 "t_ms": 0.0, "calls": 0.0, "members": []})
        g["w"] += w
        g["wp"] += w * float(row.get("roofline_pct") or 0.0)
        calls = float(c.get("calls") or 0)
        g["bytes"] += calls * float(row.get("bytes_est") or 0.0)
        g["flops"] += calls * float(row.get("flops_est") or 0.0)
        g["t_ms"] += w
        g["calls"] += calls
        g["members"].append({
            **{k: v for k, v in c.items() if k != "row"},
            "roofline_pct": row.get("roofline_pct"), "bound_type": row.get("bound_type"),
            "t_ms": row.get("t_ms"), "weight_ms": w,
            "rankable": row.get("rankable"), "confidence": row.get("confidence"),
        })

    out_axes = []
    for ax, g in sorted(axes.items(), key=lambda kv: -kv[1]["w"]):
        share = g["w"] / total_w
        if share < float(min_axis_share or 0.0):
            continue
        folded = g["wp"] / g["w"]
        # attainable_speedup and expected_e2e_gain_pct contain 1/pct, so they are NOT linear in the
        # ratio and must be recomputed from the folded value -- folding them directly would be a
        # different (and wrong) number. Same reason `classify_headroom` is re-run rather than voted.
        attainable = max(1.0, tgt / folded) if folded > 0 else 1.0
        # Each axis may only claim the slice of the kernel's GPU time it actually accounts for.
        pct_here = pct * share
        out_axes.append({
            "roof_axis": ax,
            "n_cases": len(g["members"]),
            "weight_ms": g["w"],
            "weight_share": share,
            "deployment_calls": g["calls"],
            "roofline_pct": folded,
            "target_eff": tgt,
            "attainable_speedup": attainable,
            "pct_gpu_time_share": pct_here,
            "expected_e2e_gain_pct": pct_here * (1.0 - 1.0 / attainable),
            "headroom_class": classify_headroom(folded, tgt),
            "rankable": all(m.get("rankable") for m in g["members"]),
            "confidence": ("high" if all(m.get("confidence") == "high" for m in g["members"])
                           else "low" if any(m.get("confidence") == "low" for m in g["members"])
                           else "medium"),
            "aggregate_bytes": g["bytes"],
            "aggregate_flops": g["flops"],
            "members": sorted(g["members"], key=lambda m: -m["weight_ms"]),
        })

    ex_w = sum(e["weight_ms"] for e in excluded)
    return {
        "axes": out_axes,
        "total_weight_ms": total_w,
        "excluded": {
            "n": len(excluded),
            "weight_ms": ex_w,
            "weight_share": ex_w / total_w,
            "cases": excluded,
        },
        "note": ("folded per shape, weighted by deployment calls x per-shape time; axes are NOT "
                 "combined because roofline_pct is a ratio against a different peak on each"),
    }


# ---------------------------------------------------------------- counters (stage C)

#: rocprofv3 counters to request. FETCH_SIZE/WRITE_SIZE are in KiB. fp8 has no MfmaFlopsF8 on
#: current builds -- SQ_INSTS_VALU_MFMA_MOPS_F8 stands in. Availability varies by ROCm build;
#: probe with `rocprofv3 --list-avail` and drop whatever is missing (degrades to L4, never fatal).
COUNTERS = ["FETCH_SIZE", "WRITE_SIZE", "MfmaFlops", "MfmaFlopsBF16", "MfmaFlopsF16",
            "SQ_INSTS_VALU_MFMA_MOPS_F8", "MemUnitStalled", "MfmaUtil", "OccupancyPercent"]


def parse_counter_csv(path, kernel_substr=None):
    """Sum rocprofv3 counter-collection CSV rows into {counter: value}.

    Tolerates the several column spellings rocprofv3 has shipped. Returns {} on any problem.
    """
    import csv  # noqa: PLC0415 - stdlib, kept local so a broken import can't kill module load
    out = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                low = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
                name = low.get("kernel_name") or low.get("name") or low.get("kernel") or ""
                if kernel_substr and kernel_substr.lower() not in name.lower():
                    continue
                ctr = (low.get("counter_name") or low.get("counter") or "").strip()
                val = low.get("counter_value") or low.get("value") or ""
                if not ctr:
                    continue
                try:
                    out[ctr] = out.get(ctr, 0.0) + float(val)
                except ValueError:
                    continue
    except (OSError, UnicodeDecodeError, csv.Error):
        return {}
    return out


def bytes_from_counters(counters):
    """(FETCH_SIZE + WRITE_SIZE) KiB -> bytes. None when neither counter was collected."""
    if not counters:
        return None
    fetch, write = counters.get("FETCH_SIZE"), counters.get("WRITE_SIZE")
    if fetch is None and write is None:
        return None
    return (float(fetch or 0.0) + float(write or 0.0)) * 1024.0


def flops_from_counters(counters, mfma_flops_per_mop_f8=512.0):
    """Measured FLOPs from MFMA counters. None when no usable counter is present.

    `MfmaFlops*` are already FLOP counts. fp8 only exposes an MFMA *ops* count, so it needs the
    per-instruction FLOP factor of the MFMA shape in play -- pass the right one for the kernel.
    """
    if not counters:
        return None
    for key in ("MfmaFlops", "MfmaFlopsBF16", "MfmaFlopsF16", "MfmaFlopsF32"):
        if counters.get(key):
            return float(counters[key])
    mops = counters.get("SQ_INSTS_VALU_MFMA_MOPS_F8")
    if mops:
        return float(mops) * float(mfma_flops_per_mop_f8)
    return None


def roofline_metrics_from_counters(counters, t_seconds, peak_bw, peak_flops, target_eff,
                                   pct_gpu_time=0.0, mfma_flops_per_mop_f8=512.0,
                                   launch_overhead_s=LAUNCH_OVERHEAD_S,
                                   bytes_est=None, flops_est=None):
    """Stage C: the same metric, with the achieved side COUNTED instead of assumed.

    This is the path SKILL.md section 5 calls the high-confidence one, and until now nothing
    connected the counter helpers above to the metric below them -- so every published row was an
    estimate wearing a measurement's clothes.

    Whichever side the counters do not carry falls back to `bytes_est` / `flops_est` (the analytic
    model) and is marked unmeasured, so a partial collection still yields a row and the row still
    says which half of it was counted. Returns None when neither side is usable at all, which is a
    caller-visible "stay at stage A/B", not a silent zero.
    """
    measured_bytes = bytes_from_counters(counters)
    measured_flops = flops_from_counters(counters, mfma_flops_per_mop_f8)
    b = measured_bytes if measured_bytes is not None else bytes_est
    f = measured_flops if measured_flops is not None else flops_est
    if not b and not f:
        return None
    return roofline_metrics(
        b or 0.0, f or 0.0, t_seconds, peak_bw, peak_flops, target_eff,
        pct_gpu_time=pct_gpu_time, launch_overhead_s=launch_overhead_s,
        bytes_measured=measured_bytes is not None,
        flops_measured=measured_flops is not None,
    )


# ---------------------------------------------------------------- self-test

def _selftest():
    """Reproduces the SKILL.md section 9 worked example from real measured numbers."""
    here = os.path.dirname(os.path.abspath(__file__))
    ok = True

    peaks = load_peaks(os.path.join(here, "peaks.md"), "gfx950")
    assert peaks and abs(peaks["hbm_bw_bytes_s"] - 8.0e12) < 1e9, peaks
    assert abs(peak_flops_for(peaks, "fp8") - 5.0e15) < 1e12, peaks["flops"]
    assert load_peaks(os.path.join(here, "peaks.md"), "gfxNOPE") is None      # L1 path
    print("peaks: gfx950 %.2f TB/s, fp8 %.2f PFLOP/s; unknown gfx -> None  OK"
          % (peaks["hbm_bw_bytes_s"] / 1e12, peak_flops_for(peaks, "fp8") / 1e15))

    E, M, tk, H, I, L = 256, 64, 8, 2048, 512, 40
    hit = experts_hit(E, M * tk)
    assert 215 <= hit <= 225, hit
    wbytes = hit * (2 * I * H + H * I) * dtype_bytes("fp8")
    t_layer = 2 * 49.471e-6                                   # 80 launches/step over 40 layers
    moe = roofline_metrics(wbytes, 2 * M * tk * (2 * I * H + H * I), t_layer,
                           peaks["hbm_bw_bytes_s"], peak_flops_for(peaks, "fp8"),
                           TARGET_EFF["moe"], pct_gpu_time=26.45)
    print("MoE   : hit %.0f/%d, %.0f MB/layer -> %.0f%% of roofline, %s, %.3fx, +%.2f%% e2e (%s)"
          % (hit, E, wbytes / 1e6, 100 * moe["roofline_pct"], moe["bound_type"],
             moe["attainable_speedup"], moe["expected_e2e_gain_pct"], moe["headroom_class"]))
    ok &= (moe["bound_type"] == "memory" and moe["headroom_class"] == "saturated"
           and 0.85 <= moe["roofline_pct"] <= 0.92 and moe["expected_e2e_gain_pct"] < 1.0)

    B, S, kvh, hd = 64, 1024 + 1024 // 2, 2, 256
    attn = roofline_metrics(B * S * kvh * hd * 2 * dtype_bytes("bf16"),
                            2 * B * 16 * S * hd * 2, 141.112e-6,
                            peaks["hbm_bw_bytes_s"], peak_flops_for(peaks, "bf16"),
                            TARGET_EFF["attn"], pct_gpu_time=8.86)
    print("Attn  : %.0f%% of roofline (hbm_util %.0f%%, compute_util %.1f%%), %s, %.2fx, +%.2f%% e2e (%s)"
          % (100 * attn["roofline_pct"], 100 * attn["hbm_util"], 100 * attn["compute_util"],
             attn["bound_type"], attn["attainable_speedup"], attn["expected_e2e_gain_pct"],
             attn["headroom_class"]))
    # low util on both axes above the dispatch floor -> latency/occupancy-bound, but the verdict is
    # KEPT (real headroom) so the ranking inversion below survives.
    ok &= (attn["bound_type"] == "latency" and attn["headroom_class"] == "underperforming"
           and attn["attainable_speedup"] > 1.4
           and attn["expected_e2e_gain_pct"] > moe["expected_e2e_gain_pct"])

    # the whole point: the two rankings disagree, and roofline is the one that matched reality
    ok &= (26.45 > 8.86) and (attn["expected_e2e_gain_pct"] > moe["expected_e2e_gain_pct"])
    print("Rank  : by pct -> MoE first; by expected gain -> Attn first (measured: MoE -0.064%% e2e, "
          "Attn 1.56x isolated)  OK")

    # L3 sanity band: all-expert bytes overshoot the roofline -> clamped + suspect, not discarded
    allx = roofline_metrics(E * (2 * I * H + H * I) * 1, 1.0, t_layer, peaks["hbm_bw_bytes_s"],
                            peak_flops_for(peaks, "fp8"), TARGET_EFF["moe"], pct_gpu_time=26.45)
    assert allx["suspect"] and allx["roofline_pct"] <= 1.0, allx
    print("L3    : all-expert bytes -> raw %.2f clamped to %.2f, suspect=True  OK"
          % (allx["roofline_pct_raw"], allx["roofline_pct"]))

    # Basis: the two rows above are analytic (2*M*N*K + an assumed router), so neither may rank.
    assert moe["measurement_basis"] == "model" and moe["rankable"] is False, moe
    assert attn["measurement_basis"] == "model" and attn["rankable"] is False, attn
    # The same MoE launch with its traffic actually counted: memory-side verdict, bytes measured.
    ctr = {"FETCH_SIZE": wbytes / 1024.0, "WRITE_SIZE": 0.0}
    measured = roofline_metrics_from_counters(
        ctr, t_layer, peaks["hbm_bw_bytes_s"], peak_flops_for(peaks, "fp8"), TARGET_EFF["moe"],
        pct_gpu_time=26.45, flops_est=2 * M * tk * (2 * I * H + H * I))
    assert measured["bytes_measured"] and not measured["flops_measured"], measured
    assert measured["measurement_basis"] == "mixed" and measured["bound_type"] == "memory"
    assert measured["rankable"] is True and measured["confidence"] == "medium", measured
    ok &= abs(measured["roofline_pct"] - moe["roofline_pct"]) < 1e-9   # same number, now earned
    print("Basis : modelled MoE/Attn rankable=False; counted MoE rankable=True (%s, %s)  OK"
          % (measured["measurement_basis"], measured["confidence"]))
    assert roofline_metrics_from_counters({}, t_layer, peaks["hbm_bw_bytes_s"], 1e15, 0.9) is None

    # ---- per-shape fold (section 3a)
    # A GEMM whose prefill M spans 1936..32768: the modal shape is the small one, the mean time is
    # set by the large one. This is the distribution that makes the single-row treatment wrong.
    # The op is a memory-bound fused residual+norm over M rows, NOT the GEMM: at N=2624/K=6144 a
    # bf16 GEMM has AI ~940 against a ridge of 312, so it walks toward the COMPUTE roof and a
    # bytes-derived time drives it past the FLOP peak -- correctly rejected as infeasible (L3).
    # Testing the memory-axis fold needs an op that is genuinely on the memory roof; the compute
    # roof gets its own case in (d).
    pk, pf = peaks["hbm_bw_bytes_s"], peak_flops_for(peaks, "bf16")
    shapes = [(1936, 90), (8192, 14), (32768, 10)]              # (M, deployment calls)
    H, eb = 6144, dtype_bytes("bf16")
    fold_in = []
    for m, calls in shapes:
        by = 3.0 * m * H * eb                                   # read x, read residual, write y
        fl = 4.0 * m * H                                        # AI ~0.67, well under the ridge
        t = by / (0.55 * pk)                                    # a fixed 55% of peak at every shape
        r = roofline_metrics(by, fl, t, pk, pf, TARGET_EFF["elementwise"], pct_gpu_time=6.98,
                             bytes_measured=True, flops_measured=True)
        fold_in.append({"name": "M%d" % m, "m": m, "calls": calls, "row": r})
    fold = fold_cases(fold_in, TARGET_EFF["elementwise"], pct_gpu_time=6.98)
    mem = [a for a in fold["axes"] if a["roof_axis"] == "memory"][0]
    # (a) EXACTNESS: the call-weighted mean of the per-shape ratios equals total bytes / total time
    #     / peak. This is the identity the docstring claims; if it ever stops holding, the weight is
    #     wrong, not the arithmetic.
    agg = sum(c["calls"] * c["row"]["bytes_est"] for c in fold_in) / (
        sum(c["calls"] * c["row"]["t_ms"] * 1e-3 for c in fold_in) * pk)
    assert abs(mem["roofline_pct"] - agg) < 1e-9, (mem["roofline_pct"], agg)
    assert abs(mem["roofline_pct"] - 0.55) < 1e-9, mem["roofline_pct"]
    assert mem["weight_share"] == 1.0 and fold["excluded"]["n"] == 0
    print("Fold  : 3 shapes M=1936/8192/32768 -> %.4f folded == %.4f aggregate bytes/time  OK"
          % (mem["roofline_pct"], agg))

    # (b) the fold must MOVE the answer away from the modal-shape-only reading. Same weights, but a
    #     kernel that is efficient when large and poor when small: one row per shape says 45%, the
    #     modal shape alone says 30% -- a 1.5x difference in reported headroom.
    skew = []
    for (m, calls), eff in zip(shapes, (0.30, 0.60, 0.80)):
        by = 3.0 * m * H * eb
        r = roofline_metrics(by, 4.0 * m * H, by / (eff * pk), pk, pf, TARGET_EFF["elementwise"],
                             pct_gpu_time=6.98, bytes_measured=True, flops_measured=True)
        skew.append({"name": "M%d" % m, "calls": calls, "row": r})
    sk = [a for a in fold_cases(skew, TARGET_EFF["elementwise"], pct_gpu_time=6.98)["axes"]
          if a["roof_axis"] == "memory"][0]
    modal = skew[0]["row"]["roofline_pct"]
    ok &= sk["roofline_pct"] > modal * 1.3
    print("Fold  : modal-shape-only %.0f%% vs folded %.0f%% -> single-row reading overstates "
          "headroom by %.2fx  OK" % (100 * modal, 100 * sk["roofline_pct"],
                                     (TARGET_EFF["elementwise"] / modal) / sk["attainable_speedup"]))

    # (c) no-verdict rows are EXCLUDED and their share reported, never folded at their clamped value
    with_bad = list(fold_in) + [
        {"name": "dispatch_bound", "calls": 500,
         "row": roofline_metrics(1024, 1024, 1e-6, pk, pf, TARGET_EFF["elementwise"],
                                 bytes_measured=True, flops_measured=True)},
        {"name": "no_calls", "calls": 0, "row": fold_in[0]["row"]},
    ]
    fb = fold_cases(with_bad, TARGET_EFF["elementwise"], pct_gpu_time=6.98)
    assert fb["excluded"]["n"] == 2, fb["excluded"]
    assert {e["case"]["name"] for e in fb["excluded"]["cases"]} == {"dispatch_bound", "no_calls"}
    memb = [a for a in fb["axes"] if a["roof_axis"] == "memory"][0]
    assert abs(memb["roofline_pct"] - 0.55) < 1e-9        # the bad rows did not move the ratio
    assert memb["weight_share"] < 1.0 and fb["excluded"]["weight_share"] > 0
    print("Fold  : dispatch-bound + zero-calls rows excluded (%.1f%% of weight), ratio unmoved  OK"
          % (100 * fb["excluded"]["weight_share"]))

    # (d) axes are NEVER combined: a compute-bound shape and a memory-bound shape come back as two
    #     summaries. Averaging them would divide by two different peaks in one number.
    mix = [fold_in[0], {"name": "compute_heavy", "calls": 50,
                        "row": roofline_metrics(1e6, 8e11, 1e-3, pk, pf, TARGET_EFF["elementwise"],
                                                pct_gpu_time=6.98, bytes_measured=True,
                                                flops_measured=True)}]
    mx = fold_cases(mix, TARGET_EFF["elementwise"], pct_gpu_time=6.98)
    assert {a["roof_axis"] for a in mx["axes"]} == {"memory", "compute"}, mx["axes"]
    assert abs(sum(a["pct_gpu_time_share"] for a in mx["axes"]) - 6.98) < 1e-9   # budget conserved
    ok &= len(mx["axes"]) == 2
    print("Fold  : mixed roofs -> %d separate axis summaries, pct_gpu_time split %s, never averaged"
          "  OK" % (len(mx["axes"]), "/".join("%.2f" % a["pct_gpu_time_share"] for a in mx["axes"])))
    assert fold_cases([], TARGET_EFF["elementwise"]) is None
    assert fold_cases(None, 0) is None

    # degradation: unusable inputs return None instead of raising
    for bad in [(0, 0, 1e-6, 1e12, 1e15, 0.9), (1, 1, 0, 1e12, 1e15, 0.9), (None, None, None, None, None, None)]:
        assert roofline_metrics(*bad) is None, bad
    assert dtype_bytes("who-knows") == 2 and parse_counter_csv("/nonexistent") == {}
    assert bytes_from_counters({}) is None and flops_from_counters({}) is None
    assert bytes_from_counters({"FETCH_SIZE": 1024.0, "WRITE_SIZE": 1024.0}) == 2 * 1024 * 1024
    print("L2/L4/L5: unusable input -> None/{} without raising  OK")

    print("\nSELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="reproduce the SKILL.md worked example")
    ap.add_argument("--peaks", metavar="GFX", help="print the peak table entry for GFX as JSON")
    a = ap.parse_args()
    if a.peaks:
        p = load_peaks(os.path.join(os.path.dirname(os.path.abspath(__file__)), "peaks.md"), a.peaks)
        print(json.dumps(p or derive_peaks_from_props() or {"error": "no peaks for %s" % a.peaks}, indent=2))
        raise SystemExit(0)
    raise SystemExit(_selftest() if a.selftest else ap.print_help())
