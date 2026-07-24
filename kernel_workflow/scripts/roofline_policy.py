#!/usr/bin/env python3
"""Deterministic roofline classification and comparison policy.

The module intentionally contains no file, process, or environment access.  It
accepts plain dictionaries so collectors and unit tests can use the same policy.
Empirical peaks are the decision basis; specification peaks are retained only
as context and never silently substituted for a missing empirical peak.
"""

from __future__ import division

import math


POLICY_VERSION = 1
DEFAULT_SATURATION_PCT = 60.0
EMPIRICAL_PEAK_REL_TOLERANCE = 0.05
SPECIALTIES = ("algorithm", "memory", "compute", "host_runtime")


def _number(value):
    """Return a finite float, or None for missing/non-numeric values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_number(mapping, names):
    for name in names:
        value = _number(mapping.get(name))
        if value is not None:
            return value
    return None


def compute_roofline_efficiency(metrics):
    """Compute empirical roofline efficiency without mutating *metrics*.

    Returned fields distinguish the attainable empirical roof from the optional
    specification-HBM reference.  GFLOP/s divided by GB/s is FLOP/byte, so no
    unit scale is required for the ridge or sloped memory roof.
    """
    performance = _first_number(
        metrics, ("performance_gflops", "compute_actual_gflops", "performance")
    )
    ai_hbm = _first_number(metrics, ("ai_hbm", "arithmetic_intensity_hbm"))
    compute_peak = _first_number(
        metrics,
        ("compute_empirical_peak_gflops", "empirical_compute_peak_gflops"),
    )
    hbm_peak = _first_number(
        metrics, ("hbm_empirical_peak_gbps", "empirical_hbm_peak_gbps")
    )
    hbm_spec_peak = _first_number(
        metrics, ("hbm_spec_peak_gbps", "spec_hbm_peak_gbps")
    )

    ridge = None
    empirical_roof = None
    efficiency = None
    headroom = None
    if compute_peak is not None and compute_peak > 0 and hbm_peak is not None and hbm_peak > 0:
        ridge = compute_peak / hbm_peak
        if ai_hbm is not None and ai_hbm >= 0:
            empirical_roof = min(compute_peak, ai_hbm * hbm_peak)
            if performance is not None and empirical_roof > 0:
                efficiency = 100.0 * performance / empirical_roof
                headroom = empirical_roof / performance if performance > 0 else None

    spec_hbm_roof = None
    spec_hbm_efficiency = None
    if ai_hbm is not None and ai_hbm >= 0 and hbm_spec_peak is not None and hbm_spec_peak > 0:
        spec_hbm_roof = ai_hbm * hbm_spec_peak
        if compute_peak is not None and compute_peak > 0:
            spec_hbm_roof = min(spec_hbm_roof, compute_peak)
        if performance is not None and spec_hbm_roof > 0:
            spec_hbm_efficiency = 100.0 * performance / spec_hbm_roof

    return {
        "ai_ridge_empirical": ridge,
        "roofline_empirical_ceiling_gflops": empirical_roof,
        "roofline_efficiency_pct": efficiency,
        "headroom_ratio": headroom,
        "peak_basis": "empirical" if empirical_roof is not None else "unavailable",
        "hbm_empirical_peak_gbps": hbm_peak,
        "hbm_spec_peak_gbps": hbm_spec_peak,
        "roofline_spec_hbm_ceiling_gflops": spec_hbm_roof,
        "roofline_spec_hbm_efficiency_pct": spec_hbm_efficiency,
    }


def classify_theoretical_bound(ai_hbm, ai_ridge_empirical):
    """Classify which side of the empirical HBM ridge contains the point."""
    ai_value = _number(ai_hbm)
    ridge = _number(ai_ridge_empirical)
    if ai_value is None or ridge is None or ai_value < 0 or ridge <= 0:
        return "unknown"
    return "memory_side" if ai_value < ridge else "compute_side"


def classify_observed_limit(
    compute_util_pct,
    hbm_util_pct,
    l2_util_pct=None,
    l1_util_pct=None,
    lds_util_pct=None,
    overhead_bound=False,
    saturation_pct=DEFAULT_SATURATION_PCT,
):
    """Classify observed saturation using the policy's explicit priority order."""
    threshold = _number(saturation_pct)
    if threshold is None or threshold < 0:
        raise ValueError("saturation_pct must be a non-negative finite number")
    if overhead_bound:
        return "overhead"

    compute = _number(compute_util_pct)
    hbm = _number(hbm_util_pct)
    l2 = _number(l2_util_pct)
    l1 = _number(l1_util_pct)
    lds = _number(lds_util_pct)
    if all(value is None for value in (compute, hbm, l2, l1, lds)):
        return "unknown"

    if lds is not None and lds >= threshold:
        return "lds"
    if compute is not None and hbm is not None and compute >= threshold and hbm >= threshold:
        return "balanced"
    if compute is not None and compute >= threshold:
        return "compute"
    if hbm is not None and hbm >= threshold:
        return "hbm"
    if (l2 is not None and l2 >= threshold) or (l1 is not None and l1 >= threshold):
        return "cache"
    return "latency_occupancy"


def recommend_optimization(theoretical_bound, observed_limit, roofline_efficiency_pct=None):
    """Return deterministic specialist and lever recommendations."""
    efficiency = _number(roofline_efficiency_pct)
    recommendations = {
        "overhead": (
            ["host_runtime"],
            ["reduce launches", "remove host synchronization", "use graph replay or fusion"],
        ),
        "lds": (
            ["memory", "algorithm"],
            ["reduce LDS traffic", "eliminate bank conflicts", "retile shared-memory staging"],
        ),
        "balanced": (
            ["algorithm", "memory", "compute"],
            ["raise arithmetic intensity", "co-optimize data movement and instruction scheduling"],
        ),
        "compute": (
            ["compute"],
            ["improve MFMA/VALU issue efficiency", "retile for occupancy", "reduce instruction overhead"],
        ),
        "hbm": (
            ["memory"],
            ["reduce HBM bytes", "coalesce accesses", "increase cache reuse or fuse operations"],
        ),
        "cache": (
            ["memory"],
            ["improve cache locality", "reduce cache-line waste", "retile for L1/L2 reuse"],
        ),
        "latency_occupancy": (
            ["algorithm", "compute"],
            ["increase occupancy", "expose instruction-level parallelism", "reduce serial dependencies"],
        ),
        "no_fp_work": (
            ["algorithm", "memory"],
            ["optimize the relevant integer or data-movement path", "remove redundant work"],
        ),
        "unknown": (
            [],
            ["collect complete roofline and utilization metrics"],
        ),
    }
    specialties, levers = recommendations.get(
        observed_limit, recommendations["unknown"]
    )
    specialties = list(specialties)
    levers = list(levers)

    if theoretical_bound == "memory_side" and observed_limit not in (
        "overhead",
        "hbm",
        "cache",
        "lds",
        "no_fp_work",
        "unknown",
    ):
        levers.insert(0, "increase arithmetic intensity")
    elif theoretical_bound == "compute_side" and observed_limit == "latency_occupancy":
        if "compute" not in specialties:
            specialties.append("compute")

    specialties = [item for item in specialties if item in SPECIALTIES]
    specialties = list(dict.fromkeys(specialties))
    levers = list(dict.fromkeys(levers))
    if observed_limit == "unknown" or theoretical_bound == "unknown":
        confidence = "low"
    elif efficiency is None:
        confidence = "medium"
    elif efficiency >= 80:
        confidence = "high"
    elif efficiency >= 40:
        confidence = "medium"
    else:
        confidence = "medium"
    return {
        "recommended_specialties": specialties,
        "recommended_levers": levers,
        "confidence": confidence,
    }


def build_classification(metrics, overhead_bound=False, saturation_pct=DEFAULT_SATURATION_PCT):
    """Build a complete classification, evidence, and recommendation record."""
    derived = compute_roofline_efficiency(metrics)
    ai_hbm = _first_number(metrics, ("ai_hbm", "arithmetic_intensity_hbm"))
    ridge = _first_number(metrics, ("ai_ridge_empirical",))
    if ridge is None:
        ridge = derived["ai_ridge_empirical"]
    theoretical = classify_theoretical_bound(ai_hbm, ridge)

    compute_util = _first_number(
        metrics, ("compute_utilization_pct", "compute_util_pct")
    )
    hbm_util = _first_number(metrics, ("hbm_utilization_pct", "hbm_util_pct"))
    l2_util = _first_number(metrics, ("l2_utilization_pct", "l2_util_pct"))
    l1_util = _first_number(metrics, ("l1_utilization_pct", "l1_util_pct"))
    lds_util = _first_number(metrics, ("lds_utilization_pct", "lds_util_pct"))
    explicit_no_fp = bool(metrics.get("no_fp_work", False))
    observed = classify_observed_limit(
        compute_util,
        hbm_util,
        l2_util,
        l1_util,
        lds_util,
        overhead_bound=overhead_bound,
        saturation_pct=saturation_pct,
    )
    if not overhead_bound and explicit_no_fp:
        observed = "no_fp_work"

    efficiency = _first_number(metrics, ("roofline_efficiency_pct",))
    if efficiency is None:
        efficiency = derived["roofline_efficiency_pct"]
    recommendation = recommend_optimization(theoretical, observed, efficiency)
    available = {
        "ai_hbm": ai_hbm,
        "ai_ridge_empirical": ridge,
        "compute_utilization_pct": compute_util,
        "hbm_utilization_pct": hbm_util,
        "l2_utilization_pct": l2_util,
        "l1_utilization_pct": l1_util,
        "lds_utilization_pct": lds_util,
        "roofline_efficiency_pct": efficiency,
        "overhead_bound": bool(overhead_bound),
    }
    evidence = [
        "%s=%s" % (key, value)
        for key, value in available.items()
        if value is not None and value is not False
    ]
    result = {
        "theoretical_bound": theoretical,
        "observed_limit": observed,
        "recommended_specialties": recommendation["recommended_specialties"],
        "recommended_levers": recommendation["recommended_levers"],
        "confidence": recommendation["confidence"],
        "evidence": evidence,
        "policy_version": POLICY_VERSION,
        "saturation_pct": float(saturation_pct),
    }
    result.update({key: value for key, value in derived.items() if key not in result})
    return result


def _identity_value(case, key):
    metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
    aliases = {
        "case_id": ("case_id", "id"),
        "shape": ("shape", "shapes"),
        "dtypes": ("dtypes", "dtype"),
        "kernel": (
            "matched_kernel_name",
            "kernel",
            "kernel_name",
            "selected_kernel",
        ),
        "peak_basis": ("peak_basis",),
        "compute_metric": ("compute_metric",),
    }
    for name in aliases[key]:
        if name in case:
            return case[name]
        if name in metrics:
            return metrics[name]
    return None


def _empirical_peaks_compatible(left, right):
    """Allow normal microbenchmark noise while rejecting a changed peak basis."""
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return left == right
    return math.isclose(
        left_number,
        right_number,
        rel_tol=EMPIRICAL_PEAK_REL_TOLERANCE,
        abs_tol=1e-12,
    )


def compare_cases(before, after):
    """Compare two like-for-like case records or raise ValueError.

    Identity fields are deliberately strict: a missing value on one side is
    different from a present value, and no shape/dtype/kernel coercion occurs.
    """
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("before and after must be dictionaries")
    mismatches = []
    for key in ("case_id", "shape", "dtypes", "kernel", "peak_basis", "compute_metric"):
        left = _identity_value(before, key)
        right = _identity_value(after, key)
        if left != right:
            mismatches.append("%s: %r != %r" % (key, left, right))

    before_metrics = before.get("metrics", before)
    after_metrics = after.get("metrics", after)
    for key in ("compute_empirical_peak_gflops", "hbm_empirical_peak_gbps"):
        left = before_metrics.get(key)
        right = after_metrics.get(key)
        if not _empirical_peaks_compatible(left, right):
            mismatches.append("%s: %r != %r" % (key, left, right))
    if mismatches:
        raise ValueError("incompatible roofline cases: " + "; ".join(mismatches))

    before_class = before.get("classification")
    if not isinstance(before_class, dict):
        before_class = build_classification(before_metrics)
    after_class = after.get("classification")
    if not isinstance(after_class, dict):
        after_class = build_classification(after_metrics)

    fields = (
        "performance_gflops",
        "compute_utilization_pct",
        "hbm_utilization_pct",
        "roofline_efficiency_pct",
        "headroom_ratio",
    )
    deltas = {}
    for key in fields:
        left = _first_number(before_metrics, (key,))
        if left is None:
            left = _number(before_class.get(key))
        right = _first_number(after_metrics, (key,))
        if right is None:
            right = _number(after_class.get(key))
        deltas[key] = None if left is None or right is None else right - left

    before_perf = _first_number(before_metrics, ("performance_gflops", "compute_actual_gflops"))
    after_perf = _first_number(after_metrics, ("performance_gflops", "compute_actual_gflops"))
    performance_ratio = None
    if before_perf is not None and before_perf > 0 and after_perf is not None:
        performance_ratio = after_perf / before_perf
    return {
        "case_id": _identity_value(before, "case_id"),
        "compatible": True,
        "before": before_class,
        "after": after_class,
        "deltas": deltas,
        "performance_ratio": performance_ratio,
        "improved": (
            performance_ratio > 1.0
            if performance_ratio is not None
            else (
                deltas["roofline_efficiency_pct"] > 0
                if deltas["roofline_efficiency_pct"] is not None
                else None
            )
        ),
    }
