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

# Guidance validity gate. A roofline case is only handed to the optimizer as
# actionable steering when its DATA is trustworthy AND its DIRECTION is usable.
VALID_CONFIDENCES = ("high", "medium")
# Below this measured empirical headroom there is nothing worth chasing on the
# kernel; the case is still valid data but flagged so the planner deprioritizes.
MIN_ACTIONABLE_HEADROOM = 1.05
# Roofline efficiency above this is treated as a calibration artifact, not a real
# measurement: an empirical peak lower than the achieved rate (e.g. a BF16 peak
# miscalibration) drives efficiency past 100%. Such a case yields no trustworthy
# direction, so the DATA gate rejects it rather than steering on a phantom ceiling.
EFFICIENCY_ARTIFACT_PCT = 105.0
# Latency dep/issue split (block 7.2 Wavefront Runtime Stats). A wait bucket must
# occupy at least this fraction of Wave Cycles AND clearly dominate the other bucket
# before we commit to the more specific (and oppositely-fixed) subtype.
LATENCY_WAIT_THRESHOLD_PCT = 15.0
LATENCY_DOMINANCE_RATIO = 1.25
# Raw-counter red-flag thresholds (optimization playbook six-check). Each is a
# fail-soft heuristic: a None input simply omits that flag.
COALESCING_MIN_PCT = 50.0
LDS_BANK_CONFLICT_MAX_PCT = 20.0
LOW_OCCUPANCY_PCT = 30.0
# Register-occupancy ceiling (playbook check #2): waves/SIMD = min(8, 512/(VGPR+AGPR)).
# At or below this many waves/SIMD, the register footprint -- not the launch shape -- caps
# occupancy, so cutting VGPR/AGPR is the lever that helps.
REGISTER_CEILING_WAVES = 2.0
MAX_WAVES_PER_SIMD = 8.0
VGPR_FILE_PER_SIMD = 512.0
# Achieved occupancy must reach at least this fraction of the register-limited ceiling before
# we call registers the binding constraint. If achieved sits well BELOW the ceiling, the
# register footprint is NOT what constrains occupancy and cutting VGPRs does nothing (playbook
# finding 5 / "achieved occupancy sits well below the register ceiling").
REGISTER_CEILING_PINNED_FRAC = 0.8
# A wall-time / performance change smaller than this (percent) is inside run-to-run noise
# measured on identical repeats; roofline perf deltas below it are not real wins.
NOISE_FLOOR_PCT = 3.4
# An optimization whose Amdahl e2e ceiling (time_share * reclaimable kernel fraction) is
# below this many percent cannot move end-to-end enough to be worth a round, regardless of
# how much isolated kernel headroom remains.
MIN_AMDAHL_CEILING_PCT = 1.0
# Numeric signals that prove a case was actually measured (guards the
# "matched but every metric is null" degenerate that yields no real evidence).
_MEASURABLE_SIGNALS = (
    "roofline_efficiency_pct",
    "roofline_spec_hbm_efficiency_pct",
    "headroom_ratio",
    "ai_hbm",
    "compute_utilization_pct",
    "hbm_utilization_pct",
    "l2_utilization_pct",
    "l1_utilization_pct",
    "lds_utilization_pct",
    "performance_gflops",
)


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


def refine_latency_limit(diagnostics):
    """Disambiguate a generic latency stall into dependency- vs issue-wait.

    Uses block 7.2 Wavefront Runtime Stats (each bucket already normalized to a
    percentage of Wave Cycles). Dependency-wait and issue-wait imply OPPOSITE fixes
    -- shorten chains / add registers vs. add waves / drop registers -- so the split
    only commits when one bucket is both material and clearly dominant. Otherwise it
    stays ``latency_occupancy``. Returns (limit, evidence_list).
    """
    if not isinstance(diagnostics, dict):
        return "latency_occupancy", []
    dep = _number(diagnostics.get("dependency_wait_pct"))
    issue = _number(diagnostics.get("issue_wait_pct"))
    if dep is None and issue is None:
        return "latency_occupancy", []
    dep_v = dep if dep is not None else 0.0
    issue_v = issue if issue is not None else 0.0
    evidence = []
    if dep is not None:
        evidence.append("dependency_wait_pct=%s" % dep)
    if issue is not None:
        evidence.append("issue_wait_pct=%s" % issue)
    # Dominance is a strict ratio; a zero counterpart is treated as fully dominated.
    if (
        dep_v >= LATENCY_WAIT_THRESHOLD_PCT
        and dep_v >= issue_v * LATENCY_DOMINANCE_RATIO
    ):
        return "latency_dep", evidence
    if (
        issue_v >= LATENCY_WAIT_THRESHOLD_PCT
        and issue_v >= dep_v * LATENCY_DOMINANCE_RATIO
    ):
        return "latency_issue", evidence
    return "latency_occupancy", evidence


def detect_red_flags(metrics):
    """Return raw-counter red flags from block 7/11/16 diagnostics plus efficiency.

    Fail-soft: every check that lacks its input is silently skipped. Each flag is a
    dict {flag, detail, evidence} so downstream roles can render or filter them. These
    are orthogonal to the observed_limit -- a compute-bound kernel can still spill.
    """
    flags = []
    if not isinstance(metrics, dict):
        return flags
    diag = metrics.get("diagnostics") if isinstance(metrics.get("diagnostics"), dict) else {}

    # (1) GPU fill: fewer workgroups than CUs leaves compute units idle.
    ctas = _number(diag.get("ctas"))
    num_cus = _number(diag.get("num_cus"))
    if ctas is not None and num_cus is not None and num_cus > 0 and ctas < num_cus:
        flags.append({
            "flag": "gpu_underfilled",
            "detail": "grid launches fewer workgroups than the GPU has CUs",
            "evidence": "ctas=%s < num_cus=%s" % (ctas, num_cus),
        })

    # (2) Register spill: any scratch per workitem means spills to global memory.
    scratch = _number(diag.get("scratch_per_workitem"))
    if scratch is not None and scratch > 0:
        flags.append({
            "flag": "register_spill",
            "detail": "scratch allocated per workitem indicates register spilling",
            "evidence": "scratch_per_workitem=%s" % scratch,
        })

    # (3) Occupancy: the register ceiling and achieved occupancy answer DIFFERENT questions and
    # must be compared, because the fixes are opposite. The ceiling (min(8, 512/(VGPR+AGPR)))
    # says how many waves COULD reside; achieved says how many DO. Cutting registers only helps
    # when achieved is actually pinned AT a low ceiling -- if achieved sits well below the
    # ceiling, registers are not the limiter and cutting VGPR does nothing (playbook finding 5).
    occ = _number(diag.get("achieved_occupancy_pct"))
    ceiling = _number(diag.get("waves_per_simd_ceiling"))
    if occ is not None and ceiling is not None and ceiling > 0:
        ceiling_pct = 100.0 * ceiling / MAX_WAVES_PER_SIMD
        pinned = occ >= REGISTER_CEILING_PINNED_FRAC * ceiling_pct
        if ceiling <= REGISTER_CEILING_WAVES and pinned:
            # Occupancy is pinned at a low register ceiling: cutting VGPR/AGPR (or tile) raises it.
            flags.append({
                "flag": "register_occupancy_ceiling",
                "detail": "occupancy is pinned at a low register ceiling; reduce VGPR/AGPR or tile to raise it",
                "evidence": "achieved_occupancy_pct=%s ~ ceiling %.1f%% (waves_per_simd_ceiling=%s)"
                % (occ, ceiling_pct, ceiling),
            })
        elif occ < LOW_OCCUPANCY_PCT and not pinned:
            # Achieved sits well below the register ceiling -> registers are NOT the constraint;
            # cutting VGPR will not help (playbook "you are tuning the wrong thing").
            flags.append({
                "flag": "occupancy_not_register_limited",
                "detail": "achieved occupancy is well below the register ceiling; cutting VGPR will not help -- look at LDS/barriers/launch shape",
                "evidence": "achieved_occupancy_pct=%s < ceiling %.1f%% (waves_per_simd_ceiling=%s)"
                % (occ, ceiling_pct, ceiling),
            })
    elif occ is not None and occ < LOW_OCCUPANCY_PCT:
        # Ceiling unknown (VGPR/AGPR not collected): fall back to the plain low-occupancy signal.
        flags.append({
            "flag": "low_occupancy",
            "detail": "achieved wavefront occupancy is low",
            "evidence": "achieved_occupancy_pct=%s < %s" % (occ, LOW_OCCUPANCY_PCT),
        })

    # (4) Poor coalescing: scattered global accesses waste memory bandwidth.
    coalescing = _number(diag.get("coalescing_pct"))
    if coalescing is not None and coalescing < COALESCING_MIN_PCT:
        flags.append({
            "flag": "poor_coalescing",
            "detail": "vector-L1 coalescing is below the healthy floor",
            "evidence": "coalescing_pct=%s < %s" % (coalescing, COALESCING_MIN_PCT),
        })

    # (5) LDS bank conflicts: serialized shared-memory accesses.
    lds_conflict = _number(diag.get("lds_bank_conflict_pct"))
    if lds_conflict is not None and lds_conflict > LDS_BANK_CONFLICT_MAX_PCT:
        flags.append({
            "flag": "lds_bank_conflicts",
            "detail": "LDS bank conflict rate is high",
            "evidence": "lds_bank_conflict_pct=%s > %s" % (lds_conflict, LDS_BANK_CONFLICT_MAX_PCT),
        })

    # (6) Efficiency artifact: empirical peak below achieved rate -> untrusted data.
    efficiency = _first_number(metrics, ("roofline_efficiency_pct",))
    if efficiency is not None and efficiency > EFFICIENCY_ARTIFACT_PCT:
        flags.append({
            "flag": "efficiency_artifact",
            "detail": (
                "roofline efficiency exceeds 100% -- a measurement artifact, not a record: "
                "a miscalibrated BF16 empirical peak (use HBM%/F32 MFMA% instead) or SFU ops "
                "(rsqrt/exp on rmsnorm/rope) folded into Performance(GFLOPs). Do not trust this "
                "case's limit or headroom"
            ),
            "evidence": "roofline_efficiency_pct=%s > %s" % (efficiency, EFFICIENCY_ARTIFACT_PCT),
        })

    return flags


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
        # Dependency-wait dominant: waves stall on their own result chains. Fix by
        # shortening/breaking chains and giving each thread MORE registers for ILP.
        "latency_dep": (
            ["compute", "algorithm"],
            [
                "shorten dependency chains",
                "unroll to expose instruction-level parallelism",
                "increase registers per thread to overlap independent work",
            ],
        ),
        # Issue-wait dominant: too few resident waves to cover issue latency. Fix by
        # raising occupancy -- more waves per SIMD, FEWER registers (opposite of dep).
        "latency_issue": (
            ["compute", "algorithm"],
            [
                "increase occupancy with more waves per SIMD",
                "reduce registers per thread to raise occupancy",
                "expose more independent wavefronts",
            ],
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
    elif theoretical_bound == "compute_side" and observed_limit in (
        "latency_occupancy",
        "latency_dep",
        "latency_issue",
    ):
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

    # Refine a generic latency stall into dependency- vs issue-wait when block 7.2
    # diagnostics are present; the two subtypes carry opposite register/occupancy fixes.
    diagnostics = metrics.get("diagnostics") if isinstance(metrics.get("diagnostics"), dict) else {}
    latency_evidence = []
    if observed == "latency_occupancy":
        observed, latency_evidence = refine_latency_limit(diagnostics)

    red_flags = detect_red_flags(metrics)

    efficiency = _first_number(metrics, ("roofline_efficiency_pct",))
    if efficiency is None:
        efficiency = derived["roofline_efficiency_pct"]
    recommendation = recommend_optimization(theoretical, observed, efficiency)
    # Split-K is the proven lever for an issue-wait / under-filled latency kernel (skinny-M,
    # fat-K decode GEMM). Append it only when the diagnostics actually show underfill or a
    # register ceiling, with the gating rule inline so the engineer does not misapply it.
    flag_names = {f.get("flag") for f in red_flags}
    if observed in ("latency_issue", "latency_occupancy") and (
        "gpu_underfilled" in flag_names or "register_occupancy_ceiling" in flag_names
    ):
        split_k = (
            "split-K when K>=4096 (target total CTAs ~1.5x CU count; fuse the reduction into a "
            "dedicated kernel, not tl.atomic_add)"
        )
        if split_k not in recommendation["recommended_levers"]:
            recommendation["recommended_levers"].append(split_k)
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
    evidence.extend(latency_evidence)
    evidence.extend("red_flag:%s" % flag.get("flag") for flag in red_flags)
    result = {
        "theoretical_bound": theoretical,
        "observed_limit": observed,
        "recommended_specialties": recommendation["recommended_specialties"],
        "recommended_levers": recommendation["recommended_levers"],
        "confidence": recommendation["confidence"],
        "evidence": evidence,
        "red_flags": red_flags,
        "policy_version": POLICY_VERSION,
        "saturation_pct": float(saturation_pct),
    }
    result.update({key: value for key, value in derived.items() if key not in result})
    return result


def _case_measurable(case):
    """True when at least one real numeric roofline/utilization signal exists."""
    classification = case.get("classification") if isinstance(case.get("classification"), dict) else {}
    metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
    for name in _MEASURABLE_SIGNALS:
        if _number(classification.get(name)) is not None:
            return True
        if _number(metrics.get(name)) is not None:
            return True
    return False


def _case_headroom(case):
    classification = case.get("classification") if isinstance(case.get("classification"), dict) else {}
    metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
    headroom = _number(classification.get("headroom_ratio"))
    if headroom is None:
        headroom = _number(metrics.get("headroom_ratio"))
    return headroom


def amdahl_ceiling_pct(time_share, headroom_ratio):
    """Max end-to-end percent reclaimable by fully optimizing one case.

    ``time_share`` is the case's fraction (0..1) of end-to-end time; ``headroom_ratio`` is
    the achievable kernel speedup (empirical roof / achieved). The reclaimable e2e fraction is
    ``share * (1 - 1/headroom)`` -- Amdahl's law. A large kernel headroom on a tiny-share case
    still yields a negligible e2e ceiling, which is why isolated wins so often produced 0% e2e.
    Returns None when either input is missing/degenerate.
    """
    share = _number(time_share)
    headroom = _number(headroom_ratio)
    if share is None or headroom is None or share <= 0 or headroom <= 0:
        return None
    return 100.0 * share * (1.0 - 1.0 / headroom)


def _matched_time_shares(cases):
    """Normalize matched-case weights into e2e time shares summing to 1.

    Uses each case's `weight` (defaults 1.0) as the share proxy -- the same basis the summary
    ranks on. Only matched cases participate; if every weight is missing they split evenly.
    """
    matched = [c for c in cases if isinstance(c, dict) and c.get("status") == "matched"]
    weights = {}
    total = 0.0
    for case in matched:
        weight = _number(case.get("weight"))
        if weight is None or weight < 0:
            weight = 1.0
        weights[case.get("case_id")] = weight
        total += weight
    if total <= 0:
        return {}
    return {cid: weight / total for cid, weight in weights.items()}


def _assess_case(case, result_status_ok):
    """Judge one case for DATA + DIRECTION validity; never raises."""
    if not isinstance(case, dict):
        return {"valid": False, "reason": "case_not_object"}
    classification = case.get("classification") if isinstance(case.get("classification"), dict) else {}
    case_id = case.get("case_id")
    observed = classification.get("observed_limit", "unknown")
    confidence = classification.get("confidence", "low")
    specialties = [s for s in (classification.get("recommended_specialties") or []) if s in SPECIALTIES]
    levers = list(classification.get("recommended_levers") or [])
    headroom = _case_headroom(case)
    red_flags = [f for f in (classification.get("red_flags") or []) if isinstance(f, dict)]
    has_efficiency_artifact = any(f.get("flag") == "efficiency_artifact" for f in red_flags)

    if not result_status_ok:
        reason = "result_status_not_ok"
        valid = False
    elif case.get("status") != "matched":
        reason = "case_not_matched"
        valid = False
    elif not _case_measurable(case):
        # DATA gate: matched but no measured signal is not real evidence.
        reason = "no_measured_signal"
        valid = False
    elif has_efficiency_artifact:
        # DATA gate: efficiency > 100% means the empirical peak is miscalibrated, so
        # the observed limit and headroom cannot be trusted for steering.
        reason = "efficiency_artifact"
        valid = False
    elif observed == "unknown":
        reason = "observed_limit_unknown"
        valid = False
    elif confidence not in VALID_CONFIDENCES:
        reason = "low_confidence"
        valid = False
    elif not specialties:
        # DIRECTION gate: no actionable specialty to dispatch.
        reason = "no_recommended_specialty"
        valid = False
    else:
        reason = "actionable"
        valid = True

    low_headroom = valid and headroom is not None and headroom <= MIN_ACTIONABLE_HEADROOM
    if low_headroom:
        reason = "low_headroom"
    return {
        "case_id": case_id,
        "valid": valid,
        "reason": reason,
        "low_headroom": bool(low_headroom),
        "observed_limit": observed,
        "confidence": confidence,
        "recommended_specialties": specialties,
        "recommended_levers": levers,
        "headroom_ratio": headroom,
        "weight": _number(case.get("weight")),
        "red_flags": red_flags,
    }


def assess_guidance(result):
    """Gate roofline evidence into actionable guidance for the optimizer.

    Pure and fail-soft: accepts the full roofline result dict and returns a
    compact, self-describing guidance record.  ``valid`` is True only when at
    least one case passes BOTH the data gate (matched, measured, known limit,
    non-low confidence) and the direction gate (a dispatchable specialty).
    Recommendations are unioned over the VALID cases in the priority order the
    summary already computed (weight * empirical headroom), so the planner and
    engineers can consume them directly instead of re-deriving the mapping.
    """
    if not isinstance(result, dict):
        return {
            "valid": False,
            "reason": "result_not_object",
            "policy_version": POLICY_VERSION,
            "cases": [],
            "recommended_specialties": [],
            "recommended_levers": [],
            "red_flags": [],
            "dominant_case_id": None,
            "invalid_case_ids": [],
        }
    status = result.get("status")
    result_status_ok = status in ("ok", "partial")
    cases = result.get("cases") if isinstance(result.get("cases"), list) else []

    assessed = [_assess_case(case, result_status_ok) for case in cases]
    by_id = {a.get("case_id"): a for a in assessed}

    # Amdahl worthiness: a kernel win only reaches e2e in proportion to the case's
    # share of end-to-end time. Normalize matched weights into time shares, then a
    # valid-but-tiny case is deprioritized like low_headroom (never dominant) so we
    # don't chase isolated wins that cannot move the wall clock.
    time_shares = _matched_time_shares(cases)
    for a in assessed:
        share = time_shares.get(a.get("case_id"))
        ceiling = amdahl_ceiling_pct(share, a.get("headroom_ratio"))
        a["time_share"] = share
        a["amdahl_ceiling_pct"] = ceiling
        below = bool(
            a.get("valid")
            and not a.get("low_headroom")
            and ceiling is not None
            and ceiling < MIN_AMDAHL_CEILING_PCT
        )
        a["below_amdahl_floor"] = below
        if below:
            a["reason"] = "below_amdahl_floor"

    # Consume the summary's priority_order (weight * headroom) when present so
    # the most impactful valid case leads; fall back to source order otherwise.
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    order = []
    for rank in (summary.get("priority_order") or []):
        cid = rank.get("case_id") if isinstance(rank, dict) else None
        if cid in by_id and cid not in order:
            order.append(cid)
    for a in assessed:
        cid = a.get("case_id")
        if cid not in order:
            order.append(cid)

    specialties, levers, dominant = [], [], None
    for cid in order:
        a = by_id.get(cid)
        if not a or not a.get("valid"):
            continue
        if dominant is None and not a.get("low_headroom") and not a.get("below_amdahl_floor"):
            dominant = cid
        for s in a.get("recommended_specialties", []):
            if s not in specialties:
                specialties.append(s)
        for lever in a.get("recommended_levers", []):
            if lever not in levers:
                levers.append(lever)
    # Red flags are surfaced from ALL assessed cases (including ones the gate
    # rejected -- e.g. an efficiency artifact IS a flag), deduped by flag name.
    red_flags, seen_flags = [], set()
    for a in assessed:
        for flag in a.get("red_flags", []) or []:
            name = flag.get("flag") if isinstance(flag, dict) else None
            if name and name not in seen_flags:
                seen_flags.add(name)
                red_flags.append(flag)
    if dominant is None:
        dominant = next((cid for cid in order if by_id[cid].get("valid")), None)

    valid = dominant is not None or any(a.get("valid") for a in assessed)
    if not result_status_ok:
        reason = "roofline_status_%s" % (status or "missing")
    elif valid:
        reason = "actionable_cases_present"
    elif not cases:
        reason = "no_cases"
    else:
        reason = "no_case_passed_validity_gate"
    return {
        "valid": bool(valid),
        "reason": reason,
        "policy_version": POLICY_VERSION,
        "cases": assessed,
        "recommended_specialties": specialties,
        "recommended_levers": levers,
        "red_flags": red_flags,
        "dominant_case_id": dominant,
        "invalid_case_ids": [a.get("case_id") for a in assessed if not a.get("valid")],
    }


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
    # `kernel` is deliberately NOT an identity field: comparing the roofline before vs after an
    # optimization inherently compares DIFFERENT device kernels -- a backend swap (Triton ->
    # CK a8w8_blockscale) or even a rewrite renames the selected symbol. That is the expected,
    # measured delta, not an incompatibility. The workload/device identity (case_id/shape/dtypes/
    # peak_basis/compute_metric) stays strict so we still reject an apples-to-oranges compare.
    for key in ("case_id", "shape", "dtypes", "peak_basis", "compute_metric"):
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
    before_kernel = _identity_value(before, "kernel")
    after_kernel = _identity_value(after, "kernel")
    # Run-to-run noise floor: a performance move smaller than NOISE_FLOOR_PCT is inside the
    # variance measured on identical repeats, so it is not a real win/loss. This is a roofline
    # CONSISTENCY signal only -- wall time (COMMANDMENT) remains the authoritative keep verdict.
    noise = NOISE_FLOOR_PCT / 100.0
    within_noise = (
        abs(performance_ratio - 1.0) < noise if performance_ratio is not None else None
    )
    if performance_ratio is not None:
        improved = performance_ratio >= 1.0 + noise
    elif deltas["roofline_efficiency_pct"] is not None:
        improved = deltas["roofline_efficiency_pct"] > 0
    else:
        improved = None
    # Utilization/efficiency rose but achieved performance did not clear the noise floor: the
    # denominator (empirical peak) likely moved, not the kernel -- do not claim this as a win.
    eff_delta = deltas.get("roofline_efficiency_pct")
    utilization_moved_perf_did_not = bool(
        within_noise and eff_delta is not None and eff_delta > 0
    )
    return {
        "case_id": _identity_value(before, "case_id"),
        "compatible": True,
        "before_kernel": before_kernel,
        "after_kernel": after_kernel,
        "kernel_changed": before_kernel != after_kernel,
        "before": before_class,
        "after": after_class,
        "deltas": deltas,
        "performance_ratio": performance_ratio,
        "noise_floor_pct": NOISE_FLOOR_PCT,
        "within_noise": within_noise,
        "utilization_moved_perf_did_not": utilization_moved_perf_did_not,
        "improved": improved,
    }
