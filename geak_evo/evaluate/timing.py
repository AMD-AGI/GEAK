"""Offline, source-bound timing analysis. This module never runs a workload.

Timing blocks estimate one fixed endpoint, not independent search outcomes.
Externally justified latency bounds remain distinct from measured references.
"""

import json

from .._util import EvoError, canonical_json, require, sha256_hex
import math
import random
import re
import textwrap
from statistics import fmean


class MeasurementError(EvoError, ValueError):
    """Evidence does not satisfy the declared measurement contract."""


_SECONDS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
_BINDING = ("job_id", "baseline_source_id", "candidate_source_id", "measurement_epoch_id")
_INTERVAL_LIMITS = (
    "Percentile bootstrap over paired block means, conditional on this fixed "
    "source and measurement epoch. Assumes exchangeable independent blocks; "
    "does not correct drift, autocorrelation, candidate selection, multiple "
    "comparisons, or uncertainty in workload frequencies. Timing blocks are "
    "not independent search starts. Small block counts can give misleadingly "
    "narrow intervals."
)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise MeasurementError(name + " must be a nonempty string")
    return value


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise MeasurementError(name + " must be a lowercase SHA256 digest")
    return value


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementError(name + " must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise MeasurementError(name + " must be a finite number") from None
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise MeasurementError(name + " must be finite and " + ("positive" if positive else "nonnegative"))
    return value


def _mapping(value, name):
    if not isinstance(value, dict):
        raise MeasurementError(name + " must be an object")
    return value


def _list(value, name, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise MeasurementError(name + " must be " + ("a nonempty list" if nonempty else "a list"))
    return value


def _seconds(value, unit, name):
    if not isinstance(unit, str) or unit not in _SECONDS:
        raise MeasurementError(name + " unit must be ns, us, ms, or s")
    return _number(_number(value, name, positive=True) * _SECONDS[unit], name, positive=True)


def _match(record, expected, name):
    for key, value in expected.items():
        if record.get(key) != value:
            raise MeasurementError(name + " " + key + " mismatch")


def _frequencies(cases):
    result = {}
    for case in _list(cases, "cases", nonempty=True):
        case = _mapping(case, "case")
        case_id = _text(case.get("case_id"), "case_id")
        if case_id in result:
            raise MeasurementError("duplicate case_id: " + case_id)
        frequency = _mapping(case.get("observed_frequency"), "observed_frequency")
        count = frequency.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise MeasurementError("observed_frequency count must be a positive integer; unobserved weights are forbidden")
        if frequency.get("unit") != "calls":
            raise MeasurementError("observed_frequency unit must be calls")
        result[case_id] = {
            "count": count,
            "unit": "calls",
            "provenance_ref": _text(frequency.get("provenance_ref"), "frequency provenance_ref"),
        }
    return result


def _timing(record, source_id, work_id, name):
    record = _mapping(record, name)
    _match(record, {"source_id": source_id, "work_id": work_id}, name)
    if record.get("kind") == "unprofiled_interval_mean":
        if "samples" in record:
            raise MeasurementError(name + " interval mean must not contain samples")
        timed_calls = record.get("timed_calls")
        if isinstance(timed_calls, bool) or not isinstance(timed_calls, int) or timed_calls <= 0:
            raise MeasurementError(name + " timed_calls must be a positive integer")
        report_ref = _mapping(record.get("raw_report_ref"), name + " raw_report_ref")
        if set(report_ref) != {"path", "sha256"}:
            raise MeasurementError(name + " raw_report_ref must contain exactly path and sha256")
        _text(report_ref["path"], name + " raw_report_ref path")
        _digest(report_ref["sha256"], name + " raw_report_ref sha256")
        # The acquisition report already divided its interval by timed_calls.
        # Each observation supplies one block mean, regardless of call count.
        return _seconds(record.get("value"), record.get("unit"), name)
    if record.get("kind") != "unprofiled":
        raise MeasurementError(name + " must contain unprofiled timing")
    samples = _list(record.get("samples"), name + " samples", nonempty=True)
    return fmean(_seconds(value, record.get("unit"), name) for value in samples)


def _quantile(values, probability):
    values = sorted(values)
    position = probability * (len(values) - 1)
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (position - left)


def _estimate(baseline_s, candidate_s):
    baseline_s = _number(baseline_s, "derived baseline latency", positive=True)
    candidate_s = _number(candidate_s, "derived candidate latency", positive=True)
    saving_s = baseline_s - candidate_s
    speedup = _number(baseline_s / candidate_s, "derived speedup", positive=True)
    saving_percent = 100.0 * (saving_s / baseline_s)
    if not math.isfinite(saving_percent):
        raise MeasurementError("derived saving percentage is not finite")
    return {
        "baseline_s": baseline_s,
        "candidate_s": candidate_s,
        "saving_s": saving_s,
        "saving_percent": saving_percent,
        "speedup": speedup,
    }


def _interval(draws):
    return {
        key: [_quantile([draw[key] for draw in draws], 0.025),
              _quantile([draw[key] for draw in draws], 0.975)]
        for key in ("baseline_s", "candidate_s", "saving_s", "saving_percent", "speedup")
    }


def _rng(seed, case_id):
    # Avoid Python's process-randomized hash and dependence on case input order.
    material = canonical_json([seed, case_id])
    return random.Random(int(sha256_hex(material), 16))


def _unknown(reason):
    return {"status": "unknown", "reason": reason, "bound_ratio_percent": None}


def assess_roofline(evidence, binding, measured_s):
    """Check metadata and compare an externally supplied bound with latency.

    No kernel formula, roof selection, profiler interpretation, or physical
    calibration is performed here. Qualification remains in linked evidence.
    """
    if evidence is None:
        return _unknown("no externally justified latency bound supplied")
    evidence = _mapping(evidence, "roofline")
    _match(evidence, binding, "roofline")
    if evidence.get("status") == "unknown":
        return _unknown(_text(evidence.get("reason"), "unknown roofline reason"))
    kind = evidence.get("kind")
    if kind not in ("modeled_bound", "counter_informed_bound"):
        raise MeasurementError("roofline kind must be modeled_bound or counter_informed_bound; references are separate")
    missing = []

    def qualified_text(record, key, label):
        value = record.get(key)
        if value is None or value in ("", "unknown"):
            missing.append(label)
            return None
        return _text(value, label)

    _digest(evidence.get("bound_id"), "bound_id")
    qualified_text(evidence, "justification_ref", "bound justification")
    assumptions = _list(evidence.get("assumptions", []), "bound assumptions")
    if not assumptions:
        missing.append("explicit bound assumptions")
    for assumption in assumptions:
        _text(assumption, "bound assumption")
    bound = _mapping(evidence.get("latency_bound"), "latency_bound")
    bound_s = _seconds(bound.get("value"), bound.get("unit"), "latency_bound")
    work = _mapping(evidence.get("operation_work"), "operation_work")
    _match(work, {"work_id": binding["work_id"]}, "operation_work")
    for key in ("dtype", "operation_class", "unit", "scope", "provenance_ref"):
        qualified_text(work, key, "operation_work " + key)
    if work.get("amount") is None:
        missing.append("operation work amount")
    else:
        _number(work["amount"], "operation work amount")
    traffic = _mapping(evidence.get("physical_traffic"), "physical_traffic")
    if traffic.get("unit") != "bytes":
        raise MeasurementError("physical_traffic unit must be bytes")
    for key in ("level", "scope", "provenance_ref"):
        qualified_text(traffic, key, "physical_traffic " + key)
    if traffic.get("amount") is None:
        missing.append("physical traffic amount")
    else:
        _number(traffic["amount"], "physical traffic amount")
    required_traffic_kind = "modeled" if kind == "modeled_bound" else "empirical_counter"
    if traffic.get("kind") != required_traffic_kind:
        raise MeasurementError("bound kind and physical traffic evidence kind disagree")
    if traffic.get("kind") == "empirical_counter":
        for key in ("profiler_id", "raw_artifact_ref", "counter_definition_ref"):
            qualified_text(traffic, key, "physical_traffic " + key)
        counters = _list(traffic.get("counter_ids", []), "counter_ids")
        if not counters:
            missing.append("official counter identities")
        for counter in counters:
            _text(counter, "counter identity")
    rates = _list(evidence.get("calibrated_rates", []), "calibrated_rates")
    if not rates:
        missing.append("calibrated rates")
    rate_ids = set()
    for rate in rates:
        rate = _mapping(rate, "calibrated rate")
        rate_id = _digest(rate.get("calibration_id"), "calibration_id")
        if rate_id in rate_ids:
            raise MeasurementError("duplicate calibration_id")
        rate_ids.add(rate_id)
        _number(rate.get("value"), "calibrated rate", positive=True)
        for key in ("device_id", "runtime_id", "calibration_epoch_id", "dtype",
                    "operation_class", "scope", "provenance_ref", "compatibility_ref"):
            qualified_text(rate, key, "calibrated rate " + key)
        quantity = rate.get("quantity")
        if quantity not in ("compute", "bandwidth"):
            raise MeasurementError("calibrated rate quantity must be compute or bandwidth")
        if quantity == "compute":
            if work.get("unit") not in (None, "", "unknown") and rate.get("unit") != work["unit"] + "/s":
                raise MeasurementError("compute rate unit does not match operation work unit")
            for key in ("dtype", "operation_class"):
                if (rate.get(key) not in (None, "", "unknown") and
                        work.get(key) not in (None, "", "unknown") and rate[key] != work[key]):
                    raise MeasurementError("compute calibration " + key + " mismatch; no fallback roof")
        elif rate.get("unit") != "bytes/s":
            raise MeasurementError("bandwidth rate unit must be bytes/s")
    if missing:
        return _unknown("unqualified bound metadata: " + "; ".join(missing))
    if measured_s is None:
        return _unknown("candidate latency unknown; bound retained in raw evidence")
    measured_s = _number(measured_s, "measured latency", positive=True)
    ratio = _number(100.0 * (bound_s / measured_s), "bound ratio", positive=True)
    return {
        "status": "inconsistent_above_100" if ratio > 100.0 else "assessed",
        "kind": kind,
        "bound_id": evidence["bound_id"],
        "work_id": binding["work_id"],
        "latency_bound_s": bound_s,
        "bound_ratio_percent": ratio,
        "calibration_ids": sorted(rate_ids),
        "justification_ref": evidence["justification_ref"],
        "qualification": "metadata checked; physical validity depends on linked external evidence",
    }


def _reference(evidence, binding, measured_s):
    if evidence is None:
        return None
    evidence = _mapping(evidence, "measured_reference")
    _match(evidence, {key: binding[key] for key in ("job_id", "case_id", "measurement_epoch_id")}, "measured_reference")
    if evidence.get("kind") != "measured_reference":
        raise MeasurementError("measured reference kind must be measured_reference")
    _text(evidence.get("provenance_ref"), "measured reference provenance_ref")
    source_id = _digest(evidence.get("source_id"), "reference source_id")
    work_id = _digest(evidence.get("work_id"), "reference work_id")
    timing = _mapping(evidence.get("timing"), "reference timing")
    reference_s = _timing(timing, source_id, work_id, "reference timing")
    return {
        "kind": "measured_reference",
        "source_id": source_id,
        "work_id": work_id,
        "latency_s": reference_s,
        "reference_over_candidate": None if measured_s is None else _number(reference_s / measured_s, "reference ratio", positive=True),
        "provenance_ref": evidence["provenance_ref"],
        "interpretation": "attainable comparison, not a lower latency bound; unpaired descriptive ratio",
    }


def analyze_measurements(record, *, timing_aggregation, expected_binding=None, observed_cases=None, seed=0, resamples=2000):
    """Analyze JSON evidence; optionally bind it to trusted host/job identities.

    Pass timing_aggregation=validated_job['timing']['aggregation']; only mean
    aggregation of raw per-call samples or explicit unprofiled interval means
    is supported. Pass observed_cases from that same job to enforce frozen
    weights and the complete case registry. Pass expected_binding from host
    state, not this record. Acquisition and report authentication remain
    upstream; no correctness or provenance authentication is implied. A record
    with block_schedule='shared' resamples matching block indices across cases.
    """
    if timing_aggregation != "mean":
        raise MeasurementError("measurement analysis supports mean timing only; declared " +
                               repr(timing_aggregation) + " aggregation is unsupported")
    record = _mapping(record, "measurement record")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise MeasurementError("measurement schema_version must be 1")
    shared_schedule = record.get("block_schedule") == "shared"
    if "block_schedule" in record and not shared_schedule:
        raise MeasurementError("block_schedule must be shared when supplied")
    binding = {key: record.get(key) for key in _BINDING}
    for key in _BINDING[:-1]:
        _digest(binding[key], key)
    _text(binding["measurement_epoch_id"], "measurement_epoch_id")
    if expected_binding is not None:
        expected_binding = _mapping(expected_binding, "expected_binding")
        if set(expected_binding) != set(_BINDING):
            raise MeasurementError("expected_binding must provide all four binding keys")
        _match(record, expected_binding, "measurement record")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MeasurementError("bootstrap seed must be an integer")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise MeasurementError("bootstrap resamples must be an integer >= 100")
    cases = _list(record.get("cases"), "cases", nonempty=True)
    frequencies = _frequencies(cases)
    if observed_cases is not None and frequencies != _frequencies(observed_cases):
        raise MeasurementError("cases or frozen observed frequencies mismatch registered job")
    search_start_id = record.get("search_start_id")
    if search_start_id is not None:
        _text(search_start_id, "search_start_id")
    results = []
    draws_by_case = {}
    shared_block_ids = None
    for case in sorted(cases, key=lambda item: item["case_id"]):
        case_id = case["case_id"]
        work_ids = {role: _digest(case.get(role + "_work_id"), role + "_work_id")
                    for role in ("baseline", "candidate")}
        blocks = _list(case.get("blocks"), "timing blocks")
        pairs = []
        block_ids = set()
        for block in blocks:
            block = _mapping(block, "timing block")
            block_id = _text(block.get("block_id"), "block_id")
            if block_id in block_ids:
                raise MeasurementError("duplicate block_id within case: " + block_id)
            block_ids.add(block_id)
            _match(block, {"job_id": binding["job_id"], "case_id": case_id,
                           "measurement_epoch_id": binding["measurement_epoch_id"]}, "timing block")
            if "baseline" not in block or "candidate" not in block:
                raise MeasurementError("unpaired block: baseline and candidate are both required")
            pairs.append(tuple(_timing(block[role], binding[role + "_source_id"],
                                       work_ids[role], role) for role in ("baseline", "candidate")))
        if shared_schedule and blocks:
            ordered_ids = [block["block_id"] for block in blocks]
            if shared_block_ids is None:
                shared_block_ids = ordered_ids
            elif ordered_ids != shared_block_ids:
                raise MeasurementError("shared block schedule requires the same ordered block_id list across cases")
        result = {"case_id": case_id, "observed_frequency": frequencies[case_id],
                  "baseline_work_id": work_ids["baseline"], "candidate_work_id": work_ids["candidate"],
                  "block_count": len(pairs)}
        if not pairs:
            result.update({"status": "unknown", "reason": _text(case.get("unknown_reason"), "unknown timing reason"),
                           "estimate": None, "interval_95": None, "interval_status": "no paired blocks"})
        else:
            estimate = _estimate(fmean(pair[0] for pair in pairs), fmean(pair[1] for pair in pairs))
            result.update({"status": "regression" if estimate["saving_s"] < 0 else
                                     "improvement" if estimate["saving_s"] > 0 else "unchanged",
                           "estimate": estimate})
            if len(pairs) < 2:
                result.update({"interval_95": None, "interval_status": "insufficient paired blocks"})
            else:
                # Reinitializing the shared RNG replays identical block-index
                # draws, preserving the acquisition schedule across cases.
                rng = _rng(seed, None if shared_schedule else case_id)
                draws = []
                for unused in range(resamples):
                    sampled = [pairs[rng.randrange(len(pairs))] for unused_pair in pairs]
                    draws.append(_estimate(fmean(pair[0] for pair in sampled), fmean(pair[1] for pair in sampled)))
                draws_by_case[case_id] = draws
                result.update({"interval_95": _interval(draws), "interval_status": "exploratory paired block bootstrap"})
        candidate_s = result["estimate"]["candidate_s"] if result["estimate"] else None
        case_binding = {"job_id": binding["job_id"], "case_id": case_id,
                        "source_id": binding["candidate_source_id"], "work_id": work_ids["candidate"],
                        "measurement_epoch_id": binding["measurement_epoch_id"]}
        result["roofline"] = assess_roofline(case.get("roofline"), case_binding, candidate_s)
        result["measured_reference"] = _reference(case.get("measured_reference"), case_binding, candidate_s)
        results.append(result)
    total_count = sum(frequency["count"] for frequency in frequencies.values())
    total_count_float = _number(total_count, "total observed count", positive=True)
    missing = [case["case_id"] for case in results if case["estimate"] is None]
    aggregate = {"kind": "observed_frequency_weighted_latency", "total_observed_calls": total_count,
                 "missing_case_ids": missing, "estimate": None, "interval_95": None}
    if missing:
        aggregate.update({"status": "unknown", "reason": "missing case timing; no renormalization over measured cases"})
    else:
        def weighted(estimates):
            return _estimate(
                math.fsum(frequencies[case_id]["count"] / total_count_float * value["baseline_s"] for case_id, value in estimates),
                math.fsum(frequencies[case_id]["count"] / total_count_float * value["candidate_s"] for case_id, value in estimates))
        aggregate["estimate"] = weighted([(case["case_id"], case["estimate"]) for case in results])
        aggregate["status"] = "complete"
        aggregate["observed_total_baseline_s"] = _number(aggregate["estimate"]["baseline_s"] * total_count_float, "observed baseline total", positive=True)
        aggregate["observed_total_candidate_s"] = _number(aggregate["estimate"]["candidate_s"] * total_count_float, "observed candidate total", positive=True)
        if len(draws_by_case) == len(results):
            # Shared schedules use identical index draws in every case;
            # otherwise the case-specific RNGs assume independence.
            draws = [weighted([(case_id, samples[index]) for case_id, samples in sorted(draws_by_case.items())])
                     for index in range(resamples)]
            aggregate["interval_95"] = _interval(draws)
        aggregate["interval_assumption"] = (
            "shared block resampling preserves the cross-case acquisition schedule; blocks assumed independent"
            if shared_schedule else
            "independent case block series; shared cross-case drift is not modeled")
    return {
        "schema_version": 1, **binding, "search_start_id": search_start_id,
        "analysis": {"seed": seed, "resamples": resamples, "confidence": 0.95,
                     "timing_aggregation": timing_aggregation,
                     "block_schedule": "shared" if shared_schedule else "independent",
                     "method": "paired block percentile bootstrap; equal weight per block mean",
                     "limits": _INTERVAL_LIMITS},
        "cases": results, "aggregate": aggregate,
    }


def format_text(report):
    """Render one case per short record, preserving unknowns and regressions."""
    lines = ["Measurement analysis (offline evidence only)"]
    lines.extend(key + ": " + report[key] for key in _BINDING)
    for case in report["cases"]:
        lines.append("case " + case["case_id"] + ": " + case["status"] +
                     "; observed calls=" + str(case["observed_frequency"]["count"]) +
                     "; paired blocks=" + str(case["block_count"]))
        estimate = case["estimate"]
        if estimate is None:
            lines.append("  timing unknown: " + case["reason"])
        else:
            lines.append("  baseline {:.6g} us; candidate {:.6g} us".format(estimate["baseline_s"] * 1e6, estimate["candidate_s"] * 1e6))
            lines.append("  saving {:+.3f}%; speedup {:.6g}x".format(estimate["saving_percent"], estimate["speedup"]))
            interval = case["interval_95"]
            lines.append("  exploratory 95% saving interval [{:+.3f}, {:+.3f}]%".format(*interval["saving_percent"]) if interval else
                         "  uncertainty unknown: " + case["interval_status"])
        roof = case["roofline"]
        if roof["bound_ratio_percent"] is None:
            lines.append("  roofline unknown: " + roof["reason"])
        else:
            lines.append("  " + roof["kind"] + ": {:.3f}% bound/latency; ".format(roof["bound_ratio_percent"]) + roof["status"])
        reference = case["measured_reference"]
        if reference:
            lines.append("  measured reference {:.6g} us (attainable comparison)".format(reference["latency_s"] * 1e6))
    aggregate = report["aggregate"]
    if aggregate["estimate"]:
        estimate = aggregate["estimate"]
        lines.append("observed-frequency weighted latency: baseline {:.6g} us; candidate {:.6g} us".format(estimate["baseline_s"] * 1e6, estimate["candidate_s"] * 1e6))
        lines.append("  saving {:+.3f}%; speedup {:.6g}x; all {} observed cases retained".format(estimate["saving_percent"], estimate["speedup"], len(report["cases"])))
    else:
        lines.append("observed-frequency weighted latency: unknown; " + aggregate["reason"])
        lines.append("  missing cases: " + ", ".join(aggregate["missing_case_ids"]))
    lines.append("Uncertainty: " + report["analysis"]["limits"])
    lines.append("This report does not establish correctness, GPU custody, or search-policy advantage.")
    return "\n".join(textwrap.fill(line, width=88, subsequent_indent="  ") for line in lines) + "\n"


def format_json(report, *, raw_record=None):
    """Optionally include raw blocks and all bound/reference provenance."""
    payload = report if raw_record is None else {"analysis": report, "raw_evidence": raw_record}
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
