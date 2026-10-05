# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assess three fresh scores under the fixed stopping inference model.

This module establishes no source, process, ownership, or execution authority.
The controller must bind the floor and samples to its immutable run records.
"""

import math
import statistics

SCHEMA = "quality-stop-score-assessment-v1"
SAMPLE_COUNT = 3
MAX_LOOKS = 3
FAMILY_ALPHA = 0.05
LOOK_ALPHA = 1.0 / 60.0
DEFAULT_FLOOR = 1.05

# For two degrees of freedom, P(T > t) = 1/2 - t/(2*sqrt(t*t + 2)).
# At tail probability 1/60, the exact critical value is sqrt(1682/59).
# The small positive cushion keeps the numerical critical value conservative.
CRITICAL_T = math.sqrt(1682.0 / 59.0) + 1e-12


def _positive_float(value):
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def assess_scores(scores, *, look_index, floor=DEFAULT_FLOOR):
    """Return a JSON-safe numerical assessment, without certifying execution."""
    result = {
        "schema": SCHEMA,
        "valid": False,
        "certified": False,
        "reason": None,
        "sample_count": len(scores) if type(scores) in (list, tuple) else None,
        "look_index": look_index if type(look_index) is int and 1 <= look_index <= MAX_LOOKS else None,
        "max_looks": MAX_LOOKS,
        "alpha_family": FAMILY_ALPHA,
        "alpha_look": LOOK_ALPHA,
        "degrees_of_freedom": SAMPLE_COUNT - 1,
        "critical_t": CRITICAL_T,
        "floor": None,
        "score_values": None,
        "mean_log": None,
        "sample_log_sd": None,
        "geometric_mean": None,
        "lower_log": None,
        "lower_speedup": None,
        "numeric_margin_log": None,
        "inference_model": "IID normal log scores under the declared timing law",
        "execution_authority_established": False,
    }
    if type(look_index) is not int or not 1 <= look_index <= MAX_LOOKS:
        result["reason"] = "invalid_or_exhausted_look"
        return result
    target = _positive_float(floor)
    if target is None:
        result["reason"] = "invalid_floor"
        return result
    result["floor"] = target
    if type(scores) not in (list, tuple) or len(scores) != SAMPLE_COUNT:
        result["reason"] = "exactly_three_scores_required"
        return result
    values = [_positive_float(value) for value in scores]
    if any(value is None for value in values):
        result["reason"] = "scores_must_be_finite_positive_numbers"
        return result
    result["score_values"] = values
    logs = [math.log(value) for value in values]
    mean_log = statistics.mean(logs)
    sd = statistics.stdev(logs)
    if not sd > 0:
        result["reason"] = "zero_log_score_variance"
        return result
    raw_lower = mean_log - CRITICAL_T * sd / math.sqrt(SAMPLE_COUNT)
    margin = 1e-12 * (1 + abs(raw_lower))
    lower_log = raw_lower - margin
    geometric_mean = math.exp(mean_log)
    lower_speedup = math.exp(lower_log)
    certified = lower_log > math.log(target) and lower_speedup > target
    result.update(
        valid=True,
        certified=certified,
        reason="floor_passed" if certified else "lower_bound_not_above_floor",
        mean_log=mean_log,
        sample_log_sd=sd,
        geometric_mean=geometric_mean,
        lower_log=lower_log,
        lower_speedup=lower_speedup,
        numeric_margin_log=margin,
    )
    return result
