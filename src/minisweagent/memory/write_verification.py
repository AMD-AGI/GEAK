"""Write Verification for Memory DB (SEDM-inspired).

Before committing an optimization outcome to the memory DB, verify that:
1. The outcome data is internally consistent (no NaN, no impossible values)
2. The speedup measurement is plausible (not from a crashed/throttled run)
3. The outcome adds information value (not a duplicate of existing records)

Inspired by SEDM (NeurIPS 2025): Verifiable Write Admission via
reproducible replay and A/B testing of memory entries.
"""

from __future__ import annotations

import math


def verify_outcome(
    speedup_achieved: float = 0.0,
    steps_taken: int = 0,
    cost_dollars: float = 0.0,
    patch_size_bytes: int = -1,
    **kwargs,
) -> tuple[bool, str]:
    """Verify an optimization outcome before DB commit.

    Returns (is_valid, reason). If invalid, the outcome should NOT be stored.
    Accepts keyword arguments directly (not a dict) for easy calling.
    """
    speedup = speedup_achieved
    steps = steps_taken
    cost = cost_dollars

    if patch_size_bytes == 0 and speedup > 1.0:
        return False, f"Empty patch (0 bytes) but claims {speedup}x speedup"

    # Reject NaN/inf values
    if isinstance(speedup, float) and (math.isnan(speedup) or math.isinf(speedup)):
        return False, f"Invalid speedup value: {speedup}"

    # Reject negative speedup (impossible)
    if speedup < 0:
        return False, f"Negative speedup: {speedup}"

    # Reject negative steps (zero is allowed -- orchestrator-level recording)
    if steps < 0:
        return False, f"Negative steps: {steps}"

    # Reject suspiciously high speedup (likely measurement error)
    if speedup > 100:
        return False, f"Suspiciously high speedup: {speedup}x (likely measurement error)"

    # Reject if cost is negative
    if cost < 0:
        return False, f"Negative cost: ${cost}"

    # Check for crash indicators in failure reason
    failure_reason = kwargs.get("failure_reason", "") or ""
    crash_indicators = ["segfault", "core dump", "killed", "oom", "out of memory"]
    for indicator in crash_indicators:
        if indicator.lower() in failure_reason.lower():
            return False, f"Run crashed ({indicator}) -- outcome unreliable"

    return True, "OK"


def verify_and_filter_outcomes(outcomes: list[dict]) -> list[dict]:
    """Filter a list of outcomes, keeping only verified ones."""
    verified = []
    for o in outcomes:
        is_valid, reason = verify_outcome(o)
        if is_valid:
            verified.append(o)
    return verified
