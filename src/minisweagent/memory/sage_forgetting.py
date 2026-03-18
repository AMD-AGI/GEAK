"""SAGE-inspired Ebbinghaus forgetting for memory outcomes.

Strategies/outcomes not accessed in recent sessions decay exponentially.
Successful reuse strengthens the memory; neglect weakens it.

Formula: retention = e^(-t / strength)
Where:
  t = time since last access (in days)
  strength = base_strength * (1 + 0.2 * success_count)

When retention drops below min_retention (default 0.1), the memory is
prunable.

Reference: SAGE (2025) -- Ebbinghaus decay for agent memory management.
Reported: 30% memory reduction with <2% performance loss.
"""

from __future__ import annotations

import math
import time
from typing import Any

BASE_STRENGTH = 5.0
MIN_RETENTION = 0.1


def compute_retention(
    last_access_ts: str | float,
    success_count: int = 0,
    base_strength: float = BASE_STRENGTH,
) -> float:
    if isinstance(last_access_ts, str):
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(last_access_ts.replace("Z", "+00:00"))
            last_epoch = dt.timestamp()
        except (ValueError, TypeError):
            last_epoch = time.time()
    else:
        last_epoch = float(last_access_ts)

    t_days = max(0, (time.time() - last_epoch) / 86400)
    strength = base_strength * (1.0 + 0.2 * success_count)
    return max(0.0, min(1.0, math.exp(-t_days / strength)))


def should_forget(
    last_access_ts: str | float,
    success_count: int = 0,
    min_retention: float = MIN_RETENTION,
) -> bool:
    return compute_retention(last_access_ts, success_count) < min_retention


def apply_forgetting_filter(
    outcomes: list[dict[str, Any]],
    min_retention: float = MIN_RETENTION,
) -> list[dict[str, Any]]:
    filtered = []
    for o in outcomes:
        ts = o.get("last_retrieved") or o.get("timestamp") or ""
        sc = int(o.get("success_count", o.get("use_count", 0)))
        if not should_forget(ts, sc, min_retention):
            o["_retention"] = compute_retention(ts, sc)
            filtered.append(o)
    return filtered


def rank_by_retention(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for o in outcomes:
        ts = o.get("last_retrieved") or o.get("timestamp") or ""
        sc = int(o.get("success_count", o.get("use_count", 0)))
        o["_retention"] = compute_retention(ts, sc)
    return sorted(
        outcomes,
        key=lambda o: (-o.get("_retention", 0), -float(o.get("speedup_achieved", 0))),
    )
