"""RGMem-inspired confidence scoring for memory outcomes.

Each stored outcome carries a confidence score that evolves:
  - Boosted on successful reuse (+0.1)
  - Decayed on failure (-0.15)
  - Slight decay on retrieval without feedback (-0.02)

Reference: RGMem (2025) -- Retrieval-guided confidence scoring.
Reported: +25% retrieval precision.
"""

from __future__ import annotations

from typing import Any

DEFAULT_CONFIDENCE = 0.5
BOOST_ON_SUCCESS = 0.1
DECAY_ON_FAILURE = 0.15
DECAY_ON_RETRIEVAL = 0.02
MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 1.0


def get_confidence(outcome: dict[str, Any]) -> float:
    return float(outcome.get("_confidence", outcome.get("confidence", DEFAULT_CONFIDENCE)))


def boost_confidence(outcome: dict[str, Any], amount: float = BOOST_ON_SUCCESS) -> dict[str, Any]:
    c = get_confidence(outcome)
    outcome["_confidence"] = min(MAX_CONFIDENCE, c + amount)
    return outcome


def decay_confidence(outcome: dict[str, Any], amount: float = DECAY_ON_FAILURE) -> dict[str, Any]:
    c = get_confidence(outcome)
    outcome["_confidence"] = max(MIN_CONFIDENCE, c - amount)
    return outcome


def decay_on_retrieval(outcome: dict[str, Any]) -> dict[str, Any]:
    c = get_confidence(outcome)
    outcome["_confidence"] = max(MIN_CONFIDENCE, c - DECAY_ON_RETRIEVAL)
    return outcome


def filter_by_confidence(
    outcomes: list[dict[str, Any]],
    min_confidence: float = MIN_CONFIDENCE,
) -> list[dict[str, Any]]:
    return [o for o in outcomes if get_confidence(o) >= min_confidence]


def rank_by_confidence(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        outcomes,
        key=lambda o: (-get_confidence(o), -float(o.get("speedup_achieved", 0))),
    )


def apply_confidence_layer(
    outcomes: list[dict[str, Any]],
    min_confidence: float = MIN_CONFIDENCE,
) -> list[dict[str, Any]]:
    filtered = filter_by_confidence(outcomes, min_confidence)
    for o in filtered:
        decay_on_retrieval(o)
    return rank_by_confidence(filtered)
