"""Mem0-inspired memory reconciliation (ADD/UPDATE/DELETE/NOOP).

When a new optimization outcome arrives, classify it against existing
memories to decide the appropriate action:
  ADD   -- genuinely new insight (no similar existing entry)
  UPDATE -- refines/strengthens an existing entry (same strategy, better data)
  DELETE -- contradicts an existing entry (conflicting strategies)
  NOOP  -- redundant (essentially duplicate of existing)

This prevents memory bloat and resolves contradictions automatically.

Reference: Mem0 (2025/2026) -- ADD/UPDATE/DELETE/NONE reconciliation.
Reported: +26% accuracy over OpenAI Memory.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _similarity(a: dict, b: dict) -> float:
    """Simple field-overlap similarity between two outcome dicts."""
    score = 0.0
    total = 0.0
    for key in ["kernel_category", "kernel_language", "bottleneck_type", "strategy_name"]:
        total += 1.0
        va = str(a.get(key, "")).lower().strip()
        vb = str(b.get(key, "")).lower().strip()
        if va and vb and va == vb:
            score += 1.0
        elif va and vb:
            common = set(va.split()) & set(vb.split())
            if common:
                score += 0.5
    return score / total if total > 0 else 0.0


def _are_contradicting(a: dict, b: dict) -> bool:
    """Check if two outcomes represent contradicting strategies.

    Two outcomes contradict if they target the same kernel category and
    bottleneck but one succeeded where the other failed with the same strategy,
    or they use directly opposing techniques.
    """
    if a.get("kernel_category") != b.get("kernel_category"):
        return False
    if a.get("strategy_name") != b.get("strategy_name"):
        return False
    a_success = bool(a.get("success"))
    b_success = bool(b.get("success"))
    if a_success != b_success:
        return True

    OPPOSING_PAIRS = [
        ("unroll", "reduce register"),
        ("vectorize", "scalar"),
        ("increase block", "decrease block"),
        ("fuse", "split"),
    ]
    a_tech = str(a.get("optimization_technique", "")).lower()
    b_tech = str(b.get("optimization_technique", "")).lower()
    for t1, t2 in OPPOSING_PAIRS:
        if (t1 in a_tech and t2 in b_tech) or (t2 in a_tech and t1 in b_tech):
            return True
    return False


def classify_action(
    new_outcome: dict[str, Any],
    existing_outcomes: list[dict[str, Any]],
    similarity_threshold: float = 0.75,
    contradiction_check: bool = True,
) -> tuple[str, dict[str, Any] | None]:
    """Classify how to handle a new outcome vs existing memories.

    Returns (action, matched_existing) where action is one of:
      "ADD"    -- store as new
      "UPDATE" -- update the matched existing entry
      "DELETE" -- remove the contradicted existing entry, then add new
      "NOOP"   -- skip (redundant)
    """
    best_match: dict[str, Any] | None = None
    best_sim = 0.0

    for existing in existing_outcomes:
        sim = _similarity(new_outcome, existing)
        if sim > best_sim:
            best_sim = sim
            best_match = existing

    if best_sim < 0.5:
        return "ADD", None

    if best_match is None:
        return "ADD", None

    if best_sim >= similarity_threshold:
        new_sp = float(new_outcome.get("speedup_achieved", 1.0))
        old_sp = float(best_match.get("speedup_achieved", 1.0))
        if abs(new_sp - old_sp) < 0.01:
            return "NOOP", best_match
        if contradiction_check and _are_contradicting(new_outcome, best_match):
            if new_sp > old_sp:
                return "DELETE", best_match
            else:
                return "NOOP", best_match
        return "UPDATE", best_match

    if contradiction_check and _are_contradicting(new_outcome, best_match):
        new_sp = float(new_outcome.get("speedup_achieved", 1.0))
        old_sp = float(best_match.get("speedup_achieved", 1.0))
        if new_sp > old_sp:
            return "DELETE", best_match

    return "ADD", None


def reconcile_and_store(
    new_outcome: dict[str, Any],
    store,
    existing_outcomes: list[dict[str, Any]] | None = None,
) -> str:
    """Full reconciliation pipeline: classify + execute action.

    Args:
        new_outcome: The new outcome to store.
        store: A MemoryStore instance with store/update/delete methods.
        existing_outcomes: Cached list of existing outcomes, or None to retrieve.

    Returns the action taken ("ADD", "UPDATE", "DELETE+ADD", "NOOP").
    """
    if existing_outcomes is None:
        existing_outcomes = store.retrieve(
            kernel_category=new_outcome.get("kernel_category"),
            limit=50,
        )

    action, matched = classify_action(new_outcome, existing_outcomes)

    if action == "ADD":
        store.store(new_outcome)
        logger.info("RECONCILE: ADD new outcome for %s", new_outcome.get("kernel_category"))
    elif action == "UPDATE" and matched:
        oid = str(matched.get("id", ""))
        if oid:
            store.update(oid, {
                "speedup_achieved": new_outcome.get("speedup_achieved", 1.0),
                "optimization_technique": new_outcome.get("optimization_technique", ""),
                "timestamp": new_outcome.get("timestamp", ""),
            })
            logger.info("RECONCILE: UPDATE existing outcome %s", oid)
        else:
            store.store(new_outcome)
            action = "ADD"
    elif action == "DELETE" and matched:
        oid = str(matched.get("id", ""))
        if oid:
            store.delete(oid)
        store.store(new_outcome)
        action = "DELETE+ADD"
        logger.info("RECONCILE: DELETE contradicting + ADD new for %s",
                     new_outcome.get("kernel_category"))
    else:
        logger.info("RECONCILE: NOOP for %s (redundant)", new_outcome.get("kernel_category"))

    return action
