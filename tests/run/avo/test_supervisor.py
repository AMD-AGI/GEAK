"""Unit tests for supervisor directive self-validation (objectivity guard)."""

from __future__ import annotations

import json

from minisweagent.run.avo.supervisor import _validate_directive


def _bundle(failed, current):
    return json.dumps(
        {"strategy_state": {"failed": failed, "successful": []}, "current_direction": {"strategy": current}}
    )


def test_validate_drops_already_tried_and_current():
    bundle = _bundle(["tiling", "vectorize"], "tiling")
    directive = {
        "diagnosis": "x",
        "mark_failed": [],
        "new_strategies": [
            {"name": "tiling"},  # already failed
            {"name": "warp_specialization"},  # novel
            {"name": "VECTORIZE"},  # already failed (case-insensitive)
        ],
        "backtrack_to_id": None,
    }
    out = _validate_directive(dict(directive), bundle)
    names = [s["name"] for s in out["new_strategies"]]
    assert names == ["warp_specialization"]


def test_validate_dedups_within_proposal():
    bundle = _bundle([], "")
    directive = {"new_strategies": [{"name": "pipeline"}, {"name": "pipeline"}, {"name": "occupancy"}]}
    out = _validate_directive(dict(directive), bundle)
    names = [s["name"] for s in out["new_strategies"]]
    assert names == ["pipeline", "occupancy"]


def test_validate_splices_fallback_when_all_tried():
    bundle = _bundle(["tiling", "vectorize"], "tiling")
    directive = {"new_strategies": [{"name": "tiling"}, {"name": "vectorize"}]}
    out = _validate_directive(dict(directive), bundle)
    assert len(out["new_strategies"]) >= 1
    assert out["new_strategies"][0]["name"].lower() not in {"tiling", "vectorize"}
