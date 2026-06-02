"""Unit tests for the AVO LineageStore commit gate + persistence (GPU-free)."""

from __future__ import annotations

import json
from pathlib import Path

from minisweagent.run.avo.lineage_store import LineageStore
from minisweagent.run.avo.result import AttemptRecord, VariationResult


def _make_result(step: int, speedup: float | None, *, correct: bool, patch_dir: Path) -> VariationResult:
    patch_path = None
    if speedup is not None and correct:
        patch_path = patch_dir / f"patch_{step}.patch"
        patch_path.write_text(f"--- a\n+++ b\n@@ step {step} speedup {speedup}\n", encoding="utf-8")
    return VariationResult(
        step_index=step,
        step_dir=patch_dir,
        strategy=f"strat_{step}",
        attempts=[AttemptRecord(strategy=f"strat_{step}", correctness_passed=correct, verified_speedup=speedup)],
        best_patch_path=patch_path,
        best_speedup=speedup,
        best_correct=correct,
    )


def test_seed_baseline_creates_v0(tmp_path: Path):
    (tmp_path / "baseline_metrics.json").write_text(json.dumps({"latency_ms": 10.0}), encoding="utf-8")
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    assert store.best_id == "v0"
    assert store.best_speedup == 1.0
    assert store.best_node.latency_ms == 10.0


def test_commit_gate_accepts_improvement(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    committed = store.maybe_commit(_make_result(1, 1.20, correct=True, patch_dir=tmp_path))
    assert committed is True
    assert store.best_id == "v1"
    assert store.best_speedup == 1.20


def test_commit_gate_rejects_regression(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    store.maybe_commit(_make_result(1, 1.20, correct=True, patch_dir=tmp_path))
    # A slower candidate must not be committed.
    committed = store.maybe_commit(_make_result(2, 1.05, correct=True, patch_dir=tmp_path))
    assert committed is False
    assert store.best_id == "v1"


def test_commit_gate_rejects_incorrect(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    committed = store.maybe_commit(_make_result(1, 2.0, correct=False, patch_dir=tmp_path))
    assert committed is False
    assert store.best_id == "v0"


def test_commit_gate_rejects_unverified(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    # correct but no verified speedup → not a commit candidate
    committed = store.maybe_commit(_make_result(1, None, correct=True, patch_dir=tmp_path))
    assert committed is False


def test_attempts_appended(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    store.record_attempts(_make_result(1, 1.05, correct=True, patch_dir=tmp_path))
    store.record_attempts(_make_result(2, None, correct=False, patch_dir=tmp_path))
    rows = [json.loads(line) for line in store.attempts_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["step_index"] == 1


def test_persistence_round_trip(tmp_path: Path):
    state_dir = tmp_path / "avo_state"
    store = LineageStore(state_dir)
    store.seed_from_baseline(tmp_path)
    store.maybe_commit(_make_result(1, 1.30, correct=True, patch_dir=tmp_path))

    reloaded = LineageStore(state_dir)
    assert reloaded.best_id == "v1"
    assert reloaded.best_speedup == 1.30
    assert len(reloaded.committed) == 2


def test_direction_round_trip(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.set_direction("vectorized_load", assigned_by="supervisor", supervisor_cycle=2)
    d = store.current_direction()
    assert d["strategy"] == "vectorized_load"
    assert d["assigned_by"] == "supervisor"
    assert d["supervisor_cycle"] == 2
