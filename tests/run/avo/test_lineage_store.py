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


def test_commit_gate_rejects_lazy_no_gain(tmp_path: Path):
    # A correct, verified, but ~1.0x ("lazy optimization") candidate must NOT
    # enter the lineage as the first improvement (Kernel-Smith anti-lazy guard).
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    committed = store.maybe_commit(_make_result(1, 1.0, correct=True, patch_dir=tmp_path))
    assert committed is False
    assert store.best_id == "v0"


def test_commit_gate_respects_custom_min_speedup(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state", min_commit_speedup=1.10)
    store.seed_from_baseline(tmp_path)
    # 1.05x is a real gain but below the configured 1.10 floor → rejected.
    assert store.maybe_commit(_make_result(1, 1.05, correct=True, patch_dir=tmp_path)) is False
    # 1.20x clears the floor → committed.
    assert store.maybe_commit(_make_result(2, 1.20, correct=True, patch_dir=tmp_path)) is True
    assert store.best_speedup == 1.20


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


def test_active_pointer_advances_on_commit(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    store.maybe_commit(_make_result(1, 1.20, correct=True, patch_dir=tmp_path))
    store.maybe_commit(_make_result(2, 1.50, correct=True, patch_dir=tmp_path))
    assert store.active_best_id == "v2"
    assert store.best_speedup == 1.50


def test_backtrack_pointer(tmp_path: Path):
    state_dir = tmp_path / "avo_state"
    store = LineageStore(state_dir)
    store.seed_from_baseline(tmp_path)
    store.maybe_commit(_make_result(1, 1.20, correct=True, patch_dir=tmp_path))
    store.maybe_commit(_make_result(2, 1.50, correct=True, patch_dir=tmp_path))

    assert store.set_best_pointer("v1") is True
    assert store.best_id == "v1"
    assert store.best_speedup == 1.20

    # persists across reload
    reloaded = LineageStore(state_dir)
    assert reloaded.active_best_id == "v1"
    assert reloaded.best_id == "v1"

    # unknown target is rejected
    assert store.set_best_pointer("v404") is False


def test_backtrack_then_commit_branches_from_pointer(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    store.maybe_commit(_make_result(1, 1.20, correct=True, patch_dir=tmp_path))
    store.maybe_commit(_make_result(2, 1.50, correct=True, patch_dir=tmp_path))
    store.set_best_pointer("v1")  # back to 1.20

    # 1.30 beats the backtracked tip (1.20) so it commits, and the tip advances.
    assert store.maybe_commit(_make_result(3, 1.30, correct=True, patch_dir=tmp_path)) is True
    assert store.active_best_id == store.best_id
    assert store.best_speedup == 1.30


class _FB:
    def __init__(self, v):
        self.verified_speedup = v


class _RoundEval:
    def __init__(self, patch, verified, bench):
        self.best_patch = patch
        self.full_benchmark = _FB(verified)
        self.benchmark_speedup = bench


def test_commit_from_round_prefers_verified(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    patch = tmp_path / "rescue.patch"
    patch.write_text("rescue", encoding="utf-8")
    assert store.commit_from_round(_RoundEval(str(patch), 2.0, 1.8)) is True
    assert store.best_speedup == 2.0


def test_commit_from_round_no_patch_rejected(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    assert store.commit_from_round(_RoundEval("", 2.0, 1.8)) is False
    assert store.best_id == "v0"


def _make_result_with_shapes(step: int, speedup: float, per_shape: dict, patch_dir: Path):
    patch = patch_dir / f"p{step}.patch"
    patch.write_text(f"patch {step}", encoding="utf-8")
    return VariationResult(
        step_index=step,
        step_dir=patch_dir,
        strategy=f"s{step}",
        attempts=[AttemptRecord(correctness_passed=True, verified_speedup=speedup)],
        best_patch_path=patch,
        best_speedup=speedup,
        best_correct=True,
        per_shape_speedups=per_shape,
    )


def test_per_shape_guard_blocks_regressed_shape(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state", min_per_shape_speedup=0.95)
    store.seed_from_baseline(tmp_path)
    # geomean passes but shapeB regresses below 0.95 → rejected.
    blocked = _make_result_with_shapes(1, 1.30, {"A": 2.0, "B": 0.85}, tmp_path)
    assert store.maybe_commit(blocked) is False
    assert store.best_id == "v0"
    # all shapes clear the floor → committed.
    ok = _make_result_with_shapes(2, 1.30, {"A": 1.4, "B": 1.2}, tmp_path)
    assert store.maybe_commit(ok) is True
    assert store.best_id == "v1"


def test_per_shape_guard_disabled_by_default(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")  # min_per_shape_speedup defaults to 0.0
    store.seed_from_baseline(tmp_path)
    weak = _make_result_with_shapes(1, 1.30, {"A": 2.0, "B": 0.5}, tmp_path)
    assert store.maybe_commit(weak) is True  # guard off → geomean-only behavior
