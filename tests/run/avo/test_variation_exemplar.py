"""Unit tests for the best-program exemplar injection (P-B, GPU-free)."""

from __future__ import annotations

from pathlib import Path

from minisweagent.run.avo.lineage_store import LineageStore
from minisweagent.run.avo.result import AttemptRecord, VariationResult
from minisweagent.run.avo.variation_step import build_best_exemplar, compose_task


def _commit(store: LineageStore, step: int, speedup: float, tmp: Path) -> None:
    patch = tmp / f"p{step}.patch"
    patch.write_text(f"--- a/kernel.py\n+++ b/kernel.py\n@@ step {step} @@\n+# {speedup}x change\n", encoding="utf-8")
    store.maybe_commit(
        VariationResult(
            step_index=step,
            step_dir=tmp,
            strategy=f"strat_{step}",
            attempts=[AttemptRecord(correctness_passed=True, verified_speedup=speedup)],
            best_patch_path=patch,
            best_speedup=speedup,
            best_correct=True,
        )
    )


def test_no_exemplar_for_baseline(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    # v0 baseline has no patch → no exemplar.
    assert build_best_exemplar(store) is None


def test_exemplar_includes_best_diff_and_metrics(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    _commit(store, 1, 1.25, tmp_path)

    exemplar = build_best_exemplar(store)
    assert exemplar is not None
    assert "Current best implementation" in exemplar
    assert "v1" in exemplar
    assert "1.25" in exemplar
    assert "strat_1" in exemplar
    assert "```diff" in exemplar
    assert "kernel.py" in exemplar


def test_exemplar_diff_is_truncated(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    big_patch = tmp_path / "big.patch"
    big_patch.write_text("x" * 10000, encoding="utf-8")
    store.maybe_commit(
        VariationResult(
            step_index=1,
            step_dir=tmp_path,
            strategy="big",
            attempts=[AttemptRecord(correctness_passed=True, verified_speedup=1.5)],
            best_patch_path=big_patch,
            best_speedup=1.5,
            best_correct=True,
        )
    )
    exemplar = build_best_exemplar(store)
    assert exemplar is not None
    assert "diff truncated" in exemplar


def test_compose_task_orders_exemplar_before_memory(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    body = compose_task(
        "BASE_TASK",
        store,
        {"strategy": "vectorize"},
        memory_summary="MEMORY_BLOCK",
        exemplar="EXEMPLAR_BLOCK",
    )
    assert "EXEMPLAR_BLOCK" in body
    assert "MEMORY_BLOCK" in body
    assert "BASE_TASK" in body
    # contract → exemplar → memory → base task
    assert body.index("EXEMPLAR_BLOCK") < body.index("MEMORY_BLOCK") < body.index("BASE_TASK")
