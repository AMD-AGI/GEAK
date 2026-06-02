"""Unit tests for the best-program exemplar injection (P-B, GPU-free)."""

from __future__ import annotations

from pathlib import Path

from minisweagent.run.avo.lineage_store import LineageStore
from minisweagent.run.avo.result import AttemptRecord, VariationResult
from minisweagent.run.avo.variation_step import build_best_exemplar, build_lineage_context, compose_task


def _commit(store: LineageStore, step: int, speedup: float, tmp: Path, per_shape: dict | None = None) -> None:
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
            per_shape_speedups=per_shape or {},
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


def test_lineage_context_none_with_only_baseline(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    assert build_lineage_context(store, k=3) is None


def test_lineage_context_excludes_best_shows_others(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    _commit(store, 1, 1.20, tmp_path, {"sA": 1.5, "sB": 0.95})
    _commit(store, 2, 1.45, tmp_path, {"sA": 1.6, "sB": 1.3})
    _commit(store, 3, 1.60, tmp_path, {"sA": 1.9, "sB": 1.4})

    ctx = build_lineage_context(store, k=3)
    assert ctx is not None
    # best (v3) is the exemplar; lineage context shows OTHER versions only.
    version_blocks = [b.split(" ")[0] for b in ctx.split("### ")[1:]]
    assert "v3" not in version_blocks
    assert "v2" in version_blocks and "v1" in version_blocks
    assert "per-shape:" in ctx and "```diff" in ctx


def test_lineage_context_disabled(tmp_path: Path):
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path)
    _commit(store, 1, 1.20, tmp_path)
    assert build_lineage_context(store, k=0) is None


def test_per_shape_persists_on_node(tmp_path: Path):
    state_dir = tmp_path / "avo_state"
    store = LineageStore(state_dir)
    store.seed_from_baseline(tmp_path)
    _commit(store, 1, 1.3, tmp_path, {"sA": 1.6, "sB": 1.1})
    reloaded = LineageStore(state_dir)
    assert reloaded.committed[1].per_shape == {"sA": 1.6, "sB": 1.1}


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
