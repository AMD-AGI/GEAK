"""Tests for the unified round loop helpers in ``run/unified.py``.

Tests the deterministic helpers that the unified loop relies on:
  - ``_enrich_prompt_for_round``: round > 1 injects previous-best summary
  - ``_should_stop_before_round``: soft_stop / deadline gating
  - ``_build_postprocess_ctx``: maps PipelineContext to dict
  - ``_resolve_task_file_meta``: resolves metadata paths
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from minisweagent.run.unified import (
    PipelineContext,
    _build_postprocess_ctx,
    _enrich_prompt_for_round,
    _resolve_task_file_meta,
    _should_stop_before_round,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _make_ctx(tmp_path: Path | None = None, **overrides) -> PipelineContext:
    base = {
        "preprocess_ctx": {"kernel_path": "/tmp/k.py"},
        "user_prompt": "Optimize this kernel.",
        "output_dir": tmp_path or Path("/tmp/out"),
        "gpu_ids": [0],
    }
    base.update(overrides)
    return PipelineContext(**base)


# ──────────────────────────────────────────────────────────────────────
# _enrich_prompt_for_round
# ──────────────────────────────────────────────────────────────────────


class TestEnrichPrompt:
    def test_round1_returns_base_prompt(self):
        result = _enrich_prompt_for_round("base prompt", "fixed", 1, [])
        assert result == "base prompt"

    def test_fixed_round2_with_prior_results_adds_summary(self):
        evals = [{"benchmark_speedup": 1.5}]
        result = _enrich_prompt_for_round("base prompt", "fixed", 2, evals)
        assert "Previous Rounds" in result
        assert "1.500x" in result
        assert "base prompt" in result

    def test_fixed_round2_no_improvement_returns_base(self):
        evals = [{"benchmark_speedup": 1.0}]
        result = _enrich_prompt_for_round("base prompt", "fixed", 2, evals)
        assert result == "base prompt"

    def test_planned_mode_returns_base_unchanged(self):
        evals = [{"benchmark_speedup": 2.0}]
        result = _enrich_prompt_for_round("base prompt", "planned", 2, evals)
        assert result == "base prompt"

    def test_mixed_mode_returns_base_unchanged(self):
        evals = [{"benchmark_speedup": 2.0}]
        result = _enrich_prompt_for_round("base prompt", "mixed", 2, evals)
        assert result == "base prompt"

    def test_empty_evals_returns_base(self):
        result = _enrich_prompt_for_round("base prompt", "fixed", 2, [])
        assert result == "base prompt"


# ──────────────────────────────────────────────────────────────────────
# _should_stop_before_round
# ──────────────────────────────────────────────────────────────────────


class TestShouldStop:
    def test_no_stop_signals(self):
        ctx = _make_ctx()
        assert _should_stop_before_round(ctx) is False

    def test_soft_stop_set(self):
        evt = threading.Event()
        evt.set()
        ctx = _make_ctx(soft_stop=evt)
        assert _should_stop_before_round(ctx) is True

    def test_soft_stop_not_set(self):
        evt = threading.Event()
        ctx = _make_ctx(soft_stop=evt)
        assert _should_stop_before_round(ctx) is False

    def test_deadline_expired(self):
        deadline = SimpleNamespace(expired=lambda: True)
        ctx = _make_ctx(deadline=deadline)
        assert _should_stop_before_round(ctx) is True

    def test_deadline_not_expired(self):
        deadline = SimpleNamespace(expired=lambda: False)
        ctx = _make_ctx(deadline=deadline)
        assert _should_stop_before_round(ctx) is False


# ──────────────────────────────────────────────────────────────────────
# _build_postprocess_ctx
# ──────────────────────────────────────────────────────────────────────


class TestBuildPostprocessCtx:
    def test_basic_fields(self, tmp_path: Path):
        ctx = _make_ctx(
            tmp_path,
            preprocess_ctx={
                "kernel_path": "/tmp/k.py",
                "harness_path": "/tmp/h.py",
                "baseline_metrics": {"latency": 10.0},
            },
            repo=tmp_path,
            model=MagicMock(),
            model_factory=MagicMock(),
            user_prompt="optimize it",
            rag_enabled=True,
        )
        result = _build_postprocess_ctx(ctx)

        assert result["output_dir"] == str(tmp_path)
        assert result["repo_root"] == str(tmp_path)
        assert result["harness_path"] == "/tmp/h.py"
        assert result["gpu_ids"] == [0]
        assert result["kernel_path"] == "/tmp/k.py"
        assert result["baseline_metrics"] == {"latency": 10.0}
        assert result["starting_patch"] == ""
        assert result["_best_global_speedup"] == 0
        assert result["user_instructions"] == "optimize it"
        assert result["rag_enabled"] is True

    def test_preprocess_ctx_keys_merged(self, tmp_path: Path):
        ctx = _make_ctx(
            tmp_path,
            preprocess_ctx={"kernel_path": "/tmp/k.py", "extra_key": "value"},
        )
        result = _build_postprocess_ctx(ctx)
        assert result["extra_key"] == "value"


# ──────────────────────────────────────────────────────────────────────
# _resolve_task_file_meta
# ──────────────────────────────────────────────────────────────────────


class TestResolveTaskFileMeta:
    def test_existing_files_resolved(self, tmp_path: Path):
        (tmp_path / "COMMANDMENT.md").write_text("cmd")
        (tmp_path / "baseline_metrics.json").write_text("{}")

        meta = _resolve_task_file_meta(
            tmp_path, "/tmp/k.py", "/tmp/repo", "/tmp/h.py", "pytest"
        )
        assert meta["commandment"] == str(tmp_path / "COMMANDMENT.md")
        assert meta["baseline_metrics"] == str(tmp_path / "baseline_metrics.json")
        assert meta["harness_path"] == "/tmp/h.py"
        assert meta["kernel_path"] == "/tmp/k.py"
        assert meta["repo_root"] == "/tmp/repo"
        assert meta["test_command"] == "pytest"

    def test_missing_files_are_none(self, tmp_path: Path):
        meta = _resolve_task_file_meta(tmp_path, "", "", "", None)
        assert meta["commandment"] is None
        assert meta["baseline_metrics"] is None
        assert meta["profiling"] is None
        assert meta["harness_path"] is None
        assert meta["kernel_path"] is None
        assert meta["test_command"] is None
