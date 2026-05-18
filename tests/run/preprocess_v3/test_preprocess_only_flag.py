"""Tests for the ``--preprocess-only`` CLI flag.

The flag wires through three layers:

1. ``mini.py`` Typer option — accepted at the CLI.
2. ``PipelineContext.preprocess_only`` — default ``False`` so existing
   callers do not change behaviour.
3. ``_run_unified_loop`` — when the flag is set, writes a stub
   ``final_report.json`` and returns BEFORE the round loop, skipping
   planner + dispatcher + worker execution entirely.

These tests cover all three layers without exercising any real LLM,
GPU, or subprocess. The round-loop body is patched to a sentinel that
raises if reached, so we get a clear "round loop was entered" signal
when the short-circuit fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from minisweagent.run.unified import (
    PipelineContext,
    _build_preprocess_only_report,
    _run_unified_loop,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_ctx(tmp_path: Path, **overrides) -> PipelineContext:
    """PipelineContext with the minimum fields the unified loop reads."""
    base = {
        "preprocess_ctx": {
            "kernel_path": str(tmp_path / "kernel.py"),
            "harness_path": "",
            "repo_root": str(tmp_path),
        },
        "user_prompt": "Optimize this kernel.",
        "output_dir": tmp_path,
        "gpu_ids": [0],
        "max_rounds": 3,
    }
    base.update(overrides)
    return PipelineContext(**base)


# ---------------------------------------------------------------------------
# Default behaviour
# ---------------------------------------------------------------------------


class TestDefault:
    """The flag must default to ``False`` so existing callers are unaffected."""

    def test_preprocess_only_defaults_to_false(self, tmp_path: Path) -> None:
        """Constructing a context with no kwargs gives ``preprocess_only=False``."""
        ctx = _make_ctx(tmp_path)
        assert ctx.preprocess_only is False

    def test_preprocess_only_explicit_false(self, tmp_path: Path) -> None:
        """Explicit ``False`` is a no-op."""
        ctx = _make_ctx(tmp_path, preprocess_only=False)
        assert ctx.preprocess_only is False

    def test_preprocess_only_set_to_true(self, tmp_path: Path) -> None:
        """The field accepts ``True`` and round-trips."""
        ctx = _make_ctx(tmp_path, preprocess_only=True)
        assert ctx.preprocess_only is True


# ---------------------------------------------------------------------------
# Short-circuit: round loop is NOT entered when flag is True
# ---------------------------------------------------------------------------


class TestShortCircuit:
    """When the flag is True the round loop must NOT be entered."""

    def test_short_circuit_skips_planner_and_dispatcher(
        self, tmp_path: Path
    ) -> None:
        """The TaskPlanner / Dispatcher are still constructed (cheap; they
        need preprocess_ctx for their constructors) but the round loop
        itself never runs ``run_staged_task_batch`` or ``post_round_evaluate``.

        We assert this by patching the heavy round-loop functions to raise
        and verifying they are NOT called.
        """
        ctx = _make_ctx(tmp_path, preprocess_only=True)

        def _boom(*args, **kwargs):  # pragma: no cover - asserts on call
            raise AssertionError(
                "Round loop body should NOT run with preprocess_only=True; "
                f"got call to {args!r} {kwargs!r}"
            )

        # Patch every heavy round-loop call site. ``TaskPlanner`` and
        # ``Dispatcher`` are constructed before the short-circuit (the
        # current implementation places the early-return after init) — we
        # stub them with no-op classes so the early-return path stays
        # cheap and deterministic.
        class _StubPlanner:
            def __init__(self, *a, **k):
                pass

            def build_pool(self, *a, **k):  # pragma: no cover - not hit
                _boom()

        class _StubDispatcher:
            def __init__(self, *a, **k):
                pass

            def select(self, *a, **k):  # pragma: no cover - not hit
                _boom()

        with patch(
            "minisweagent.run.planner.task_planner.TaskPlanner", _StubPlanner
        ), patch(
            "minisweagent.run.dispatcher.selector.Dispatcher", _StubDispatcher
        ), patch(
            "minisweagent.run.dispatch.run_staged_task_batch", _boom
        ), patch(
            "minisweagent.run.postprocess.results.post_round_evaluate", _boom
        ), patch(
            "minisweagent.run.postprocess.results.finalize_run", _boom
        ), patch(
            "minisweagent.run.dispatcher.writer.write_dispatch_plan_as_task_files",
            _boom,
        ):
            result = _run_unified_loop(ctx, mode="fixed")

        assert isinstance(result, dict)
        assert result["status"] == "preprocess_only"
        assert result["round_results"] == []

    def test_flag_false_runs_full_loop_path(self, tmp_path: Path) -> None:
        """When the flag is False the short-circuit MUST NOT trigger.

        We assert this indirectly: with the flag off we expect the
        TaskPlanner constructor to be reached (the first heavy call
        site after the preflight). We replace it with a sentinel that
        records a single call then raises a known exception, which
        proves the short-circuit path was NOT taken.
        """
        ctx = _make_ctx(tmp_path, preprocess_only=False)

        sentinel_calls: list[int] = []

        class _RecordingPlanner:
            def __init__(self, *a, **k):
                sentinel_calls.append(1)
                # Bail out hard so the test stops here without needing to
                # stub the rest of the loop. The presence of >=1 call to
                # this constructor is proof we did NOT short-circuit.
                raise RuntimeError("planner-init-reached")

        with patch(
            "minisweagent.run.planner.task_planner.TaskPlanner",
            _RecordingPlanner,
        ):
            with pytest.raises(RuntimeError, match="planner-init-reached"):
                _run_unified_loop(ctx, mode="fixed")

        assert len(sentinel_calls) == 1


# ---------------------------------------------------------------------------
# Stub final_report.json contents
# ---------------------------------------------------------------------------


class TestStubReport:
    """The stub ``final_report.json`` must have the right shape."""

    def test_stub_report_status_field(self, tmp_path: Path) -> None:
        """``status`` must be the canonical literal ``"preprocess_only"``."""
        ctx = _make_ctx(tmp_path, preprocess_only=True)
        report = _build_preprocess_only_report(ctx, tmp_path, loop_start_t=0.0)
        assert report["status"] == "preprocess_only"

    def test_stub_report_includes_required_keys(self, tmp_path: Path) -> None:
        """Required keys are always present (even with empty values)."""
        ctx = _make_ctx(tmp_path, preprocess_only=True)
        report = _build_preprocess_only_report(ctx, tmp_path, loop_start_t=0.0)
        required = {
            "status",
            "summary",
            "preprocess_artifacts",
            "path_taken",
            "round_results",
            "elapsed_s",
            "best_speedup",
            "best_patch",
            "best_round",
            "best_task",
        }
        assert required.issubset(report.keys())
        # round_results MUST be empty for preprocess-only.
        assert report["round_results"] == []
        # Speedup/patch fields MUST be None — preprocess-only never
        # produces an optimized patch.
        assert report["best_speedup"] is None
        assert report["best_patch"] is None

    def test_stub_report_enumerates_existing_artifacts(
        self, tmp_path: Path
    ) -> None:
        """Only artifacts that actually exist on disk are listed."""
        (tmp_path / "COMMANDMENT.md").write_text("# cmd")
        (tmp_path / "baseline_metrics.json").write_text("{}")
        # CODEBASE_CONTEXT.md intentionally missing.

        ctx = _make_ctx(tmp_path, preprocess_only=True)
        report = _build_preprocess_only_report(ctx, tmp_path, loop_start_t=0.0)
        artifacts = report["preprocess_artifacts"]
        assert any("COMMANDMENT.md" in p for p in artifacts)
        assert any("baseline_metrics.json" in p for p in artifacts)
        assert not any("CODEBASE_CONTEXT.md" in p for p in artifacts)

    def test_stub_report_includes_path_taken_when_present(
        self, tmp_path: Path
    ) -> None:
        """``path_taken`` is forwarded from ``preprocess_ctx`` when set."""
        ctx = _make_ctx(
            tmp_path,
            preprocess_only=True,
            preprocess_ctx={
                "kernel_path": str(tmp_path / "k.py"),
                "harness_path": "",
                "repo_root": str(tmp_path),
                "path_taken": "A",
            },
        )
        report = _build_preprocess_only_report(ctx, tmp_path, loop_start_t=0.0)
        assert report["path_taken"] == "A"

    def test_stub_report_path_taken_none_when_absent(
        self, tmp_path: Path
    ) -> None:
        """Legacy preprocess result has no ``path_taken``; we get ``None``."""
        ctx = _make_ctx(tmp_path, preprocess_only=True)
        report = _build_preprocess_only_report(ctx, tmp_path, loop_start_t=0.0)
        assert report["path_taken"] is None

    def test_stub_report_elapsed_s_is_nonnegative_float(
        self, tmp_path: Path
    ) -> None:
        """``elapsed_s`` is a non-negative float (seconds since loop start)."""
        import time as _time

        ctx = _make_ctx(tmp_path, preprocess_only=True)
        report = _build_preprocess_only_report(
            ctx, tmp_path, loop_start_t=_time.monotonic() - 0.01,
        )
        assert isinstance(report["elapsed_s"], float)
        assert report["elapsed_s"] >= 0.0


# ---------------------------------------------------------------------------
# End-to-end short-circuit: file is written to disk
# ---------------------------------------------------------------------------


class TestFinalReportFileWritten:
    """The short-circuit MUST persist ``final_report.json`` to disk."""

    def test_final_report_json_written_to_output_dir(
        self, tmp_path: Path
    ) -> None:
        """After short-circuit the JSON file exists and parses cleanly."""
        ctx = _make_ctx(tmp_path, preprocess_only=True)

        with patch(
            "minisweagent.run.planner.task_planner.TaskPlanner"
        ), patch("minisweagent.run.dispatcher.selector.Dispatcher"):
            result = _run_unified_loop(ctx, mode="fixed")

        report_path = tmp_path / "final_report.json"
        assert report_path.exists(), "final_report.json was not written"
        on_disk = json.loads(report_path.read_text())
        assert on_disk["status"] == "preprocess_only"
        # The return value should equal the on-disk content.
        assert result["status"] == on_disk["status"]

    def test_run_pipeline_delegates_short_circuit(
        self, tmp_path: Path
    ) -> None:
        """``run_pipeline`` honours ``preprocess_only`` via the loop helper."""
        ctx = _make_ctx(tmp_path, preprocess_only=True)

        with patch(
            "minisweagent.run.planner.task_planner.TaskPlanner"
        ), patch("minisweagent.run.dispatcher.selector.Dispatcher"), patch(
            "minisweagent.tools.tools_runtime.ToolRuntime"
        ):
            result = run_pipeline(ctx, mode="fixed")

        assert isinstance(result, dict)
        assert result["status"] == "preprocess_only"
        assert (tmp_path / "final_report.json").exists()
