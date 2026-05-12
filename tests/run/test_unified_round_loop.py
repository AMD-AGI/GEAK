"""Tests for the unified round loop in ``run/unified.py`` (fixed mode).

Pins the new behavior added in the unified-round-loop commit:

  - Fixed mode iterates ``ctx.max_rounds`` times (default 1 preserves
    legacy "one round × N parallel" shape).
  - Best result across rounds wins (track ``best_speedup``).
  - Per-round artefacts go to ``output_dir/round_N/`` when
    ``max_rounds > 1``; single-round callers keep flat output layout.
  - Round > 1 task bodies include the previous-best summary as an
    extra addendum.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.unified import (
    PipelineContext,
    _invoke_fixed_runner,
    _run_fixed,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _make_ctx(tmp_path: Path, *, max_rounds: int | None = None) -> PipelineContext:
    """Minimal PipelineContext for fixed-mode tests."""
    return PipelineContext(
        preprocess_ctx={"kernel_path": "/tmp/k.py"},
        user_prompt="Optimize this kernel.",
        kernel_language="triton",
        output_dir=tmp_path,
        gpu_ids=[0],
        model=MagicMock(),
        max_rounds=max_rounds,
        env=MagicMock(),
        env_class=MagicMock(),
        env_kwargs={},
        repo=tmp_path,
        num_parallel=2,
    )


def _fake_round_result(speedup: float | None = 1.2) -> SimpleNamespace:
    return SimpleNamespace(
        best_speedup=speedup,
        patch_id="p",
        agent_id=0,
        patch_dir=None,
        best_patch_file=None,
        llm_conclusion="",
    )


# ──────────────────────────────────────────────────────────────────────
# Round loop basics
# ──────────────────────────────────────────────────────────────────────


class TestRoundCount:
    def test_default_max_rounds_runs_once(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)  # max_rounds=None -> coerced to 1
        runner = MagicMock(return_value=_fake_round_result())

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            runner,
        ):
            result = _run_fixed(ctx)

        assert runner.call_count == 1
        assert result.best_speedup == 1.2

    def test_max_rounds_three_runs_three_times(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, max_rounds=3)
        runner = MagicMock(side_effect=[
            _fake_round_result(1.1),
            _fake_round_result(1.3),
            _fake_round_result(1.2),
        ])

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            runner,
        ):
            result = _run_fixed(ctx)

        assert runner.call_count == 3
        # Best across rounds wins (1.3 > 1.2 > 1.1)
        assert result.best_speedup == 1.3

    def test_max_rounds_zero_coerced_to_one(self, tmp_path: Path) -> None:
        """Defensive: 0 or negative max_rounds shouldn't skip the loop."""
        ctx = _make_ctx(tmp_path, max_rounds=0)
        runner = MagicMock(return_value=_fake_round_result())

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            runner,
        ):
            _run_fixed(ctx)

        assert runner.call_count == 1


# ──────────────────────────────────────────────────────────────────────
# Best-across-rounds selection
# ──────────────────────────────────────────────────────────────────────


class TestBestSelection:
    def test_picks_highest_speedup(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, max_rounds=3)
        results = [
            _fake_round_result(1.1),
            _fake_round_result(2.0),  # winner
            _fake_round_result(1.5),
        ]
        runner = MagicMock(side_effect=results)

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            runner,
        ):
            result = _run_fixed(ctx)
        assert result is results[1]
        assert result.best_speedup == 2.0

    def test_returns_none_when_all_rounds_fail(self, tmp_path: Path) -> None:
        """When every round returns None (no best), overall best is None."""
        ctx = _make_ctx(tmp_path, max_rounds=2)
        runner = MagicMock(side_effect=[None, None])

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            runner,
        ):
            result = _run_fixed(ctx)
        assert result is None

    def test_picks_winning_round_when_others_are_none(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, max_rounds=3)
        winner = _fake_round_result(1.4)
        runner = MagicMock(side_effect=[None, winner, None])

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            runner,
        ):
            result = _run_fixed(ctx)
        assert result is winner


# ──────────────────────────────────────────────────────────────────────
# Per-round artefact nesting
# ──────────────────────────────────────────────────────────────────────


class TestPerRoundArtefactDirs:
    def test_single_round_keeps_flat_output_dir(self, tmp_path: Path) -> None:
        """max_rounds=1 preserves legacy flat layout — no round_1/ subdir."""
        ctx = _make_ctx(tmp_path, max_rounds=1)
        captured_kwargs: list[dict] = []

        def _runner(**kwargs):
            captured_kwargs.append(kwargs)
            return _fake_round_result()

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            _runner,
        ):
            _run_fixed(ctx)

        assert len(captured_kwargs) == 1
        # output_dir stays flat (tmp_path itself, not tmp_path/round_1)
        assert captured_kwargs[0]["output_dir"] == tmp_path

    def test_multi_round_nests_output_dirs(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, max_rounds=3)
        captured_kwargs: list[dict] = []

        def _runner(**kwargs):
            captured_kwargs.append(kwargs)
            return _fake_round_result()

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            _runner,
        ):
            _run_fixed(ctx)

        assert len(captured_kwargs) == 3
        for i, call_kwargs in enumerate(captured_kwargs, start=1):
            expected = tmp_path / f"round_{i}"
            assert call_kwargs["output_dir"] == expected
            assert expected.is_dir()


# ──────────────────────────────────────────────────────────────────────
# Round > 1 task body enrichment
# ──────────────────────────────────────────────────────────────────────


class TestPreviousBestInTaskBody:
    def test_first_round_body_has_no_previous_best(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, max_rounds=2)
        captured_bodies: list[str] = []

        def _runner(**kwargs):
            captured_bodies.append(kwargs["task_content"])
            return _fake_round_result(1.5)

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            _runner,
        ):
            _run_fixed(ctx)

        # Round 1 body: no "Previous Rounds" section
        assert "Previous Rounds" not in captured_bodies[0]

    def test_second_round_body_mentions_previous_best(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, max_rounds=2)
        captured_bodies: list[str] = []

        def _runner(**kwargs):
            captured_bodies.append(kwargs["task_content"])
            return _fake_round_result(1.5)

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            _runner,
        ):
            _run_fixed(ctx)

        assert len(captured_bodies) == 2
        # Round 2 body mentions the previous best speedup
        assert "Previous Rounds" in captured_bodies[1]
        assert "1.500x" in captured_bodies[1]
        # AND still carries the original user prompt
        assert "Optimize this kernel." in captured_bodies[1]

    def test_no_previous_best_when_round1_returns_none(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, max_rounds=2)
        captured_bodies: list[str] = []

        def _runner(**kwargs):
            captured_bodies.append(kwargs["task_content"])
            return None  # round 1 produces no best

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            _runner,
        ):
            _run_fixed(ctx)

        # Round 2 body has no previous-best block because round 1 gave None
        assert "Previous Rounds" not in captured_bodies[1]


# ──────────────────────────────────────────────────────────────────────
# Kwargs plumbing
# ──────────────────────────────────────────────────────────────────────


class TestKwargsPlumbing:
    def test_model_and_config_flow_through(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, max_rounds=1)
        ctx.config = {"agent": {"extra_setting": "foo"}, "model": {"name": "m"}}
        ctx.test_command = "python3 /tmp/h.py --correctness"
        captured_kwargs: list[dict] = []

        def _runner(**kwargs):
            captured_kwargs.append(kwargs)
            return _fake_round_result()

        with patch(
            "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
            _runner,
        ):
            _run_fixed(ctx)

        kwargs = captured_kwargs[0]
        assert kwargs["config"] == ctx.config
        assert kwargs["model"] is ctx.model
        assert kwargs["agent_config"]["save_patch"] is True
        assert kwargs["agent_config"]["test_command"] == ctx.test_command
        assert kwargs["agent_config"]["extra_setting"] == "foo"
        assert kwargs["gpu_ids"] == "0"

    def test_invoke_helper_preserves_num_parallel(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, max_rounds=1)
        ctx.num_parallel = 4
        captured_kwargs: list[dict] = []

        _invoke_fixed_runner(
            ctx=ctx,
            body="hello",
            run_fixed_mode=lambda **kwargs: captured_kwargs.append(kwargs) or None,
            round_num=1,
        )
        assert captured_kwargs[0]["num_parallel"] == 4
