"""Tests for the preprocess phases scaffolding — contract + orchestrator shape.

These tests cover the new ``preprocess/phases/`` package and
``PreprocessOrchestrator``.  They verify the contract shape without
running a full pipeline (which needs a kernel + GPU + the ATD MCP).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from minisweagent.run.preprocess.orchestrator import (
    PreprocessOrchestrator,
)
from minisweagent.run.preprocess.phases.base import Phase, PhaseContext
from minisweagent.run.preprocess.phases.baseline import BaselinePhase
from minisweagent.run.preprocess.phases.discovery import DiscoveryPhase
from minisweagent.run.preprocess.phases.explore import ExplorePhase
from minisweagent.run.preprocess.phases.harness import HarnessPhase
from minisweagent.run.preprocess.phases.translation import TranslationPhase


# ── Base contract ─────────────────────────────────────────────────────


class TestPhaseContract:
    def test_phase_subclasses_have_distinct_names(self) -> None:
        names = [p.name for p in PreprocessOrchestrator().phases]
        assert len(set(names)) == len(names)
        assert names == ["translation", "discovery", "harness", "baseline", "explore"]

    def test_phase_run_raises_if_not_overridden(self) -> None:
        ctx = PhaseContext()
        with pytest.raises(NotImplementedError):
            Phase().run(ctx)

    def test_phase_is_applicable_default_true(self) -> None:
        assert Phase().is_applicable(PhaseContext()) is True


# ── TranslationPhase gating ───────────────────────────────────────────


class TestTranslationGate:
    def test_not_applicable_when_target_language_unset(self) -> None:
        ctx = PhaseContext(target_language=None)
        assert TranslationPhase().is_applicable(ctx) is False

    def test_applicable_when_target_language_set(self) -> None:
        ctx = PhaseContext(target_language="hip")
        assert TranslationPhase().is_applicable(ctx) is True

    def test_short_circuit_when_source_equals_target(self) -> None:
        """When the source kernel extension maps to the same canonical
        language as the target, TranslationPhase no-ops instead of
        raising NotImplementedError."""
        ctx = PhaseContext(
            kernel_url="/tmp/kernel.hip",  # inferred source=hip
            target_language="hip",
        )
        TranslationPhase().run(ctx)
        assert any(name == "translation" for name, _reason in ctx.phases_skipped)

    def test_runs_translation_agent_when_translation_needed(self, tmp_path) -> None:
        """TranslationPhase invokes the TranslationAgent verify-retry
        loop when source != target.  We stub the LLM via ctx-level
        injection to avoid calling a real model.
        """
        src = tmp_path / "kernel.py"
        src.write_text("def add(a, b): return a + b\n")

        ctx = PhaseContext(
            kernel_url=str(src),
            target_language="hip",
        )

        from unittest.mock import patch

        # Patch the agent-builder to inject a mocked agent whose loop
        # returns a successful TranslationResult.
        from minisweagent.pipeline_workers.translation import TranslationAgent
        from minisweagent.pipeline_workers.translation.translator import TranslationResult

        fake_result = TranslationResult(
            ok=True,
            candidate_code="__global__ void add(...) { ... }",
            attempts_used=1,
        )

        with patch.object(TranslationAgent, "loop", return_value=fake_result):
            TranslationPhase().run(ctx)

        # Kernel path got swapped to the translated file (.hip suffix)
        assert ctx.kernel_path.endswith(".hip")
        assert "translation" in ctx.phases_run


# ── PreprocessContext output shape ────────────────────────────────────


class TestPhaseContext:
    def test_to_dict_contains_expected_keys(self) -> None:
        ctx = PhaseContext()
        d = ctx.to_dict()
        for key in (
            "kernel_path",
            "repo_root",
            "resolved",
            "codebase_context_path",
            "discovery",
            "harness_path",
            "test_command",
            "baseline_metrics",
            "commandment",
        ):
            assert key in d

    def test_to_dict_omits_input_fields(self) -> None:
        """Inputs (kernel_url, output_dir, gpu_id, etc.) must NOT leak
        into the output contract dict that downstream consumers read."""
        ctx = PhaseContext(kernel_url="/tmp/x.py", gpu_id=3, translate_only=True)
        d = ctx.to_dict()
        for input_only_key in ("kernel_url", "gpu_id", "translate_only", "target_language"):
            assert input_only_key not in d


# ── Orchestrator behaviour (mocked phases) ────────────────────────────


class _RecordingPhase(Phase):
    """Test helper: records that it ran + lets us control outputs."""

    def __init__(self, name: str, applicable: bool = True, outputs: dict | None = None) -> None:
        self.name = name
        self._applicable = applicable
        self._outputs = outputs or {}

    def is_applicable(self, ctx: PhaseContext) -> bool:
        return self._applicable

    def run(self, ctx: PhaseContext) -> None:
        for k, v in self._outputs.items():
            setattr(ctx, k, v)
        ctx.phases_run.append(self.name)


class TestOrchestratorFlow:
    def test_all_phases_run_in_order(self) -> None:
        phases = [
            _RecordingPhase("a"),
            _RecordingPhase("b"),
            _RecordingPhase("c"),
        ]
        ctx = PhaseContext()
        PreprocessOrchestrator(phases=phases).run(ctx)
        assert ctx.phases_run == ["a", "b", "c"]

    def test_skips_inapplicable_phases(self) -> None:
        phases = [
            _RecordingPhase("always"),
            _RecordingPhase("skipme", applicable=False),
            _RecordingPhase("also_always"),
        ]
        ctx = PhaseContext()
        PreprocessOrchestrator(phases=phases).run(ctx)
        assert ctx.phases_run == ["always", "also_always"]
        assert ctx.phases_skipped == [("skipme", "not applicable")]

    def test_translate_only_returns_after_translation(self) -> None:
        class _FakeTranslation(Phase):
            name = "translation"

            def run(self, ctx: PhaseContext) -> None:
                ctx.phases_run.append(self.name)

        phases = [
            _FakeTranslation(),
            _RecordingPhase("discovery"),
            _RecordingPhase("harness"),
        ]
        ctx = PhaseContext(target_language="hip", translate_only=True)
        PreprocessOrchestrator(phases=phases).run(ctx)
        # Only translation ran; discovery/harness were not reached
        assert ctx.phases_run == ["translation"]

    def test_phase_exception_propagates(self) -> None:
        class _Boom(Phase):
            name = "boom"

            def run(self, ctx: PhaseContext) -> None:
                raise RuntimeError("fatal")

        phases = [_Boom()]
        with pytest.raises(RuntimeError, match="fatal"):
            PreprocessOrchestrator(phases=phases).run(PhaseContext())


# ── Orchestrator integration with legacy fallback ─────────────────────


class TestCliUsesOrchestrator:
    """Regression guard: cli.py must import the preprocessor entry from
    the orchestrator shim, not the legacy monolith directly.  Anyone
    who reverts the flip gets a failing test pointing at the right
    file.
    """

    def test_cli_imports_run_preprocessor_from_orchestrator(self) -> None:
        from minisweagent.run import mini

        # Inspect the imported ``run_preprocessor`` symbol
        rp = mini.run_preprocessor
        # The shim is exported as ``run_preprocessor_via_orchestrator``
        # in the orchestrator module; after ``as run_preprocessor``
        # aliasing the __name__ attribute still reveals the real source.
        assert rp.__module__ == "minisweagent.run.preprocess.orchestrator", (
            f"mini.run_preprocessor must come from the orchestrator shim; got {rp.__module__}"
        )


class TestUserTaskPlumbing:
    """Regression guards for the ``user_task`` plumbing introduced by
    PR #226's import of upstream commit ``15e113c``.

    Two failure modes the original sync PR missed:

    1.  ``mini.py`` splats ``user_task=task_content`` into
        ``run_preprocessor`` which on ``gwiab`` resolves to the
        orchestrator shim.  Before this fix the shim did not accept
        ``user_task`` and every preprocess call raised ``TypeError``.

    2.  Even if the shim accepts the kwarg, the value must survive the
        trip through :class:`PhaseContext` into
        :class:`~minisweagent.run.preprocess.phases.harness.HarnessPhase`
        so the UnitTestAgent / ShapeFixerAgent prompts actually receive
        a ``USER TASK CONTEXT`` block.  Without this the feature ships
        dead-on-arrival on the modular pipeline.
    """

    _USER_TASK_PROBE = "PROBE_USER_TASK_42::production-contract"

    def test_orchestrator_shim_accepts_user_task_kwarg(self, tmp_path: Path) -> None:
        """Direct regression for the crash: passing ``user_task`` into
        the shim used to raise ``TypeError`` because the kwarg was not
        declared on the signature.  All phases stubbed out so this is a
        pure plumbing test."""
        from minisweagent.run.preprocess import orchestrator as orch_mod

        class _NoOpPhase(Phase):
            name = "noop"

            def run(self, ctx):
                # Mark mandatory outputs as populated so the legacy
                # fallback is not invoked — we only want to exercise the
                # PhaseContext construction here.
                ctx.harness_path = "/tmp/h.py"
                ctx.baseline_metrics_path = str(tmp_path / "bm.json")

        with (
            patch(
                "minisweagent.run.preprocess.orchestrator.DiscoveryPhase",
                _NoOpPhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.HarnessPhase",
                _NoOpPhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.BaselinePhase",
                _NoOpPhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.ExplorePhase",
                _NoOpPhase,
            ),
        ):
            orch_mod.run_preprocessor_via_orchestrator(
                kernel_url="dummy",
                output_dir=tmp_path,
                user_task=self._USER_TASK_PROBE,
            )

    def test_user_task_round_trips_through_phase_context(self, tmp_path: Path) -> None:
        """Verify the value passed into the shim reaches phases via
        ``ctx.user_task`` (rather than being dropped on the floor)."""
        from minisweagent.run.preprocess import orchestrator as orch_mod

        captured: dict[str, str | None] = {"user_task": "SENTINEL_NOT_SET"}

        class _CapturePhase(Phase):
            name = "capture"

            def run(self, ctx):
                captured["user_task"] = ctx.user_task
                ctx.harness_path = "/tmp/h.py"
                ctx.baseline_metrics_path = str(tmp_path / "bm.json")

        with (
            patch(
                "minisweagent.run.preprocess.orchestrator.DiscoveryPhase",
                _CapturePhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.HarnessPhase",
                _CapturePhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.BaselinePhase",
                _CapturePhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.ExplorePhase",
                _CapturePhase,
            ),
        ):
            orch_mod.run_preprocessor_via_orchestrator(
                kernel_url="dummy",
                output_dir=tmp_path,
                user_task=self._USER_TASK_PROBE,
            )

        assert captured["user_task"] == self._USER_TASK_PROBE

    def test_user_task_forwarded_to_legacy_fallback(self, tmp_path: Path) -> None:
        """When phases leave mandatory outputs empty, the legacy
        fallback runs — and must receive ``user_task`` so the legacy
        UTA / ShapeFixer chain stays consistent with the new pipeline.
        """
        from minisweagent.run.preprocess import orchestrator as orch_mod

        captured: dict = {}

        def _fake_legacy(**kwargs):
            captured.update(kwargs)
            return {
                "kernel_path": "/tmp/k.py",
                "repo_root": str(tmp_path),
                "harness_path": "/tmp/harness.py",
                "baseline_metrics_path": str(tmp_path / "bm.json"),
                "commandment": "BASE",
            }

        class _NoOpPhase(Phase):
            name = "noop"

            def run(self, ctx):
                pass

        with (
            patch(
                "minisweagent.run.preprocess.orchestrator.DiscoveryPhase",
                _NoOpPhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.HarnessPhase",
                _NoOpPhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.BaselinePhase",
                _NoOpPhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.ExplorePhase",
                _NoOpPhase,
            ),
            patch(
                "minisweagent.run.preprocess.preprocessor.run_preprocessor",
                _fake_legacy,
            ),
        ):
            orch_mod.run_preprocessor_via_orchestrator(
                kernel_url="dummy",
                output_dir=tmp_path,
                user_task=self._USER_TASK_PROBE,
            )

        assert captured.get("user_task") == self._USER_TASK_PROBE


class TestUserTaskAgentPrompt:
    """End-of-pipeline check: the ``USER TASK CONTEXT`` block actually
    lands in the agent task strings the LLM sees.  The YAML
    system-prompt overrides in ``mini_unit_test_agent.yaml`` /
    ``mini_shape_fixer.yaml`` only fire when this block is present, so
    the prepend logic is part of the feature contract.
    """

    _USER_TASK_PROBE = "PROBE_USER_TASK_42::production-contract"

    def test_shape_fixer_task_prepends_user_task_block(self, tmp_path: Path) -> None:
        from minisweagent.run.preprocess.shape_fixer_agent import _build_shape_fixer_task

        bench = tmp_path / "bench.py"
        bench.write_text("# fake benchmark\n")
        harness = tmp_path / "harness.py"
        harness.write_text("# fake harness\n")

        body = _build_shape_fixer_task(
            benchmark_file=bench,
            harness_path=harness,
            kernel_path=None,
            gpu_id=0,
            validation_feedback=None,
            user_task=self._USER_TASK_PROBE,
        )
        assert "USER TASK CONTEXT (production workload contract -- HIGHEST PRIORITY):" in body
        assert self._USER_TASK_PROBE in body
        # Sentinel for the ordering: the block must come BEFORE the
        # "SHAPE SOURCE FILE" header, otherwise the YAML override rule
        # cannot recognise it as the highest-priority section.
        assert body.index(self._USER_TASK_PROBE) < body.index("SHAPE SOURCE FILE")

    def test_shape_fixer_task_unchanged_when_user_task_none(self, tmp_path: Path) -> None:
        """Backward-compat: when ``user_task`` is None or empty the
        prefix must not be added (the legacy discovery-driven behaviour
        is preserved verbatim).
        """
        from minisweagent.run.preprocess.shape_fixer_agent import _build_shape_fixer_task

        bench = tmp_path / "bench.py"
        bench.write_text("# fake\n")
        harness = tmp_path / "harness.py"
        harness.write_text("# fake\n")

        for empty in (None, "", "   "):
            body = _build_shape_fixer_task(
                benchmark_file=bench,
                harness_path=harness,
                kernel_path=None,
                gpu_id=0,
                validation_feedback=None,
                user_task=empty,
            )
            assert "USER TASK CONTEXT" not in body

    def test_unit_test_agent_task_prepends_user_task_block(self, tmp_path: Path) -> None:
        """End-to-end on the UnitTestAgent path: when ``run_unit_test_agent``
        is invoked with a non-empty ``user_task``, the constructed agent
        task string must start with the ``USER TASK CONTEXT`` block.

        We stub the model and the agent's ``run`` method so this test
        is hermetic — no LLM call, no real repo.
        """
        from unittest.mock import MagicMock

        from minisweagent.run.preprocess import unit_test_agent as uta_mod

        captured: dict[str, str] = {}

        class _StubAgent:
            log_file = None

            def __init__(self, *args, **kwargs):
                pass

            def run(self, task: str):
                captured["task"] = task
                return ("Submitted", "TEST_COMMAND: python /tmp/h.py --correctness")

        with (
            patch.object(uta_mod, "UnitTestAgent", _StubAgent),
            patch.object(uta_mod, "LocalEnvironment", MagicMock()),
            patch.object(
                uta_mod,
                "load_preprocess_agent_config",
                return_value=({}, {}),
            ),
        ):
            uta_mod.run_unit_test_agent(
                model=MagicMock(),
                repo=tmp_path,
                kernel_name="probe_kernel",
                user_task=self._USER_TASK_PROBE,
            )

        task = captured["task"]
        assert "USER TASK CONTEXT (production workload contract -- HIGHEST PRIORITY):" in task
        assert self._USER_TASK_PROBE in task
        # Ordering: prefix must precede the per-kernel preamble so the
        # YAML rule sees it as HIGHEST PRIORITY.
        assert task.index(self._USER_TASK_PROBE) < task.index("Create a fixed test harness")


class TestLegacyFallback:
    def test_falls_back_when_phases_leave_mandatory_outputs_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """During the refactor transition: if the new phases don't
        populate mandatory outputs (harness_path, baseline_metrics_path),
        ``run_preprocessor_via_orchestrator`` calls into the legacy
        monolith to fill them in.  We mock the legacy helper to
        confirm it's invoked."""
        from minisweagent.run.preprocess import orchestrator as orch_mod

        captured: dict = {}

        def _fake_legacy(**kwargs):
            captured.update(kwargs)
            return {
                "kernel_path": "/tmp/k.py",
                "repo_root": str(tmp_path),
                "harness_path": "/tmp/harness.py",
                "baseline_metrics_path": str(tmp_path / "baseline_metrics.json"),
                "commandment": "BASE",
            }

        # Patch DiscoveryPhase to no-op so no real preprocessing runs.
        class _NoOpPhase(Phase):
            name = "noop"

            def run(self, ctx):
                pass

        with (
            patch(
                "minisweagent.run.preprocess.orchestrator.DiscoveryPhase",
                _NoOpPhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.HarnessPhase",
                _NoOpPhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.BaselinePhase",
                _NoOpPhase,
            ),
            patch(
                "minisweagent.run.preprocess.orchestrator.ExplorePhase",
                _NoOpPhase,
            ),
            patch(
                "minisweagent.run.preprocess.preprocessor.run_preprocessor",
                _fake_legacy,
            ),
        ):
            result = orch_mod.run_preprocessor_via_orchestrator(
                kernel_url="dummy",
                output_dir=tmp_path,
            )

        # Legacy path got invoked with our args
        assert captured["kernel_url"] == "dummy"
        assert captured["output_dir"] == tmp_path
        # Result contains the legacy outputs
        assert result["harness_path"] == "/tmp/harness.py"
        assert result["commandment"] == "BASE"
