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
            f"mini.run_preprocessor must come from the orchestrator shim; "
            f"got {rp.__module__}"
        )


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

        with patch(
            "minisweagent.run.preprocess.orchestrator.DiscoveryPhase",
            _NoOpPhase,
        ), patch(
            "minisweagent.run.preprocess.orchestrator.HarnessPhase",
            _NoOpPhase,
        ), patch(
            "minisweagent.run.preprocess.orchestrator.BaselinePhase",
            _NoOpPhase,
        ), patch(
            "minisweagent.run.preprocess.orchestrator.ExplorePhase",
            _NoOpPhase,
        ), patch(
            "minisweagent.run.preprocess.preprocessor.run_preprocessor",
            _fake_legacy,
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
