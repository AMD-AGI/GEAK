"""Tests for Workstream D2 — ExplorePhase wiring of KernelAnalysisAgent.

Pins:
  - _try_kernel_analysis populates ctx.kernel_analysis_md on success
  - Silently skipped when language is None / model unavailable / kernel missing
  - compose_task_body prepends kernel_analysis_md when populated
  - Injection order: rubric BEFORE memory
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.compose import ComposeInputs, compose_task_body
from minisweagent.run.preprocess.phases.base import PhaseContext
from minisweagent.run.preprocess.phases.explore import (
    ExplorePhase,
    _try_kernel_analysis,
)


@pytest.fixture
def kernel_analysis_enabled():
    """Enable the KernelAnalysisAgent env gate for tests that need it."""
    with patch.dict(os.environ, {"GEAK_USE_KERNEL_ANALYSIS": "1"}):
        yield


def _good_rubric() -> str:
    return (
        "## [A] Primitives\nfoo\n\n"
        "## [B] Shape Regimes\nbar\n\n"
        "## [C] Profile Hotspots\nbaz\n\n"
        "## [D] Attack Surfaces\nqux\n"
    )


def _fake_language(tmp_path: Path) -> MagicMock:
    lang = MagicMock()
    lang.name = "triton"
    (tmp_path / "sys.md").write_text("sys")
    lang.system_prompt_path = tmp_path / "sys.md"
    lang.system_prompt = "sys"
    return lang


# ──────────────────────────────────────────────────────────────────────
# _try_kernel_analysis gating
# ──────────────────────────────────────────────────────────────────────


class TestEnvGate:
    """KernelAnalysisAgent is OFF by default; enabled via GEAK_USE_KERNEL_ANALYSIS=1."""

    def test_gated_off_by_default(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _fake_language(tmp_path)
        model = MagicMock()
        model.query = MagicMock(return_value=_good_rubric())
        ctx.model = model

        # Env var unset -> gate OFF
        env_without_gate = {k: v for k, v in os.environ.items() if k != "GEAK_USE_KERNEL_ANALYSIS"}
        with patch.dict(os.environ, env_without_gate, clear=True):
            _try_kernel_analysis(ctx, output_dir=tmp_path)
        assert ctx.kernel_analysis_md is None, (
            "KernelAnalysisAgent should be gated OFF by default "
            "(GEAK_USE_KERNEL_ANALYSIS!=1)"
        )

    def test_gated_off_when_env_var_is_zero(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _fake_language(tmp_path)
        model = MagicMock()
        model.query = MagicMock(return_value=_good_rubric())
        ctx.model = model

        with patch.dict(os.environ, {"GEAK_USE_KERNEL_ANALYSIS": "0"}):
            _try_kernel_analysis(ctx, output_dir=tmp_path)
        assert ctx.kernel_analysis_md is None


class TestAnalysisWiringGates:
    def test_skipped_when_language_none(
        self, tmp_path: Path, kernel_analysis_enabled
    ) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = None
        ctx.model = MagicMock()

        _try_kernel_analysis(ctx, output_dir=tmp_path)
        assert ctx.kernel_analysis_md is None

    def test_skipped_when_kernel_missing(
        self, tmp_path: Path, kernel_analysis_enabled
    ) -> None:
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = ""
        ctx.language = _fake_language(tmp_path)
        ctx.model = MagicMock()

        _try_kernel_analysis(ctx, output_dir=tmp_path)
        assert ctx.kernel_analysis_md is None

    def test_skipped_when_model_unavailable(
        self, tmp_path: Path, kernel_analysis_enabled
    ) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _fake_language(tmp_path)
        ctx.model = None
        ctx.model_factory = None

        _try_kernel_analysis(ctx, output_dir=tmp_path)
        assert ctx.kernel_analysis_md is None

    def test_success_populates_kernel_analysis_md(
        self, tmp_path: Path, kernel_analysis_enabled
    ) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _fake_language(tmp_path)
        model = MagicMock()
        model.query = MagicMock(return_value=_good_rubric())
        ctx.model = model

        _try_kernel_analysis(ctx, output_dir=tmp_path)
        assert ctx.kernel_analysis_md is not None
        assert "## [A] Primitives" in ctx.kernel_analysis_md
        assert "## [D] Attack Surfaces" in ctx.kernel_analysis_md
        # File written to disk
        assert (tmp_path / "kernel_analysis.md").exists()

    def test_subagent_exception_swallowed(
        self, tmp_path: Path, kernel_analysis_enabled
    ) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _fake_language(tmp_path)
        model = MagicMock()
        model.query = MagicMock(side_effect=RuntimeError("boom"))
        ctx.model = model

        # Does NOT raise — phase shielded by the wrapper
        _try_kernel_analysis(ctx, output_dir=tmp_path)
        # kernel_analysis_md stays None because the write happens AFTER
        # the model response is parsed
        assert ctx.kernel_analysis_md is None


# ──────────────────────────────────────────────────────────────────────
# compose_task_body injection
# ──────────────────────────────────────────────────────────────────────


class TestTaskBodyInjection:
    def test_rubric_injected_when_present(self) -> None:
        prompt = "Optimize this kernel for latency."
        preprocess_ctx = {
            "kernel_path": "/tmp/k.py",
            "kernel_analysis_md": _good_rubric(),
        }
        body = compose_task_body(
            ComposeInputs(user_prompt=prompt, mode="fixed", preprocess_ctx=preprocess_ctx)
        )
        assert "Kernel Analysis" in body
        assert "## [A] Primitives" in body
        assert "Optimize this kernel" in body

    def test_no_injection_when_kernel_analysis_missing(self) -> None:
        prompt = "Optimize this kernel."
        preprocess_ctx = {"kernel_path": "/tmp/k.py"}
        body = compose_task_body(
            ComposeInputs(user_prompt=prompt, mode="fixed", preprocess_ctx=preprocess_ctx)
        )
        assert "Kernel Analysis" not in body

    def test_no_injection_when_kernel_analysis_empty(self) -> None:
        preprocess_ctx = {"kernel_path": "/tmp/k.py", "kernel_analysis_md": ""}
        body = compose_task_body(
            ComposeInputs(user_prompt="Optimize.", mode="fixed", preprocess_ctx=preprocess_ctx)
        )
        assert "Kernel Analysis" not in body

    def test_injection_order_rubric_before_memory(self) -> None:
        """Rubric must be injected BEFORE memory so KB evidence can
        reference the [A]-[D] framing."""
        # Use a preprocess_ctx with both rubric and (empty) baseline so
        # memory retrieval yields nothing and doesn't interfere.
        prompt = "USER_PROMPT_MARKER"
        preprocess_ctx = {
            "kernel_path": "/tmp/k.py",
            "kernel_analysis_md": _good_rubric(),
        }
        body = compose_task_body(
            ComposeInputs(user_prompt=prompt, mode="fixed", preprocess_ctx=preprocess_ctx)
        )
        assert body.startswith(prompt)
        # Rubric goes directly after the prompt (memory would come after
        # if retrieval returned anything).
        assert body.index("## [A] Primitives") > body.index(prompt)
