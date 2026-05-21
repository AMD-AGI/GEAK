"""Tests for Workstream D4 full — TranslationPhase golden-tensor source-side validation.

Pins:
  - When ``ctx.harness`` is set, TranslationPhase runs the source
    harness ``--correctness`` once before the retry loop.
  - If source passes: translation proceeds (pass-through).
  - If source fails: translation is aborted with RuntimeError (no
    LLM attempts wasted on a broken source).
  - If no harness / harness file missing / subprocess timeout /
    indeterminate exit: gracefully skip (None) and proceed to
    Layer 1+2 verification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.preprocess.phases.base import PhaseContext
from minisweagent.run.preprocess.phases.translation import TranslationPhase


# ──────────────────────────────────────────────────────────────────────
# _validate_source_correctness direct tests
# ──────────────────────────────────────────────────────────────────────


class TestValidateSourceCorrectness:
    def test_returns_none_when_no_harness(self, tmp_path: Path) -> None:
        src_path = tmp_path / "k.py"
        src_path.write_text("pass")
        ctx = PhaseContext()
        ctx.harness = None
        ctx.harness_path = None

        result = TranslationPhase._validate_source_correctness(ctx, src_path)
        assert result is None

    def test_returns_none_when_harness_file_missing(self, tmp_path: Path) -> None:
        src_path = tmp_path / "k.py"
        src_path.write_text("pass")
        ctx = PhaseContext()
        ctx.harness = str(tmp_path / "does_not_exist.py")

        result = TranslationPhase._validate_source_correctness(ctx, src_path)
        assert result is None

    def test_pass_when_source_correctness_exits_zero_ok(self, tmp_path: Path) -> None:
        src_path = tmp_path / "k.py"
        src_path.write_text("pass")
        harness = tmp_path / "harness.py"
        harness.write_text("import sys; print('OK'); sys.exit(0)")
        ctx = PhaseContext()
        ctx.harness = str(harness)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["py", "h"], returncode=0, stdout="OK\n", stderr=""
            )
            result = TranslationPhase._validate_source_correctness(ctx, src_path)
        assert result is True

    def test_fail_when_source_correctness_exits_nonzero_fail(self, tmp_path: Path) -> None:
        src_path = tmp_path / "k.py"
        src_path.write_text("pass")
        harness = tmp_path / "harness.py"
        harness.write_text("print('FAIL'); sys.exit(1)")
        ctx = PhaseContext()
        ctx.harness = str(harness)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["py", "h"], returncode=1, stdout="FAIL\n", stderr=""
            )
            result = TranslationPhase._validate_source_correctness(ctx, src_path)
        assert result is False

    def test_skip_on_timeout(self, tmp_path: Path) -> None:
        src_path = tmp_path / "k.py"
        src_path.write_text("pass")
        harness = tmp_path / "harness.py"
        harness.write_text("import time; time.sleep(100)")
        ctx = PhaseContext()
        ctx.harness = str(harness)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="", timeout=0.1)
            result = TranslationPhase._validate_source_correctness(ctx, src_path)
        assert result is None

    def test_skip_on_indeterminate_exit(self, tmp_path: Path) -> None:
        """Exit code that's neither 0+OK nor nonzero+FAIL is treated
        as indeterminate (None), not fail."""
        src_path = tmp_path / "k.py"
        src_path.write_text("pass")
        harness = tmp_path / "harness.py"
        harness.write_text("print('weird'); sys.exit(42)")
        ctx = PhaseContext()
        ctx.harness = str(harness)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["py", "h"], returncode=42, stdout="something weird\n", stderr=""
            )
            result = TranslationPhase._validate_source_correctness(ctx, src_path)
        assert result is None

    def test_uses_harness_path_when_harness_is_none(self, tmp_path: Path) -> None:
        """Falls back to ``ctx.harness_path`` when ``ctx.harness`` unset."""
        src_path = tmp_path / "k.py"
        src_path.write_text("pass")
        harness = tmp_path / "harness.py"
        harness.write_text("print('OK')")
        ctx = PhaseContext()
        ctx.harness = None
        ctx.harness_path = str(harness)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="OK\n", stderr=""
            )
            result = TranslationPhase._validate_source_correctness(ctx, src_path)
        assert result is True


# ──────────────────────────────────────────────────────────────────────
# TranslationPhase.run integration
# ──────────────────────────────────────────────────────────────────────


class TestTranslationPhaseIntegration:
    def _make_ctx(self, tmp_path: Path, *, with_harness: bool = True) -> PhaseContext:
        src_path = tmp_path / "kernel.py"
        src_path.write_text("@triton.jit\ndef kernel(): pass\n")
        ctx = PhaseContext(
            output_dir=tmp_path,
            target_language="hip",
            kernel_url=str(src_path),
        )
        if with_harness:
            harness = tmp_path / "harness.py"
            harness.write_text("print('OK')")
            ctx.harness = str(harness)
        return ctx

    def test_aborts_on_source_correctness_failure(self, tmp_path: Path) -> None:
        ctx = self._make_ctx(tmp_path, with_harness=True)

        with patch("subprocess.run") as mock_run:
            # Source harness fails --correctness
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="FAIL\n", stderr=""
            )

            phase = TranslationPhase()
            with pytest.raises(RuntimeError, match="source kernel fails its own --correctness"):
                phase.run(ctx)

    def test_proceeds_when_source_correctness_passes(self, tmp_path: Path) -> None:
        """When source passes correctness, translation proceeds to the
        LLM agent (we stub the agent to avoid the actual LLM call)."""
        ctx = self._make_ctx(tmp_path, with_harness=True)

        # Mock out agent.loop so we don't call a real model
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.candidate_code = "__global__ void kernel(float* x) {}\n"
        mock_result.attempts_used = 1
        mock_result.feedback_history = []
        mock_agent.loop.return_value = mock_result

        with patch("subprocess.run") as mock_run, patch.object(
            TranslationPhase, "_build_agent", return_value=mock_agent
        ):
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="OK\n", stderr=""
            )
            TranslationPhase().run(ctx)

        # Phase completed, agent was invoked, translated file written
        assert mock_agent.loop.called
        assert ctx.kernel_path is not None
        assert Path(ctx.kernel_path).exists()

    def test_proceeds_when_no_harness_available(self, tmp_path: Path) -> None:
        """No harness -> skip source validation, proceed to LLM."""
        ctx = self._make_ctx(tmp_path, with_harness=False)

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.candidate_code = "__global__ void kernel(float* x) {}\n"
        mock_result.attempts_used = 1
        mock_result.feedback_history = []
        mock_agent.loop.return_value = mock_result

        with patch.object(
            TranslationPhase, "_build_agent", return_value=mock_agent
        ):
            TranslationPhase().run(ctx)

        assert mock_agent.loop.called

    def test_skips_gracefully_on_subprocess_failure(self, tmp_path: Path) -> None:
        """Harness timeout / OSError -> skip source validation (None);
        translation proceeds rather than bailing."""
        ctx = self._make_ctx(tmp_path, with_harness=True)

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.candidate_code = "__global__ void kernel(float* x) {}\n"
        mock_result.attempts_used = 1
        mock_result.feedback_history = []
        mock_agent.loop.return_value = mock_result

        with patch("subprocess.run") as mock_run, patch.object(
            TranslationPhase, "_build_agent", return_value=mock_agent
        ):
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="", timeout=0.1)
            # Should NOT raise — skip gracefully
            TranslationPhase().run(ctx)

        assert mock_agent.loop.called
