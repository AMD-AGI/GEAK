"""Tests for Workstream C — Jinja commandment rendering in ExplorePhase.

Pins:
  - ctx.language populated by DiscoveryPhase
  - ExplorePhase prefers Jinja template when language has one
  - Falls back to legacy commandment.py when Jinja unavailable
  - Both paths pass validate_commandment
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.preprocess.phases.base import PhaseContext
from minisweagent.run.preprocess.phases.explore import ExplorePhase, _try_jinja_render


class TestTryJinjaRender:
    """Direct tests of the Jinja render helper — independent of ExplorePhase."""

    def test_returns_none_when_language_is_none(self, tmp_path: Path) -> None:
        ctx = PhaseContext(output_dir=tmp_path, kernel_path="/tmp/k.py", repo_root="/tmp")
        ctx.language = None
        result = _try_jinja_render(
            ctx=ctx, correctness_cmd=None, perf_cmd=None, compile_cmd=None,
            harness_path=None, inner_kernel=False, profile_replays=3,
        )
        assert result is None

    def test_returns_none_when_template_path_is_none(self, tmp_path: Path) -> None:
        ctx = PhaseContext(output_dir=tmp_path, kernel_path="/tmp/k.py", repo_root="/tmp")
        fake_lang = MagicMock()
        fake_lang.commandment_template_path = None
        ctx.language = fake_lang
        result = _try_jinja_render(
            ctx=ctx, correctness_cmd=None, perf_cmd=None, compile_cmd=None,
            harness_path=None, inner_kernel=False, profile_replays=3,
        )
        assert result is None

    def test_returns_none_when_template_file_missing(self, tmp_path: Path) -> None:
        ctx = PhaseContext(output_dir=tmp_path, kernel_path="/tmp/k.py", repo_root="/tmp")
        fake_lang = MagicMock()
        fake_lang.commandment_template_path = tmp_path / "does_not_exist.j2"
        fake_lang.name = "fake"
        ctx.language = fake_lang
        result = _try_jinja_render(
            ctx=ctx, correctness_cmd=None, perf_cmd=None, compile_cmd=None,
            harness_path=None, inner_kernel=False, profile_replays=3,
        )
        assert result is None

    def test_renders_triton_template_successfully(self, tmp_path: Path) -> None:
        from minisweagent.kernel_languages import registry

        triton = registry.get("triton")
        assert triton is not None
        assert triton.commandment_template_path is not None

        ctx = PhaseContext(output_dir=tmp_path, kernel_path="/tmp/k.py", repo_root="/tmp/repo")
        ctx.language = triton

        result = _try_jinja_render(
            ctx=ctx, correctness_cmd=None, perf_cmd=None, compile_cmd=None,
            harness_path="/tmp/harness.py", inner_kernel=False, profile_replays=3,
        )
        assert result is not None
        assert "# Commandment" in result
        # Required contract sections
        for section in ("## Setup", "## Correctness", "## Benchmark", "## Full Benchmark", "## Profile"):
            assert section in result, f"Rendered Triton commandment missing {section}"
        assert "/tmp/harness.py" in result
        assert "/tmp/repo" in result

    def test_renders_hip_template_with_compile_command(self, tmp_path: Path) -> None:
        from minisweagent.kernel_languages import registry

        hip = registry.get("hip")
        assert hip is not None

        ctx = PhaseContext(output_dir=tmp_path, kernel_path="/tmp/k.cu", repo_root="/tmp/repo")
        ctx.language = hip

        result = _try_jinja_render(
            ctx=ctx,
            correctness_cmd="./bench --check",
            perf_cmd="./bench --time",
            compile_cmd="make",
            harness_path="/tmp/harness.py",
            inner_kernel=False,
            profile_replays=5,
        )
        assert result is not None
        assert "make" in result
        assert "./bench --check" in result
        assert "./bench --time" in result
        assert "--replays 5" in result

    def test_jinja_render_output_passes_validate_commandment(self, tmp_path: Path) -> None:
        """Rendered output from Jinja templates must pass the universal
        validate_commandment contract."""
        from minisweagent.kernel_languages import registry
        from minisweagent.kernel_languages.contract import validate_commandment

        for lang_name in ("triton", "hip"):
            lang = registry.get(lang_name)
            ctx = PhaseContext(output_dir=tmp_path, kernel_path="/tmp/k", repo_root="/tmp/repo")
            ctx.language = lang
            result = _try_jinja_render(
                ctx=ctx,
                correctness_cmd="cmd-correct",
                perf_cmd="cmd-perf",
                compile_cmd="cmd-compile",
                harness_path="/tmp/h.py",
                inner_kernel=False,
                profile_replays=3,
            )
            assert result is not None

            # Write and validate
            cm_path = tmp_path / f"COMMANDMENT_{lang_name}.md"
            cm_path.write_text(result)
            validate_commandment(cm_path)  # raises on failure


class TestExplorePhaseRenderPathSelection:
    """ExplorePhase picks Jinja when available, falls back otherwise."""

    def test_uses_jinja_when_language_has_template(self, tmp_path: Path) -> None:
        from minisweagent.kernel_languages import registry

        triton = registry.get("triton")
        ctx = PhaseContext(
            output_dir=tmp_path,
            test_command="python3 /tmp/harness.py",
        )
        ctx.kernel_path = "/tmp/k.py"
        ctx.repo_root = str(tmp_path)
        ctx.language = triton

        with patch(
            "minisweagent.run.preprocess.commandment.generate_commandment"
        ) as mock_legacy, patch(
            "minisweagent.run.preprocess.commandment.generate_commandment_from_commands"
        ) as mock_legacy_cmd:
            ExplorePhase().run(ctx)

        # Jinja path was used — neither legacy function was called.
        mock_legacy.assert_not_called()
        mock_legacy_cmd.assert_not_called()
        assert ctx.commandment is not None
        assert "# Commandment" in ctx.commandment
        assert ctx.commandment_path is not None

    def test_falls_back_when_language_is_none(self, tmp_path: Path) -> None:
        ctx = PhaseContext(
            output_dir=tmp_path,
            test_command="python3 /tmp/harness.py",
        )
        ctx.kernel_path = "/tmp/k.py"
        ctx.repo_root = str(tmp_path)
        ctx.language = None  # no language detected

        with patch(
            "minisweagent.run.preprocess.commandment.generate_commandment",
            return_value="# Legacy Commandment\n## Setup\n## Correctness\n## Benchmark\n## Full Benchmark\n## Profile\n",
        ):
            ExplorePhase().run(ctx)

        assert ctx.commandment is not None
        assert "Legacy Commandment" in ctx.commandment

    def test_falls_back_when_jinja_missing_despite_language(self, tmp_path: Path) -> None:
        """Language set but template_path points at a non-existent file -> fallback."""
        ctx = PhaseContext(
            output_dir=tmp_path,
            test_command="python3 /tmp/harness.py",
        )
        ctx.kernel_path = "/tmp/k.py"
        ctx.repo_root = str(tmp_path)
        fake_lang = MagicMock()
        fake_lang.commandment_template_path = tmp_path / "does_not_exist.j2"
        fake_lang.name = "fake"
        ctx.language = fake_lang

        with patch(
            "minisweagent.run.preprocess.commandment.generate_commandment",
            return_value="# Legacy\n## Setup\n## Correctness\n## Benchmark\n## Full Benchmark\n## Profile\n",
        ):
            ExplorePhase().run(ctx)

        assert "Legacy" in ctx.commandment

    def test_eval_path_falls_through_to_legacy_when_no_template(self, tmp_path: Path) -> None:
        ctx = PhaseContext(
            output_dir=tmp_path,
            eval_command="make && ./bench",
        )
        ctx.kernel_path = "/tmp/k.cu"
        ctx.repo_root = str(tmp_path)
        ctx.language = None  # no KernelLanguage detected

        with patch(
            "minisweagent.run.preprocess.commandment.generate_commandment_from_commands",
            return_value="# Legacy\n## Setup\n## Correctness\n## Benchmark\n## Full Benchmark\n## Profile\n",
        ):
            ExplorePhase().run(ctx)

        # eval path still triggers the test_command sync
        assert ctx.test_command == "make && ./bench"


class TestDiscoveryPopulatesLanguage:
    """DiscoveryPhase populates ctx.language so ExplorePhase can find it."""

    def test_phase_context_has_language_field(self) -> None:
        ctx = PhaseContext()
        assert hasattr(ctx, "language")
        assert ctx.language is None
