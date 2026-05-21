"""Tests for Workstream D2 — KernelAnalysisAgent subagent.

Pins:
  - Prompt includes all 4 required [A]-[D] section headers verbatim
  - Success path writes markdown + sets ok=True
  - Retry on missing headers
  - Best-effort fallback writes last attempt even when retries exhausted
  - Profile blob rendering picks high-signal fields
  - Code-fence stripping
  - Config YAML exists + is loadable
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from minisweagent.pipeline_workers.base import SubagentConfig
from minisweagent.pipeline_workers.preprocess.kernel_analysis import (
    REQUIRED_RUBRIC_HEADERS,
    KernelAnalysisAgent,
    KernelAnalysisResult,
)


def _good_rubric() -> str:
    return """\
## [A] Primitives
Elementwise add on [N] tensors.  Memory-bound.

## [B] Shape Regimes
BLOCK_SIZE=1024; N up to 1_000_000.

## [C] Profile Hotspots
No profile data supplied — unable to identify hotspots.

## [D] Attack Surfaces
1. Vectorised loads (high) — stride-1 access.
2. Persistent kernel pattern (medium) — short-kernel launch amortisation.
"""


def _missing_d_rubric() -> str:
    """Rubric missing the [D] header — should trigger retry."""
    return """\
## [A] Primitives
Elementwise.

## [B] Shape Regimes
BLOCK_SIZE=1024.

## [C] Profile Hotspots
None supplied.
"""


def _fake_language(tmp_path: Path) -> MagicMock:
    lang = MagicMock()
    lang.name = "triton"
    (tmp_path / "system.md").write_text("you are the triton worker")
    lang.system_prompt_path = tmp_path / "system.md"
    lang.system_prompt = (tmp_path / "system.md").read_text()
    return lang


def _make_agent(
    tmp_path: Path,
    *,
    responses: list[str],
    max_retries: int = 1,
) -> KernelAnalysisAgent:
    lang = _fake_language(tmp_path)
    config = SubagentConfig(
        name="kernel_analysis",
        model_name="fake",
        system_template="sys {language_name}",
        instance_template="inst {language_name}",
        step_limit=1,
        cost_limit=3.0,
        temperature=0.2,
        extra={"max_retries": max_retries},
    )
    agent = KernelAnalysisAgent(language=lang, config=config)
    model = MagicMock()
    it = iter(responses)
    model.query = lambda _msgs: next(it)
    agent.model = model  # type: ignore[attr-defined]
    return agent


# ──────────────────────────────────────────────────────────────────────
# Success path
# ──────────────────────────────────────────────────────────────────────


class TestSuccessOnFirstAttempt:
    def test_all_four_headers_present(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        out = tmp_path / "analysis.md"

        agent = _make_agent(tmp_path, responses=[_good_rubric()])
        result = agent.run(kernel_path=kernel, out_path=out)
        assert result["ok"] is True
        assert result["attempts_used"] == 1
        assert result["missing_headers"] == []
        assert out.exists()
        content = out.read_text()
        for h in REQUIRED_RUBRIC_HEADERS:
            assert h in content

    def test_strips_markdown_fences(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        out = tmp_path / "analysis.md"
        fenced = "```markdown\n" + _good_rubric() + "```\n"
        agent = _make_agent(tmp_path, responses=[fenced])
        result = agent.run(kernel_path=kernel, out_path=out)
        assert result["ok"] is True
        assert not out.read_text().strip().startswith("```")


# ──────────────────────────────────────────────────────────────────────
# Retry / best-effort
# ──────────────────────────────────────────────────────────────────────


class TestRetryOnMissingHeader:
    def test_retry_recovers(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        out = tmp_path / "analysis.md"
        agent = _make_agent(
            tmp_path,
            responses=[_missing_d_rubric(), _good_rubric()],
            max_retries=1,
        )
        result = agent.run(kernel_path=kernel, out_path=out)
        assert result["ok"] is True
        assert result["attempts_used"] == 2

    def test_best_effort_when_all_retries_miss_header(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        out = tmp_path / "analysis.md"
        agent = _make_agent(
            tmp_path,
            responses=[_missing_d_rubric(), _missing_d_rubric()],
            max_retries=1,
        )
        result = agent.run(kernel_path=kernel, out_path=out)
        assert result["ok"] is False
        assert "## [D] Attack Surfaces" in result["missing_headers"]
        # File IS still written (best-effort)
        assert out.exists()


# ──────────────────────────────────────────────────────────────────────
# Prompt composition
# ──────────────────────────────────────────────────────────────────────


class TestPromptComposition:
    def test_all_four_headers_in_prompt(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        out = tmp_path / "analysis.md"
        captured: list[str] = []
        agent = _make_agent(tmp_path, responses=[_good_rubric()])

        def _q(messages):
            captured.append(messages[1]["content"])
            return _good_rubric()

        agent.model.query = _q  # type: ignore[attr-defined]
        agent.run(kernel_path=kernel, out_path=out)
        inst = captured[0]
        for h in REQUIRED_RUBRIC_HEADERS:
            assert h in inst

    def test_retry_prompt_names_missing_headers(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        out = tmp_path / "analysis.md"
        captured: list[str] = []
        agent = _make_agent(
            tmp_path,
            responses=[_missing_d_rubric(), _good_rubric()],
            max_retries=1,
        )

        responses = iter([_missing_d_rubric(), _good_rubric()])

        def _q(messages):
            captured.append(messages[1]["content"])
            return next(responses)

        agent.model.query = _q  # type: ignore[attr-defined]
        agent.run(kernel_path=kernel, out_path=out)
        assert len(captured) == 2
        retry_prompt = captured[1]
        assert "PREVIOUS ATTEMPT" in retry_prompt
        assert "## [D] Attack Surfaces" in retry_prompt

    def test_profile_data_rendered_when_present(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        out = tmp_path / "analysis.md"
        captured: list[str] = []
        agent = _make_agent(tmp_path, responses=[_good_rubric()])

        def _q(messages):
            captured.append(messages[1]["content"])
            return _good_rubric()

        agent.model.query = _q  # type: ignore[attr-defined]
        agent.run(
            kernel_path=kernel,
            out_path=out,
            baseline_metrics={
                "duration_us": 123.4,
                "bottleneck": "memory",
                "metrics": {"memory.hbm_bandwidth_utilization": 85.1},
                "top_kernels": [
                    {"name": "foo", "duration_share": "42%"},
                    {"name": "bar", "duration_share": "31%"},
                ],
            },
            profile={"success": True},
        )
        inst = captured[0]
        assert "123.4" in inst
        assert "memory" in inst
        assert "85.1" in inst
        assert "foo" in inst and "42%" in inst

    def test_profile_absence_explicitly_flagged(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        out = tmp_path / "analysis.md"
        captured: list[str] = []
        agent = _make_agent(tmp_path, responses=[_good_rubric()])

        def _q(messages):
            captured.append(messages[1]["content"])
            return _good_rubric()

        agent.model.query = _q  # type: ignore[attr-defined]
        agent.run(kernel_path=kernel, out_path=out)
        assert "none supplied" in captured[0].lower() or "skip the measured" in captured[0].lower()


# ──────────────────────────────────────────────────────────────────────
# Config YAML
# ──────────────────────────────────────────────────────────────────────


class TestKernelAnalysisConfig:
    def test_config_yaml_exists(self) -> None:
        from minisweagent.pipeline_workers import preprocess

        cfg = Path(preprocess.__file__).parent / "configs" / "kernel_analysis.yaml"
        assert cfg.exists()

    def test_config_yaml_loadable(self) -> None:
        import yaml

        from minisweagent.pipeline_workers import preprocess

        cfg = Path(preprocess.__file__).parent / "configs" / "kernel_analysis.yaml"
        data = yaml.safe_load(cfg.read_text())
        assert data["name"] == "kernel_analysis"
        assert "max_retries" in data["extra"]
