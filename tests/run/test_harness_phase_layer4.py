"""Tests for HarnessPhase — the 7-layer resolution chain.

Covers:
  - Layer 1 (harness_path already set) short-circuits
  - Layer 2 (explicit --harness) validates + promotes
  - Layer 3 (split-harness-hint) promotes when valid
  - Layer 4 (testcase_cache) — pinned elsewhere via smoke path
  - Layer 5 (HarnessBuilder subagent) — success + failure paths
  - Layer 6 (UnitTestAgent legacy) — skipped when model absent
  - Layer 7 (discovery fallback) — focused_test / tests[0]

The runtime validator ``execute_harness_validation`` is subprocess-
heavy so every test patches it to return OK without actually running
python against the kernel.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.preprocess.phases.base import PhaseContext
from minisweagent.run.preprocess.phases.harness import (
    HarnessPhase,
    _build_test_command,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _make_language(
    tmp_path: Path,
    *,
    template_body: str = "# jinja harness skeleton\n",
) -> MagicMock:
    lang = MagicMock()
    lang.name = "triton"
    lang.harness_template = template_body
    lang.builder_hints = "hints"
    lang.system_prompt = "you are the worker"
    (tmp_path / "system_prompt.md").write_text(lang.system_prompt)
    lang.system_prompt_path = tmp_path / "system_prompt.md"
    return lang


def _valid_harness_code() -> str:
    return """\
import argparse
p = argparse.ArgumentParser()
mutex = p.add_mutually_exclusive_group(required=True)
mutex.add_argument("--correctness", action="store_true")
mutex.add_argument("--benchmark", action="store_true")
mutex.add_argument("--full-benchmark", action="store_true")
mutex.add_argument("--profile", action="store_true")
a = p.parse_args()
print("GEAK_RESULT_LATENCY_MS=1.0")
print("GEAK_RESULT_SPEEDUP=1.0")
"""


@pytest.fixture
def mock_runtime_ok():
    """Patch execute_harness_validation to return OK everywhere it's imported."""
    import_sites = [
        "minisweagent.run.preprocess.harness_utils.execute_harness_validation",
    ]
    patches = [
        patch(site, return_value=(True, [], [
            {"mode": "correctness", "success": True, "duration_s": 0.1},
            {"mode": "benchmark", "success": True, "duration_s": 0.1},
            {"mode": "full-benchmark", "success": True, "duration_s": 0.1},
            {"mode": "profile", "success": True, "duration_s": 0.1},
        ]))
        for site in import_sites
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


@pytest.fixture
def mock_cache_miss():
    """Make the testcase_cache always report a miss."""
    with patch(
        "minisweagent.run.preprocess.testcase_cache.get_testcase_cache_entry",
        return_value=None,
    ):
        yield


# ──────────────────────────────────────────────────────────────────────
# Layer 1 — harness_path already set
# ──────────────────────────────────────────────────────────────────────


class TestLayer1AlreadySet:
    def test_short_circuits_when_harness_path_set(self, tmp_path: Path) -> None:
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.harness_path = "/tmp/preexisting.py"
        ctx.test_command = "python3 /tmp/preexisting.py --correctness"

        with patch(
            "minisweagent.subagents.preprocess.harness_builder.HarnessBuilder"
        ) as mock_builder:
            HarnessPhase().run(ctx)
            mock_builder.assert_not_called()

        assert ctx.harness_path == "/tmp/preexisting.py"


# ──────────────────────────────────────────────────────────────────────
# Layer 5 — HarnessBuilder (D1)
# ──────────────────────────────────────────────────────────────────────


class TestLayer5HarnessBuilder:
    def test_skipped_when_language_is_none(
        self, tmp_path: Path, mock_runtime_ok, mock_cache_miss
    ) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = None
        ctx.model = MagicMock()

        HarnessPhase().run(ctx)
        assert ctx.harness_path is None or ctx.harness_path == ""

    def test_skipped_when_harness_template_is_empty(
        self, tmp_path: Path, mock_runtime_ok, mock_cache_miss
    ) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _make_language(tmp_path, template_body="")
        ctx.model = MagicMock()

        HarnessPhase().run(ctx)
        assert ctx.harness_path is None or ctx.harness_path == ""

    def test_skipped_when_model_unavailable(
        self, tmp_path: Path, mock_runtime_ok, mock_cache_miss
    ) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _make_language(tmp_path)
        ctx.model = None
        ctx.model_factory = None

        HarnessPhase().run(ctx)
        assert ctx.harness_path is None or ctx.harness_path == ""

    def test_success_populates_all_harness_fields(
        self, tmp_path: Path, mock_runtime_ok, mock_cache_miss
    ) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("@triton.jit\ndef foo(): pass\n")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.repo_root = str(tmp_path)
        ctx.language = _make_language(tmp_path)
        model = MagicMock()
        model.query = MagicMock(return_value=_valid_harness_code())
        ctx.model = model

        HarnessPhase().run(ctx)

        # Auto-generated harness uses the ``_geak_`` ownership prefix
        # so it can never collide with a user file named ``harness.py``.
        expected = tmp_path / "_geak_auto_harness.py"
        assert ctx.harness_path == str(expected)
        assert ctx.test_command is not None
        assert "--correctness" in ctx.test_command
        assert str(expected) in ctx.test_command
        assert expected.exists()

    def test_falls_back_when_harness_builder_fails(
        self,
        tmp_path: Path,
        mock_runtime_ok,
        mock_cache_miss,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Model returns garbage -> HarnessBuildFailed -> Layer 5 returns None.

        Layer 6 (UTA) is skipped because we mock its dependency, and
        Layer 7 is also skipped because discovery is empty.  Final:
        no harness resolved, phase leaves ctx.harness_path unset.

        We override ``GEAK_HARNESS_BUILDER_BUDGET_S`` to a sub-second
        value so the validate-retry loop terminates quickly in tests
        even though the production default is 30 min.
        """
        monkeypatch.setenv("GEAK_HARNESS_BUILDER_BUDGET_S", "0.1")

        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _make_language(tmp_path)
        ctx.repo_root = None  # disables Layer 6 (UTA)
        model = MagicMock()
        model.query = MagicMock(return_value="def main(): pass\n")
        ctx.model = model

        HarnessPhase().run(ctx)
        assert ctx.harness_path is None or ctx.harness_path == ""

    def test_model_factory_used_when_model_is_none(
        self, tmp_path: Path, mock_runtime_ok, mock_cache_miss
    ) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _make_language(tmp_path)
        ctx.model = None
        factory_model = MagicMock()
        factory_model.query = MagicMock(return_value=_valid_harness_code())
        ctx.model_factory = lambda: factory_model

        HarnessPhase().run(ctx)
        assert ctx.harness_path == str(tmp_path / "_geak_auto_harness.py")


# ──────────────────────────────────────────────────────────────────────
# Layer 7 — discovery fallback
# ──────────────────────────────────────────────────────────────────────


class TestLayer7DiscoveryFallback:
    def test_uses_focused_test_when_available(
        self, tmp_path: Path, mock_cache_miss
    ) -> None:
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(tmp_path / "k.py")
        (tmp_path / "k.py").write_text("pass")
        ctx.discovery = {
            "focused_test": {
                "focused_command": "pytest /tmp/focused.py --correctness"
            }
        }

        HarnessPhase().run(ctx)
        assert ctx.test_command == "pytest /tmp/focused.py --correctness"

    def test_uses_tests_zero_when_no_focused(
        self, tmp_path: Path, mock_cache_miss
    ) -> None:
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(tmp_path / "k.py")
        (tmp_path / "k.py").write_text("pass")
        ctx.discovery = {
            "tests": [{"command": "pytest /tmp/unit.py"}]
        }

        HarnessPhase().run(ctx)
        assert ctx.test_command == "pytest /tmp/unit.py"


# ──────────────────────────────────────────────────────────────────────
# _build_test_command shape
# ──────────────────────────────────────────────────────────────────────


class TestBuildTestCommand:
    def test_shape_matches_legacy(self, tmp_path: Path) -> None:
        """``_build_test_command`` emits ``python3 <absolute> --correctness``
        (same shape as legacy ``_build_deterministic_test_command``)."""
        import shlex
        import sys

        harness = tmp_path / "h.py"
        harness.write_text("")
        cmd = _build_test_command(str(harness))
        assert cmd == (
            f"{shlex.quote(sys.executable)} "
            f"{shlex.quote(str(harness.resolve()))} --correctness"
        )

    def test_uses_resolved_absolute_path(self, tmp_path: Path) -> None:
        harness = tmp_path / "h.py"
        harness.write_text("")
        cmd = _build_test_command(str(harness))
        assert str(harness.resolve()) in cmd
