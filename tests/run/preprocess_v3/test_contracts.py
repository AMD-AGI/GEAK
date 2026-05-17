"""Tests for ``minisweagent.run.preprocess_v3.contracts``.

The legacy ``kernel_languages/contract.py`` validators are
deliberately permissive (PR-1 stubs); the wrapper still has to
exercise the failure surface they DO enforce — missing files,
totally non-conformant harnesses — and surface them as
:class:`ContractResult` issues. We build small fixture files under
``tmp_path`` to drive each branch.
"""

from __future__ import annotations

from pathlib import Path

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3.contracts import (
    ContractResult,
    validate_step_outputs,
)


def _fakeling() -> KernelLanguage:
    """Build a minimal ``KernelLanguage`` for tests that don't care about language."""
    return KernelLanguage(
        name="fakeling",
        file_extensions=frozenset({".py"}),
        kb_namespace="fakeling",
    )


def _good_harness(tmp_path: Path) -> Path:
    """A harness body containing every required flag + every required marker."""
    harness = tmp_path / "harness.py"
    harness.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--correctness', action='store_true')\n"
        "p.add_argument('--benchmark', action='store_true')\n"
        "p.add_argument('--full-benchmark', action='store_true')\n"
        "p.add_argument('--profile', action='store_true')\n"
        "print('GEAK_RESULT_LATENCY_MS=1.0')\n"
        "print('GEAK_RESULT_SPEEDUP=1.0')\n",
        encoding="utf-8",
    )
    return harness


def _bad_harness(tmp_path: Path) -> Path:
    """Harness lacking BOTH the required flags AND the required markers."""
    harness = tmp_path / "harness.py"
    harness.write_text("# nothing here\n", encoding="utf-8")
    return harness


def _good_commandment(tmp_path: Path) -> Path:
    """A commandment with the v3-style five level-2 sections in order."""
    commandment = tmp_path / "COMMANDMENT.md"
    commandment.write_text(
        "## Setup\nrun.sh\n\n"
        "## Correctness\nrun.sh --correctness\n\n"
        "## Benchmark\nrun.sh --benchmark\n\n"
        "## Full Benchmark\nrun.sh --full-benchmark\n\n"
        "## Profile\nkernel-profile ...\n",
        encoding="utf-8",
    )
    return commandment


# ---------------------------------------------------------------------------
# harness step
# ---------------------------------------------------------------------------


def test_validate_step_outputs_harness_passes_for_valid_harness(tmp_path: Path) -> None:
    """A fully-conformant harness yields ``ok=True`` with no issues."""
    harness = _good_harness(tmp_path)

    result = validate_step_outputs(
        "harness",
        {"harness": harness},
        _fakeling(),
    )

    assert isinstance(result, ContractResult)
    assert result.ok is True
    assert result.step == "harness"
    assert result.issues == []


def test_validate_step_outputs_harness_fails_for_non_conformant_harness(tmp_path: Path) -> None:
    """A harness missing both flags AND markers raises a ``ContractViolation``.

    The legacy validator is permissive on partial misses but raises
    when both sets are missing — exactly the case we synthesize here.
    """
    harness = _bad_harness(tmp_path)

    result = validate_step_outputs(
        "harness",
        {"harness": harness},
        _fakeling(),
    )

    assert result.ok is False
    assert result.step == "harness"
    assert len(result.issues) == 1
    assert "missing required flags" in result.issues[0]
    assert "missing required markers" in result.issues[0] or "required markers" in result.issues[0]


def test_validate_step_outputs_harness_fails_for_missing_file(tmp_path: Path) -> None:
    """A path that doesn't exist on disk is reported as an issue, not raised."""
    missing = tmp_path / "nope.py"

    result = validate_step_outputs(
        "harness",
        {"harness": missing},
        _fakeling(),
    )

    assert result.ok is False
    assert any("does not exist" in issue for issue in result.issues)


def test_validate_step_outputs_harness_fails_when_artifact_key_absent(tmp_path: Path) -> None:
    """The orchestrator forgot to plumb the harness key — surface, don't raise."""
    result = validate_step_outputs("harness", {}, _fakeling())

    assert result.ok is False
    assert any("missing artifact 'harness'" in issue for issue in result.issues)


# ---------------------------------------------------------------------------
# commandment step
# ---------------------------------------------------------------------------


def test_validate_step_outputs_commandment_passes_for_valid_commandment(tmp_path: Path) -> None:
    """The legacy validator accepts the v3 5-section format."""
    commandment = _good_commandment(tmp_path)

    result = validate_step_outputs(
        "commandment",
        {"commandment": commandment},
        _fakeling(),
    )

    assert result.ok is True
    assert result.issues == []


def test_validate_step_outputs_commandment_passes_for_legacy_uppercase_format(tmp_path: Path) -> None:
    """The legacy validator is permissive — uppercase format also passes today.

    ``kernel_languages/contract.py::validate_commandment`` returns
    silently when the v3 sections aren't found (PR-1 permissive
    stub), so the legacy-style ``## SETUP / ## CORRECTNESS / …``
    output our :func:`render_commandment` legacy fallback emits is
    treated as ``ok=True``. Locks that behaviour in so the wrapper
    doesn't accidentally tighten the legacy semantics.
    """
    commandment = tmp_path / "legacy_COMMANDMENT.md"
    commandment.write_text(
        "## SETUP\nrun.sh\n\n"
        "## CORRECTNESS\nrun.sh --correctness\n\n"
        "## PROFILE\nkernel-profile ...\n\n"
        "## BENCHMARK\nrun.sh --benchmark\n\n"
        "## FULL_BENCHMARK\nrun.sh --full-benchmark\n",
        encoding="utf-8",
    )

    result = validate_step_outputs(
        "commandment",
        {"commandment": commandment},
        _fakeling(),
    )

    assert result.ok is True


def test_validate_step_outputs_commandment_fails_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "no_commandment.md"

    result = validate_step_outputs(
        "commandment",
        {"commandment": missing},
        _fakeling(),
    )

    assert result.ok is False
    assert any("does not exist" in issue for issue in result.issues)


def test_validate_step_outputs_commandment_fails_when_artifact_key_absent(tmp_path: Path) -> None:
    result = validate_step_outputs(
        "commandment",
        {"harness": tmp_path / "h.py"},  # wrong key
        _fakeling(),
    )

    assert result.ok is False
    assert any("missing artifact 'commandment'" in issue for issue in result.issues)


# ---------------------------------------------------------------------------
# unknown step
# ---------------------------------------------------------------------------


def test_validate_step_outputs_reports_unknown_step(tmp_path: Path) -> None:
    """An unknown ``step`` name produces a structured issue, not an exception."""
    result = validate_step_outputs(
        "not-a-real-step",
        {"harness": tmp_path / "h.py"},
        _fakeling(),
    )

    assert result.ok is False
    assert result.step == "not-a-real-step"
    assert any("unknown step" in issue for issue in result.issues)
    assert any("'fakeling'" in issue for issue in result.issues)


# ---------------------------------------------------------------------------
# Result dataclass surface
# ---------------------------------------------------------------------------


def test_contract_result_default_issues_is_empty_list(tmp_path: Path) -> None:
    """Constructing a passing result without an explicit ``issues`` defaults to ``[]``."""
    r = ContractResult(ok=True, step="harness")
    assert r.issues == []
