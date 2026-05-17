"""Contract validation wrapper for v3 preprocess step outputs.

Wraps the per-artifact validators that already live in
:mod:`minisweagent.kernel_languages.contract` (today: ``validate_harness``,
``validate_commandment``) under a single, step-oriented entry point
the orchestrator can call between phases.

The wrapper exists to:

* Translate ``ContractViolation`` exceptions into a structured
  :class:`ContractResult` that carries ``ok`` / ``step`` / ``issues``,
  so the orchestrator never has to wrap every gate call in a
  try/except.
* Map abstract "step" names (``"harness"``, ``"commandment"``, …)
  onto the concrete artifact dict keys the rest of the pipeline
  produces. This decouples the orchestrator's state-machine
  vocabulary from the validator surface.
* Keep the validation surface uniform for future steps (a v3
  ``"baseline"`` validator would slot in here without changing the
  caller).

Strict: no LLM calls, no network. Pure file-level inspection.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.kernel_languages.contract import (
    ContractViolation,
    validate_commandment,
    validate_harness,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContractResult:
    """Outcome of a contract gate check.

    Attributes:
        ok:
            ``True`` when every required artifact for ``step`` was
            present and passed its contract validator.
        step:
            The step name passed to :func:`validate_step_outputs`
            (echoed back so callers logging multiple results have a
            stable index).
        issues:
            Human-readable strings describing each failure. Empty
            when ``ok`` is ``True``. The first entry is typically
            the most actionable (e.g. ``"harness path does not
            exist: ..."``).
    """

    ok: bool
    step: str
    issues: list[str] = field(default_factory=list)


def _validate_via(
    fn: Callable[[Path], None],
    artifact_path: Path,
) -> list[str]:
    """Invoke a per-artifact validator and capture ``ContractViolation`` as strings."""
    issues: list[str] = []
    try:
        fn(Path(artifact_path))
    except ContractViolation as exc:
        issues.append(str(exc))
    except FileNotFoundError as exc:
        issues.append(str(exc))
    except Exception as exc:
        # Unexpected errors are surfaced so they don't get swallowed
        # silently, but tagged with type so the caller can tell them
        # apart from contract violations.
        issues.append(f"{type(exc).__name__}: {exc}")
    return issues


_KNOWN_ARTIFACT_KEYS: dict[str, tuple[str, Callable[[Path], None]]] = {
    "harness": ("harness", validate_harness),
    "commandment": ("commandment", validate_commandment),
}


def validate_step_outputs(
    step: str,
    artifacts: dict[str, Path],
    kernel_language: KernelLanguage,
) -> ContractResult:
    """Validate the artifacts produced by a v3 preprocess step.

    The dispatch table is keyed on ``step``:

    * ``"harness"`` — runs
      :func:`minisweagent.kernel_languages.contract.validate_harness`
      on ``artifacts["harness"]``.
    * ``"commandment"`` — runs
      :func:`minisweagent.kernel_languages.contract.validate_commandment`
      on ``artifacts["commandment"]``.

    Unknown ``step`` names produce a non-``ok`` :class:`ContractResult`
    rather than raising — the orchestrator can log the issue and
    decide whether to abort or continue.

    Args:
        step:
            Step name (one of the dispatch keys above, or anything
            else for the not-yet-implemented case).
        artifacts:
            Mapping ``artifact_kind -> path``. Must contain the
            artifact key the dispatch table associates with
            ``step``; missing keys produce an issue, not an
            exception.
        kernel_language:
            Resolved language for this run. Currently unused by the
            dispatch table (the per-artifact validators in
            ``kernel_languages/contract.py`` are language-agnostic
            permissive stubs today), but threaded through so future
            language-aware checks can hook in without changing the
            caller surface.

    Returns:
        A :class:`ContractResult`. ``ok`` is ``True`` when every
        validator returned cleanly; otherwise ``issues`` lists
        every ``ContractViolation`` and surrogate error encountered.
    """
    if step not in _KNOWN_ARTIFACT_KEYS:
        logger.debug(
            "validate_step_outputs: unknown step %r (known: %s); reporting as failure.",
            step,
            sorted(_KNOWN_ARTIFACT_KEYS),
        )
        return ContractResult(
            ok=False,
            step=step,
            issues=[
                f"unknown step {step!r}; supported: "
                f"{sorted(_KNOWN_ARTIFACT_KEYS)} (kernel_language={kernel_language.name!r})"
            ],
        )

    artifact_key, validator = _KNOWN_ARTIFACT_KEYS[step]
    if artifact_key not in artifacts:
        return ContractResult(
            ok=False,
            step=step,
            issues=[f"missing artifact {artifact_key!r} for step {step!r}"],
        )

    issues = _validate_via(validator, artifacts[artifact_key])
    return ContractResult(ok=not issues, step=step, issues=issues)


__all__ = [
    "ContractResult",
    "validate_step_outputs",
]
