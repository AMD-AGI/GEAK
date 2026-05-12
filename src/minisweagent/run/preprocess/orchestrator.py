"""PreprocessOrchestrator — drives the 4 mandatory + 1 conditional phase.

The orchestrator is deliberately thin: it iterates a fixed list of
phase instances and calls ``is_applicable`` / ``run`` on each, in
order.  Phases communicate via a shared ``PhaseContext``.

Phase order:

  1. TranslationPhase  — conditional; only when ``target_language``
                          differs from the source
  2. DiscoveryPhase    — always
  3. HarnessPhase      — always
  4. BaselinePhase     — always
  5. ExplorePhase      — always

During the preprocessing refactor transition period, the phase bodies
are thin — they either:

  (a) own their logic completely (e.g. ``DiscoveryPhase``), OR
  (b) early-exit so the orchestrator falls back to the legacy
      ``run_preprocessor`` monolith for the rest of the pipeline.

Case (b) is marked by the phase setting its output fields to empty /
None; the orchestrator detects any mandatory output still missing
after all phases ran and falls back to
``run_preprocessor_legacy`` for the missing steps.  This keeps every
interim commit rollback-safe without creating a second runtime path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext
from minisweagent.run.preprocess.phases.baseline import BaselinePhase
from minisweagent.run.preprocess.phases.discovery import DiscoveryPhase
from minisweagent.run.preprocess.phases.explore import ExplorePhase
from minisweagent.run.preprocess.phases.harness import HarnessPhase
from minisweagent.run.preprocess.phases.translation import TranslationPhase

logger = logging.getLogger(__name__)


class PreprocessOrchestrator:
    """Drive all preprocessing phases in order."""

    def __init__(self, phases: list[Phase] | None = None) -> None:
        self.phases: list[Phase] = phases or [
            TranslationPhase(),
            DiscoveryPhase(),
            HarnessPhase(),
            BaselinePhase(),
            ExplorePhase(),
        ]

    def run(self, ctx: PhaseContext) -> PhaseContext:
        """Execute every applicable phase.  Mutates ``ctx`` in place.

        If ``ctx.translate_only`` is True and TranslationPhase
        succeeds, we return early after TranslationPhase (the
        standalone ``geak translate`` path).
        """
        for phase in self.phases:
            if not phase.is_applicable(ctx):
                logger.debug("Phase %s skipped (is_applicable=False).", phase.name)
                ctx.phases_skipped.append((phase.name, "not applicable"))
                continue

            try:
                phase.run(ctx)
            except NotImplementedError:
                # Translation phase body isn't built yet; cli.py
                # pre-empts this case, but if someone invokes the
                # orchestrator directly we surface the error with
                # context.
                raise
            except Exception as exc:
                logger.error(
                    "Phase %s failed with %s: %s",
                    phase.name,
                    type(exc).__name__,
                    exc,
                )
                raise

            if ctx.translate_only and phase.name == TranslationPhase.name:
                logger.info("translate_only=True: returning after TranslationPhase")
                return ctx

            # §13.2-A row 3: honour ``GEAK_HARNESS_ONLY=1`` by returning
            # early after HarnessPhase.  Test harnesses that only need
            # the harness generated + validated (no profiling, no
            # baseline metrics, no commandment) use this env flag.
            if phase.name == HarnessPhase.name:
                from minisweagent.run.preprocess.phases.harness import (
                    is_harness_only_mode,
                )

                if is_harness_only_mode():
                    logger.info(
                        "GEAK_HARNESS_ONLY=1: returning after HarnessPhase "
                        "(skipping baseline + explore)."
                    )
                    return ctx

        return ctx


def run_preprocessor_via_orchestrator(
    kernel_url: str,
    output_dir: Path,
    gpu_id: int = 0,
    *,
    model: Any = None,
    model_factory: Any = None,
    console: Any = None,
    harness: str | None = None,
    repo: str | Path | None = None,
    eval_command: str | None = None,
    correctness_command: str | list[str] | None = None,
    performance_command: str | list[str] | None = None,
    benchmark_timeout: int = 3600,
    target_language: str | None = None,
    translate_only: bool = False,
    budget: Any = None,
    state: Any = None,
) -> dict[str, Any]:
    """Drop-in shim for the legacy ``run_preprocessor`` signature.

    Builds a ``PhaseContext`` from the kwargs, runs
    ``PreprocessOrchestrator``, and returns the output dict.  During
    the refactor transition, phases that haven't absorbed their
    logic yet cause the orchestrator to delegate to
    ``run_preprocessor`` (legacy) for the missing steps.  This keeps
    every commit rollback-safe.
    """
    ctx = PhaseContext(
        kernel_url=kernel_url,
        output_dir=Path(output_dir),
        gpu_id=gpu_id,
        harness=harness,
        repo=repo,
        eval_command=eval_command,
        correctness_command=correctness_command,
        performance_command=performance_command,
        benchmark_timeout=benchmark_timeout,
        model=model,
        model_factory=model_factory,
        console=console,
        target_language=target_language,
        translate_only=translate_only,
    )

    orch = PreprocessOrchestrator()
    orch.run(ctx)

    # Legacy fallback for any mandatory output not yet populated by
    # the new phase bodies.  Only triggered during the refactor
    # transition — deletes once every phase owns its logic.
    result = ctx.to_dict()
    if "harness" in ctx.phases_run and not result.get("baseline_metrics_path"):
        try:
            from minisweagent.run.preprocess.phases.harness import is_harness_only_mode

            if is_harness_only_mode():
                _validate_contract_artifacts(result, output_dir=Path(output_dir))
                return result
        except Exception:
            logger.debug("Harness-only detection failed; considering legacy fallback.", exc_info=True)
    if not result.get("harness_path") or not result.get("baseline_metrics_path"):
        logger.debug(
            "Orchestrator produced partial ctx (harness_path=%s, baseline_metrics_path=%s); "
            "falling back to legacy run_preprocessor for remaining steps.",
            bool(result.get("harness_path")),
            bool(result.get("baseline_metrics_path")),
        )
        from minisweagent.run.preprocess.preprocessor import run_preprocessor as _legacy

        legacy_result = _legacy(
            kernel_url=kernel_url,
            output_dir=output_dir,
            gpu_id=gpu_id,
            model=model,
            model_factory=model_factory,
            console=console,
            harness=harness,
            repo=repo,
            eval_command=eval_command,
            correctness_command=correctness_command,
            performance_command=performance_command,
            benchmark_timeout=benchmark_timeout,
            target_language=target_language,
            budget=budget,
            state=state,
        )
        # Legacy wins for fields where the new phases didn't produce
        # output; new phases win for fields they did populate.
        for key, value in legacy_result.items():
            if not result.get(key):
                result[key] = value

    # Universal-contract validation on the final artefacts, regardless
    # of whether they came from the new phases or the legacy fallback.
    # Permissive today (logs warnings only); tightens to FAIL once the
    # Jinja commandment templates + HarnessBuilder land.
    _validate_contract_artifacts(result, output_dir=Path(output_dir))

    return result


def _validate_contract_artifacts(result: dict[str, Any], *, output_dir: Path) -> None:
    """Validate harness + commandment against the universal contract.

    Logs warnings when an artifact is partially non-compliant.  Does
    not raise — keeps the behaviour backwards-compatible with the
    legacy path.  The FAIL-strict transition lands when PR-2's full
    template migration completes.
    """
    try:
        from minisweagent.kernel_languages.contract import (
            validate_commandment,
            validate_harness,
        )
    except Exception:
        return

    harness_path = result.get("harness_path")
    if harness_path:
        try:
            validate_harness(Path(harness_path))
        except Exception as exc:
            logger.warning("[yellow]validate_harness: %s[/yellow]", exc)

    cm_path = output_dir / "COMMANDMENT.md"
    if cm_path.exists():
        try:
            validate_commandment(cm_path)
        except Exception as exc:
            logger.warning("[yellow]validate_commandment: %s[/yellow]", exc)


__all__ = ["PreprocessOrchestrator", "run_preprocessor_via_orchestrator"]
