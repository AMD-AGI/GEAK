"""Base Phase contract for the preprocessing orchestrator.

Every preprocessing phase implements the same tiny interface:

    class MyPhase(Phase):
        name = "my_phase"

        def is_applicable(self, ctx: PhaseContext) -> bool:
            return True   # override for conditional phases (e.g. Translation)

        def run(self, ctx: PhaseContext) -> None:
            ...           # read/write ctx; raise on fatal errors

Kept deliberately minimal — the orchestrator in
``preprocess/orchestrator.py`` iterates over a list of phase instances
and calls ``is_applicable`` / ``run`` on each.  Phases do not know about
each other; they communicate via ``PhaseContext``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PhaseContext:
    """Shared mutable state threaded through preprocess phases.

    This is the phase-level view of ``PreprocessContext`` plus the
    inputs the caller (``cli.py``) provides.  Phases read the fields
    relevant to their concern and write the fields they produce; the
    orchestrator converts this to a final ``PreprocessContext`` at the
    end for downstream consumers.

    Inputs (set by caller before running phases):
      - kernel_url           — path / URL to source kernel
      - output_dir           — where artefacts land
      - gpu_id               — GPU device for profiling
      - harness              — explicit harness path (optional)
      - repo                 — repository root (optional; inferred if missing)
      - eval_command / correctness_command / performance_command — CLI eval hooks
      - benchmark_timeout    — baseline subprocess timeout (seconds)
      - model / model_factory / console — LLM + UI plumbing
      - target_language      — when set and different from the detected
                                source language, TranslationPhase runs
      - translate_only       — standalone ``geak translate``; stop after
                                TranslationPhase

    Outputs (populated by phases):
      - kernel_path, repo_root   — set by DiscoveryPhase
      - discovery                 — set by DiscoveryPhase
      - codebase_context_path     — set by DiscoveryPhase
      - harness_path, test_command, harness_results,
        testcase_selection        — set by HarnessPhase
      - profiling                 — set by BaselinePhase (profile.json payload)
      - baseline_metrics_path,
        baseline_metrics,
        benchmark_baseline,
        full_benchmark_baseline   — set by BaselinePhase
      - commandment_path,
        commandment               — set by ExplorePhase
      - kernel_analysis_md        — set by ExplorePhase (future KernelAnalysisAgent)
      - resolved                  — set by DiscoveryPhase

    Phase status:
      - phases_run                — ordered list of phase names that executed
      - phases_skipped            — ordered list of (name, reason) pairs
    """

    # ── Inputs (required) ───────────────────────────────────────────────
    kernel_url: str = ""
    output_dir: Path = field(default_factory=Path)
    gpu_id: int = 0

    # ── Inputs (optional plumbing) ──────────────────────────────────────
    harness: str | None = None
    repo: str | Path | None = None
    eval_command: str | None = None
    correctness_command: str | list[str] | None = None
    performance_command: str | list[str] | None = None
    benchmark_timeout: int = 3600
    model: Any = None
    model_factory: Any = None
    console: Any = None

    # ── Inputs (translation signals) ────────────────────────────────────
    target_language: str | None = None
    translate_only: bool = False

    # ── Inputs (user task contract) ─────────────────────────────────────
    user_task: str | None = None
    """User-supplied ``-t`` prompt forwarded from ``mini.py``.

    When set and non-empty, harness-producing phases prepend it as a
    ``USER TASK CONTEXT (HIGHEST PRIORITY)`` block to the UnitTestAgent
    and ShapeFixerAgent task prompts.  The matching system-prompt YAMLs
    (``mini_unit_test_agent.yaml`` / ``mini_shape_fixer.yaml``) then
    override the discovered benchmark file's default shape/dtype sweep
    with the production contract from the user task.

    Wired by :func:`run_preprocessor_via_orchestrator`; consumed by
    :class:`~minisweagent.run.preprocess.phases.harness.HarnessPhase`
    (``_layer6_unit_test_agent`` and ``_run_shape_fixer``).  Backward-
    compatible: ``None`` falls back to the legacy discovery-driven
    behaviour."""

    # ── Outputs (populated by phases) ───────────────────────────────────
    kernel_path: str = ""
    repo_root: str = ""
    resolved: dict | None = None
    codebase_context_path: str | None = None
    discovery: dict | None = None
    language: Any = None
    """Resolved ``KernelLanguage`` instance for the (possibly translated)
    kernel.  Populated by DiscoveryPhase via
    ``kernel_languages.registry.detect_best(Path(kernel_path))``.  Downstream
    phases (ExplorePhase for Jinja commandment, etc.) read
    ``ctx.language.<path_field>`` to render language-driven templates
    without re-doing detection.  Typed as ``Any`` to avoid an import
    cycle with the kernel_languages package."""

    harness_path: str = ""
    test_command: str | None = None
    harness_results: list[dict] | None = None
    testcase_selection: dict | None = None

    profiling: Any = None
    baseline_metrics_path: str | None = None
    baseline_metrics: dict | None = None
    benchmark_baseline: str | None = None
    full_benchmark_baseline: str | None = None
    correctness: dict | None = None
    """Set by BaselinePhase on the eval_command path — mirrors the legacy
    ``ctx["correctness"]`` dict (command / returncode / stdout_path /
    stderr_path).  §13.2-A row 1."""

    commandment: str | None = None
    commandment_path: str | None = None
    kernel_analysis_md: str | None = None
    evaluation_contract: dict[str, Any] | None = None

    split_harness_hint: str | None = None
    """Absolute path to a harness file produced by DiscoveryPhase when
    it detected a merged kernel file (kernel + test logic) and split
    them.  HarnessPhase uses this as a candidate harness when
    ``ctx.harness`` is unset.  §13.2-A row 6."""

    harness_seed: str | None = None
    """Absolute path to a USER-SUPPLIED harness that failed to satisfy
    the language's full contract (Layer 2 set it on its way out).
    HarnessBuilder (Layer 5) consumes this as a starting template for
    its validate-retry loop — the LLM gets the user's harness in the
    prompt with a "fix this to pass the universal contract" directive,
    which converges faster than generating from scratch.  None when
    the user didn't supply a harness or their harness already passed."""

    # ── Status ──────────────────────────────────────────────────────────
    phases_run: list[str] = field(default_factory=list)
    phases_skipped: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the OUTPUT fields as a dict compatible with the legacy
        ``run_preprocessor()`` return shape.

        Inputs are intentionally omitted — the dict is what downstream
        orchestration consumes.
        """
        return {
            "kernel_path": self.kernel_path,
            "repo_root": self.repo_root,
            "resolved": self.resolved,
            "codebase_context_path": self.codebase_context_path,
            "discovery": self.discovery,
            "harness_path": self.harness_path,
            "test_command": self.test_command,
            "harness_results": self.harness_results,
            "testcase_selection": self.testcase_selection,
            "profiling": self.profiling,
            "baseline_metrics_path": self.baseline_metrics_path,
            "baseline_metrics": self.baseline_metrics,
            "benchmark_baseline": self.benchmark_baseline,
            "full_benchmark_baseline": self.full_benchmark_baseline,
            "correctness": self.correctness,
            "commandment": self.commandment,
            "commandment_path": self.commandment_path,
            "kernel_analysis_md": self.kernel_analysis_md,
            "evaluation_contract": self.evaluation_contract,
        }


class Phase:
    """Base class for a single preprocessing phase.

    Subclasses set ``name`` (class attr) and override ``run``.  Override
    ``is_applicable`` only for conditional phases.
    """

    name: str = "phase"

    def is_applicable(self, ctx: PhaseContext) -> bool:
        """Return False to have the orchestrator skip this phase.

        Default: always run.  ``TranslationPhase`` overrides to gate on
        ``ctx.target_language`` differing from the detected source.
        """
        return True

    def run(self, ctx: PhaseContext) -> None:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__}.run must be overridden")

    # Convenience helper subclasses may use for logging phase boundaries
    def _log_enter(self) -> None:
        logger.info("[bold cyan]--- Phase: %s ---[/bold cyan]", self.name)


__all__ = ["Phase", "PhaseContext"]
