"""Explore phase — render commandment + (future) kernel analysis rubric.

Inputs  (read from ctx):
  - kernel_path, harness_path, repo_root
  - test_command, eval_command, correctness_command, performance_command
  - discovery  (to infer kernel language when ``ctx.language`` unset)
  - language   (optional ``KernelLanguage`` populated by DiscoveryPhase)
  - baseline_metrics, profiling, codebase_context_path  (future: feed
    ``KernelAnalysisAgent``)

Outputs (written to ctx):
  - commandment           (str markdown)
  - commandment_path      (str path to COMMANDMENT.md)
  - kernel_analysis_md    (future — populated by KernelAnalysisAgent)

Render path selection (plan §13.2-E row 26):

  1. **Jinja (preferred)** — if ``ctx.language.commandment_template_path``
     is set, render via Jinja.  The template receives all the same
     variables the legacy Python generator consumed plus the full
     ``ctx`` object for extensibility.
  2. **Legacy fallback** — if the KernelLanguage has no Jinja
     template OR Jinja rendering fails, fall back to the legacy
     ``run/preprocess/commandment.py`` functions.

Both paths write ``{output_dir}/COMMANDMENT.md`` and run the universal
``validate_commandment`` contract on the output before returning.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

logger = logging.getLogger(__name__)


def _join_cmd(cmd: str | list[str] | None) -> str | None:
    if cmd is None:
        return None
    if isinstance(cmd, list):
        return " && ".join(c.strip() for c in cmd if c.strip()) or None
    return cmd.strip() or None


def _try_jinja_render(
    *,
    ctx: PhaseContext,
    correctness_cmd: str | None,
    perf_cmd: str | None,
    compile_cmd: str | None,
    harness_path: str | None,
    inner_kernel: bool,
    profile_replays: int,
) -> str | None:
    """Render the per-language Jinja commandment template if available.

    Returns:
        Rendered markdown string on success.
        ``None`` when the language has no template OR rendering fails
        (caller then falls back to legacy path).
    """
    language = ctx.language
    if language is None or language.commandment_template_path is None:
        return None

    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError:
        logger.debug("jinja2 not installed; falling back to legacy commandment.py")
        return None

    template_path = Path(language.commandment_template_path)
    if not template_path.exists():
        return None

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=True,
        undefined=StrictUndefined,  # catch template variable typos at render time
    )
    try:
        tmpl = env.get_template(template_path.name)
        return tmpl.render(
            kernel_path=ctx.kernel_path,
            harness_path=harness_path or "",
            repo_root=ctx.repo_root,
            inner_kernel=inner_kernel,
            profile_replays=profile_replays,
            correctness_command=correctness_cmd,
            performance_command=perf_cmd,
            compile_command=compile_cmd,
        )
    except Exception as exc:
        logger.warning(
            "[yellow]Jinja commandment render failed for language=%s: %s; "
            "falling back to legacy commandment.py[/yellow]",
            getattr(language, "name", "?"),
            exc,
        )
        return None


class ExplorePhase(Phase):
    """Render the commandment markdown.

    Two paths for commandment sourcing (plan §13.2-E row 26):

      1. Jinja (preferred) — language.commandment_template_path.
      2. Legacy Python (``commandment.py``).

    Both paths feed through the universal ``validate_commandment``
    contract before returning.
    """

    name = "explore"

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()
        if ctx.commandment_path:
            logger.info("  commandment already rendered; skipping.")
            return

        # Require prerequisites — defer to legacy fallback when missing.
        if not ctx.kernel_path:
            logger.debug("ExplorePhase: no kernel_path yet; deferring to legacy fallback.")
            return
        if not (ctx.test_command or ctx.eval_command):
            logger.info("  Skipping commandment (no test_command or eval_command)")
            ctx.phases_run.append(self.name)
            return

        output_dir = Path(ctx.output_dir)
        correctness_cmd = _join_cmd(ctx.correctness_command)
        perf_cmd = _join_cmd(ctx.performance_command)

        # Build harness_path the same way both paths need it.
        harness_path: str | None = None
        if ctx.test_command:
            try:
                from minisweagent.run.preprocess.harness_utils import extract_harness_path

                harness_path = ctx.harness_path or extract_harness_path(ctx.test_command)
            except Exception:
                harness_path = ctx.harness_path

        # Profile replay count — legacy default is 3.  (Future: read
        # from language-specific config or baseline metrics.)
        profile_replays = 3

        commandment: str | None = _try_jinja_render(
            ctx=ctx,
            correctness_cmd=correctness_cmd,
            perf_cmd=perf_cmd,
            compile_cmd=None,
            harness_path=harness_path,
            inner_kernel=False,  # TODO: detect from kernel content; legacy does this too
            profile_replays=profile_replays,
        )

        if commandment is not None:
            logger.info(
                "  COMMANDMENT.md rendered via Jinja (language=%s)",
                getattr(ctx.language, "name", "?"),
            )
        elif ctx.eval_command:
            try:
                from minisweagent.run.preprocess.commandment import (
                    generate_commandment_from_commands,
                )

                commandment = generate_commandment_from_commands(
                    kernel_path=ctx.kernel_path,
                    compile_command=None,
                    correctness_command=correctness_cmd,
                    performance_command=perf_cmd or ctx.eval_command,
                    repo_root=ctx.repo_root,
                )
                logger.info("  COMMANDMENT.md generated (legacy, from eval command)")
            except Exception as exc:
                logger.warning("[yellow]Commandment from command failed: %s[/yellow]", exc, exc_info=True)

            # §13.2-A row 5: legacy preprocessor.py:1204 sets
            # ``ctx["test_command"] = eval_command`` on the eval path so
            # downstream consumers see a consistent test_command regardless
            # of which path produced it.  Match that behaviour.
            if not ctx.test_command:
                ctx.test_command = ctx.eval_command
        elif ctx.test_command:
            try:
                from minisweagent.run.preprocess.commandment import generate_commandment
                from minisweagent.run.preprocess.discovery_types import (
                    _infer_kernel_language,
                )

                kernel_type = (ctx.discovery or {}).get("kernel", {}).get("type", "")
                kernel_language = _infer_kernel_language(Path(ctx.kernel_path), kernel_type)
                commandment = generate_commandment(
                    kernel_path=ctx.kernel_path,
                    harness_path=harness_path,
                    repo_root=ctx.repo_root,
                    kernel_language=kernel_language,
                )
                logger.info("  COMMANDMENT.md generated (legacy, from harness)")
            except Exception as exc:
                logger.warning("[yellow]Commandment failed: %s[/yellow]", exc, exc_info=True)

        ctx.commandment = commandment
        if commandment:
            cm_path = output_dir / "COMMANDMENT.md"
            cm_path.write_text(commandment)
            ctx.commandment_path = str(cm_path)

            # Universal contract validator — same call for Jinja and
            # legacy outputs so either path must satisfy the 5-section
            # contract enforced by kernel_languages/contract.py.
            try:
                from minisweagent.kernel_languages.contract import (
                    validate_commandment,
                )

                validate_commandment(cm_path)
            except Exception as exc:
                logger.warning("[yellow]validate_commandment: %s[/yellow]", exc)

        # ── Kernel analysis rubric (D2) ─────────────────────────────
        #
        # Best-effort [A]-[D] rubric produced by KernelAnalysisAgent
        # and written to ``{output_dir}/kernel_analysis.md``.  When
        # populated, ``ctx.kernel_analysis_md`` is prepended to every
        # task body by ``compose_task_body`` so both fixed and planned
        # modes see the same structured analysis.  Missing language
        # or model -> silent skip (advisory context, not a hard
        # requirement).
        _try_kernel_analysis(ctx, output_dir=output_dir)

        ctx.phases_run.append(self.name)


def _try_kernel_analysis(ctx: PhaseContext, *, output_dir: Path) -> None:
    """Produce the [A]-[D] rubric via KernelAnalysisAgent (D2).

    GATED BEHIND ``GEAK_USE_KERNEL_ANALYSIS=1`` — OFF BY DEFAULT.

    Rationale: the rubric subagent is NEW in the refactor pipeline
    (no equivalent in ``origin/main``).  When we run parity tests
    against the old pipeline, we need an apples-to-apples comparison
    with this subagent OFF.  Set ``GEAK_USE_KERNEL_ANALYSIS=1`` in
    the environment to enable it.

    Silently skips when any of:
      - ``GEAK_USE_KERNEL_ANALYSIS`` is not ``"1"`` (default).
      - ``ctx.language`` is None (DiscoveryPhase did not resolve a
        KernelLanguage — nothing to feed the subagent with).
      - No model is available (``ctx.model`` unset AND no model
        factory).
      - ``ctx.kernel_path`` is missing.
      - The subagent raises for any reason (analysis is advisory).

    When it succeeds, sets ``ctx.kernel_analysis_md`` to the rendered
    markdown string (so ``compose_task_body`` can inject it without
    re-reading from disk).
    """
    if os.environ.get("GEAK_USE_KERNEL_ANALYSIS", "0") != "1":
        logger.debug("  KernelAnalysisAgent: gated off (GEAK_USE_KERNEL_ANALYSIS!=1); skipping.")
        return
    if ctx.language is None:
        logger.debug("  KernelAnalysisAgent: ctx.language is None; skipping rubric.")
        return
    if not ctx.kernel_path or not Path(ctx.kernel_path).is_file():
        logger.debug("  KernelAnalysisAgent: no kernel_path; skipping rubric.")
        return

    model = _resolve_model(ctx)
    if model is None:
        logger.debug("  KernelAnalysisAgent: no model available; skipping rubric.")
        return

    try:
        from minisweagent.pipeline_workers.base import SubagentConfig
        from minisweagent.pipeline_workers.preprocess.kernel_analysis import (
            KernelAnalysisAgent,
        )
    except Exception as exc:
        logger.warning("[yellow]KernelAnalysisAgent import failed: %s[/yellow]", exc)
        return

    out_path = output_dir / "kernel_analysis.md"
    config = SubagentConfig(
        name="kernel_analysis",
        model_name=getattr(model, "name", "kernel_analysis_model"),
        system_template="",
        instance_template="",
        step_limit=1,
        cost_limit=3.0,
        temperature=0.2,
        extra={"max_retries": 1},
    )
    agent = KernelAnalysisAgent(language=ctx.language, config=config)
    agent.model = model  # type: ignore[attr-defined]

    try:
        result = agent.run(
            kernel_path=Path(ctx.kernel_path),
            out_path=out_path,
            profile=ctx.profiling,
            baseline_metrics=ctx.baseline_metrics,
            codebase_context_path=(Path(ctx.codebase_context_path) if ctx.codebase_context_path else None),
            max_retries=int(config.extra.get("max_retries", 1)),
        )
    except Exception as exc:  # noqa: BLE001 — shield the phase
        logger.warning(
            "[yellow]KernelAnalysisAgent raised %s: %s; skipping rubric.[/yellow]",
            type(exc).__name__,
            exc,
        )
        return

    if not isinstance(result, dict):
        return

    analysis_path = result.get("analysis_path")
    if not analysis_path:
        return

    # Read the written markdown back so ctx.kernel_analysis_md carries
    # the rubric string (compose_task_body consumes this directly, no
    # disk-read needed).
    try:
        ctx.kernel_analysis_md = Path(analysis_path).read_text(encoding="utf-8")
    except Exception:
        ctx.kernel_analysis_md = None
        return

    logger.info(
        "  KernelAnalysisAgent produced rubric: %s (ok=%s, attempts=%s)",
        analysis_path,
        result.get("ok"),
        result.get("attempts_used"),
    )


def _resolve_model(ctx: PhaseContext) -> Any:
    """Return a model instance, constructing one from the factory if needed."""
    if ctx.model is not None:
        return ctx.model
    factory = getattr(ctx, "model_factory", None)
    if callable(factory):
        try:
            return factory()
        except Exception as exc:
            logger.debug("model_factory raised %s: %s", type(exc).__name__, exc)
            return None
    return None


__all__ = ["ExplorePhase"]
