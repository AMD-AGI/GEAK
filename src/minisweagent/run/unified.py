"""Unified pipeline entry for fixed, planned, and mixed modes.

All modes run the SAME ``OptimizationAgent`` class on each worker.
The only thing that differs is the **task body** each worker receives:

  - ``fixed``   — one task body, replicated across ``num_parallel`` copies.
                  Variance comes from LLM sampling alone.  Good when the
                  language or problem already has strong priors (e.g. HIP
                  kernels with a single obvious optimization axis).
  - ``planned`` — A planner LLM emits N diverse strategy prompts, one
                  per worker.  Good when the search space is large and
                  distinct strategies are likely to find different optima
                  (e.g. Triton kernels with many viable tiling choices).
  - ``mixed``   — Default.  Splits N workers 50/50: half get identical
                  fixed prompts (variance from LLM sampling), half get
                  LLM-planner-generated diverse strategies.  Combines
                  the reliability of best-of-N with the exploration
                  breadth of planned mode.

The legacy "homogeneous" / "heterogeneous" terminology is an artifact of
pre-refactor code that had separate agent CLASSES for each dispatch
style.  With the unified ``OptimizationAgent`` those names no longer
describe anything real — the worker class is the same; only the task
body differs.  All public APIs and logs now use ``fixed`` / ``planned``
/ ``mixed``.

NOTE on translation: source→target language translation is NOT a
``run_pipeline`` mode.  It is a **conditional preprocess phase**
(``preprocess/phases/translation.py``, not yet implemented) that runs
BEFORE the optimization loop when the user requests a target language
different from the source.  Translation owns its own ``TranslationAgent``
subagent (a standalone ``SubagentBase`` subclass with a verify-retry
loop against golden tensors) and does not reuse ``OptimizationAgent``.
After the phase completes, ``ctx.kernel_path`` and ``ctx.language`` are
swapped to the translated kernel and the normal fixed/planned/auto
pipeline continues.

``run_pipeline`` is responsible for resolving the tool set, composing the
task body (via ``run/compose.py``), and dispatching to the shared pool
runner.  ``_run_fixed`` drives the round loop for fixed mode directly,
while ``_run_planned`` delegates to ``run_orchestrator`` (whose
``run_planned_orchestrator`` internals own the round loop already).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from minisweagent.run.compose import ComposeInputs, Mode, compose_task_body

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Context passed through ``run_pipeline``.

    This is a light wrapper over the existing dict-shaped ``preprocess_ctx``
    to keep the migration tax low.  Fields added here are the *new* pieces
    the unified path needs that were previously scattered across call sites
    (tool profile, RAG toggle, GPU ids, model factory).
    """

    preprocess_ctx: dict[str, Any]
    user_prompt: str
    kernel_language: str | None = None
    output_dir: Path | None = None
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    model: Any = None
    model_factory: Callable[[], Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    max_rounds: int | None = None
    env: Any = None
    env_class: Any = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    repo: Path | None = None
    test_command: str | None = None
    metric: str | None = None
    rag_enabled: bool = False
    extra_addenda: list[str] = field(default_factory=list)
    num_parallel: int | None = None
    model_name: str | None = None
    console: Any = None
    deadline: Any = None
    soft_stop: Any = None
    registry: Any = None


# ── Tool resolution ───────────────────────────────────────────────────
#
# Single site for deciding which tools each mode exposes to the agent.
# This replaces the scattered per-call-site ``ToolRuntime(tool_profile=...,
# use_strategy_manager=...)`` constructions.


def _resolve_tools(ctx: PipelineContext, mode: Mode):
    """Return a ``ToolRuntime`` instance configured for ``mode``.

    Kept as a free function so callers can inspect the resolved tool set
    (for tests, debugging, the `one resolution site' CI gate).
    """
    from minisweagent.tools.tools_runtime import ToolRuntime

    runtime = ToolRuntime(
        tool_profile="full",
        use_strategy_manager=True,
    )

    if not ctx.rag_enabled:
        runtime.disable_tools(["query", "optimize"])
    else:
        try:
            runtime.wrap_rag_tools_with_postprocessor()
        except Exception as exc:
            logger.warning("Failed to wrap RAG tools with postprocessor: %s", exc)

    logger.debug("_resolve_tools: mode=%s rag=%s", mode, ctx.rag_enabled)
    return runtime


# ── Pipeline dispatch ─────────────────────────────────────────────────


def run_pipeline(ctx: PipelineContext, mode: Mode):
    """Drive one full optimization pipeline and return the final report.

    This is the canonical entry point.  It resolves tools once, composes
    the task body once (for ``fixed``), and dispatches.  Fixed mode is
    driven by ``_run_fixed`` (round loop inline here); planned mode
    delegates to ``run_orchestrator``; mixed mode splits workers 50/50
    between fixed and planned strategies.  All paths ultimately
    instantiate the same ``OptimizationAgent`` on each worker.
    """
    logger.info(
        "run_pipeline: mode=%s kernel_language=%s output_dir=%s max_rounds=%s rag_enabled=%s",
        mode,
        ctx.kernel_language,
        ctx.output_dir,
        ctx.max_rounds,
        ctx.rag_enabled,
    )

    _ = _resolve_tools(ctx, mode)

    if mode == "planned":
        return _run_planned(ctx)
    if mode == "mixed":
        return _run_planned(ctx, task_generation="mixed")
    if mode == "fixed":
        return _run_fixed(ctx)
    if mode == "translate":
        # Translation is a *preprocess phase*, not a run_pipeline mode.
        # Reject early with a pointer to the correct entry point so
        # callers migrate rather than silently fall back.
        raise ValueError(
            "mode='translate' is not a run_pipeline mode.  Translation runs as a "
            "conditional preprocess phase (see preprocess/phases/translation.py) "
            "before run_pipeline.  Use ``geak --target-language ...`` or "
            "``geak translate`` at the CLI, not run_pipeline(..., mode='translate')."
        )
    raise ValueError(f"Unknown pipeline mode: {mode!r}")


def _run_planned(ctx: PipelineContext, task_generation: str = "planned"):
    """Dispatch into the planner-driven parallel path.

    The planner (``task_generator``) composes its own per-task bodies, so
    here we only massage the top-level preprocess context (commandment
    presence check, constraint / directive addenda, rag flag) before
    delegating.

    ``mixed`` asks the orchestrator to split workers between fixed and
    planner-generated task bodies.
    """
    from minisweagent.run.orchestrator import run_orchestrator

    pctx = dict(ctx.preprocess_ctx)
    commandment = pctx.get("commandment")
    if not commandment:
        # Planned mode requires the commandment because the planner LLM
        # references it per sub-task.  Fixed mode skips this check.
        raise RuntimeError(
            "planned mode requires ``commandment`` in preprocess_ctx; "
            "check preprocessor logs for failures."
        )

    pctx.setdefault("user_instructions", ctx.user_prompt)
    pctx["rag_enabled"] = ctx.rag_enabled
    pctx["output_dir"] = str(ctx.output_dir) if ctx.output_dir else pctx.get("output_dir")
    if ctx.num_parallel is not None:
        pctx["parallel_worker_count"] = ctx.num_parallel
    if task_generation in {"fixed", "mixed"}:
        pctx["fixed_parallel_task_body"] = compose_task_body(
            ComposeInputs(
                user_prompt=ctx.user_prompt,
                mode="fixed",
                preprocess_ctx=ctx.preprocess_ctx,
                kernel_language=ctx.kernel_language,
                extra_addenda=list(ctx.extra_addenda),
            )
        )

    # Extra addenda (user-specified constraints / directives extracted by
    # the caller) get appended to the commandment so every sub-task sees
    # them.
    if ctx.extra_addenda:
        addendum = "\n\n".join(a.strip() for a in ctx.extra_addenda if a and a.strip())
        if addendum:
            pctx["commandment"] = (commandment + "\n\n" + addendum).strip()
            if ctx.output_dir is not None:
                try:
                    _cm_path = Path(ctx.output_dir) / "COMMANDMENT.md"
                    _cm_path.write_text(pctx["commandment"], encoding="utf-8")
                    logger.info("Enriched commandment written to %s", _cm_path)
                except Exception as exc:
                    logger.warning("Failed to persist enriched commandment: %s", exc)

    return run_orchestrator(
        preprocess_ctx=pctx,
        gpu_ids=ctx.gpu_ids,
        model=ctx.model,
        model_factory=ctx.model_factory,
        output_dir=ctx.output_dir,
        max_rounds=ctx.max_rounds,
        heterogeneous=True,
        task_generation=task_generation,
        deadline=ctx.deadline,
        soft_stop=ctx.soft_stop,
        registry=ctx.registry,
    )


def _run_fixed(ctx: PipelineContext):
    """Run fixed mode with an explicit round loop.

    The round loop is the single most important structural difference
    between the legacy fixed-mode path and the diagram's end state:

      - Legacy (pre-refactor): the fixed runner ran ONCE with
        ``num_parallel`` copies.  Fixed mode was effectively "1 round ×
        N parallel agents".
      - Diagram (this implementation): iterate ``max_rounds`` times;
        each round spawns ``num_parallel`` copies that re-attempt the
        optimization.  The best result across all rounds wins.

    Between rounds, the task body is enriched with a short summary of
    the best speedup found so far so subsequent rounds can build on
    (or diverge from) earlier wins.  Planned mode is unchanged — its
    planner already does explicit multi-round iteration.
    """
    from minisweagent.agents.homogeneous.homogeneous_agent import run_homogeneous_agent

    max_rounds = max(1, int(ctx.max_rounds or 1))
    best_result = None
    best_speedup: float | None = None
    last_round_result: Any = None

    for round_num in range(1, max_rounds + 1):
        logger.info(
            "[bold cyan]%s[/bold cyan]\n  [bold]Fixed-mode round %d/%d[/bold]\n[bold cyan]%s[/bold cyan]",
            "=" * 60,
            round_num,
            max_rounds,
            "=" * 60,
        )

        # Compose the task body for this round.  On rounds > 1 we
        # append the previous best so the LLM has a concrete target
        # to beat — same signal the planned-mode planner gets from
        # its ``round_N_evaluation.json`` input.
        round_addenda = list(ctx.extra_addenda)
        if round_num > 1 and best_speedup is not None:
            round_addenda.append(
                f"## Previous Rounds\n\n"
                f"The best candidate across rounds 1..{round_num - 1} "
                f"achieved a verified speedup of {best_speedup:.3f}x.  "
                f"Use this as a lower bound — the current round's goal is "
                f"to find an approach that beats it, OR to confirm the "
                f"strategy generalises across seeds.  Explore strategies "
                f"not already exhausted in earlier rounds."
            )

        body = compose_task_body(
            ComposeInputs(
                user_prompt=ctx.user_prompt,
                mode="fixed",
                preprocess_ctx=ctx.preprocess_ctx,
                kernel_language=ctx.kernel_language,
                extra_addenda=round_addenda,
            )
        )

        round_best = _invoke_fixed_runner(
            ctx=ctx,
            body=body,
            run_fixed_mode=run_homogeneous_agent,
            round_num=round_num,
        )
        last_round_result = round_best

        # Track the best result across all rounds.
        round_speedup = getattr(round_best, "best_speedup", None) if round_best else None
        if round_speedup is not None and (
            best_speedup is None or round_speedup > best_speedup
        ):
            best_speedup = round_speedup
            best_result = round_best

        logger.info(
            "Fixed-mode round %d complete (this round best: %s; overall best: %s)",
            round_num,
            f"{round_speedup:.3f}x" if round_speedup is not None else "—",
            f"{best_speedup:.3f}x" if best_speedup is not None else "—",
        )

    # Prefer the best-by-speedup result across rounds.  When no round
    # produced a measurable speedup, fall back to the last round's raw
    # result so callers see the same shape as legacy single-round
    # invocations (which always returned whatever ``run_fixed_mode``
    # gave them, with or without a ``best_speedup`` attribute).
    return best_result if best_result is not None else last_round_result


def _invoke_fixed_runner(
    *,
    ctx: PipelineContext,
    body: str,
    run_fixed_mode: Callable[..., Any],
    round_num: int,
) -> Any:
    """Call ``run_fixed_mode`` with kwargs built from ``ctx``.

    Isolated so the round loop body stays readable and tests can mock
    just the invocation without tangling with kwarg plumbing.  Takes
    ``round_num`` so the fixed-mode runner writes per-round artefacts
    into ``<output_dir>/round_N/`` subdirs (matching planned mode's
    convention).
    """
    agent_config = dict(ctx.config.get("agent", {}))
    agent_config["save_patch"] = True
    if ctx.test_command is not None:
        agent_config["test_command"] = ctx.test_command
    if ctx.metric is not None:
        agent_config["metric"] = ctx.metric

    round_output_dir: Path | None = None
    if ctx.output_dir is not None:
        # Only nest per-round when max_rounds > 1 so single-round
        # callers still see artefacts directly under output_dir
        # (legacy behaviour preserved).
        max_rounds = max(1, int(ctx.max_rounds or 1))
        round_output_dir = (
            ctx.output_dir / f"round_{round_num}" if max_rounds > 1 else ctx.output_dir
        )
        round_output_dir.mkdir(parents=True, exist_ok=True)
        agent_config["patch_output_dir"] = str(round_output_dir)

    kwargs: dict[str, Any] = dict(
        config=ctx.config,
        task_content=body,
        model=ctx.model,
        env=ctx.env,
        env_class=ctx.env_class,
        env_kwargs=ctx.env_kwargs,
        agent_config=agent_config,
        repo=ctx.repo,
    )
    if ctx.num_parallel is not None:
        kwargs["num_parallel"] = ctx.num_parallel
    if ctx.gpu_ids:
        # run_fixed_mode takes a string and re-parses internally;
        # re-serialize the canonical list[int] form.
        kwargs["gpu_ids"] = ",".join(str(g) for g in ctx.gpu_ids)
    if round_output_dir is not None:
        kwargs["output_dir"] = round_output_dir
    if ctx.model_name is not None:
        kwargs["model_name"] = ctx.model_name
    if ctx.console is not None:
        kwargs["console"] = ctx.console
    if ctx.deadline is not None:
        kwargs["deadline"] = ctx.deadline
    if ctx.soft_stop is not None:
        kwargs["soft_stop"] = ctx.soft_stop
    if ctx.registry is not None:
        kwargs["registry"] = ctx.registry

    return run_fixed_mode(**kwargs)


__all__ = ["PipelineContext", "run_pipeline"]
