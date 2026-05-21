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

``run_pipeline`` is responsible for resolving the tool set, initializing
the planner and dispatcher, and driving the unified round loop via
``_run_unified_loop``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    # When True, ``_run_unified_loop`` writes a stub ``final_report.json``
    # right before the round loop and returns. Used by the v3 preprocess
    # test sweep to validate preprocessing without paying the 30-90 minute
    # optimization round loop cost per kernel-scenario.
    preprocess_only: bool = False


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
    """Drive one full optimization pipeline and return a FinalReport.

    Single entry point for all modes.  Resolves tools once, then
    delegates to ``_run_unified_loop`` which runs the same deterministic
    round loop regardless of mode.
    """
    logger.info(
        "run_pipeline: mode=%s kernel_language=%s output_dir=%s max_rounds=%s",
        mode,
        ctx.kernel_language,
        ctx.output_dir,
        ctx.max_rounds,
    )
    _ = _resolve_tools(ctx, mode)

    if mode not in ("fixed", "planned", "mixed"):
        raise ValueError(f"Unknown pipeline mode: {mode!r}")

    return _run_unified_loop(ctx, mode)


# ── Postprocess context builder ──────────────────────────────────────


def _build_postprocess_ctx(pipeline_ctx: PipelineContext) -> dict[str, Any]:
    """Build the ctx dict that post_round_evaluate and finalize_run expect.

    Maps PipelineContext fields to the dict keys consumed by:
    - evaluate_round_best: output_dir, preprocess_dir, repo_root,
      harness_path, gpu_ids
    - post_round_evaluate: starting_patch, _best_global_speedup
      (mutated each round)
    - finalize_run / auto_finalize: output_dir, kernel_path,
      baseline_metrics
    """
    return {
        **pipeline_ctx.preprocess_ctx,
        "output_dir": str(pipeline_ctx.output_dir),
        "preprocess_dir": str(pipeline_ctx.output_dir),
        "repo_root": str(pipeline_ctx.repo or ""),
        "harness_path": str(pipeline_ctx.preprocess_ctx.get("harness_path", "")),
        "gpu_ids": list(pipeline_ctx.gpu_ids),
        "kernel_path": str(pipeline_ctx.preprocess_ctx.get("kernel_path", "")),
        "baseline_metrics": pipeline_ctx.preprocess_ctx.get("baseline_metrics", {}),
        "model": pipeline_ctx.model,
        "model_factory": pipeline_ctx.model_factory,
        "starting_patch": "",
        "_best_global_speedup": 0,
        "deadline": pipeline_ctx.deadline,
        "soft_stop": pipeline_ctx.soft_stop,
        "registry": pipeline_ctx.registry,
        "user_instructions": pipeline_ctx.user_prompt,
        "rag_enabled": pipeline_ctx.rag_enabled,
    }


# ── Round-loop helpers ───────────────────────────────────────────────


def _should_stop_before_round(ctx: PipelineContext) -> bool:
    """Check if soft_stop or deadline have fired."""
    if ctx.soft_stop is not None and ctx.soft_stop.is_set():
        return True
    if ctx.deadline is not None and ctx.deadline.expired():
        return True
    return False


def _resolve_task_file_meta(
    pp_dir: Path,
    kernel_path: str,
    repo_root: str,
    harness_path: str,
    test_command: str | None,
) -> dict[str, str | None]:
    """Resolve paths for ``write_dispatch_plan_as_task_files`` kwargs."""

    def _if_exists(p: Path) -> str | None:
        return str(p) if p.exists() else None

    return {
        "commandment": _if_exists(pp_dir / "COMMANDMENT.md"),
        "baseline_metrics": _if_exists(pp_dir / "baseline_metrics.json"),
        "profiling": _if_exists(pp_dir / "profile.json"),
        "codebase_context": _if_exists(pp_dir / "CODEBASE_CONTEXT.md"),
        "benchmark_baseline": _if_exists(pp_dir / "benchmark_baseline.txt"),
        "harness_path": harness_path or None,
        "kernel_path": kernel_path or None,
        "repo_root": repo_root or None,
        "test_command": str(test_command) if test_command else None,
    }


def _enrich_prompt_for_round(
    base_prompt: str,
    mode: Mode,
    round_num: int,
    round_evals: list[dict[str, Any]],
) -> str:
    """Enrich user prompt with prior round data for the planner.

    Fixed mode: append best speedup from prior rounds so the LLM has a
    concrete target.  Planned/mixed: base prompt unchanged (the planner
    receives ``round_evals`` separately via ``build_pool``).
    """
    if mode == "fixed" and round_num > 1 and round_evals:
        best_so_far = max(
            (e.get("benchmark_speedup", 1.0) for e in round_evals),
            default=1.0,
        )
        if best_so_far > 1.0:
            return base_prompt + (
                f"\n\n## Previous Rounds\n\n"
                f"The best candidate across rounds 1..{round_num - 1} "
                f"achieved {best_so_far:.3f}x speedup.  Beat it or explore "
                f"strategies not yet tried."
            )
    return base_prompt


# ── Incremental report persistence ──────────────────────────────────


def _save_incremental_report(
    ctx: dict[str, Any],
    output_dir: Path,
    completed_round: int,
) -> None:
    """Best-effort incremental final_report.json after each round.

    If the process is killed by a GPU fault before finalize_run(),
    the last incremental report preserves the best verified result.
    Overwritten by finalize_run() at natural completion.
    """
    try:
        from minisweagent.run.postprocess.results import auto_finalize

        report = auto_finalize(ctx)
        report["status"] = f"incremental_after_round_{completed_round}"
        report_path = output_dir / "final_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
    except Exception:
        logger.debug("incremental report save failed (non-fatal)", exc_info=True)


# ── Budget-timeout select_patch ───────────────────────────────────────


def _run_timeout_select_patch(
    ctx: "PipelineContext",
    output_dir: Path,
    postprocess_ctx: dict[str, Any],
) -> None:
    """Best-effort select_patch + auto_finalize on budget timeout.

    Two phases (mirroring the hard-kill path):

    1. Run the LLM ``SelectPatchAgent`` to fill in any missing per-task
       ``best_results.json`` files (non-fatal if it fails).
    2. Call ``auto_finalize`` — the same canonical path used by normal
       completion — to rewrite ``final_report.json`` with the complete
       report format.

    Fully wrapped in try/except — failure is non-fatal because
    ``finalize_run`` has already written a valid ``final_report.json``.
    """
    # Phase 1: best-effort LLM select_patch (non-fatal)
    try:
        from minisweagent.agents.select_patch_agent import SelectPatchAgent
        from minisweagent.config import load_agent_config
        from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig

        results_dir = output_dir / "results"
        if not results_dir.is_dir():
            logger.warning("timeout select_patch: no results/ dir; skipping agent")
        else:
            task_dirs = sorted({
                p.parent for p in results_dir.glob("round_*/*/best_results.json")
            })
            if not task_dirs:
                task_dirs = sorted({
                    d for d in results_dir.glob("round_*/*")
                    if d.is_dir() and d.name != "worktrees"
                })
            if not task_dirs:
                logger.warning("timeout select_patch: no task dirs found; skipping agent")
            else:
                metric = postprocess_ctx.get("metric") or (
                    ctx.config.get("patch", {}).get("metric")
                    if isinstance(ctx.config, dict)
                    else None
                )
                model = ctx.model_factory()
                agent_config, _ = load_agent_config("mini_select_patch")

                env_config = LocalEnvironmentConfig(cwd=str(results_dir))
                env = LocalEnvironment(**env_config.__dict__)

                agent = SelectPatchAgent(model, env, **agent_config)
                agent.log_file = results_dir / "timeout_select_agent.log"
                agent.patch_dir = results_dir

                metric_section = metric if metric else "None"
                dir_listing = "\n".join(f"  - {d}" for d in task_dirs)
                task = (
                    f"\n## User-provided metric\n{metric_section}\n\n"
                    f"## Inputs\n"
                    f"- Work directory (absolute): {results_dir}\n"
                    f"- This is a TIMEOUT selection: the run was interrupted by budget.\n"
                    f"- Results are organized under round_*/ subdirectories.\n"
                    f"  Each subdirectory may contain:\n"
                    f"  - patch_*.patch files\n"
                    f"  - patch_*_test.txt test output logs\n"
                    f"  - best_results.json (per-task selection by previous agents)\n"
                    f"- You should scan ALL directories below to find the "
                    f"best patch across all rounds.\n"
                    f"- Use patch_0_test.txt from any directory as baseline "
                    f"(patch_0 = original unmodified kernel).\n"
                    f"- Found {len(task_dirs)} task directories:\n{dir_listing}\n"
                )

                logger.info(
                    "Budget timeout: running select_patch on %d task dirs in %s",
                    len(task_dirs),
                    results_dir,
                )
                agent.run(task)
                best_patch_id = agent.extract_final_result()
                if best_patch_id:
                    logger.info("Budget timeout: select_patch chose %s", best_patch_id)
                else:
                    logger.warning("Budget timeout: select_patch did not produce a result")
    except Exception:
        logger.exception("Budget timeout: select_patch agent failed (non-fatal)")

    # Phase 2: auto_finalize — same path as normal completion
    try:
        from minisweagent.run.postprocess.results import auto_finalize

        report = auto_finalize(postprocess_ctx)

        # Stamp timeout metadata onto the report written by auto_finalize
        report_path = output_dir / "final_report.json"
        if report_path.is_file():
            final = json.loads(report_path.read_text())
            final["status"] = "timeout_auto_finalized"
            report_path.write_text(json.dumps(final, indent=2, default=str))
            logger.info(
                "Budget timeout: wrote final_report.json via auto_finalize "
                "(best_speedup=%s)",
                final.get("best_speedup"),
            )
    except Exception:
        logger.exception(
            "Budget timeout: auto_finalize failed (non-fatal; finalize_run report preserved)"
        )


# ── --preprocess-only stub report ────────────────────────────────────


# Canonical preprocess artifact filenames the v3 pipeline writes into
# the run output dir. Listed here so the stub report enumerates every
# file the round loop would have consumed — making the smoke test
# assertion ``len(artifacts) > 0`` meaningful.
_PREPROCESS_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "COMMANDMENT.md",
    "CODEBASE_CONTEXT.md",
    "baseline_metrics.json",
    "profile.json",
    "benchmark_baseline.txt",
    "preprocess_context.json",
    "resolved.json",
)


def _build_preprocess_only_report(
    ctx: PipelineContext,
    pp_dir: Path,
    loop_start_t: float,
) -> dict[str, Any]:
    """Build the stub final_report.json payload for ``--preprocess-only``.

    Mirrors the canonical ``final_report.json`` shape (status, summary,
    best_speedup, best_patch) so existing parsers do not break, plus
    three preprocess-specific fields:

    * ``preprocess_artifacts`` — list of absolute paths to every known
      preprocess artifact present on disk (skips missing ones rather
      than failing).
    * ``path_taken`` — ``"A"`` / ``"B"`` / ``None`` from
      ``ctx.preprocess_ctx``. The v3 orchestrator records this; the
      legacy preprocessor does not, so ``None`` is expected for legacy
      runs.
    * ``round_results`` — always empty for preprocess-only.
    * ``elapsed_s`` — time spent in ``_run_unified_loop`` so far.
    """
    artifacts = [
        str((pp_dir / name).resolve())
        for name in _PREPROCESS_ARTIFACT_FILENAMES
        if (pp_dir / name).exists()
    ]
    return {
        "status": "preprocess_only",
        "summary": (
            "Preprocessing artifacts written; round loop skipped "
            "(--preprocess-only)."
        ),
        "preprocess_artifacts": artifacts,
        "path_taken": ctx.preprocess_ctx.get("path_taken"),
        "round_results": [],
        "elapsed_s": round(time.monotonic() - loop_start_t, 3),
        "best_speedup": None,
        "best_patch": None,
        "best_round": None,
        "best_task": None,
    }


# ── Unified round loop ───────────────────────────────────────────────


def _run_unified_loop(ctx: PipelineContext, mode: Mode) -> Any:
    """Single mode-blind round loop for all pipeline modes.

    Mode differences collapse to one integer K inside
    ``Dispatcher._k_for_mode``.  The loop itself is identical for
    fixed, planned, and mixed.
    """
    from minisweagent.agents.heterogeneous.task_generator import _extract_kernel_meta
    from minisweagent.agents.optimization_agent import OptimizationAgent
    from minisweagent.run.dispatch import run_staged_task_batch
    from minisweagent.run.dispatcher.selector import Dispatcher
    from minisweagent.run.dispatcher.writer import write_dispatch_plan_as_task_files
    from minisweagent.run.planner.task_planner import TaskPlanner
    from minisweagent.run.postprocess.evaluation import (
        preflight_commandment_contract,
    )
    from minisweagent.run.postprocess.results import finalize_run, post_round_evaluate
    from minisweagent.subagents import SubAgentRegistry

    _loop_start_t = time.monotonic()

    output_dir = Path(ctx.output_dir)
    pp_dir = output_dir
    postprocess_ctx = _build_postprocess_ctx(ctx)
    max_rounds = max(1, int(ctx.max_rounds or 5))
    n_workers = ctx.num_parallel or len(ctx.gpu_ids) or 1

    # ── Extract kernel metadata ──────────────────────────────────
    disc_dict = ctx.preprocess_ctx.get("discovery") or {}
    kernel_path = str(ctx.preprocess_ctx.get("kernel_path", ""))
    kernel_meta = _extract_kernel_meta(disc_dict, kernel_path)

    # ── Preflight COMMANDMENT contract ───────────────────────────
    commandment_path = pp_dir / "COMMANDMENT.md"
    repo_root = str(ctx.repo or ctx.preprocess_ctx.get("repo_root", ""))
    harness_path = str(ctx.preprocess_ctx.get("harness_path", ""))
    gpu_id = ctx.gpu_ids[0] if ctx.gpu_ids else 0

    if repo_root and commandment_path.exists():
        try:
            preflight_commandment_contract(
                commandment_path,
                repo_root,
                harness_path,
                gpu_id,
                output_dir=output_dir,
            )
        except Exception as exc:
            logger.error("Preflight contract failed: %s", exc)
            raise

    # ── Initialize planner + dispatcher ──────────────────────────
    planner = TaskPlanner(
        model=ctx.model,
        subagent_registry=SubAgentRegistry(),
        preprocess_ctx=ctx.preprocess_ctx,
        kernel_meta=kernel_meta,
    )
    dispatcher = Dispatcher()

    # ── Resolve metadata paths for task file writing ─────────────
    task_file_kwargs = _resolve_task_file_meta(
        pp_dir, kernel_path, repo_root, harness_path, ctx.test_command,
    )

    round_evals: list[dict[str, Any]] = []

    # ── --preprocess-only short-circuit ──────────────────────────
    # When the CLI requested preprocess-only validation, write a stub
    # final_report.json that mirrors the canonical shape (so downstream
    # tooling can still parse it) but skips the round loop entirely.
    # The test sweep relies on this to avoid paying the 30-90 minute
    # optimization cost per kernel-scenario.
    if ctx.preprocess_only:
        report = _build_preprocess_only_report(ctx, pp_dir, _loop_start_t)
        report_path = output_dir / "final_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info(
            "--preprocess-only set: skipping round loop; wrote stub final_report.json "
            "(%d artifacts, path_taken=%s, elapsed=%.1fs)",
            len(report["preprocess_artifacts"]),
            report["path_taken"],
            report["elapsed_s"],
        )
        return report

    # ── Round loop ───────────────────────────────────────────────
    _budget_stopped = False
    for round_num in range(1, max_rounds + 1):
        if _should_stop_before_round(ctx):
            logger.warning(
                "Budget reached before round %d; finalizing.", round_num,
            )
            _budget_stopped = True
            break

        is_last = round_num == max_rounds
        tag = " (FINAL)" if is_last else ""
        logger.info(
            "\n════════════════════════════════════════════════════════════\n"
            "  Round %d/%d%s  (mode=%s, workers=%d)\n"
            "════════════════════════════════════════════════════════════",
            round_num,
            max_rounds,
            tag,
            mode,
            n_workers,
        )

        if postprocess_ctx.get("starting_patch"):
            logger.info(
                "Starting from best patch so far: %s",
                postprocess_ctx["starting_patch"],
            )

        # 1. PLAN — generate M candidate tasks
        user_prompt_for_round = _enrich_prompt_for_round(
            ctx.user_prompt, mode, round_num, round_evals,
        )
        pool = planner.build_pool(
            round_num=round_num,
            user_prompt=user_prompt_for_round,
            round_evals=round_evals,
            mode=mode,
            agent_class=OptimizationAgent,
            output_dir=output_dir,
            num_gpus=len(ctx.gpu_ids),
            rag_enabled=ctx.rag_enabled,
        )

        # 2. SELECT — pick N tasks from pool
        plan = dispatcher.select(pool, mode, n_workers)

        # 3. WRITE — .md task files for traceability
        task_files = write_dispatch_plan_as_task_files(
            plan,
            output_dir,
            round_num=round_num,
            starting_patch=postprocess_ctx.get("starting_patch") or None,
            **task_file_kwargs,
        )

        # 4. EXECUTE — staged dispatch with early exit on improvement
        results_dir = output_dir / "results" / f"round_{round_num}"
        results_dir.mkdir(parents=True, exist_ok=True)
        run_staged_task_batch(
            task_files=task_files,
            gpu_ids=ctx.gpu_ids,
            output_dir=results_dir,
            model_factory=ctx.model_factory,
            console=ctx.console,
            deadline=ctx.deadline,
            soft_stop=ctx.soft_stop,
            registry=ctx.registry,
        )

        # 5. EVALUATE — FULL_BENCHMARK verification (all modes)
        round_eval = post_round_evaluate(
            postprocess_ctx, round_num, output_dir,
        )
        if round_eval is not None:
            round_eval_dict = (
                round_eval.to_dict()
                if hasattr(round_eval, "to_dict")
                else round_eval
            )
            round_evals.append(round_eval_dict)

        logger.info("Round %d complete.", round_num)
        _save_incremental_report(postprocess_ctx, output_dir, round_num)

    # ── Finalize ─────────────────────────────────────────────────
    report = finalize_run(postprocess_ctx, output_dir)

    # ── Budget-triggered select_patch (runs AFTER finalize so it
    #    can overwrite final_report.json with the LLM result) ────
    if _budget_stopped:
        _run_timeout_select_patch(ctx, output_dir, postprocess_ctx)

    return report


__all__ = ["PipelineContext", "_build_preprocess_only_report", "run_pipeline"]
