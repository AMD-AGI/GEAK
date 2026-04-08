"""Heterogeneous orchestrator: LLM-driven multi-round optimization.

In heterogeneous mode, an LLM agent drives the optimization loop by
calling tools in sequence each round:

1. ``generate_tasks`` -- create diverse optimization task files
2. ``dispatch_tasks`` -- run them in parallel across GPUs
3. ``collect_results`` -- review what each task achieved
4. ``finalize`` -- signal completion (final round only)

The LLM decides what strategies to try each round based on profiling
data, prior results, and COMMANDMENT constraints.  The round loop is
implicit -- the LLM's behavior drives iteration, not a Python for-loop.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from minisweagent.agents.heterogeneous.prompts import INSTANCE_TEMPLATE, SYSTEM_PROMPT
from minisweagent.agents.heterogeneous.schemas import build_tools_schema
from minisweagent.agents.heterogeneous.tools import dispatch_tool_call
from minisweagent.debug_runtime import emit_debug_log, model_tools_snapshot

logger = logging.getLogger(__name__)


# ── LLM step loop ────────────────────────────────────────────────────


def run_llm_steps(
    model,
    messages: list[dict],
    ctx: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any] | None:
    """Run LLM tool-call steps until the LLM responds with text or calls ``finalize``.

    Returns a finalize report dict if the LLM called ``finalize``,
    otherwise ``None`` (the LLM responded with text, signalling it is
    ready for the next phase).
    """
    max_steps = int(os.getenv("GEAK_ORCHESTRATOR_STEP_LIMIT", "200"))
    step = 0
    _wm = ctx.get("working_memory")

    while step < max_steps:
        step += 1
        logger.debug("[dim]%s step %d[/dim]", phase, step)

        if _wm and phase != "explore":
            _wm.update_step(step, 0.0)
            _wm_text = _wm.format_for_injection()
            if _wm_text and not any("[Working Memory" in m.get("content", "") for m in messages[-3:]):
                messages.append({"role": "user", "content": f"[Working Memory Update]\n{_wm_text}"})

        response = model.query(messages)

        content_text = response.get("content", "") if isinstance(response, dict) else ""
        tool_call = response.get("tools") if isinstance(response, dict) else None

        if not tool_call:
            if phase.startswith("round_") and any(
                name in content_text for name in ("dispatch_tasks", "collect_results", "finalize")
            ):
                emit_debug_log(
                    "heterogeneous_orchestrator:run_llm_steps:no_tool_call",
                    "Orchestrator produced text mentioning missing orchestration tools",
                    {
                        "phase": phase,
                        "step": step,
                        "content_preview": content_text[:300],
                        "model_tools": model_tools_snapshot(model),
                    },
                    hypothesis_id="H3",
                )
            if content_text:
                logger.info("  Orchestrator: %s", content_text[:300])
            messages.append({"role": "assistant", "content": content_text})
            return None

        tool_name = tool_call.get("function", {}).get("name", "")
        tool_args = tool_call.get("function", {}).get("arguments", {})
        tool_id = tool_call.get("id", f"call_{phase}_{step}")

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {}

        logger.debug("  Tool: %s(%s)", tool_name, json.dumps(tool_args)[:200])

        messages.append(
            {
                "role": "assistant",
                "content": content_text,
                "tool_calls": tool_call,
            }
        )

        result_str = dispatch_tool_call(ctx, tool_name, tool_args, phase=phase)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result_str,
            }
        )

        logger.debug("  Result: %s", result_str[:300])

        if _wm:
            try:
                from minisweagent.memory.working_memory import (  # pylint: disable=import-error,no-name-in-module
                    extract_insight_from_tool_result,
                )

                insight = extract_insight_from_tool_result(tool_name, result_str, 0)
                if insight:
                    _wm.ingest_insight(insight)
            except Exception:
                pass

        if tool_name == "finalize":
            try:
                report = json.loads(result_str)
            except json.JSONDecodeError:
                report = {"summary": result_str}
            logger.info("[bold green]Orchestrator: Optimisation finalised.[/bold green]")
            return report

    logger.warning(
        "Orchestrator hit step limit (%d) for phase %s — proceeding to next phase",
        max_steps,
        phase,
    )
    return None


# ── Main entry point ─────────────────────────────────────────────────


def run_heterogeneous_orchestrator(
    preprocess_ctx: dict[str, Any],
    gpu_ids: list[int],
    model,
    model_factory,
    output_dir: Path,
    max_rounds: int,
    start_round: int,
) -> dict[str, Any]:
    """Run the heterogeneous orchestrator with LLM-driven tool calling.

    This is the main heterogeneous entry point, called by
    ``run/orchestrator.py:run_orchestrator`` when ``heterogeneous=True``.
    """
    from minisweagent.agents.heterogeneous.task_generator import _extract_kernel_meta
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent
    from minisweagent.run.postprocess.results import (
        finalize_run,
        post_round_evaluate,
    )
    from minisweagent.tools.tools_runtime import ToolRuntime

    disc_dict = preprocess_ctx.get("discovery") or {}
    kernel_path = preprocess_ctx.get("kernel_path", "")
    kernel_meta = _extract_kernel_meta(disc_dict, kernel_path)

    preprocess_dir = output_dir
    for candidate in ("resolved.json", "discovery.json", "profile.json"):
        if (output_dir / candidate).exists():
            preprocess_dir = output_dir
            break

    toolruntime = ToolRuntime(tool_profile="full", use_strategy_manager=True)

    ctx: dict[str, Any] = {
        **preprocess_ctx,
        "kernel_meta": kernel_meta,
        "output_dir": str(output_dir),
        "preprocess_dir": str(preprocess_dir),
        "gpu_ids": gpu_ids,
        "model": model,
        "model_factory": model_factory,
        "agent_class": StrategyInteractiveAgent,
        "toolruntime": toolruntime,
    }

    tools_schema = build_tools_schema(toolruntime)
    model_impl = getattr(model, "_impl", model)
    _orig = getattr(model_impl, "tools", None)
    original_tools = list(_orig) if isinstance(_orig, list) else _orig
    model_impl.tools = tools_schema

    bm = preprocess_ctx.get("baseline_metrics") or {}
    bm_summary = json.dumps(bm, indent=2, default=str) if bm else "Not available"

    prof = preprocess_ctx.get("profiling") or {}
    prof_summary = json.dumps(prof, indent=2, default=str)[:2000] if prof else "Not available"

    cmd = preprocess_ctx.get("commandment") or ""
    cmd_excerpt = cmd[:1500] + ("..." if len(cmd) > 1500 else "") if cmd else "Not available"

    codebase_ctx = ""
    _codebase_ctx_path = preprocess_dir / "CODEBASE_CONTEXT.md"
    if _codebase_ctx_path.exists():
        codebase_ctx = _codebase_ctx_path.read_text().strip()

    _memory_context = ""
    try:
        from minisweagent.memory.integration import (  # pylint: disable=import-error,no-name-in-module
            assemble_memory_context,
        )

        _bm = preprocess_ctx.get("baseline_metrics") or {}
        _memory_context = assemble_memory_context(
            kernel_path=str(preprocess_ctx.get("kernel_path", "")),
            bottleneck_type=_bm.get("bottleneck"),
            profiling_metrics=_bm,
        )
        if _memory_context:
            _memory_context = "### Optimization Memory (from past runs)\n" + _memory_context
    except Exception as _mem_exc:
        logger.debug("Memory context assembly failed: %s", _mem_exc)

    _working_mem = None
    try:
        from minisweagent.memory.integration import (  # pylint: disable=import-error,no-name-in-module
            is_working_memory_enabled,
        )

        if is_working_memory_enabled():
            from minisweagent.memory.cross_session_memory import (  # pylint: disable=import-error,no-name-in-module
                classify_kernel_category,
            )
            from minisweagent.memory.working_memory import (  # pylint: disable=import-error,no-name-in-module
                WorkingMemory,
            )

            _kpath = str(preprocess_ctx.get("kernel_path", ""))
            _wm_notebook_dir = str(output_dir / "_working_memory")
            _working_mem = WorkingMemory(
                kernel_category=classify_kernel_category(_kpath) if _kpath else "unknown",
                max_steps=int(os.getenv("GEAK_AGENT_STEP_LIMIT", "100")),
                notebook_dir=_wm_notebook_dir,
                notebook_writer_id="orchestrator",
            )
            _working_mem.load_baseline_from_artifacts(
                baseline_metrics_path=str(output_dir / "baseline_metrics.json"),
                benchmark_baseline_path=str(output_dir / "benchmark_baseline.txt"),
            )
            _working_mem.sync_notebook_baseline()
            ctx["working_memory"] = _working_mem
    except Exception as _wm_exc:
        logger.debug("WorkingMemory init failed: %s", _wm_exc)

    instance_msg = INSTANCE_TEMPLATE.format(
        kernel_path=str(preprocess_ctx.get("kernel_path", "N/A")),
        repo_root=str(preprocess_ctx.get("repo_root", "N/A")),
        test_command=str(preprocess_ctx.get("test_command", "N/A")),
        gpu_ids=str(gpu_ids),
        output_dir=str(output_dir),
        codebase_context=codebase_ctx or "Not available",
        baseline_metrics_summary=bm_summary,
        profiling_summary=prof_summary,
        commandment_excerpt=cmd_excerpt,
        memory_context=_memory_context,
    )

    start_label = f"rounds {start_round}-{max_rounds}" if start_round > 1 else f"{max_rounds} rounds"
    logger.info(
        "[bold cyan]--- Orchestrator starting (%s, %d GPUs) ---[/bold cyan]",
        f"{start_label}",
        len(gpu_ids),
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instance_msg},
    ]

    if start_round > 1:
        for prev_round in range(1, start_round):
            eval_path = output_dir / f"round_{prev_round}_evaluation.json"
            if eval_path.exists():
                try:
                    round_eval = json.loads(eval_path.read_text())
                    ctx[f"round_{prev_round}_eval"] = round_eval
                    eval_summary = json.dumps(round_eval, indent=2, default=str)[:2000]
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"## Round {prev_round} Evaluation (prior run)\n\n"
                                f"The best kernel from round {prev_round} was evaluated "
                                f"with FULL_BENCHMARK and PROFILE:\n```\n{eval_summary}\n```\n"
                                "Use this data to inform your strategy."
                            ),
                        }
                    )
                    logger.info("  Loaded prior evaluation: %s", eval_path.name)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("  Could not load %s: %s", eval_path.name, exc)

    try:
        if start_round <= 1:
            logger.info("[bold cyan]--- Exploration phase ---[/bold cyan]")
            finalize_result = run_llm_steps(
                model,
                messages,
                ctx,
                phase="explore",
            )
            if finalize_result is not None:
                return finalize_result

        for round_num in range(start_round, max_rounds + 1):
            is_last = round_num == max_rounds
            round_header = f"--- Round {round_num}/{max_rounds}{' (final round)' if is_last else ''} ---"
            logger.info("[bold cyan]%s[/bold cyan]", round_header)

            if is_last:
                round_instruction = (
                    f"Begin round {round_num} (FINAL round). "
                    "Call generate_tasks, dispatch_tasks, collect_results, "
                    "then call **finalize** with a full summary of the best "
                    "results across all rounds."
                )
            else:
                round_instruction = (
                    f"Begin round {round_num}/{max_rounds}. "
                    "Call generate_tasks, dispatch_tasks, collect_results. "
                    "Then evaluate the results and respond with your analysis. "
                    "Focus on strategies not yet tried or that build on "
                    "previous successes. For later-round decisions, prefer the "
                    "system-provided FULL_BENCHMARK verified outcomes over raw "
                    "task-local speedup claims."
                )
            messages.append({"role": "user", "content": round_instruction})

            finalize_result = run_llm_steps(
                model,
                messages,
                ctx,
                phase=f"round_{round_num}",
            )

            round_eval = post_round_evaluate(ctx, round_num, output_dir)
            if round_eval:
                if _working_mem:
                    round_eval_dict = round_eval.to_dict() if hasattr(round_eval, "to_dict") else round_eval
                    _working_mem.record_round_evaluation(round_eval_dict)
                eval_summary = json.dumps(
                    round_eval.to_dict() if hasattr(round_eval, "to_dict") else round_eval,
                    indent=2,
                    default=str,
                )[:2000]
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"## Round {round_num} Evaluation\n\n"
                            f"The best kernel from round {round_num} was evaluated "
                            f"with FULL_BENCHMARK and PROFILE:\n```\n{eval_summary}\n```\n"
                            "Use this data to inform your next-round strategy. "
                            "Treat the FULL_BENCHMARK result as canonical and use "
                            "task-local speedups only as supporting evidence."
                        ),
                    }
                )

            if finalize_result is not None:
                return finalize_run(
                    ctx,
                    output_dir,
                    finalize_result=finalize_result,
                    round_eval=round_eval,
                )
    finally:
        if original_tools is not None:
            model_impl.tools = original_tools
        elif hasattr(model_impl, "tools"):
            model_impl.tools = []

    logger.info("Orchestrator completed all rounds without calling finalize - auto-selecting best result...")

    return finalize_run(ctx, output_dir)
