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
import time
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

        _t0 = time.monotonic()
        response = model.query(messages)
        _elapsed = time.monotonic() - _t0
        logger.debug("%s step %d: model.query returned in %.1fs", phase, step, _elapsed)

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
                _first_line = content_text.strip().split("\n", 1)[0][:200]
                _suffix = "..." if len(content_text.strip()) > len(_first_line) else ""
                logger.info("  Orchestrator: %s%s", _first_line, _suffix)
            messages.append({"role": "assistant", "content": content_text})
            return None

        tool_name = tool_call.get("function", {}).get("name", "")
        tool_args = tool_call.get("function", {}).get("arguments", {})
        tool_id = tool_call.get("id", f"call_{phase}_{step}")

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                logger.warning("Bad JSON in tool_args for %s; resetting to empty dict.", tool_name)
                tool_args = {}

        logger.debug("  Tool: %s(%s)", tool_name, json.dumps(tool_args)[:200])

        messages.append(
            {
                "role": "assistant",
                "content": content_text,
                "tool_calls": tool_call,
            }
        )

        _t0 = time.monotonic()
        result_str = dispatch_tool_call(ctx, tool_name, tool_args, phase=phase)
        _elapsed = time.monotonic() - _t0
        if _elapsed > 5.0:
            logger.info("[dim]Tool %s completed in %.1fs[/dim]", tool_name, _elapsed)

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
            except Exception as exc:
                logger.debug("Working-memory insight extraction failed for %s: %s", tool_name, exc)

        if tool_name == "finalize":
            try:
                report = json.loads(result_str)
            except json.JSONDecodeError:
                logger.warning("Finalize payload is not valid JSON; wrapping as summary text.")
                report = {"summary": result_str}
            logger.info("[bold green]Orchestrator: Optimisation finalised.[/bold green]")
            return report

    logger.warning(
        "Orchestrator hit step limit (%d) for phase %s — proceeding to next phase",
        max_steps,
        phase,
    )
    return None


def _log_final_summary(report) -> None:
    """Log a human-readable conclusion at the end of a heterogeneous run."""
    if report is None:
        return
    best_speedup = getattr(report, "best_speedup", None) or 0
    best_patch = getattr(report, "best_patch", None) or "unknown"
    best_round = getattr(report, "best_round", None) or "unknown"
    summary = getattr(report, "summary", "") or ""
    logger.info(
        "Heterogeneous run completed. Best patch: %s (round %s, %.4fx speedup)",
        best_patch,
        best_round,
        best_speedup,
    )
    if summary:
        logger.info("Summary: %s", summary[:500])


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
    kernel_path = str(preprocess_ctx.get("kernel_path", ""))
    kernel_meta = _extract_kernel_meta(disc_dict, kernel_path)

    preprocess_dir = output_dir
    for candidate in ("resolved.json", "discovery.json", "profile.json"):
        if (output_dir / candidate).exists():
            preprocess_dir = output_dir
            logger.debug("preprocess_dir set to output_dir (found %s).", candidate)
            break

    toolruntime = ToolRuntime(tool_profile="full", use_strategy_manager=True)
    rag_enabled = preprocess_ctx.get("rag_enabled", False)
    if not rag_enabled:
        toolruntime.disable_tools(["query", "optimize"])
    else:
        try:
            toolruntime.wrap_rag_tools_with_postprocessor(api_key=model.config.api_key)
        except Exception as e:
            logger.warning("Failed to wrap RAG tools with RAG postprocessor: %s", e)

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
    if not bm:
        logger.warning("No baseline metrics found in preprocess_ctx; using empty dict.")
    bm_summary = json.dumps(bm, indent=2, default=str) if bm else "Not available"

    prof = preprocess_ctx.get("profiling") or {}
    if not prof:
        logger.warning("No profiling data found in preprocess_ctx; using empty dict.")
    prof_summary = json.dumps(prof, indent=2, default=str)[:2000] if prof else "Not available"

    cmd = preprocess_ctx.get("commandment") or ""
    if not cmd:
        logger.error("No commandment found in preprocess_ctx.")
        raise ValueError("No commandment found in preprocess_ctx.")
    cmd_excerpt = cmd[:4000] + ("..." if len(cmd) > 4000 else "") if cmd else "Not available"

    codebase_ctx = ""
    _codebase_ctx_path = preprocess_dir / "CODEBASE_CONTEXT.md"
    if _codebase_ctx_path.exists():
        codebase_ctx = _codebase_ctx_path.read_text().strip()
        logger.debug("Loaded CODEBASE_CONTEXT.md (%d bytes) from %s", len(codebase_ctx), preprocess_dir)
    else:
        logger.warning("CODEBASE_CONTEXT.md not found in preprocess_dir; using empty string.")

    _memory_context = ""
    try:
        from minisweagent.memory.integration import (  # pylint: disable=import-error,no-name-in-module
            assemble_memory_context,
        )

        _memory_context = assemble_memory_context(
            kernel_path=kernel_path,
            bottleneck_type=bm.get("bottleneck"),
            profiling_metrics=bm,
        )
        if _memory_context:
            _memory_context = "### Optimization Memory (from past runs)\n" + _memory_context
            logger.info("Cross-session memory context injected (%d chars)", len(_memory_context))
        else:
            logger.info("Cross-session memory: no relevant experiences found")
    except Exception as _mem_exc:
        logger.warning("Cross-session memory assembly failed: %s", _mem_exc)

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

            _wm_notebook_dir = str(output_dir / "_working_memory")
            _working_mem = WorkingMemory(
                kernel_category=classify_kernel_category(kernel_path) if kernel_path else "unknown",
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
    except Exception as exc:
        logger.debug("WorkingMemory init failed: %s", exc)

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
        "\n[bold cyan]%s[/bold cyan]\n  [bold]Heterogeneous Orchestrator[/bold] (%s, %d GPUs)\n[bold cyan]%s[/bold cyan]",
        "=" * 60,
        start_label,
        len(gpu_ids),
        "=" * 60,
    )

    if rag_enabled:
        rag_tools_desc = (
            "\n\n**Knowledge Base Lookup** (Recommended)\n"
            "- Use `query` tool to search for optimization techniques, "
            "hardware-specific tips, and code patterns relevant to this kernel\n"
            "- Use `optimize` tool to get targeted optimization suggestions "
            "based on your kernel type and bottleneck analysis\n"
            "- Integrate retrieved knowledge into your strategy planning\n"
        )
    else:
        rag_tools_desc = ""
    system_prompt = SYSTEM_PROMPT.format(rag_tools_description=rag_tools_desc)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_msg},
    ]

    if start_round > 1:
        logger.info("Resuming from round %d; loading prior evaluations 1..%d.", start_round, start_round - 1)
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
            logger.info(
                "\n[dim]%s[/dim]\n  [bold yellow]Exploration Phase[/bold yellow] (this may take a few minutes)\n[dim]%s[/dim]",
                "-" * 60,
                "-" * 60,
            )
            _explore_t0 = time.monotonic()
            finalize_result = run_llm_steps(
                model,
                messages,
                ctx,
                phase="explore",
            )
            _explore_elapsed = time.monotonic() - _explore_t0
            logger.info("[bold green]Exploration completed[/bold green] in %.0fs.", _explore_elapsed)
            if finalize_result is not None:
                return finalize_result

        for round_num in range(start_round, max_rounds + 1):
            is_last = round_num == max_rounds
            final_tag = " [bold red](FINAL)[/bold red]" if is_last else ""
            color = "bold green" if not is_last else "bold red"
            logger.info(
                "\n[%s]%s[/%s]\n  [bold]Round %d/%d[/bold]%s\n[%s]%s[/%s]",
                color,
                "=" * 60,
                color,
                round_num,
                max_rounds,
                final_tag,
                color,
                "=" * 60,
                color,
            )

            if ctx.get("starting_patch"):
                logger.info("Starting from best patch so far: %s", ctx["starting_patch"])

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

            _round_t0 = time.monotonic()
            finalize_result = run_llm_steps(
                model,
                messages,
                ctx,
                phase=f"round_{round_num}",
            )
            _round_elapsed = time.monotonic() - _round_t0
            logger.info("Round %d LLM loop completed in %.0fs.", round_num, _round_elapsed)

            round_eval = post_round_evaluate(ctx, round_num, output_dir)
            logger.info("Round %d post-evaluation complete.", round_num)

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
                logger.info("Finalizing run")
                report = finalize_run(
                    ctx,
                    output_dir,
                    finalize_result=finalize_result,
                    round_eval=round_eval,
                )
                _log_final_summary(report)
                return report
    finally:
        logger.debug("Restoring original model tools schema.")
        if original_tools is not None:
            model_impl.tools = original_tools
        elif hasattr(model_impl, "tools"):
            model_impl.tools = []

    logger.info("Orchestrator completed all rounds without calling finalize - auto-selecting best result...")

    report = finalize_run(ctx, output_dir)
    _log_final_summary(report)
    return report
