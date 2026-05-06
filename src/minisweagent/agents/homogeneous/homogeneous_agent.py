#!/usr/bin/env python3
"""Direct ParallelAgent runner for N identical task bodies (tests / rare entry).

Production fixed mode uses ``run_pipeline`` → ``run_orchestrator``; see
``run/unified.py``.  Shared GPU parsing: ``run/utils/gpu_ids.py``.
"""

import copy
import json
import logging
import time
from pathlib import Path

from rich.console import Console

from minisweagent.agents.parallel_agent import BestPatchResult, ParallelAgent
from minisweagent.agents.optimization_agent import OptimizationAgent
from minisweagent.models import get_model
from minisweagent.run.pool_runner import build_fixed_tasks
from minisweagent.run.utils.gpu_ids import parse_gpu_ids

logger = logging.getLogger(__name__)


def run_fixed_mode(
    config: dict,
    task_content: str,
    model,
    env,
    env_class,
    env_kwargs: dict,
    agent_config: dict,
    repo: Path | None = None,
    gpu_ids: str | None = None,
    output_dir: Path | None = None,
    model_name: str | None = None,
    console: Console | None = None,
) -> BestPatchResult | None:
    """Run ``fixed`` mode: one identical task body per GPU (``len(gpu_ids)`` workers).

    Dispatch width always matches the GPU list.  Variance across workers
    comes from LLM sampling alone (temperature > 0 or trajectory seeds).

    Prefer ``run_pipeline(..., mode=\"fixed\")``; direct calls are rare.
    """
    if console is None:
        console = Console(highlight=False)

    parallel_config = config.get("parallel", {})

    final_gpu_ids = parse_gpu_ids(gpu_ids or parallel_config.get("gpu_ids") or config.get("agent", {}).get("gpu_ids"))
    parallel_workers = max(1, len(final_gpu_ids))
    logger.debug("gpu_ids=%s parallel_workers=%d", final_gpu_ids, parallel_workers)

    # Repository path
    final_repo = repo
    if not final_repo:
        final_repo = parallel_config.get("repo") or config.get("agent", {}).get("repo")

    final_repo = Path(final_repo).resolve()
    if not final_repo.exists():
        raise ValueError(f"Repository path does not exist: {final_repo}")

    # GEAK homogeneous flow always uses strategy interactive agent.
    base_agent_class = OptimizationAgent

    # Configure agent for homogeneous mode
    agent_config["mode"] = "yolo"
    agent_config["confirm_exit"] = False
    agent_config.setdefault("use_strategy_manager", True)
    agent_config["gpu_ids"] = final_gpu_ids
    agent_config["repo"] = str(final_repo)
    agent_config["agent_class"] = base_agent_class

    # Create output directory (pop from agent_config as ParallelAgentConfig doesn't accept it)
    final_output_dir = Path(agent_config.pop("output_dir", None) or output_dir or "optimization_logs")
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Set patch_output_dir to output_dir so patches are saved alongside logs
    agent_config["patch_output_dir"] = str(final_output_dir)

    # Get model config for factory
    model_config = config.get("model", {})

    logger.info(
        "\n[bold cyan]%s[/bold cyan]\n  [bold]Fixed-mode parallel run[/bold] (%d workers, GPUs %s)\n[bold cyan]%s[/bold cyan]",
        "=" * 60,
        parallel_workers,
        final_gpu_ids,
        "=" * 60,
    )
    logger.info("  repo=%s, output_dir=%s", final_repo, final_output_dir)
    logger.info("[dim]Sub-agents are working — expect no output for several minutes.[/dim]")

    # Build an identical-copies AgentTask list so fixed mode flows through
    # the shared ``run_pool`` scheduler.  Same pool, same worktrees, same
    # logs as the planned-mode path — only the task body differs.
    task_body_with_wt = task_content + "\n\n" + "The current worktree is: " + str(final_repo)
    fixed_tasks = build_fixed_tasks(
        parallel_workers,
        base_agent_class,
        task_body_with_wt,
        base_label="parallel",
    )
    # ParallelAgentConfig carries ``tasks`` alongside ``agent_class`` — when
    # ``tasks`` is set, ParallelAgent.run_parallel skips its inline fixed
    # branch and calls run_pool directly.
    agent_config["tasks"] = fixed_tasks

    agent = ParallelAgent(model, env, **agent_config)

    try:
        _t0 = time.monotonic()
        best_result = agent.run(
            task_body_with_wt,
            console=console,
            model_factory=lambda: get_model(model_name, model_config.copy()),
            env_factory=lambda: env_class(**copy.deepcopy(env_kwargs)),
        )
        _elapsed = time.monotonic() - _t0

        if best_result:
            logger.info(
                "Homogeneous run completed in %.0fs. Best patch: %s (agent %d)",
                _elapsed,
                best_result.patch_id,
                best_result.agent_id,
            )
            console.print(
                f"\n[bold green]Best patch:[/bold green] {best_result.patch_id} (agent {best_result.agent_id})"
            )
        else:
            logger.info("Homogeneous run completed in %.0fs. No best patch selected.", _elapsed)
            console.print("\n[bold yellow]No best patch selected[/bold yellow]")

        # Write final_report.json (aligned with heterogeneous output structure)
        speedup = best_result.best_speedup if best_result else None
        best_patch_path = (
            str(best_result.patch_dir / best_result.patch_id) if best_result and best_result.patch_dir else None
        )
        report = {
            "status": "complete" if best_result else "complete_no_patch",
            "best_patch": (best_result.best_patch_file or best_patch_path) if best_result else None,
            "best_speedup": speedup,
            "summary": best_result.llm_conclusion if best_result else "No best patch selected",
        }
        report_path = final_output_dir / "final_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Wrote final_report.json to %s", report_path)

    except Exception as e:
        logger.error("Homogeneous agent failed: %s", e, exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {e}")
        error_report = {
            "status": "error",
            "best_patch": None,
            "best_speedup": None,
            "summary": str(e),
        }
        error_report_path = final_output_dir / "final_report.json"
        try:
            error_report_path.write_text(json.dumps(error_report, indent=2, default=str))
        except Exception:
            logger.debug("Failed to write error report to %s", error_report_path)
        raise

    return best_result


run_homogeneous_agent = run_fixed_mode
