#!/usr/bin/env python3
"""
Homogeneous Agent Runner - Run multiple identical agents in parallel.

This module provides a simplified interface to run ParallelAgent with
homogeneous configuration (all agents run the same task with identical settings).
"""

import copy
from pathlib import Path

from rich.console import Console

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.agents.parallel_agent import BestPatchResult, ParallelAgent
from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj


def parse_gpu_ids(gpu_ids_str: str | None) -> list[int]:
    """Parse comma-separated GPU IDs string to list of integers."""
    if not gpu_ids_str:
        return [0]
    return [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]


def run_homogeneous_agent(
    config: dict,
    task_content: str,
    model,
    env,
    env_class,
    env_kwargs: dict,
    tools_settings: dict,
    agent_config: dict,
    repo: Path | None = None,
    num_parallel: int | None = None,
    gpu_ids: str | None = None,
    output_dir: Path | None = None,
    model_name: str | None = None,
    console: Console | None = None,
) -> BestPatchResult | None:
    """
    Run homogeneous parallel agents.

    This function is called from mini.py when agent_mode is 'homogeneous'.
    Configuration is already loaded and merged by mini.py.

    Args:
        config: Merged configuration dict
        task_content: Task description
        model: Model instance
        env: Environment instance
        env_class: Environment class for factory
        env_kwargs: Environment kwargs for factory
        tools_settings: Tools settings from config
        agent_config: Base agent configuration
        repo: Repository path for git worktree management
        num_parallel: Number of parallel agents
        gpu_ids: Comma-separated GPU IDs
        output_dir: Output directory
        model_name: Model name for factory
        console: Rich console for output

    Returns:
        The ParallelAgent instance after execution
    """
    if console is None:
        console = Console(highlight=False)

    # Parse configuration values
    parallel_config = config.get("parallel", {})

    # Number of parallel agents
    final_num_parallel = (
        num_parallel or parallel_config.get("num_parallel") or config.get("agent", {}).get("num_parallel") or 1
    )

    # GPU IDs
    final_gpu_ids = parse_gpu_ids(gpu_ids or parallel_config.get("gpu_ids") or config.get("agent", {}).get("gpu_ids"))

    # Repository path
    final_repo = repo
    if not final_repo:
        final_repo = parallel_config.get("repo") or config.get("agent", {}).get("repo")

    final_repo = Path(final_repo).resolve()
    if not final_repo.exists():
        raise ValueError(f"Repository path does not exist: {final_repo}")

    # Select agent class based on tools.strategy_manager config
    if tools_settings["strategy_manager"]:
        base_agent_class = StrategyInteractiveAgent
        console.print(f"[bold cyan]Using Strategy Agent[/bold cyan] (strategy file: {tools_settings['strategy_file']})")
    else:
        base_agent_class = InteractiveAgent
        console.print("[bold cyan]Using Interactive Agent[/bold cyan]")

    # Configure agent for homogeneous mode
    agent_config["mode"] = "yolo"
    agent_config["confirm_exit"] = False
    agent_config["num_parallel"] = final_num_parallel
    agent_config["gpu_ids"] = final_gpu_ids
    agent_config["repo"] = str(final_repo)
    agent_config["agent_class"] = base_agent_class

    # Create output directory (pop from agent_config as ParallelAgentConfig doesn't accept it)
    final_output_dir = Path(agent_config.pop("output_dir", None) or output_dir or "optimization_logs")
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Set patch_output_dir to output_dir so patches are saved alongside logs
    agent_config["patch_output_dir"] = str(final_output_dir)

    # Setup logging
    log_file = final_output_dir / "homogeneous_agent.log"
    traj_output = final_output_dir / "trajectory.json"

    # Print configuration summary
    console.print("\n[bold]Configuration Summary:[/bold]")
    console.print(f"  Parallel agents: [cyan]{final_num_parallel}[/cyan]")
    console.print(f"  GPU IDs: [cyan]{final_gpu_ids}[/cyan]")
    console.print(f"  Repository: [cyan]{final_repo}[/cyan]")
    console.print(f"  Output: [cyan]{final_output_dir}[/cyan]")
    if agent_config.get("test_command"):
        console.print(f"  Test command: [cyan]{agent_config['test_command']}[/cyan]")
    console.print(f"  Log file: [cyan]{log_file}[/cyan]")
    console.print()

    # Get model config for factory
    model_config = config.get("model", {})

    # Create and run ParallelAgent
    agent = ParallelAgent(model, env, **agent_config)
    agent.log_file = log_file

    try:
        best_result = agent.run(
            task_content,
            output=traj_output,
            save_traj_fn=save_traj,
            console=console,
            model_factory=lambda: get_model(model_name, model_config.copy()),
            env_factory=lambda: env_class(**copy.deepcopy(env_kwargs)),
        )

        if best_result:
            console.print(
                f"\n[bold green]Best patch:[/bold green] {best_result.patch_id} (agent {best_result.agent_id})"
            )
            if best_result.llm_conclusion:
                console.print(f"[bold green]Conclusion:[/bold green] {best_result.llm_conclusion}")
        else:
            console.print("\n[bold yellow]No best patch selected[/bold yellow]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise

    return best_result
