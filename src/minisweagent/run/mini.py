#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import copy
import os
import traceback
from pathlib import Path
from typing import Any

import typer
import yaml
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.agents.interactive_textual import TextualAgent
from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent
from minisweagent.agents.parallel_agent import ParallelAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.run.utils.save import save_traj
from minisweagent.run.utils.config_editor import load_and_merge_configs
from minisweagent.utils.log import logger
from minisweagent.agents.unit_test_agent import run_unit_test_agent

DEFAULT_CONFIG = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT = global_config_dir / "last_mini_run.traj.json"

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")
prompt_session = PromptSession(history=FileHistory(global_config_dir / "mini_task_history.txt"))


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
_HELP_TEXT = """Run mini-SWE-agent in your local environment.

[not dim]
There are two different user interfaces:

[bold green]mini[/bold green] Simple REPL-style interface
[bold green]mini -v[/bold green] Pager-style interface (Textual)

More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/mini/[/bold green]
[/not dim]
"""


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    visual: bool = typer.Option(False, "-v", "--visual", help="Toggle (pager-style) UI (Textual) depending on the MSWEA_VISUAL_MODE_DEFAULT environment setting",),
    model_name: str | None = typer.Option( None, "-m", "--model", help="Model to use",),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path | None = typer.Option(None, "-c", "--config", help="Path to config file (overrides template selection)"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
    exit_immediately: bool = typer.Option( False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
    # Strategy mode configuration
    enable_strategies: bool = typer.Option(True, "--enable-strategies/--no-enable-strategies", help="Enable optimization strategy management (optool command). Auto-selects appropriate template.", rich_help_panel="Advanced"),
    strategy_file: str = typer.Option(".optimization_strategies.md", "--strategy-file", help="Path to strategy file (relative to workspace)", rich_help_panel="Advanced"),
    # Patch mode configuration (always enabled)
    test_command: str | None = typer.Option(None, "--test_command", "--test-command", help="Test command to run for patch validation"),
    create_test: bool = typer.Option(
        False,
        "--create-test",
        "--create_test",
        help="Auto-create/search unit tests and infer test_command when missing (or to override it).",
        rich_help_panel="Advanced",
    ),
    patch_output: Path | None = typer.Option(None, "--patch-output", help="Output directory for patch files and test results"),
    metric: str | None = typer.Option(None, "--metric", help="Metric extraction task description for LLM"),
    num_parallel: int | None = typer.Option(None, "--num-parallel", help="Number of parallel patch agents to run (only effective with --save-patch). If not specified, reads from config file."),
    repo: Path | None = typer.Option(None, "--repo", help="Repository path for parallel execution. Required when num_parallel > 1. Each agent will get an isolated workdir using git worktree."),
    gpu_ids: str | None = typer.Option(None, "--gpu-ids", help="Comma-separated GPU IDs for agents (e.g., '0,1,2,3'). For single agent, uses first GPU. Defaults to '0'."),
) -> Any:
    # fmt: on
    configure_if_first_time()
    
    # 1. Load base config (mini.yaml - always loaded as foundation)
    base_config_path = builtin_config_dir / "mini.yaml"
    console.print(f"Loading base config: [bold green]'{base_config_path.name}'[/bold green]")
    config = yaml.safe_load(base_config_path.read_text())
    
    # 2. Select and merge template based on enable_strategies flag
    if enable_strategies:
        template_name = "mini_kernel_strategy_list.yaml"
    else:
        template_name = "mini_system_prompt.yaml"
    
    template_path = builtin_config_dir / template_name
    console.print(f"Applying template: [bold green]'{template_name}'[/bold green] (save_patch always enabled)")
    template_config = yaml.safe_load(template_path.read_text())
    config = _deep_merge(config, template_config)
    
    # 3. Load user config if explicitly specified (final override)
    if config_spec:
        config_path = get_config_path(config_spec)
        console.print(f"[dim]Applying user config from '{config_path}' (final override)[/dim]")
        user_config = yaml.safe_load(config_path.read_text())
        config = _deep_merge(config, user_config)

    tools_cfg = config.get("tools") or {}
    if tools_cfg:
        if "bash" in tools_cfg:
            config.setdefault("model", {}).setdefault("bash_tool", tools_cfg["bash"])
        if "profiling" in tools_cfg:
            config.setdefault("model", {}).setdefault("profiling", tools_cfg["profiling"])
        if "profiling_type" in tools_cfg:
            config.setdefault("agent", {}).setdefault("profiling_type", tools_cfg["profiling_type"])
        if tools_cfg.get("profiling") and "profiling_type" not in tools_cfg:
            config.setdefault("agent", {}).setdefault("profiling_type", "profiling")
        if "strategy_manager" in tools_cfg:
            config.setdefault("agent", {}).setdefault("use_strategy_manager", tools_cfg["strategy_manager"])
            config.setdefault("model", {}).setdefault("use_strategy_manager", tools_cfg["strategy_manager"])

    # Backward compatibility: legacy top-level tool flags
    if "profiling" in config:
        config.setdefault("model", {}).setdefault("profiling", config["profiling"])
    if "profiling_type" in config:
        config.setdefault("agent", {}).setdefault("profiling_type", config["profiling_type"])
    if config.get("model", {}).get("profiling") and not config.get("agent", {}).get("profiling_type"):
        config.setdefault("agent", {})["profiling_type"] = "profiling"

    # Read task content - if task is a file path, read its content; otherwise use task as-is
    task_content = task
    if task:
        task_path = Path(task)
        if task_path.exists() and task_path.is_file():
            # Read file content regardless of extension (txt, md, etc.)
            task_content = task_path.read_text(encoding="utf-8")
            console.print(f"[bold green]Read task from file: {task_path}[/bold green]")
        elif not task.strip():
            # Empty task, prompt user
            task_content = None
    
    if not task_content:
        console.print("[bold yellow]What do you want to do?")
        task_content = prompt_session.prompt(
            "",
            multiline=True,
            bottom_toolbar=HTML(
                "Submit task: <b fg='yellow' bg='black'>Esc+Enter</b> | "
                "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
                "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
            ),
        )
        console.print("[bold green]Got that, thanks![/bold green]")
    elif task and '.md' in task:
        with open(task, "r", encoding="utf-8") as f:
            task = f.read()

    if yolo:
        config.setdefault("agent", {})["mode"] = "yolo"
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
    if exit_immediately:
        config.setdefault("agent", {})["confirm_exit"] = False
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    # Set use_strategy_manager in model config based on enable_strategies flag
    config.setdefault("model", {})["use_strategy_manager"] = enable_strategies
    model = get_model(model_name, config.get("model", {}))
    env = LocalEnvironment(**config.get("env", {}))

    # Load and merge configurations: Command-line > parallel_config from yaml > auto-detect
    result = load_and_merge_configs(
        config, repo, test_command, metric, num_parallel, gpu_ids, patch_output,
        task_content, yolo, model, console
    )
    if result == (None, None, None, None, None, None, None):
        console.print("[bold yellow]Continuing without automatic patch saving. You can still interact with the agent.[/bold yellow]")
        # Keep original None values since user aborted
        repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output, kernel_name = None, None, None, None, [0], None, None
    else:
        repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output, kernel_name = result

    if create_test or not test_command:
        if not repo:
            raise ValueError("repo is required for --create-test or when test_command is missing. Please pass --repo.")
        console.print(
            "[bold yellow]No test_command provided (or --create-test enabled). "
            "Will auto-create/search unit tests and infer a test command via UnitTestAgent...[/bold yellow]"
        )
        test_command = run_unit_test_agent(
            model=get_model(model_name, config.get("model", {})),
            repo=repo,
            kernel_name=kernel_name or "unknown",
            log_dir=patch_output,
        )
        console.print(f"[bold green]Using UnitTestAgent test_command:[/bold green] {test_command}")
    
    # ============ Step 1: Choose base agent class ============
    # Based on enable_strategies flag, select appropriate agent and template
    if enable_strategies:
        # Use strategy agent with mini_kernel_strategy_list.yaml template
        base_agent_class = StrategyInteractiveAgent
        console.print(f"[bold cyan]Using Strategy Agent with strategy file: {strategy_file}[/bold cyan]")
    else:
        # Use interactive agent with mini_system_prompt.yaml template
        # Choose between visual (Textual) and non-visual (Interactive) mode
        if visual == (os.getenv("MSWEA_VISUAL_MODE_DEFAULT", "false") == "false"):
            base_agent_class = TextualAgent
        else:
            base_agent_class = InteractiveAgent
        console.print(f"[bold cyan]Using Interactive Agent (visual={'on' if base_agent_class == TextualAgent else 'off'})[/bold cyan]")
    
    # Mode (yolo/confirm/human) is set via config and applies to all InteractiveAgent subclasses
    
    # ============ Step 2: Configure agent settings ============
    agent_config = config.get("agent", {})
    
    # Add strategy manager settings
    agent_config["use_strategy_manager"] = enable_strategies
    if enable_strategies:
        agent_config["strategy_file_path"] = strategy_file
    
    # Configure save_patch settings (always enabled)
    agent_config["save_patch"] = True
    agent_config["test_command"] = test_command or config.get("patch", {}).get("test_command")
    patch_dir = patch_output or config.get("patch", {}).get("patch_output_dir") or (global_config_dir / "patches")
    agent_config["patch_output_dir"] = str(patch_dir)
    agent_config["metric"] = metric or config.get("patch", {}).get("metric")
    
    # ============ Step 3: Use ParallelAgent (supports both single and parallel execution) ============
    agent_class = ParallelAgent
    agent_config["agent_class"] = base_agent_class
    agent_config["num_parallel"] = num_parallel or 1
    agent_config["gpu_ids"] = parsed_gpu_ids
    
    if num_parallel and num_parallel > 1:
        console.print(f"[bold cyan]Using Parallel Mode: {num_parallel} agents[/bold cyan]")
        console.print(f"[dim]GPU IDs: {parsed_gpu_ids}[/dim]")
        
        # Configure repo path for parallel execution
        repo_path = repo or config.get("patch", {}).get("repo")
        if repo_path:
            agent_config["repo"] = str(Path(repo_path).resolve())
            console.print(f"[dim]Repository: {agent_config['repo']}[/dim]")
        else:
            console.print("[bold yellow]Warning: No repo path specified for parallel execution[/bold yellow]")
    else:
        console.print(f"[bold cyan]Using Single Agent Mode[/bold cyan]")
        console.print(f"[dim]Using GPU: {parsed_gpu_ids[0]}[/dim]")
        # Set HIP_VISIBLE_DEVICES for single agent GPU isolation
        env.config.env = env.config.env or {}
        env.config.env["HIP_VISIBLE_DEVICES"] = str(parsed_gpu_ids[0])
    
    # Create and run agent
    agent = agent_class(model, env, **agent_config)
    try:
        exit_status, result = agent.run(
            task_content,
            output=output,
            save_traj_fn=save_traj,
            console=console,
            model_factory=lambda: get_model(model_name, config.get("model", {})),
            env_factory=lambda: LocalEnvironment(**copy.deepcopy(config.get("env", {}))),
        )
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
    
    return agent


if __name__ == "__main__":
    app()
