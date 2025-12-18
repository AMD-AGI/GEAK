#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

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
from minisweagent.agents.patch_agent import PatchAgent
from minisweagent.agents.parallel_agent import ParallelAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger

DEFAULT_CONFIG = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT = global_config_dir / "last_mini_run.traj.json"

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")
prompt_session = PromptSession(history=FileHistory(global_config_dir / "mini_task_history.txt"))
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
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
    exit_immediately: bool = typer.Option( False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
    save_patch: bool = typer.Option(False, "--save-patch", help="Save git patches and test results"),
    test_command: str | None = typer.Option(None, "--test-command", help="Test command to run for patch validation"),
    patch_output: Path | None = typer.Option(None, "--patch-output", help="Output directory for patch files and test results"),
    metric: str | None = typer.Option(None, "--metric", help="Metric extraction task description for LLM"),
    num_parallel: int | None = typer.Option(None, "--num-parallel", help="Number of parallel patch agents to run (only effective with --save-patch). If not specified, reads from config file."),
    repo: Path | None = typer.Option(None, "--repo", help="Repository path for parallel execution. Required when num_parallel > 1. Each agent will get an isolated workdir using git worktree."),
) -> Any:
    # fmt: on
    configure_if_first_time()
    config_path = get_config_path(config_spec)
    console.print(f"Loading agent config from [bold green]'{config_path}'[/bold green]")
    config = yaml.safe_load(config_path.read_text())

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

    if yolo:
        config.setdefault("agent", {})["mode"] = "yolo"
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
    if exit_immediately:
        config.setdefault("agent", {})["confirm_exit"] = False
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    model = get_model(model_name, config.get("model", {}))
    env = LocalEnvironment(**config.get("env", {}))

    # Both visual flag and the MSWEA_VISUAL_MODE_DEFAULT flip the mode, so it's essentially a XOR
    agent_class = InteractiveAgent
    if visual == (os.getenv("MSWEA_VISUAL_MODE_DEFAULT", "false") == "false"):
        agent_class = TextualAgent
    agent_config = config.get("agent", {})
    
    # Get num_parallel from config or command line (command line takes precedence)
    effective_num_parallel = num_parallel if num_parallel is not None else config.get("patch", {}).get("num_parallel", 1)
    
    # Auto-enable save_patch if num_parallel > 1, or use explicit save_patch flag
    save_patch = save_patch or effective_num_parallel > 1
    
    # Configure patch agent if save_patch is enabled
    if save_patch:
        # Use PatchAgent when num_parallel=1, ParallelAgent when num_parallel>1
        if effective_num_parallel > 1:
            agent_class = ParallelAgent
        else:
            agent_class = PatchAgent
        agent_config["save_patch"] = True
        agent_config["test_command"] = test_command or config.get("patch", {}).get("test_command")
        patch_dir = patch_output or config.get("patch", {}).get("patch_output_dir") or (global_config_dir / "patches")
        agent_config["patch_output_dir"] = str(patch_dir)
        agent_config["metric"] = metric or config.get("patch", {}).get("metric")
    
    # Get repo path from config or command line (command line takes precedence)
    repo_path = repo or config.get("patch", {}).get("repo")
    if repo_path:
        repo_path = Path(repo_path).resolve()

    # Configure agent with parallel settings if needed
    if save_patch and effective_num_parallel > 1:
        agent_config["num_parallel"] = effective_num_parallel
        if repo_path:
            agent_config["repo"] = str(repo_path)
        agent_config["parallel_gpu_ids"] = config.get("patch", {}).get("parallel_gpu_ids", [])
    
    # Create and run agent
    agent = agent_class(model, env, **agent_config)
    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run(
            task_content,
            output=output,
            save_traj_fn=save_traj,
            console=console,
            model_factory=lambda: get_model(model_name, config.get("model", {})),
            env_factory=lambda: LocalEnvironment(**config.get("env", {})),
        )
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        if output and not (save_patch and effective_num_parallel > 1):
            save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
    
    # Return agent for single execution, result for parallel execution
    return result if (save_patch and effective_num_parallel > 1) else agent


if __name__ == "__main__":
    app()
