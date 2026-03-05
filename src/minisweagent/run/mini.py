#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import copy
import os
from pathlib import Path
from typing import Any

import typer
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.run.config.global_config import configure_if_first_time
from minisweagent.run.utils.mini_helpers import (
    apply_runtime_settings,
    build_agent,
    load_config,
    load_instructions,
    resolve_task_input,
    run_full_pipeline,
)
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger

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
    config_spec: Path | None = typer.Option(None, "-c", "--config", help="Path to config file (overrides template selection)"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT, "--traj-output", "--output", help="Output trajectory file (legacy; trajectories are also saved inside --patch-output)"),
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
    patch_output: Path | None = typer.Option(None, "-o", "--patch-output", help="Output directory for pipeline results, patches, and test artifacts"),
    metric: str | None = typer.Option(None, "--metric", help="Metric extraction task description for LLM"),
    num_parallel: int | None = typer.Option(None, "--num-parallel", help="Number of parallel patch agents to run (only effective with --save-patch). If not specified, reads from config file."),
    repo: Path | None = typer.Option(None, "--repo", help="Repository path for parallel execution. Required when num_parallel > 1. Each agent will get an isolated workdir using git worktree."),
    gpu_ids: str | None = typer.Option(None, "--gpu-ids", help="Comma-separated GPU IDs for agents (e.g., '0,1,2,3'). For single agent, uses first GPU. Defaults to '0'."),
    # Runtime environment configuration (ported from MSA branch)
    runtime: str = typer.Option("local", "--runtime", help="Runtime environment: local, docker, or auto (auto-detects GPU availability).", rich_help_panel="Advanced"),
    docker_image: str | None = typer.Option(None, "--docker-image", help="Docker image to use when --runtime=docker.", rich_help_panel="Advanced"),
    workspace: Path | None = typer.Option(None, "--workspace", help="Workspace directory to mount in Docker.", rich_help_panel="Advanced"),
    kernel_url: str | None = typer.Option(None, "--kernel-url", help="Kernel as URL (e.g. https://github.com/.../file.py#L106). Resolved path/line/kernel name are injected into the task.", rich_help_panel="Kernel"),
    max_rounds: int | None = typer.Option(None, "--max-rounds", help="Maximum optimisation rounds for the orchestrator (default: GEAK_MAX_ROUNDS env or 5).", rich_help_panel="Advanced"),
    allowed_agents: str | None = typer.Option(None, "--allowed-agents", help="Comma-separated list of allowed agent types (e.g. swe_agent,strategy_agent). Sets GEAK_ALLOWED_AGENTS.", rich_help_panel="Advanced"),
    excluded_agents: str | None = typer.Option(None, "--excluded-agents", help="Comma-separated list of excluded agent types (e.g. openevolve). Sets GEAK_EXCLUDED_AGENTS.", rich_help_panel="Advanced"),
    heterogeneous: bool = typer.Option(False, "--heterogeneous", help="Use LLM-generated diverse optimization tasks (requires preprocessing/discovery). Default: homogeneous.", rich_help_panel="Advanced"),
    from_task: Path | None = typer.Option(None, "--from-task", help="Deprecated: use --task with a YAML-frontmatter .md file instead.", hidden=True),
    rag: bool = typer.Option(False, "--rag", help="Enable RAG retrieval from AMD/NVIDIA knowledge base"),
    debug: bool = typer.Option(False, "-d", "--debug", help="Enable debug output (only with --rag)"),
) -> Any:
    # fmt: on
    configure_if_first_time()

    if allowed_agents:
        os.environ["GEAK_ALLOWED_AGENTS"] = allowed_agents
    if excluded_agents:
        os.environ["GEAK_EXCLUDED_AGENTS"] = excluded_agents

    # Deprecated --from-task: map to --task
    if from_task:
        if not task:
            task = str(from_task)
        console.print("[dim]Note: --from-task is deprecated. Use --task with a YAML-frontmatter .md file instead.[/dim]")

    # Runtime environment check (diagnostic only)
    if runtime in ("auto", "docker"):
        try:
            from minisweagent.runtime_env import detect_runtime_environment
            rt_env = detect_runtime_environment()
            if runtime == "auto" and not rt_env.has_gpu:
                console.print("[bold yellow]No GPU detected locally. Consider --runtime docker.[/bold yellow]")
            elif runtime == "docker":
                console.print(f"[bold cyan]Docker runtime selected. Image: {docker_image or 'default'}[/bold cyan]")
        except ImportError:
            if runtime == "docker":
                console.print("[bold yellow]runtime_env module not available; proceeding with local.[/bold yellow]")

    # --- Phase 1: Config ---
    config = load_config(enable_strategies, config_spec, console)

    # --- Phase 2: Task resolution ---
    ti = resolve_task_input(task, repo, test_command, patch_output, gpu_ids,
                            kernel_url, workspace, console)

    # Interactive prompt fallback (terminal I/O stays in the CLI shell)
    if not ti.task_content and not kernel_url:
        console.print("[bold yellow]What do you want to do?")
        ti.task_content = prompt_session.prompt(
            "",
            multiline=True,
            bottom_toolbar=HTML(
                "Submit task: <b fg='yellow' bg='black'>Esc+Enter</b> | "
                "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
                "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
            ),
        )
        console.print("[bold green]Got that, thanks![/bold green]")

    # --- Phase 3: Runtime settings (model, env, merge configs) ---
    rt = apply_runtime_settings(
        config, ti, yolo=yolo, cost_limit=cost_limit,
        exit_immediately=exit_immediately, model_name=model_name,
        model_class=model_class, enable_strategies=enable_strategies,
        rag=rag, debug=debug, console=console,
    )

    # --- Phase 4: Full pipeline early-return ---
    if kernel_url and not ti.tf_meta:
        run_full_pipeline(
            kernel_url, config, rt.model, model_name, rt.parsed_gpu_ids,
            rt.patch_output, max_rounds, heterogeneous, console,
        )
        return None

    # --- Phase 5: Build agent ---
    agent, agent_log_file = build_agent(
        config, rt.model, model_name, rt.env, ti, rt,
        visual=visual, enable_strategies=enable_strategies,
        strategy_file=strategy_file, create_test=create_test,
        heterogeneous=heterogeneous, console=console,
    )

    # --- Phase 6: Load instructions + run ---
    instructions_content = load_instructions(workspace)
    if instructions_content:
        console.print("Loaded pipeline instructions [bold green](INSTRUCTIONS.md)[/bold green]")

    try:
        exit_status, result = agent.run(
            ti.task_content, instructions=instructions_content,
            output=output,
            save_traj_fn=save_traj,
            console=console,
            model_factory=lambda: _make_model(model_name, config),
            env_factory=lambda _repo=rt.repo: rt.env_class(
                **{**copy.deepcopy(rt.env_kwargs), **({"cwd": str(Path(_repo).resolve())} if _repo else {})}
            ),
        )
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)

    return agent


def _make_model(model_name, config):
    from minisweagent.models import get_model
    return get_model(model_name, config.get("model", {}))


if __name__ == "__main__":
    app()
