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

from geakagent import global_config_dir
from geakagent.agents.interactive import InteractiveAgent
from geakagent.agents.interactive_textual import TextualAgent
from geakagent.agents.patch_agent import PatchAgent
from geakagent.config import builtin_config_dir, get_config_path
from geakagent.environments.local import LocalEnvironment
from geakagent.environments.docker import DockerEnvironment
from geakagent.models import get_model
from geakagent.run.extra.config import configure_if_first_time
from geakagent.run.utils.save import save_traj
from geakagent.runtime_env import (
    prompt_runtime_environment,
    get_runtime_config_for_agent,
    RuntimeType,
    display_runtime_info,
)
from geakagent.utils.log import logger

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
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'geakagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
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
    runtime: str | None = typer.Option(None, "--runtime", help="Runtime environment: 'local', 'docker', or 'auto' (default: auto)", rich_help_panel="Runtime"),
    docker_image: str | None = typer.Option(None, "--docker-image", help="Docker image to use (default: lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x)", rich_help_panel="Runtime"),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace directory to mount in Docker", rich_help_panel="Runtime"),
    no_runtime_check: bool = typer.Option(False, "--no-runtime-check", help="Skip runtime environment detection", rich_help_panel="Runtime"),
) -> Any:
    # fmt: on
    configure_if_first_time()
    config_path = get_config_path(config_spec)
    console.print(f"Loading agent config from [bold green]'{config_path}'[/bold green]")
    config = yaml.safe_load(config_path.read_text())

    if not task:
        console.print("[bold yellow]What do you want to do?")
        task = prompt_session.prompt(
            "",
            multiline=True,
            bottom_toolbar=HTML(
                "Submit task: <b fg='yellow' bg='black'>Esc+Enter</b> | "
                "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
                "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
            ),
        )
        console.print("[bold green]Got that, thanks![/bold green]")

    # Runtime environment detection and configuration
    runtime_env = None
    if not no_runtime_check:
        if runtime == "local":
            # Force local environment
            from geakagent.runtime_env import RuntimeEnvironment
            runtime_env = RuntimeEnvironment(runtime_type=RuntimeType.LOCAL)
            display_runtime_info(runtime_env)
        elif runtime == "docker":
            # Force Docker environment
            from geakagent.runtime_env import RuntimeEnvironment, DEFAULT_DOCKER_IMAGE
            image = docker_image or DEFAULT_DOCKER_IMAGE
            runtime_env = RuntimeEnvironment(
                runtime_type=RuntimeType.DOCKER,
                docker_image=image,
                docker_devices=["/dev/kfd", "/dev/dri"],
                has_gpu=True,
                has_triton=True,
                has_torch=True
            )
            display_runtime_info(runtime_env)
        else:
            # Auto-detect (default)
            runtime_env = prompt_runtime_environment(auto_confirm=yolo)
        
        # Update config based on runtime environment
        if runtime_env and hasattr(runtime_env, 'runtime_type') and runtime_env.runtime_type == RuntimeType.DOCKER:
            workspace_path = workspace or os.getcwd()
            runtime_config = get_runtime_config_for_agent(runtime_env, workspace_path)
            config["env"] = runtime_config

    if yolo:
        config.setdefault("agent", {})["mode"] = "yolo"
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
    if exit_immediately:
        config.setdefault("agent", {})["confirm_exit"] = False
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    model = get_model(model_name, config.get("model", {}))
    
    # Create environment based on runtime configuration
    env_config = config.get("env", {})
    if runtime_env and runtime_env.runtime_type == RuntimeType.DOCKER:
        env = DockerEnvironment(**env_config)
    else:
        env = LocalEnvironment(**env_config)

    # Both visual flag and the MSWEA_VISUAL_MODE_DEFAULT flip the mode, so it's essentially a XOR
    agent_class = InteractiveAgent
    if visual == (os.getenv("MSWEA_VISUAL_MODE_DEFAULT", "false") == "false"):
        agent_class = TextualAgent
    agent_config = config.get("agent", {})
    if save_patch:
        agent_class = PatchAgent
        agent_config["save_patch"] = True
        agent_config["test_command"] = test_command
        patch_dir = patch_output or (global_config_dir / "patches")
        agent_config["patch_output_dir"] = str(patch_dir)
        agent_config["metric"] = metric

    agent = agent_class(model, env, **agent_config)
    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run(task)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        if output:
            save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
    return agent


if __name__ == "__main__":
    app()
