#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import os
import sys
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

def _inject_resolved_kernel(kernel_url: str, workspace: str | None, task: str) -> tuple[str, str | None]:
    """Resolve kernel URL to local path/line/kernel name and append to task. Returns (task, kernel_name or None). Raises on resolve error."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from geak_agent.resolve_kernel_url import resolve_kernel_url, get_kernel_name_at_line
    except ImportError as e:
        raise SystemExit(f"Cannot resolve --kernel-url: geak_agent not found ({e}). Run from repo root or install geak_agent.") from e
    clone_into = (Path(workspace) if workspace else Path.cwd())
    resolved = resolve_kernel_url(kernel_url, clone_into=clone_into)
    if resolved.get("error"):
        raise SystemExit(f"Kernel URL resolve failed: {resolved['error']}")
    path = resolved["local_file_path"]
    line_num = resolved.get("line_number")
    kernel_name = get_kernel_name_at_line(path, line_num) if line_num else None
    if line_num:
        line_info = f" Line: {line_num}"
        kernel_info = f", kernel name: {kernel_name!r}" if kernel_name else ""
        profile_hint = (
            " When profiling, the kernel of interest is set automatically; do not use --auto-select."
            if kernel_name else " When profiling, use --filter for the kernel of interest if the file has multiple kernels."
        )
    else:
        line_info = ""
        kernel_info = ""
        profile_hint = " Line number was not specified; discovery should identify the kernel(s) in the file."
    profile_usage = (
        "To profile: run kernel-profile '<command>' (one quoted argument = command that runs the kernel). "
        f"Example: kernel-profile 'python3 {path} --profile' — or use the project's benchmark script if the file has no --profile. "
    )
    block = (
        f"\n\n--- Resolved kernel (from --kernel-url) ---\n"
        f"Use this path for all steps (discover, test, profile, optimize): {path}.{line_info}{kernel_info}\n"
        f"{profile_usage}{profile_hint}\n"
        f"---\n"
    )
    return task + block, kernel_name


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
    kernel_url: str | None = typer.Option(None, "--kernel-url", help="Kernel as URL (e.g. https://github.com/.../file.py#L106). Resolved path/line/kernel name are injected into the task.", rich_help_panel="Kernel"),
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

    # Resolve --kernel-url to local path, line, and kernel name; inject into task (profiler filter set after env config)
    resolved_kernel_name = None
    if kernel_url:
        task, resolved_kernel_name = _inject_resolved_kernel(kernel_url, workspace, task)

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
    if os.getenv("GEAK_PROTECTED_FILES"):
        config.setdefault("env", {})["protected_files"] = [
            f.strip() for f in os.getenv("GEAK_PROTECTED_FILES", "").split(",") if f.strip()
        ]
    if os.getenv("GEAK_SUMMARY_ON_COST_LIMIT", "").lower() in ("1", "true", "yes"):
        config.setdefault("agent", {})["summary_on_cost_limit"] = True
    if os.getenv("GEAK_SUMMARY_ON_LIMIT_PROMPT"):
        config.setdefault("agent", {})["summary_on_limit_prompt"] = os.getenv("GEAK_SUMMARY_ON_LIMIT_PROMPT")
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    model = get_model(model_name, config.get("model", {}))
    
    # Put resolved kernel filter into env vars for the agent's shell (kernel-profile will use it)
    if resolved_kernel_name:
        config.setdefault("env", {}).setdefault("env", {})["GEAK_PROFILE_KERNEL_FILTER"] = resolved_kernel_name
    
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
     if output:
            save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
    return agent


if __name__ == "__main__":
    app()
