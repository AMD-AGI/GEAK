#!/usr/bin/env python3
"""CLI entry point for running subagents directly.

Usage examples::

    geak-subagent --list
    geak-subagent --agent reverse-knowledge --prompt 'analyze repo at /path/to/repo'
    geak-subagent --agent reverse-knowledge /path/to/repo
    geak-subagent --agent reverse-knowledge /path/baseline /path/optimized
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from minisweagent.subagents.subagent_registry import SubAgentRegistry

logger = logging.getLogger(__name__)
console = Console(highlight=False)
app = typer.Typer(
    rich_markup_mode="rich",
    help="Run GEAK subagents directly from the command line.",
    no_args_is_help=True,
)


@app.command("list")
def list_subagents() -> None:
    """List all available subagents."""
    registry = SubAgentRegistry()

    if not registry.subagents:
        console.print("[yellow]No subagents found.[/yellow]")
        console.print("Add subagent definitions to [bold]subagents/<name>/SUBAGENT.yaml[/bold]")
        raise typer.Exit(0)

    table = Table(title="Available Subagents")
    table.add_column("Name", style="bold cyan")
    table.add_column("Mode", style="green")
    table.add_column("Description")
    table.add_column("Path")

    for name in sorted(registry.subagents):
        desc = registry.subagents[name]
        table.add_row(name, desc.execution_mode, desc.description[:80], str(desc.path))

    console.print(table)


@app.command("run")
def run_subagent(
    agent: str = typer.Option(..., "--agent", "-a", help="Name of the registered subagent to run."),
    prompt: str | None = typer.Option(None, "--prompt", "-p", help="Task prompt for the subagent."),
    config: str | None = typer.Option(None, "--config", "-c", help="Override config YAML file."),
    model_name: str | None = typer.Option(None, "--model", "-m", help="Override model name."),
    step_limit: int = typer.Option(0, "--step-limit", help="Override step limit (0 = default)."),
    cost_limit: float = typer.Option(0.0, "--cost-limit", help="Override cost limit (0.0 = default)."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Run in yolo mode (no confirmations)."),
    args: list[str] | None = typer.Argument(
        None, help="Positional arguments passed to subprocess subagents (e.g. paths)."
    ),
) -> None:
    """Run a registered subagent with the given prompt."""
    import os

    from minisweagent.run.extra.config import configure_if_first_time

    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    configure_if_first_time()

    registry = SubAgentRegistry()
    descriptor = registry.get(agent)

    if descriptor is None:
        console.print(f"[bold red]Error:[/bold red] Unknown subagent: [bold]{agent}[/bold]")
        available = ", ".join(registry.list_names()) or "(none)"
        console.print(f"Available: {available}")
        raise typer.Exit(1)

    if not prompt and not args:
        console.print("[bold red]Error:[/bold red] Provide --prompt or positional arguments.")
        raise typer.Exit(1)

    console.print(
        f"[bold cyan]Running subagent:[/bold cyan] {descriptor.name} ([green]{descriptor.execution_mode}[/green])"
    )

    if descriptor.execution_mode == "subprocess":
        _run_subprocess(descriptor, prompt, args or [], config)
    else:
        _run_inprocess(descriptor, prompt or " ".join(args or []), config, model_name, step_limit, cost_limit, yes)


def _run_subprocess(descriptor, prompt: str | None, positional_args: list[str], config_override: str | None) -> None:
    """Run a subprocess-mode subagent."""
    import subprocess

    from minisweagent import get_repo_root

    geak_root = get_repo_root()

    if not descriptor.entry_script:
        console.print("[bold red]Error:[/bold red] No entry_script defined for this subagent.")
        raise typer.Exit(1)

    entry = geak_root / descriptor.entry_script
    if not entry.exists():
        console.print(f"[bold red]Error:[/bold red] Entry script not found: {entry}")
        raise typer.Exit(1)

    import os

    env = {**os.environ}
    env["GEAK_ROOT"] = str(geak_root)
    env.setdefault("PYTHONPATH", str(geak_root / "src"))

    # Set config path: override or the SUBAGENT.yaml itself
    if config_override:
        config_path = Path(config_override).resolve()
    else:
        config_path = (descriptor.path / "SUBAGENT.yaml").resolve()

    env["GEAK_REVERSE_KL_CONFIG"] = str(config_path)

    # Build task from prompt + write to temp file if needed
    if prompt:
        import tempfile

        task_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        task_file.write(prompt)
        task_file.close()
        env["GEAK_REVERSE_KL_TASK_FILE"] = task_file.name

    cmd = [str(entry)] + positional_args
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")

    try:
        result = subprocess.run(cmd, env=env, cwd=str(geak_root))
        raise typer.Exit(result.returncode)
    finally:
        if prompt:
            os.unlink(task_file.name)


def _run_inprocess(
    descriptor,
    task: str,
    config_override: str | None,
    model_name: str | None,
    step_limit_override: int,
    cost_limit_override: float,
    yolo: bool,
) -> None:
    """Run an inprocess-mode subagent."""
    import yaml

    from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig
    from minisweagent.environments import get_environment_class
    from minisweagent.models import get_model

    # Load config: from override file, or from embedded config in descriptor
    agent_config: dict = {}
    model_config: dict = {}
    env_config: dict = {}

    if config_override:
        config_path = Path(config_override)
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            agent_config = dict(raw.get("agent") or {})
            model_config = dict(raw.get("model") or {})
            env_config = dict(raw.get("env") or {})
    else:
        agent_config = dict(descriptor.agent_config)
        model_config = dict(descriptor.model_config)
        env_config = dict(descriptor.env_config)

    # Apply overrides
    if step_limit_override > 0:
        agent_config["step_limit"] = step_limit_override
    elif descriptor.step_limit > 0:
        agent_config["step_limit"] = descriptor.step_limit

    if cost_limit_override > 0:
        agent_config["cost_limit"] = cost_limit_override
    elif descriptor.cost_limit > 0:
        agent_config["cost_limit"] = descriptor.cost_limit

    if yolo:
        agent_config["mode"] = "yolo"

    # Filter to InteractiveAgentConfig fields
    from dataclasses import fields as dc_fields

    allowed = {f.name for f in dc_fields(InteractiveAgentConfig)}
    agent_config = {k: v for k, v in agent_config.items() if k in allowed}

    # Strip placeholder API key
    api_key = model_config.get("api_key")
    if api_key is None or (isinstance(api_key, str) and api_key.strip().lower() in ("", "none", "null")):
        model_config.pop("api_key", None)

    # Create model and env
    model = get_model(model_name, model_config)

    env_kwargs = dict(env_config)
    env_type = str(env_kwargs.pop("type", env_kwargs.pop("environment_class", "local"))).strip().lower() or "local"
    env_class = get_environment_class(env_type)
    env = env_class(**env_kwargs)

    # Run
    agent = InteractiveAgent(model, env, **agent_config)
    console.print(f"[bold green]Starting subagent:[/bold green] {descriptor.name}")
    exit_status, msg = agent.run(task)
    console.print(f"\n[bold]Finished:[/bold] {exit_status}")
    if msg:
        console.print(msg)

    raise typer.Exit(0 if exit_status == "Submitted" else 1)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    list_all: bool = typer.Option(False, "--list", "-l", help="List all available subagents."),
) -> None:
    """GEAK Subagent CLI -- run registered subagents directly."""
    if list_all:
        list_subagents()
        raise typer.Exit(0)
    if ctx.invoked_subcommand is None:
        console.print(
            "Use [bold]--list[/bold] to see available subagents or [bold]run --agent NAME[/bold] to execute one."
        )
        console.print("Run [bold]geak-subagent --help[/bold] for full usage.")


if __name__ == "__main__":
    app()
