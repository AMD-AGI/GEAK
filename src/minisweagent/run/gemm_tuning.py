"""CLI entry for the GEMM tuning agent (``geak-gemm-tuning``).

Creates ``<cwd>/optimization_logs/gemm_tuning_<timestamp>/``, loads config from
``SubAgentRegistry.get("gemm-tuning")``, and runs the agent via the same
``_run_inprocess()`` path used by ``geak-subagent run``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console(highlight=False)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    task: str = typer.Option(
        ...,
        "-t",
        "--task",
        help="Task or instructions for the GEMM tuning agent.",
        show_default=False,
    ),
    config: str | None = typer.Option(
        None,
        "-c",
        "--config",
        help="Optional override config YAML file.",
    ),
    model_name: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Override model name.",
    ),
    cwd: Path | None = typer.Option(
        None,
        "--cwd",
        help="Base directory for workspace (default: current directory).",
        file_okay=False,
        resolve_path=True,
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Run in yolo mode (no confirmations)."),
) -> None:
    """Run one GEMM tuning agent session."""
    import os

    from minisweagent.run.extra.config import configure_if_first_time
    from minisweagent.subagents.subagent_registry import SubAgentRegistry

    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    configure_if_first_time()

    registry = SubAgentRegistry()
    descriptor = registry.get("gemm-tuning")

    if descriptor is None:
        console.print("[bold red]Error:[/bold red] gemm-tuning subagent not found in registry.")
        console.print("Ensure subagents/gemm-tuning/SUBAGENT.yaml exists.")
        raise typer.Exit(1)

    # Create workspace
    run_cwd = (cwd or Path.cwd()).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    workspace = run_cwd / "optimization_logs" / f"gemm_tuning_{stamp}"
    workspace.mkdir(parents=True, exist_ok=False)

    # Augment task with workspace info
    task_for_agent = (
        f"{task.rstrip()}\n\n"
        f"Your workspace is under: {workspace.resolve()}\n"
        "The shell working directory for this run is set to that path; keep benchmarks, "
        "tuner output, logs, and final_report.json there unless the task requires otherwise."
    )

    console.print(f"[bold cyan]GEMM tuning workspace:[/bold cyan] {workspace.resolve()}")

    # Delegate to the standard inprocess runner from subagent_cli
    from minisweagent.run.subagent_cli import _run_inprocess

    _run_inprocess(
        descriptor,
        task_for_agent,
        config,
        model_name,
        step_limit_override=0,
        cost_limit_override=0.0,
        yolo=yes,
    )


if __name__ == "__main__":
    app()
