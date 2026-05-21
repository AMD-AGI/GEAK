"""CLI entry for the GEMM tuning agent (`geak-gemm-tuning`).

Creates ``<cwd>/optimization_logs/gemm_tuning_<timestamp>/``, uses it as the agent shell
workspace, appends that path to the task text for ``{{task}}``. Agent config loads from
``mini_gemm_tuning.yaml``; model/env always load from ``geak.yaml`` so every
entry-point shares a single model configuration. Optional ``-c`` YAML overlays only
``model_class``, ``base_url``, ``model_name``, and ``api_key``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import typer

from minisweagent.agents.gemm_tuning_agent import run_gemm_tuning_agent
from minisweagent.config import load_config
from minisweagent.models import get_model

logger = logging.getLogger(__name__)

_GEMM_AGENT_CONFIG = "mini_gemm_tuning"
_GEAK_CONFIG = "geak"
_MODEL_OVERLAY_KEYS = ("model_class", "base_url", "model_name", "api_key")

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_gemm_tuning_config(overlay_spec: str | None) -> tuple[dict, dict, dict]:
    """Load agent config from ``mini_gemm_tuning.yaml``; model/env from ``geak.yaml``.

    Optional ``-c`` overlay may override model keys only.
    """
    agent_full = load_config(_GEMM_AGENT_CONFIG)
    agent_kw = dict(agent_full.get("agent") or {})

    geak_full = load_config(_GEAK_CONFIG)
    model_kw = dict(geak_full.get("model") or {})
    env_kw = dict(geak_full.get("env") or {})

    if not overlay_spec:
        return agent_kw, model_kw, env_kw

    try:
        overlay = load_config(overlay_spec)
    except FileNotFoundError:
        logger.warning(
            "Overlay config %r not found; using model section from %s only",
            overlay_spec,
            _GEAK_CONFIG,
        )
        return agent_kw, model_kw, env_kw

    overlay_model = overlay.get("model") or {}
    for key in _MODEL_OVERLAY_KEYS:
        if key in overlay_model and overlay_model[key] is not None:
            model_kw[key] = overlay_model[key]

    return agent_kw, model_kw, env_kw


@app.command()
def run(
    task: str = typer.Option(
        ...,
        "-t",
        "--task",
        help="Task or instructions for the GEMM tuning agent",
        show_default=False,
    ),
    config: str | None = typer.Option(
        None,
        "-c",
        "--config",
        help=(
            "Optional YAML overlay (e.g. loading.yaml). Only model_class, base_url, "
            "model_name, and api_key from its model: section override mini_gemm_tuning; "
            "agent/env always come from mini_gemm_tuning.yaml"
        ),
    ),
    cwd: Path | None = typer.Option(
        None,
        "--cwd",
        help="Base directory: creates optimization_logs/gemm_tuning_<timestamp>/ here (default: current directory)",
        file_okay=False,
        resolve_path=True,
    ),
    model_name: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Override model_name from the config's model section",
    ),
    log_dir: Path | None = typer.Option(
        None,
        "--log-dir",
        help="Agent log and traj directory (default: the created GEMM tuning workspace)",
        file_okay=False,
    ),
) -> None:
    """Run one GemmTuningAgent session."""
    run_cwd = (cwd or Path.cwd()).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    workspace = run_cwd / "optimization_logs" / f"gemm_tuning_{stamp}"
    workspace.mkdir(parents=True, exist_ok=False)

    task_for_agent = (
        f"{task.rstrip()}\n\n"
        f"Your workspace is under: {workspace.resolve()}\n"
        "The shell working directory for this run is set to that path; keep benchmarks, "
        "tuner output, logs, and final_report.json there unless the task requires otherwise."
    )
    effective_log_dir = log_dir if log_dir is not None else workspace

    typer.echo(
        f"GEMM tuning workspace: {workspace.resolve()}\n"
        f"Conversation log: {effective_log_dir / 'task_0.log'}\n"
        f"Trajectory: {effective_log_dir / 'traj.json'}"
    )

    agent_kw, model_kw, env_section = _load_gemm_tuning_config(config)
    _mc = typer.style(str(model_kw.get("model_class")), fg=typer.colors.GREEN, bold=True)
    _mn = typer.style(str(model_kw.get("model_name")), fg=typer.colors.GREEN, bold=True)
    typer.echo(
        f"Model/env config loaded from geak.yaml "
        f"(model_class={_mc}, model_name={_mn}). "
        f"To change model or env settings, edit geak.yaml."
    )
    if model_name:
        model_kw["model_name"] = model_name

    model = get_model(config=model_kw)
    status, msg = run_gemm_tuning_agent(
        model=model,
        cwd=workspace,
        agent_config=agent_kw,
        task=task_for_agent,
        local_env=env_section,
        log_dir=effective_log_dir,
    )

    if status != "Submitted":
        logger.warning("Agent finished with status=%s: %s", status, msg)
        typer.echo(msg, err=True)
        raise typer.Exit(code=1)
    typer.echo(msg)


if __name__ == "__main__":
    app()
