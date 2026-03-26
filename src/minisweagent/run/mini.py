#!/usr/bin/env python3

"""Backup mini entry with kernel-type routing."""

import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import typer
import yaml
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents.homogeneous.homogeneous_agent import parse_gpu_ids, run_homogeneous_agent
from minisweagent.agents.parallel_agent import BestPatchResult
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment_class
from minisweagent.models import get_model
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.run.orchestrator import run_orchestrator
from minisweagent.run.preprocess.preprocessor import run_preprocessor
from minisweagent.run.utils.task_parser import _resolve_path_case, display_parsed_config, parse_task_info

DEFAULT_CONFIG = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT = global_config_dir / "last_mini_run.traj.json"

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")
prompt_session = PromptSession(history=FileHistory(global_config_dir / "mini_task_history.txt"))


class TeeOutput:
    """Capture stdout/stderr to buffer while keeping terminal output."""

    def __init__(self, original):
        self.terminal = original
        self.buffer = StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_kernel_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "triton":
        return "triton"
    if text in {"hip", "rocm", "rocblas"}:
        return "hip"
    return "other"


def _derive_output_dir_and_traj(output: Path | None) -> tuple[Path, Path]:
    """Unify patch_output_dir and -o/--output location.

    - If output is a file path: output_dir = output.parent, traj = output
    - If output is a directory: output_dir = output, traj = output/trajectory.json
    - If output is not provided: use ./Optimization_logs as output_dir
    """
    if output is None:
        output_dir = Path.cwd() / "Optimization_logs"
        return output_dir, output_dir / "trajectory.json"

    if output.suffix:
        return output.parent, output

    return output, output / "trajectory.json"


def _final_report_to_bestpatchresult(report: Any) -> BestPatchResult | None:
    if report is None:
        return None
    report_dict = report.to_dict() if hasattr(report, "to_dict") else report
    if not isinstance(report_dict, dict):
        return None

    best_patch = report_dict.get("best_patch")
    patch_path = Path(best_patch) if best_patch else None
    return BestPatchResult(
        agent_id=0,
        patch_id=patch_path.stem if patch_path else "unknown",
        test_output="",
        metric_result={
            "best_speedup": report_dict.get("best_speedup"),
            "best_round": report_dict.get("best_round"),
            "best_task": report_dict.get("best_task"),
            "status": report_dict.get("status"),
        },
        patch_dir=patch_path.parent if patch_path else None,
        llm_conclusion=str(report_dict.get("summary") or ""),
    )


_HELP_TEXT = """Run mini-SWE-agent in your local environment.

[not dim]
There are two different user interfaces:

[bold green]mini[/bold green] Simple REPL-style interface
[bold green]mini -v[/bold green] Pager-style interface (Textual)
[/not dim]
"""


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    visual: bool = typer.Option(False, "-v", "--visual", help="Toggle UI",),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use",),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path | None = typer.Option(None, "-c", "--config", help="Path to config file"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output trajectory file or directory"),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately", rich_help_panel="Advanced"),
    repo: Path | None = typer.Option(None, "--repo", help="Target Repository path."),
    kernel_url: str | None = typer.Option(None, "--kernel-url", help="Target Kernel URL."),
    num_parallel: int | None = typer.Option(None, "--num-parallel", help="Number of parallel patch agents."),
    gpu_ids: str | None = typer.Option(None, "--gpu-ids", help="Comma-separated GPU IDs."),
    test_command: str | None = typer.Option(None, "--test_command", "--test-command", help="Test command"),
):
    # fmt: on
    del visual
    tee_out, tee_err = TeeOutput(sys.stdout), TeeOutput(sys.stderr)
    sys.stdout, sys.stderr = tee_out, tee_err

    configure_if_first_time()

    # 1) Config merge
    base_config_path = builtin_config_dir / "mini_kernel_strategy_list.yaml"
    console.print(f"Loading base config: [bold green]'{base_config_path.name}'[/bold green]")
    config = yaml.safe_load(base_config_path.read_text()) or {}
    if config_spec:
        config_path = get_config_path(config_spec)
        console.print(f"[dim]Applying user config from '{config_path}' (final override)[/dim]")
        user_config = yaml.safe_load(config_path.read_text()) or {}
        config = _deep_merge(config, user_config)

    if yolo:
        config.setdefault("agent", {})["mode"] = "yolo"
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
    if exit_immediately:
        config.setdefault("agent", {})["confirm_exit"] = False
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    tools_cfg = config.get("tools") or {}
    disabled_tools: list[str] = []
    if tools_cfg.get("bash") is False:
        disabled_tools.append("bash")
    if tools_cfg.get("profiling") is False:
        disabled_tools.append("profiling")

    if disabled_tools:
        config.setdefault("agent", {}).setdefault("disabled_tools", [])
        config["agent"]["disabled_tools"] = list(set(config["agent"]["disabled_tools"]) | set(disabled_tools))

    model = get_model(model_name, config.get("model", {}))
    _model_name = getattr(model.config, "model_name", "unknown")
    console.print(f"\\Using model: [bold cyan]{_model_name}[/bold cyan]")

    task_content = task
    if task:
        task_path = Path(task)
        if task_path.exists() and task_path.is_file():
            task_content = task_path.read_text(encoding="utf-8")
            console.print(f"[bold green]Read task from file: {task_path}[/bold green]")
        elif not task.strip():
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

    # 2) Detect configs from task
    parsed_config = parse_task_info(task_content, model)
    kernel_type = _normalize_kernel_type(parsed_config.get("kernel_type"))
    console.print(f"[bold cyan]Detected kernel_type:[/bold cyan] {kernel_type}")

    if repo is None and parsed_config.get("repo"):
        repo = Path(parsed_config["repo"])
    if test_command is None and parsed_config.get("test_command"):
        test_command = parsed_config["test_command"]
    if num_parallel is None:
        num_parallel = _as_int(parsed_config.get("num_parallel"))
    if gpu_ids is None and parsed_config.get("gpu_ids"):
        gpu_ids = parsed_config["gpu_ids"]

    kernel_target = kernel_url or parsed_config.get("kernel_url") or parsed_config.get("kernel_name")
    if not kernel_target:
        console.print("[red]Error: missing kernel target. Provide --kernel-url or include kernel info in task.[/red]")
        raise typer.Exit(1)

    parsed_gpu_ids = parse_gpu_ids(gpu_ids)
    metric = parsed_config.get("metric") or config.get("patch", {}).get("metric")

    preprocess_output_dir, traj_output_path = _derive_output_dir_and_traj(output)
    preprocess_output_dir.mkdir(parents=True, exist_ok=True)
    config.setdefault("patch", {})["patch_output_dir"] = str(preprocess_output_dir)

    _display_cfg = dict(parsed_config)
    _display_cfg["kernel_type"] = kernel_type
    if kernel_url and not _display_cfg.get("kernel_url"):
        _display_cfg["kernel_url"] = kernel_url
    console.print(display_parsed_config(_display_cfg, str(preprocess_output_dir)))

    _env_kwargs = dict(config.get("env", {}))
    env_type = str(_env_kwargs.pop("type", _env_kwargs.pop("environment_class", "local"))).strip().lower() or "local"
    try:
        env_class = get_environment_class(env_type)
        env = env_class(**_env_kwargs)
    except Exception as e:
        console.print(f"[red]Error: failed to initialize env.type={env_type}: {e}[/red]")
        raise typer.Exit(1)

    preprocess_ctx = run_preprocessor(
        kernel_url=kernel_target,
        repo=repo,
        output_dir=preprocess_output_dir,
        gpu_id=parsed_gpu_ids[0] if parsed_gpu_ids else 0,
        model_factory=lambda: get_model(model_name, config.get("model", {})),
        console=console,
        harness=config.get("patch", {}).get("harness"),
        eval_command=test_command,
    )

    if preprocess_ctx.get("test_command") and not test_command:
        test_command = preprocess_ctx["test_command"]
    if preprocess_ctx.get("repo_root") and repo is None:
        repo = Path(preprocess_ctx["repo_root"])

    commandment = preprocess_ctx.get("commandment")
    if commandment:
        task_content = f"{commandment}\n\n---\n\n{task_content}"

    # kernel_type routing:
    # - hip/other -> homogeneous agent
    # - triton -> heterogeneous orchestrator
    if kernel_type == "triton":
        report = run_orchestrator(
            preprocess_ctx=preprocess_ctx,
            gpu_ids=parsed_gpu_ids,
            model=model,
            model_factory=lambda: get_model(model_name, config.get("model", {})),
            output_dir=preprocess_output_dir,
            max_rounds=config.get("orchestrator", {}).get("max_rounds"),
            heterogeneous=True,
            console=console,
        )
        return _final_report_to_bestpatchresult(report)

    agent_config = dict(config.get("agent", {}))
    enable_strategies = _as_bool(tools_cfg.get("strategy_manager", False))
    strategy_file = tools_cfg.get("strategy_file")
    if enable_strategies and strategy_file:
        agent_config["strategy_file_path"] = strategy_file
    agent_config["save_patch"] = True
    agent_config["test_command"] = test_command or config.get("patch", {}).get("test_command")
    agent_config["metric"] = metric
    agent_config["patch_output_dir"] = str(preprocess_output_dir)

    repo_path = repo or config.get("patch", {}).get("repo")
    if repo_path:
        p = Path(repo_path)
        if not p.exists():
            resolved = _resolve_path_case(p)
            if resolved is not None:
                p = resolved
        repo_path = p.resolve()

    tools_settings = {
        "strategy_manager": enable_strategies,
        "strategy_file": strategy_file,
    }

    return run_homogeneous_agent(
        config=config,
        task_content=task_content,
        model=model,
        env=env,
        env_class=env.__class__,
        env_kwargs=_env_kwargs,
        tools_settings=tools_settings,
        agent_config=agent_config,
        repo=repo_path,
        num_parallel=num_parallel,
        gpu_ids=gpu_ids,
        output_dir=preprocess_output_dir,
        traj_output=traj_output_path,
        model_name=model_name,
        console=console,
    )


if __name__ == "__main__":
    app()
