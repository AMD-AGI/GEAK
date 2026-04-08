#!/usr/bin/env python3

"""Backup mini entry with kernel-type routing."""

import logging
import os
import shlex
import sys
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

logger = logging.getLogger(__name__)
console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")
prompt_session = PromptSession(history=FileHistory(global_config_dir / "mini_task_history.txt"))


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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


def _derive_output_dir_and_traj(output: Path | None, kernel_name: str | None) -> tuple[Path, Path]:
    """Unify patch_output_dir and -o/--output location.

    - If output is a file path: output_dir = output.parent, traj = output
    - If output is a directory: output_dir = output, traj = output/trajectory.json
    - If output is not provided: use ./optimization_logs/<kernel_name>_<timestamp> as output_dir
    """
    if output is None:
        from minisweagent.run.utils.task_parser import generate_patch_output_dir

        output_dir = (Path.cwd() / Path(generate_patch_output_dir(kernel_name))).resolve()
        return output_dir, output_dir / "trajectory.json"

    output = output.resolve()

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


def _try_promote_to_harness(test_command: str) -> str | None:
    """Check if test_command points to a harness with argparse modes.

    If so, return the harness path (to pass as harness= to the preprocessor,
    which automatically uses --profile for profiling).
    Otherwise return None (keep using eval_command= as-is).
    """
    parts = shlex.split(test_command)
    script = None
    for part in parts:
        if part.endswith(".py") and Path(part).is_file():
            script = part
            break
    if not script:
        return None

    from minisweagent.run.preprocess.harness_utils import validate_harness

    valid, _errors = validate_harness(script)
    return script if valid else None


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
    kernel_url: str | None = typer.Option(None, "--kernel-url", "--kernel-path", help="Target kernel source (path or URL)."),
    num_parallel: int | None = typer.Option(None, "--num-parallel", help="Number of parallel patch agents."),
    gpu_ids: str | None = typer.Option(None, "--gpu-ids", help="Comma-separated GPU IDs."),
    test_command: str | None = typer.Option(None, "--test_command", "--test-command", help="Test command"),
):
    # fmt: on
    del visual

    if sys.stdin.isatty():
        configure_if_first_time()

    # 1) Config merge
    base_config_path = builtin_config_dir / "mini_kernel_strategy_list.yaml"
    logger.info("Loading base config: [bold green]'%s'[/bold green]", base_config_path.name)
    config = yaml.safe_load(base_config_path.read_text()) or {}
    config_path = config_spec or (builtin_config_dir / "geak.yaml")
    logger.info("[dim]Applying user config from '%s' (final override)[/dim]", config_path)
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
        disabled_tools.append("profile_kernel")

    if disabled_tools:
        config.setdefault("agent", {}).setdefault("disabled_tools", [])
        config["agent"]["disabled_tools"] = list(set(config["agent"]["disabled_tools"]) | set(disabled_tools))

    model = get_model(model_name, config.get("model", {}))
    _model_name = getattr(model.config, "model_name", "unknown")
    logger.info("Using model: [bold cyan]%s[/bold cyan]", _model_name)

    task_content = task
    if task:
        task_path = Path(task)
        if task_path.exists() and task_path.is_file():
            task_content = task_path.read_text(encoding="utf-8")
            logger.info("[bold green]Read task from file: %s[/bold green]", task_path)
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

    # 2a) LLM-driven pipeline param extraction
    heterogeneous = None
    max_rounds = None
    if task_content:
        from minisweagent.run.utils.task_parser import parse_pipeline_params

        logger.info("[bold cyan]Checking task for pipeline parameters...[/bold cyan]")
        pipeline_params = parse_pipeline_params(task_content, model)

        # Apply non-None extracted values (CLI flags still take priority)
        if pipeline_params.get("heterogeneous") is not None and heterogeneous is None:
            heterogeneous = pipeline_params["heterogeneous"]
        if pipeline_params.get("max_rounds") is not None:
            max_rounds = pipeline_params["max_rounds"]

        # Prompt for missing required params (kernel_url) — only if not already set
        if kernel_url is None:
            from minisweagent.run.utils.config_editor import prompt_missing_pipeline_params

            pipeline_params, should_use_pipeline = prompt_missing_pipeline_params(
                pipeline_params, console, yolo
            )

            if should_use_pipeline:
                if pipeline_params.get("kernel_url"):
                    kernel_url = pipeline_params["kernel_url"]

    # 2b) Detect configs from task
    parsed_config = parse_task_info(task_content, model)
    task_kernel_type = _normalize_kernel_type(parsed_config.get("kernel_type"))
    kernel_type = task_kernel_type
    if kernel_url:
        kp = Path(kernel_url)
        if kp.exists() and kp.is_file():
            from minisweagent.agents.heterogeneous.task_generator import _infer_kernel_type

            inferred = _normalize_kernel_type(_infer_kernel_type(kp))
            if inferred in {"hip", "triton"}:
                kernel_type = inferred

    # Keep kernel_type resolution internal; avoid exposing routing details in logs.

    if repo is None and parsed_config.get("repo"):
        repo = Path(parsed_config["repo"])
    if test_command is None and parsed_config.get("test_command"):
        test_command = parsed_config["test_command"]
    if num_parallel is None:
        num_parallel = _as_int(parsed_config.get("num_parallel"))
    if gpu_ids is None and parsed_config.get("gpu_ids") is not None:
        _parsed_gpu_ids = parsed_config["gpu_ids"]
        if isinstance(_parsed_gpu_ids, list):
            gpu_ids = ",".join(map(str, _parsed_gpu_ids))
        else:
            gpu_ids = str(_parsed_gpu_ids)
        if not gpu_ids.strip():
            gpu_ids = None

    # Apply config/model/output_dir extracted from task (CLI flags take priority)
    if config_spec is None and parsed_config.get("config"):
        _task_config_path = get_config_path(Path(parsed_config["config"]))
        if _task_config_path and _task_config_path.exists():
            logger.info("[dim]Applying config from task: '%s'[/dim]", _task_config_path)
            _task_user_config = yaml.safe_load(_task_config_path.read_text()) or {}
            config = _deep_merge(config, _task_user_config)

    if model_name is None and parsed_config.get("model"):
        model_name = parsed_config["model"]
        model = get_model(model_name, config.get("model", {}))
        logger.info("Using model (from task): [bold cyan]%s[/bold cyan]", model_name)

    if output is None and parsed_config.get("output_dir"):
        output = Path(parsed_config["output_dir"])

    kernel_target = kernel_url or parsed_config.get("kernel_url") or parsed_config.get("kernel_name")
    if not kernel_target:
        logger.error(
            "[red]Error: missing kernel target. Provide --kernel-url or include kernel info in task.[/red]"
        )
        raise typer.Exit(1)

    parsed_gpu_ids = parse_gpu_ids(gpu_ids)
    if num_parallel is None and isinstance(gpu_ids, str) and gpu_ids.strip():
        num_parallel = len(parsed_gpu_ids)
    metric = parsed_config.get("metric") or config.get("patch", {}).get("metric")

    kernel_name_for_output = parsed_config.get("kernel_name")
    if not kernel_name_for_output and kernel_url:
        kernel_name_for_output = Path(kernel_url).stem
    if not kernel_name_for_output and isinstance(kernel_target, str):
        kernel_name_for_output = Path(kernel_target).stem

    preprocess_output_dir, traj_output_path = _derive_output_dir_and_traj(output, kernel_name_for_output)
    preprocess_output_dir.mkdir(parents=True, exist_ok=True)
    config.setdefault("patch", {})["patch_output_dir"] = str(preprocess_output_dir)
    logger.info(
        "[dim]Logs and artifacts for this run are under '%s' "
        "(e.g. optimization_logs/<kernel>_<timestamp>/).[/dim]",
        preprocess_output_dir,
    )

    # Display the *resolved* configuration (CLI overrides auto-detection).
    _display_cfg = dict(parsed_config)
    _display_cfg["kernel_type"] = kernel_type
    if kernel_url:
        _display_cfg["kernel_url"] = kernel_url
    if repo is not None:
        _display_cfg["repo"] = str(repo)
    if test_command is not None:
        _display_cfg["test_command"] = test_command
    if num_parallel is not None:
        _display_cfg["num_parallel"] = num_parallel
    if gpu_ids is not None:
        _display_cfg["gpu_ids"] = gpu_ids
    if model_name is not None:
        _display_cfg["model"] = model_name
    if config_spec is not None:
        _display_cfg["config"] = str(config_spec)
    console.print(display_parsed_config(_display_cfg, str(preprocess_output_dir)))

    _env_kwargs = dict(config.get("env", {}))
    env_type = str(_env_kwargs.pop("type", _env_kwargs.pop("environment_class", "local"))).strip().lower() or "local"
    try:
        env_class = get_environment_class(env_type)
        env = env_class(**_env_kwargs)
    except Exception as e:
        logger.error("[red]Error: failed to initialize env.type=%s: %s[/red]", env_type, e)
        raise typer.Exit(1)

    harness_spec = config.get("patch", {}).get("harness")
    if not harness_spec and test_command:
        promoted = _try_promote_to_harness(test_command)
        if promoted:
            harness_spec = promoted
            logger.info("[bold cyan]Promoted test command to validated harness: %s[/bold cyan]", promoted)

    _preprocess_kwargs = dict(
        kernel_url=kernel_target,
        repo=repo,
        output_dir=preprocess_output_dir,
        gpu_id=parsed_gpu_ids[0] if parsed_gpu_ids else 0,
        model_factory=lambda: get_model(model_name, config.get("model", {})),
        console=console,
    )

    if harness_spec:
        try:
            preprocess_ctx = run_preprocessor(**_preprocess_kwargs, harness=harness_spec)
        except RuntimeError as exc:
            if "harness" in str(exc).lower():
                logger.warning(
                    "[yellow]Harness validation failed, falling back to eval_command: %s[/yellow]", exc
                )
                preprocess_ctx = run_preprocessor(**_preprocess_kwargs, eval_command=test_command)
            else:
                raise
    else:
        if isinstance(test_command, str) and "&&" in test_command:
            left, right = test_command.rsplit("&&", 1)
            correctness_command = left.strip() or None
            performance_command = right.strip() or None
            preprocess_ctx = run_preprocessor(
                **_preprocess_kwargs,
                correctness_command=correctness_command,
                performance_command=performance_command,
            )
        else:
            preprocess_ctx = run_preprocessor(**_preprocess_kwargs, eval_command=test_command)

    if preprocess_ctx.get("test_command") and not test_command:
        test_command = preprocess_ctx["test_command"]
    if preprocess_ctx.get("repo_root") and repo is None:
        repo = Path(preprocess_ctx["repo_root"])

    # kernel_type routing:
    # - hip/other -> homogeneous agent
    # - triton -> heterogeneous orchestrator
    # Auto-detect kernel type if heterogeneous flag was not set by LLM extraction or task parser
    if heterogeneous is None:
        _discovery = preprocess_ctx.get("discovery") or {}
        _kernel_info = _discovery.get("kernel") or {}
        _auto_kernel_type = _kernel_info.get("type")

        if (not _auto_kernel_type or _auto_kernel_type == "unknown") and preprocess_ctx.get("kernel_path"):
            from minisweagent.agents.heterogeneous.task_generator import _infer_kernel_type
            _auto_kernel_type = _infer_kernel_type(Path(preprocess_ctx["kernel_path"]))

        if _auto_kernel_type == "triton":
            heterogeneous = True
        else:
            heterogeneous = False
    else:
        pass

    if heterogeneous:
        commandment = preprocess_ctx.get("commandment")
        if commandment:
            task_content = f"{commandment}\n\n---\n\n{task_content}"
        report = run_orchestrator(
            preprocess_ctx=preprocess_ctx,
            gpu_ids=parsed_gpu_ids,
            model=model,
            model_factory=lambda: get_model(model_name, config.get("model", {})),
            output_dir=preprocess_output_dir,
            max_rounds=max_rounds or config.get("orchestrator", {}).get("max_rounds"),
            heterogeneous=True,
        )
        return _final_report_to_bestpatchresult(report)

    agent_config = dict(config.get("agent", {}))
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

    return run_homogeneous_agent(
        config=config,
        task_content=task_content,
        model=model,
        env=env,
        env_class=env.__class__,
        env_kwargs=_env_kwargs,
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
