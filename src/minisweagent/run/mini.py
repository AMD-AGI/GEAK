#!/usr/bin/env python3

"""Backup mini entry with kernel-type routing."""

import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import typer
import yaml
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents.parallel_agent import BestPatchResult
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment_class
from minisweagent.models import get_model
from minisweagent.run.budget import BudgetSpec, RunBudget
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.run.pipeline_helpers import (
    DEFAULT_RUN_MODE,
    RUN_MODES,
    apply_mode_presets,
    resolve_max_rounds,
)
from minisweagent.run.preprocess_v3.adapter import run_preprocess_v3 as run_preprocessor
from minisweagent.run.state import (
    PreprocessState,
    preprocess_hard_stop_handler,
    preprocess_soft_stop_handler,
)
from minisweagent.run.utils.task_parser import (
    _resolve_path_case,
    display_parsed_config,
    extract_user_constraints,
    parse_task_info,
)
from minisweagent.utils.log import DEFAULT_LOG_FILENAME, add_file_handler

logger = logging.getLogger(__name__)
console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")
prompt_session = PromptSession(history=FileHistory(global_config_dir / "mini_task_history.txt"))


def _parse_gpu_ids(gpu_ids_str: str | None) -> list[int]:
    if not gpu_ids_str:
        return [0]
    return [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]


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
    if text == "pytorch2flydsl":
        return "pytorch2flydsl"
    if text == "flydsl":
        return "flydsl"
    return "other"


def _derive_output_dir(output: Path | None, kernel_name: str | None) -> tuple[Path, bool]:
    """Derive the output directory from ``-o``/``--output``.

    Returns ``(path, auto)``. ``auto`` is True iff ``output`` was ``None`` and
    geak generated the path under ``<cwd>/optimization_logs/``.

    - If output is a file path: output_dir = output.parent (auto=False)
    - If output is a directory: output_dir = output (auto=False)
    - If output is not provided: use ./optimization_logs/<kernel_name>_<timestamp>
      (auto=True)

    The returned path is always absolute. Several preprocess helpers
    (notably ``_resolve_deterministic_harness`` and the merged-file split
    helpers in ``preprocess/harness_utils.py``) interpret a relative path
    in their inputs as relative to ``repo_root``; if we let a relative
    ``output_dir`` flow through, harness paths the split helper writes
    would later be resolved against the wrong root and the run would fail
    with "Deterministic harness file not found". This invariant was
    originally added in ``3b0ff0ac`` ("fix(preprocess): resolve output_dir
    to absolute and exclude preprocessor artifacts from patches"), then
    silently reverted by ``c07285cc`` (a RAG refactor whose "remove dead
    code" line item caught this). Keep the ``.resolve()`` calls below.
    """
    if output is None:
        from minisweagent.run.utils.task_parser import generate_patch_output_dir

        return (Path.cwd() / Path(generate_patch_output_dir(kernel_name))).resolve(), True

    if output.suffix:
        return output.parent.resolve(), False

    return output.resolve(), False


def _final_report_to_bestpatchresult(report: Any) -> BestPatchResult | None:
    if report is None:
        return None
    report_dict = report.to_dict() if hasattr(report, "to_dict") else report
    if not isinstance(report_dict, dict):
        return None

    best_patch = report_dict.get("best_patch")
    patch_path = Path(best_patch) if best_patch else None
    raw_speedup = report_dict.get("best_speedup")
    return BestPatchResult(
        agent_id=0,
        patch_id=patch_path.stem if patch_path else "unknown",
        test_output="",
        best_speedup=float(raw_speedup) if raw_speedup is not None else None,
        best_patch_file=str(patch_path) if patch_path else None,
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

After a run completes, two post-processing steps run by default (disabled
when [bold green]--debug[/bold green] is set):

- Patch apply: applies the winning patch to [bold]--repo[/bold] on the
  current branch and commits it.
- Cleanup: prunes per-run artifacts to [bold]final_report.json[/bold], the
  winning [bold].diff[/bold], [bold]geak_agent.log[/bold], and
  [bold]COMMANDMENT.md[/bold].

Hard-kill (wall-clock timeout) leaves artifacts in place for forensic analysis
regardless of debug mode.
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
    preprocess_only: bool = typer.Option(False, "--preprocess-only", help="Run preprocessing only; skip the optimization round loop and exit after preprocess artifacts are written.", rich_help_panel="Advanced"),
    repo: Path | None = typer.Option(None, "--repo", help="Target Repository path."),
    kernel_url: str | None = typer.Option(None, "--kernel-url", "--kernel-path", help="Target kernel source (path or URL)."),
    num_parallel: int | None = typer.Option(None, "--num-parallel", help="Number of parallel patch agents."),
    gpu_ids: str | None = typer.Option(None, "--gpu-ids", help="Comma-separated GPU IDs."),
    test_command: str | None = typer.Option(None, "--test_command", "--test-command", help="Test command"),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Wall-clock budget profile: 'quick' (~1h) or 'full' (~2h). Default: from geak.yaml run.mode.",
    ),
    total_budget_s: float | None = typer.Option(
        None,
        "--total-budget-s",
        help="Override the mode's total wall-clock budget (seconds). Escape hatch for testing.",
    ),
    scoring_target: str = typer.Option(
        "wall",
        "--target",
        help=(
            "Which signal the harness reports as GEAK_RESULT_LATENCY_MS (the scoring metric "
            "the agent optimizes against). 'wall' = end-to-end host latency via "
            "triton.testing.do_bench (includes Python/dispatch overhead). 'kernel' = GPU-only "
            "kernel time via torch.profiler CUDA events (excludes dispatch). The dual-signal "
            "harness always reports BOTH (GEAK_RESULT_WALL_MS + GEAK_RESULT_KERNEL_MS) for "
            "agent visibility; --target only chooses which becomes the scoring signal."
        ),
    ),
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help=(
            "Debug mode: disables post-run patch apply and artifact cleanup so the "
            "full run directory is preserved for inspection. Default: off."
        ),
    ),
):
    # fmt: on
    del visual

    # --debug disables post-run patch apply and artifact cleanup
    apply_best_patch = not debug
    cleanup = not debug

    # Only run interactive first-time setup when stdin is a tty. Without the
    # guard, a CI / scripted run without ``MSWEA_CONFIGURED`` or any API key
    # env var would block on ``prompt(...)`` inside ``setup()``. The guard
    # was originally present and was removed by ``c07285cc``; restoring it.
    if sys.stdin.isatty():
        configure_if_first_time()

    # 1) Config merge — explicit UTF-8 avoids locale-dependent decoding for YAML on some platforms
    base_config_path = builtin_config_dir / "mini_kernel_strategy_list.yaml"
    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    if config:
        logger.info("Loaded base config from [bold green]'%s'[/bold green]", base_config_path.name)
    else:
        logger.warning(
            "Base config %s: null or empty YAML file.",
            base_config_path.name,
        )

    config_path = config_spec or (builtin_config_dir / "geak.yaml")
    user_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if user_config:
        logger.info(
            "Loaded user config from [bold green]'%s'[/bold green] [dim](final override)[/dim]",
            config_path.name,
        )
    else:
        logger.warning(
            "User config %s: null or empty YAML file.",
            config_path.name,
        )

    config = _deep_merge(config, user_config)

    # Validate --mode early but defer the full precedence resolution + preset
    # injection until AFTER parse_pipeline_params runs.
    if mode is not None:
        _normalized_cli_mode = mode.strip().lower()
        if _normalized_cli_mode not in RUN_MODES:
            raise typer.BadParameter(f"--mode must be one of {RUN_MODES}; got {mode!r}")
        mode = _normalized_cli_mode

    if yolo:
        config.setdefault("agent", {})["mode"] = "yolo"
        logger.info("Running in YOLO mode.")
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
        logger.info("Setting cost limit to %s.", cost_limit)
    if exit_immediately:
        config.setdefault("agent", {})["confirm_exit"] = False
        logger.info("Running in exit-immediately mode.")
    if preprocess_only:
        logger.info("Running in preprocess-only mode: round loop will be skipped.")
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
        logger.info("Using model class: %s.", model_class)

    tools_cfg = config.get("tools") or {}
    disabled_tools: list[str] = []
    for tool_name, enabled in tools_cfg.items():
        if enabled is False:
            disabled_tools.append(tool_name)

    # RAG MCP toggle: disable RAG tools when rag is not enabled
    rag_enabled = tools_cfg.get("rag", False)
    if rag_enabled:
        # Auto-install rag-mcp package if missing
        try:
            import rag_mcp  # noqa: F401
        except ImportError:
            logger.info("rag-mcp package not found, installing automatically...")
            _rag_mcp_path = Path(__file__).resolve().parents[3] / "mcp_tools" / "rag-mcp"
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(_rag_mcp_path)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Auto-install of rag-mcp failed.\n\n"
                    f"stderr:\n{result.stderr}\n\n"
                    "Please install manually:\n"
                    f"  pip install -e {_rag_mcp_path}"
                )
            logger.info("rag-mcp installed successfully.")
            # Refresh sys.path so the newly installed package is discoverable
            import importlib
            sys.path.insert(0, str(_rag_mcp_path / "src"))
            importlib.invalidate_caches()
            import rag_mcp  # noqa: F401
        # Auto-build semantic index if missing
        _index_path = Path.home() / ".cache" / "amd-ai-devtool" / "semantic-index"
        _has_faiss = (_index_path / "index.faiss").exists() or (_index_path / "faiss.index").exists()
        _has_pkl = bool(list(_index_path.glob("*.pkl"))) if _index_path.exists() else False
        if not (_has_faiss and _has_pkl):
            logger.info("RAG index not found at %s, building automatically...", _index_path)
            _build_script = Path(__file__).resolve().parents[3] / "scripts" / "build_index.py"
            result = subprocess.run(
                [sys.executable, str(_build_script), "--force"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Auto-build of RAG index failed.\n\n"
                    f"stderr:\n{result.stderr}\n\n"
                    "Please build manually:\n"
                    f"  python {_build_script} --force"
                )
            logger.info("RAG index built successfully.")
    else:
        disabled_tools.append("query")
        disabled_tools.append("optimize")

    if disabled_tools:
        config.setdefault("agent", {}).setdefault("disabled_tools", [])
        config["agent"]["disabled_tools"] = list(set(config["agent"]["disabled_tools"]) | set(disabled_tools))
        os.environ["GEAK_DISABLED_TOOLS"] = ",".join(config["agent"]["disabled_tools"])

    # Propagate use_skills to subagents via environment variable
    if config.get("agent", {}).get("use_skills"):
        os.environ["GEAK_USE_SKILLS"] = "1"

    logger.debug("config: %s", config)

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
            logger.info("Task content is empty.")

    if not task_content:
        console.print("[bold yellow]What do you want to do?")
        logger.info("Prompting user for task input (interactive).")
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
        logger.info("User task input (%d chars): %s", len(task_content), task_content[:500])

    # 2a) LLM-driven pipeline param extraction
    max_rounds = None
    task_extracted_mode: str | None = None
    if task_content:
        from minisweagent.run.utils.task_parser import parse_pipeline_params

        logger.info("[bold cyan]Checking task for pipeline parameters...[/bold cyan]")
        pipeline_params = parse_pipeline_params(task_content, model)
        logger.debug("pipeline_params: %s", pipeline_params)

        if pipeline_params.get("max_rounds") is not None:
            max_rounds = pipeline_params["max_rounds"]
            logger.info("Using max rounds: %s.", max_rounds)
        if pipeline_params.get("mode") is not None:
            task_extracted_mode = pipeline_params["mode"]
            logger.info("Task-extracted run mode: %s.", task_extracted_mode)

        # Prompt for missing required params (kernel_url) — only if not already set
        if kernel_url is None:
            from minisweagent.run.utils.config_editor import prompt_missing_pipeline_params

            logger.info("Prompting for missing pipeline parameters.")
            pipeline_params, should_use_pipeline = prompt_missing_pipeline_params(pipeline_params, console, yolo)
            logger.info("pipeline_params: %s, should_use_pipeline: %s", pipeline_params, should_use_pipeline)

            if should_use_pipeline:
                if pipeline_params.get("kernel_url"):
                    kernel_url = pipeline_params["kernel_url"]
                    logger.info("Using kernel URL: %s.", kernel_url)

    # Finalize mode precedence: CLI --mode > task-extracted mode > YAML
    # run.mode > built-in default. Apply presets exactly once. We do this
    # AFTER parse_pipeline_params so a "quick mode"/"1 hour" phrase in the
    # natural-language task can drive the budget.
    _run_cfg = config.get("run") or {}
    resolved_mode = (mode or task_extracted_mode or _run_cfg.get("mode") or DEFAULT_RUN_MODE).strip().lower()
    if resolved_mode not in RUN_MODES:
        raise typer.BadParameter(f"--mode must be one of {RUN_MODES}; got {resolved_mode!r}")
    _mode_source = "cli" if mode else "task" if task_extracted_mode else "yaml" if _run_cfg.get("mode") else "default"
    logger.info("[bold cyan]Run mode: %s (source=%s)[/bold cyan]", resolved_mode, _mode_source)
    config.setdefault("run", {})["mode"] = resolved_mode
    apply_mode_presets(config, resolved_mode)

    # 2b) Detect configs from task
    parsed_config = parse_task_info(task_content, model)
    kernel_type = _normalize_kernel_type(parsed_config.get("kernel_type"))
    logger.info("Normalized kernel_type from task content: %s", kernel_type)

    if kernel_url:
        kp = Path(kernel_url)
        if kp.exists() and kp.is_file():
            from minisweagent.agents.heterogeneous.task_generator import _infer_kernel_type

            inferred = _normalize_kernel_type(_infer_kernel_type(kp))
            if inferred in {"hip", "triton", "flydsl"}:
                kernel_type = inferred
                logger.info("Updated kernel_type using kernel path: %s", kernel_type)

    if repo is None and parsed_config.get("repo"):
        repo = Path(parsed_config["repo"])
        logger.info("Using repo from task content: %s", repo)
    if test_command is None and parsed_config.get("test_command"):
        test_command = parsed_config["test_command"]
        logger.info("Using test command from task content: %s", test_command)
    if gpu_ids is None and parsed_config.get("gpu_ids") is not None:
        _parsed_gpu_ids = parsed_config["gpu_ids"]
        if isinstance(_parsed_gpu_ids, list):
            gpu_ids = ",".join(map(str, _parsed_gpu_ids))
        else:
            gpu_ids = str(_parsed_gpu_ids)
        if not gpu_ids.strip():
            gpu_ids = None
        logger.info("Using gpu_ids: %s", gpu_ids)

    # Apply config/model/output_dir extracted from task (CLI flags take priority)
    if config_spec is None and parsed_config.get("config"):
        _task_config_path = get_config_path(Path(parsed_config["config"]))
        if _task_config_path and _task_config_path.exists():
            logger.info("[dim]Applying config from task: '%s'[/dim]", _task_config_path)
            _task_yaml = yaml.safe_load(_task_config_path.read_text(encoding="utf-8"))
            _task_user_config = _task_yaml or {}
            if not _task_yaml:
                logger.warning(
                    "Task config %s: null or empty YAML; merge layer uses {}.",
                    _task_config_path,
                )
            config = _deep_merge(config, _task_user_config)

    if model_name is None and parsed_config.get("model"):
        model_name = parsed_config["model"]
        model = get_model(model_name, config.get("model", {}))
        logger.info("Using model (from task): [bold cyan]%s[/bold cyan]", model_name)

    if output is None and parsed_config.get("output_dir"):
        output = Path(parsed_config["output_dir"])
        logger.info("Using output_dir from task content: %s", output)

    kernel_target = kernel_url or parsed_config.get("kernel_url") or parsed_config.get("kernel_name")

    if not kernel_target and repo is None:
        logger.error(
            "[red]Error: missing kernel target. Provide --kernel-url, --repo, or include kernel info in task.[/red]"
        )
        raise typer.Exit(1)

    parsed_gpu_ids = _parse_gpu_ids(gpu_ids)

    # Auto-detect num_parallel from gpu_ids when not explicitly provided.
    if num_parallel is None:
        num_parallel = _as_int(parsed_config.get("num_parallel"))
    if num_parallel is None and parsed_gpu_ids:
        num_parallel = len(parsed_gpu_ids)
        logger.info("Auto-setting num_parallel=%s from gpu_ids.", num_parallel)

    kernel_name_for_output = parsed_config.get("kernel_name")
    if not kernel_name_for_output and kernel_url:
        kernel_name_for_output = Path(kernel_url).stem
    if not kernel_name_for_output and isinstance(kernel_target, str):
        kernel_name_for_output = Path(kernel_target).stem
    if not kernel_name_for_output:
        kernel_name_for_output = "kernel_auto"
    logger.info("Using kernel_name_for_output: %s", kernel_name_for_output)

    preprocess_output_dir, _output_dir_auto = _derive_output_dir(output, kernel_name_for_output)
    preprocess_output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(preprocess_output_dir / DEFAULT_LOG_FILENAME)

    _run_t0 = time.monotonic()
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
    _resolved_config_display = display_parsed_config(_display_cfg, str(preprocess_output_dir))
    logger.info("Resolved configuration:\n%s", _resolved_config_display)

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

    _target_language = "flydsl" if kernel_type in {"pytorch2flydsl", "flydsl"} else None
    scoring_target_norm = (scoring_target or "wall").strip().lower()
    if scoring_target_norm not in {"wall", "kernel"}:
        logger.warning(
            "Unknown --target=%s, falling back to 'wall'. Valid: wall|kernel.",
            scoring_target,
        )
        scoring_target_norm = "wall"
    logger.info("Scoring target: %s (GEAK_RESULT_LATENCY_MS = %s_ms)", scoring_target_norm, scoring_target_norm)

    # ── Build RunBudget from mode + CLI overrides ────────────────────
    _budget_cfg = (config.get("run") or {}).get("budgets", {}).get(resolved_mode) or {}
    if not _budget_cfg:
        raise typer.BadParameter(
            f"No run.budgets.{resolved_mode} block in config; check geak.yaml"
        )
    _spec = BudgetSpec(
        mode=resolved_mode,  # type: ignore[arg-type]
        total_s=float(total_budget_s if total_budget_s is not None else _budget_cfg["total_s"]),
        preprocess_soft_cap_s=float(_budget_cfg["preprocess_soft_cap_s"]),
        preprocess_hard_cap_fraction=float(_budget_cfg["preprocess_hard_cap_fraction"]),
        finalize_grace_s=float(_budget_cfg["finalize_grace_s"]),
        kill_buffer_s=float(_budget_cfg.get("kill_buffer_s", 60.0)),
    )
    budget = RunBudget(spec=_spec)
    for _line in budget.banner_lines():
        console.print(f"[bold cyan]{_line}[/bold cyan]")
        logger.info(_line)

    # ── SIGINT handler: first Ctrl-C -> SoftStop; second Ctrl-C -> ──
    # ── force-terminate registry and re-raise.                     ──
    state = PreprocessState(output_dir=preprocess_output_dir)
    _sigint_count = {"n": 0}
    _orig_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(_signum, _frame):  # noqa: ANN001
        _sigint_count["n"] += 1
        if _sigint_count["n"] == 1:
            logger.warning("SIGINT received -- flipping SoftStop (graceful finalize). Press Ctrl-C again to force.")
            budget.soft_stop.set()
        else:
            logger.error("Second SIGINT received -- terminating tracked subprocesses and exiting.")
            signal.signal(signal.SIGINT, _orig_sigint or signal.SIG_DFL)
            try:
                state.registry.terminate_all()
            except Exception:
                logger.exception("registry.terminate_all() failed during SIGINT")
            raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _sigint_handler)

    # State the outer ``finally`` will read. Declare here so the finally
    # can't NameError on partial failure between this point and the first
    # branch assignment.
    result: BestPatchResult | None = None
    repo_path: Path | None = None
    effective_repo: Path | None = repo
    _run_succeeded = False

    try:
        _preprocess_kwargs = dict(
            kernel_url=kernel_target,
            repo=repo,
            output_dir=preprocess_output_dir,
            gpu_id=parsed_gpu_ids[0] if parsed_gpu_ids else 0,
            model_factory=lambda: get_model(model_name, config.get("model", {})),
            console=console,
            target_language=_target_language,
            budget=budget,
            state=state,
            user_task=task_content,
            scoring_target=scoring_target_norm,
        )
        logger.debug("Preprocess kwargs: %s", _preprocess_kwargs)

        # Schedule preprocess watchdogs that reach into ``state`` to apply the
        # stage-aware soft/hard policy. Cancelled in the finally block.
        budget.schedule_preprocess_watchdogs(
            on_soft=lambda: preprocess_soft_stop_handler(
                state,
                soft_cap_s=budget.spec.preprocess_soft_cap_s,
                hard_cap_s=budget.spec.preprocess_hard_cap_s,
                console=console,
            ),
            on_hard=lambda: preprocess_hard_stop_handler(
                state,
                hard_cap_s=budget.spec.preprocess_hard_cap_s,
                console=console,
            ),
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
                logger.info(
                    "Correctness command: %s, Performance command: %s",
                    correctness_command,
                    performance_command,
                )
                preprocess_ctx = run_preprocessor(
                    **_preprocess_kwargs,
                    correctness_command=correctness_command,
                    performance_command=performance_command,
                )
            else:
                preprocess_ctx = run_preprocessor(**_preprocess_kwargs, eval_command=test_command)
        logger.debug("Preprocessor context: %s", preprocess_ctx)

        # Cancel preprocess watchdogs and transition phase before optimization.
        budget.cancel_preprocess_watchdogs()
        _T_pp_end_elapsed = budget.elapsed()
        opt_deadline = budget.commit_preprocess(_T_pp_end_elapsed)
        budget.schedule_optimization_watchdog()

        # Hard-kill backstop: if cooperative shutdown stalls (e.g. a sub-agent
        # is mid-subprocess.run with no internal soft_stop poll), forcibly
        # terminate the registry and ``os._exit`` at ``started_at + total_s``
        # (the absolute wall-clock cap). This is the only thing that
        # guarantees the run exits within budget regardless of where the
        # stall is. Cleanup is intentionally NOT invoked here: the
        # per-run dir is preserved for forensic analysis of WHY the
        # watchdog fired.
        _HARD_KILL_SELECT_PATCH_TIMEOUT_S = 300  # 5 minutes for select_patch before os._exit

        def _hard_kill_select_patch(output_dir: Path) -> None:
            """Best-effort select_patch + auto_finalize during hard-kill.

            Runs in a daemon thread so the main hard-kill path can enforce a
            wall-clock cap on it.  Two phases:

            1. Run the LLM ``SelectPatchAgent`` to fill in any missing
               per-task ``best_results.json`` files (non-fatal if it fails).
            2. Call ``auto_finalize`` — the same canonical path used by
               normal completion — to write a complete ``final_report.json``.
            """
            # Phase 1: best-effort LLM select_patch (non-fatal)
            try:
                from minisweagent.agents.select_patch_agent import SelectPatchAgent
                from minisweagent.config import load_agent_config
                from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig

                results_dir = output_dir / "results"
                if not results_dir.is_dir():
                    logger.warning("hard-kill select_patch: no results/ dir; skipping agent")
                else:
                    task_dirs = sorted({
                        p.parent for p in results_dir.glob("round_*/*/best_results.json")
                    })
                    if not task_dirs:
                        task_dirs = sorted({
                            d for d in results_dir.glob("round_*/*")
                            if d.is_dir() and d.name != "worktrees"
                        })
                    if not task_dirs:
                        logger.warning("hard-kill select_patch: no task dirs found; skipping agent")
                    else:
                        metric = config.get("patch", {}).get("metric")
                        model = get_model(model_name, config.get("model", {}))
                        agent_config, _ = load_agent_config("mini_select_patch")

                        env_config = LocalEnvironmentConfig(cwd=str(results_dir))
                        env = LocalEnvironment(**env_config.__dict__)

                        agent = SelectPatchAgent(model, env, **agent_config)
                        agent.log_file = results_dir / "hard_kill_select_agent.log"
                        agent.patch_dir = results_dir

                        metric_section = metric if metric else "None"
                        dir_listing = "\n".join(f"  - {d}" for d in task_dirs)
                        task = (
                            f"\n## User-provided metric\n{metric_section}\n\n"
                            f"## Inputs\n"
                            f"- Work directory (absolute): {results_dir}\n"
                            f"- This is a HARD-KILL selection: the run hit the absolute wall-clock cap.\n"
                            f"- Results are organized under round_*/ subdirectories.\n"
                            f"  Each subdirectory may contain:\n"
                            f"  - patch_*.patch files\n"
                            f"  - patch_*_test.txt test output logs\n"
                            f"  - best_results.json (per-task selection by previous agents)\n"
                            f"- Scan ALL directories below to find the best patch across all rounds.\n"
                            f"- Use patch_0_test.txt from any directory as baseline "
                            f"(patch_0 = original unmodified kernel).\n"
                            f"- Found {len(task_dirs)} task directories:\n{dir_listing}\n"
                        )

                        logger.info(
                            "hard-kill select_patch: running on %d task dirs in %s",
                            len(task_dirs), results_dir,
                        )
                        agent.run(task)
                        best_patch_id = agent.extract_final_result()
                        if best_patch_id:
                            logger.info("hard-kill select_patch: chose %s", best_patch_id)
                        else:
                            logger.warning("hard-kill select_patch: agent did not produce a result")
            except Exception:
                logger.exception("hard-kill select_patch agent failed (non-fatal)")

            # Phase 2: auto_finalize — same path as normal completion
            try:
                from minisweagent.run.postprocess.results import auto_finalize

                _ctx = {"output_dir": str(output_dir)}
                auto_finalize(_ctx)

                # Stamp hard-kill metadata onto the report written by auto_finalize
                report_path = output_dir / "final_report.json"
                if report_path.is_file():
                    final = json.loads(report_path.read_text())
                    final["status"] = "hard_kill_auto_finalized"
                    final["exit_code"] = 124
                    final["elapsed_s"] = round(budget.elapsed(), 3)
                    report_path.write_text(json.dumps(final, indent=2, default=str))
                    logger.info(
                        "hard-kill: wrote final_report.json via auto_finalize "
                        "(best_speedup=%s)",
                        final.get("best_speedup"),
                    )
            except Exception:
                logger.exception("hard-kill auto_finalize failed (non-fatal)")

        def _hard_kill_handler() -> None:
            logger.error(
                "[budget] HARD KILL: started_at + total_s reached; terminating registry and exiting",
            )
            # Terminate all tracked subprocesses first so GPU resources are freed.
            try:
                state.registry.terminate_all(escalate_after_s=5.0)
            except Exception:
                logger.exception("hard-kill: registry.terminate_all() failed")

            # Best-effort: run select_patch agent to find the best result
            # across all completed rounds. Runs in a daemon thread with a
            # 5-minute timeout so it cannot block os._exit indefinitely.
            _report_path = preprocess_output_dir / "final_report.json"
            if not _report_path.exists():
                # Write a stub immediately so there's always *something*
                try:
                    _report_path.write_text(
                        json.dumps(
                            {
                                "status": "hard_kill",
                                "exit_code": 124,
                                "elapsed_s": round(budget.elapsed(), 3),
                                "reason": "started_at + total_s reached",
                            },
                            indent=2,
                        )
                    )
                except Exception:
                    logger.exception("hard-kill: writing stub final_report.json failed (non-fatal)")

            select_thread = threading.Thread(
                target=_hard_kill_select_patch,
                args=(preprocess_output_dir,),
                daemon=True,
                name="geak-hard-kill-select-patch",
            )
            select_thread.start()
            select_thread.join(timeout=_HARD_KILL_SELECT_PATCH_TIMEOUT_S)

            if select_thread.is_alive():
                logger.warning(
                    "hard-kill select_patch timed out after %ds; proceeding with os._exit",
                    _HARD_KILL_SELECT_PATCH_TIMEOUT_S,
                )

            # Loud user-facing warning identifying the artifact path and the
            # fact that cleanup did NOT run. Operators reading a CI tail or
            # subprocess.run capture can't miss it.
            _msg = (
                f"[geak HARD-KILL] Wall-clock budget exceeded "
                f"(elapsed={budget.elapsed():.0f}s, budget={budget.spec.total_s:.0f}s). "
                f"Per-run artifacts at {preprocess_output_dir} were PRESERVED for forensics. "
                f"Cleanup did NOT run; inspect and prune manually when done."
            )
            logger.warning(_msg)
            try:
                console.print(f"[bold red]{_msg}[/bold red]")
            except Exception:
                pass  # never block os._exit on a console-render error

            # 124 is the conventional exit code for a wall-clock timeout
            # (matches GNU ``timeout``). Use ``os._exit`` (not sys.exit) to
            # bypass any atexit/cleanup that might block.
            os._exit(124)

        budget.schedule_optimization_hard_kill_watchdog(_hard_kill_handler)

        logger.info(
            "[budget] preprocess finished at +%.0fs; opt_deadline @+%.0fs "
            "(softstop_at @+%.0fs, hard_kill @+%.0fs)",
            _T_pp_end_elapsed,
            _T_pp_end_elapsed + opt_deadline.remaining(),
            _T_pp_end_elapsed + max(0.0, opt_deadline.remaining() - budget.spec.finalize_grace_s),
            budget.spec.total_s,
        )

        if preprocess_ctx.get("test_command") and not test_command:
            test_command = preprocess_ctx["test_command"]
        if preprocess_ctx.get("repo_root") and repo is None:
            repo = Path(preprocess_ctx["repo_root"])

        # Resolve max_rounds via the documented precedence chain:
        # CLI --max-rounds (if any future flag added) > config (mode preset) >
        # GEAK_MAX_ROUNDS env > default. ``max_rounds`` from task parsing acts as
        # an additional override (parsed via parse_pipeline_params at the top of
        # main()), so we honor it when set.
        _resolved_max_rounds, _max_rounds_source = resolve_max_rounds(
            cli_max_rounds=max_rounds,
            config=config,
        )
        logger.info(
            "[budget] max_rounds=%d (source=%s; mode=%s)",
            _resolved_max_rounds,
            _max_rounds_source,
            resolved_mode,
        )

        preprocess_ctx["user_instructions"] = task_content
        preprocess_ctx["rag_enabled"] = rag_enabled

        extracted = extract_user_constraints(task_content, model)
        extra_addenda: list[str] = []
        if extracted["constraints"]:
            block = ["## USER-SPECIFIED CONSTRAINTS\n\nThese are mandatory. Violation means rejection.\n"]
            block.extend(f"- {c}" for c in extracted["constraints"])
            extra_addenda.append("\n".join(block))
        if extracted["directives"]:
            block = [
                "## PRESCRIBED OPTIMIZATION DIRECTIVES\n\n"
                "These are the user's prescribed optimization strategies. Prioritize them, but\n"
                "also explore additional directions beyond these.\n"
                "NOTE: Any performance numbers in the original user request come from full-model\n"
                "profiling under different conditions. Use ONLY the GEAK-measured baseline metrics\n"
                "for before/after speedup comparisons.\n"
            ]
            block.extend(f"- {d}" for d in extracted["directives"])
            extra_addenda.append("\n".join(block))

        metric = parsed_config.get("metric") or config.get("patch", {}).get("metric")
        logger.info("Using metric: %s", metric)

        repo_path = repo or config.get("patch", {}).get("repo")
        if repo_path:
            p = Path(repo_path)
            if not p.exists():
                resolved = _resolve_path_case(p)
                if resolved is not None:
                    p = resolved
            repo_path = p.resolve()
        logger.info("Resolved repo path: %s", repo_path)
        effective_repo = repo_path

        from minisweagent.run.unified import PipelineContext, run_pipeline

        pipeline_mode = "mixed"
        logger.info("Running unified pipeline mode: %s", pipeline_mode)
        pipeline_result = run_pipeline(
            PipelineContext(
                preprocess_ctx=preprocess_ctx,
                user_prompt=task_content,
                kernel_language=_normalize_kernel_type(preprocess_ctx.get("kernel_type")),
                output_dir=preprocess_output_dir,
                gpu_ids=parsed_gpu_ids,
                model=model,
                model_factory=lambda: get_model(model_name, config.get("model", {})),
                config=config,
                max_rounds=_resolved_max_rounds,
                env=env,
                env_class=env.__class__,
                env_kwargs=_env_kwargs,
                repo=repo_path,
                test_command=test_command or config.get("patch", {}).get("test_command"),
                metric=metric,
                rag_enabled=rag_enabled,
                extra_addenda=extra_addenda,
                num_parallel=num_parallel,
                model_name=model_name,
                console=console,
                deadline=opt_deadline,
                soft_stop=budget.soft_stop,
                registry=state.registry,
                preprocess_only=preprocess_only,
            ),
            mode=pipeline_mode,
        )
        # Flip success flag immediately after the agent returns, before the
        # time.monotonic() log line (which can't realistically raise, but
        # consistency with the het branch is cheap).
        _run_succeeded = True
        logger.info("Run completed in %.0fs.", time.monotonic() - _run_t0)
        result = _final_report_to_bestpatchresult(pipeline_result)
        return result  # noqa: RET504 – result read by finally block
    finally:
        # Ordering matters:
        # 1. Cancel timers FIRST so the hard-kill watchdog can't race the
        #    rest of the finally with an os._exit mid-rmtree.
        # 2. Terminate the registry next so any tracked subprocess is dead
        #    before we touch its working directory.
        # 3. Restore SIGINT before cleanup so a Ctrl-C during prune lands
        #    on the default handler (clean KeyboardInterrupt out).
        # 4. Then run finalize + retention. Both are broad-except wrapped
        #    so a hook failure can't mask the original exception.
        budget.cancel_all_timers()
        try:
            state.registry.terminate_all()
        except Exception:
            logger.exception("registry.terminate_all() failed during run cleanup")
        try:
            signal.signal(signal.SIGINT, _orig_sigint or signal.SIG_DFL)
        except Exception:
            logger.exception("restoring SIGINT handler failed")

        logger.info("[geak --cleanup] starting")
        outcome: dict | None = None
        try:
            from minisweagent.run.postprocess.finalize_apply import finalize_apply_and_cleanup

            outcome = finalize_apply_and_cleanup(
                result,
                effective_repo,
                preprocess_output_dir,
                apply_best_patch=apply_best_patch and _run_succeeded,
                cleanup=cleanup,
            )
        except Exception:
            logger.exception("finalize_apply_and_cleanup raised (non-fatal)")
        logger.info(
            "[geak --cleanup] completed: %s",
            (outcome or {}).get("cleanup_status", "unknown"),
        )

        if outcome:
            _apply_status = outcome.get("apply_status")
            if _apply_status in {"apply_failed", "commit_failed"}:
                console.print(
                    f"[bold yellow][geak apply] {_apply_status}: "
                    f"{outcome.get('reason', '')}[/bold yellow]"
                )


if __name__ == "__main__":
    app()
