#!/usr/bin/env python3

"""Backup mini entry with kernel-type routing."""

import json
import logging
import os
import shlex
import signal
import subprocess
import sys
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
from minisweagent.run.pipeline_helpers import resolve_max_rounds
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

After a run completes, two post-processing steps run by default: the winning
patch is applied and committed to [bold]--repo[/bold], and per-run artifacts are
pruned to [bold]final_report.json[/bold], the winning [bold].diff[/bold],
[bold]geak_agent.log[/bold], and [bold]COMMANDMENT.md[/bold].

Use [bold green]--debug[/bold green] to disable both steps (no patch apply, no
artifact cleanup) so that the full run directory is preserved for inspection.
Hard-kill (wall-clock timeout) always leaves artifacts in place regardless.
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
    total_budget_s: float = typer.Option(
        7200,
        "--total-budget-s",
        help="Total wall-clock budget in seconds. Default: 7200 (2 hours).",
    ),
    max_rounds: int | None = typer.Option(
        None,
        "--max-rounds",
        help="Maximum number of optimization rounds. Default: 3. Env: GEAK_MAX_ROUNDS.",
    ),
    pipeline_mode: str | None = typer.Option(
        None,
        "--pipeline-mode",
        help="Dispatch mode: fixed | planned | mixed. Overrides GEAK_PIPELINE_MODE env.",
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

    # Derive apply_best_patch / cleanup from --debug
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
    _task_max_rounds: int | None = None
    if task_content:
        from minisweagent.run.utils.task_parser import parse_pipeline_params

        logger.info("[bold cyan]Checking task for pipeline parameters...[/bold cyan]")
        pipeline_params = parse_pipeline_params(task_content, model)
        logger.debug("pipeline_params: %s", pipeline_params)

        if pipeline_params.get("max_rounds") is not None:
            _task_max_rounds = pipeline_params["max_rounds"]
            logger.info("Task-extracted max rounds: %s.", _task_max_rounds)

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

    # Use task-extracted max_rounds as fallback when CLI --max-rounds not set
    if max_rounds is None and _task_max_rounds is not None:
        max_rounds = _task_max_rounds

    logger.info("[bold cyan]Budget: %ds, max_rounds: %s[/bold cyan]", total_budget_s, max_rounds or 3)

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
    if not kernel_target:
        logger.error(
            "[red]Error: missing kernel target. Provide --kernel-url or include kernel info in task.[/red]"
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
    _spec = BudgetSpec(
        total_s=float(total_budget_s),
        preprocess_soft_cap_s=total_budget_s * 0.125,       # 12.5% (was 900/7200)
        preprocess_hard_cap_fraction=0.5,                    # already a fraction
        finalize_grace_s=total_budget_s * 0.042,             # ~4.2% (was 300/7200)
        kill_buffer_s=total_budget_s * 0.05,                 # 5%   (was 360/7200)
    )
    budget_mgr = RunBudget(spec=_spec)
    for _line in budget_mgr.banner_lines():
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
            budget_mgr.soft_stop.set()
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
            budget=budget_mgr,
            state=state,
            user_task=task_content,
            scoring_target=scoring_target_norm,
        )
        logger.debug("Preprocess kwargs: %s", _preprocess_kwargs)

        # Schedule preprocess watchdogs that reach into ``state`` to apply the
        # stage-aware soft/hard policy. Cancelled in the finally block.
        budget_mgr.schedule_preprocess_watchdogs(
            on_soft=lambda: preprocess_soft_stop_handler(
                state,
                soft_cap_s=budget_mgr.spec.preprocess_soft_cap_s,
                hard_cap_s=budget_mgr.spec.preprocess_hard_cap_s,
                console=console,
            ),
            on_hard=lambda: preprocess_hard_stop_handler(
                state,
                hard_cap_s=budget_mgr.spec.preprocess_hard_cap_s,
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
        budget_mgr.cancel_preprocess_watchdogs()
        _T_pp_end_elapsed = budget_mgr.elapsed()
        opt_deadline = budget_mgr.commit_preprocess(_T_pp_end_elapsed)
        budget_mgr.schedule_optimization_watchdog()

        # Hard-kill backstop: if cooperative shutdown stalls (e.g. a sub-agent
        # is mid-subprocess.run with no internal soft_stop poll), forcibly
        # terminate the registry and ``os._exit`` at ``started_at + total_s``
        # (the absolute wall-clock cap). This is the only thing that
        # guarantees the run exits within budget regardless of where the
        # stall is. Cleanup is intentionally NOT invoked here: the
        # per-run dir is preserved for forensic analysis of WHY the
        # watchdog fired.
        def _hard_kill_handler() -> None:
            logger.error(
                "[budget] HARD KILL: started_at + total_s reached; terminating registry and exiting",
            )
            # Try auto_finalize to produce a real final_report from existing
            # round results (pure I/O, no LLM, takes milliseconds). Fall back
            # to a minimal stub if auto_finalize fails for any reason.
            try:
                from minisweagent.run.postprocess.results import auto_finalize
                _ctx = {"output_dir": str(preprocess_output_dir)}
                report = auto_finalize(_ctx)
                report["status"] = "hard_kill"
                report["exit_code"] = 124
                report["elapsed_s"] = round(budget_mgr.elapsed(), 3)
                (preprocess_output_dir / "final_report.json").write_text(
                    json.dumps(report, indent=2, default=str))
            except Exception:
                logger.exception("hard-kill: auto_finalize failed, writing stub")
                try:
                    _stub_path = preprocess_output_dir / "final_report.json"
                    if not _stub_path.exists():
                        _stub_path.write_text(
                            json.dumps(
                                {
                                    "status": "hard_kill",
                                    "exit_code": 124,
                                    "elapsed_s": round(budget_mgr.elapsed(), 3),
                                    "reason": "started_at + total_s reached",
                                },
                                indent=2,
                            )
                        )
                except Exception:
                    pass
            try:
                state.registry.terminate_all(escalate_after_s=5.0)
            except Exception:
                logger.exception("hard-kill: registry.terminate_all() failed")

            # Loud user-facing warning identifying the artifact path and the
            # fact that cleanup did NOT run. Operators reading a CI tail or
            # subprocess.run capture can't miss it.
            _msg = (
                f"[geak HARD-KILL] Wall-clock budget exceeded "
                f"(elapsed={budget_mgr.elapsed():.0f}s, budget={budget_mgr.spec.total_s:.0f}s). "
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

        budget_mgr.schedule_optimization_hard_kill_watchdog(_hard_kill_handler)

        logger.info(
            "[budget] preprocess finished at +%.0fs; opt_deadline @+%.0fs "
            "(softstop_at @+%.0fs, hard_kill @+%.0fs)",
            _T_pp_end_elapsed,
            _T_pp_end_elapsed + opt_deadline.remaining(),
            _T_pp_end_elapsed + max(0.0, opt_deadline.remaining() - budget_mgr.spec.finalize_grace_s),
            budget_mgr.spec.total_s,
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
            "[budget] max_rounds=%d (source=%s)",
            _resolved_max_rounds,
            _max_rounds_source,
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

        _valid_modes = {"fixed", "planned", "mixed"}
        if pipeline_mode is not None and pipeline_mode in _valid_modes:
            _mode_source = "cli"
        elif (_env_mode := os.environ.get("GEAK_PIPELINE_MODE")) in _valid_modes:
            pipeline_mode, _mode_source = _env_mode, "env"
        elif (_yaml_mode := (config.get("pipeline") or {}).get("mode")) in _valid_modes:
            pipeline_mode, _mode_source = _yaml_mode, "yaml"
        else:
            pipeline_mode, _mode_source = "mixed", "default"
        logger.info("Running unified pipeline mode: %s (source=%s)", pipeline_mode, _mode_source)
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
                soft_stop=budget_mgr.soft_stop,
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
        budget_mgr.cancel_all_timers()
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
            if _apply_status == "skipped_dirty":
                console.print(
                    "[bold yellow][geak apply] Skipped: --repo has uncommitted tracked changes. "
                    "Commit/stash and re-run apply manually.[/bold yellow]"
                )
            elif _apply_status in {"apply_failed", "commit_failed"}:
                console.print(
                    f"[bold yellow][geak apply] {_apply_status}: "
                    f"{outcome.get('reason', '')}[/bold yellow]"
                )



if __name__ == "__main__":
    app()
