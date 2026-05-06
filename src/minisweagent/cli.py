#!/usr/bin/env python3

"""The single GEAK CLI entry point.

Exposes ``geak -t "<prompt>"`` and delegates every execution mode
(``fixed`` / ``planned`` / ``mixed``) through
``run/unified.py::run_pipeline(ctx, mode)``.  Discovery, harness building,
baseline measurement, and optimization are all reachable from here; the
underlying modules remain callable programmatically.

Translation (source→target language) is NOT a run_pipeline mode.  It is
a conditional preprocess phase triggered by ``--target-language`` (or the
LLM-extracted ``target_language`` field); the phase runs before
run_pipeline, uses a standalone ``TranslationAgent`` (not
``OptimizationAgent``), and hands a translated kernel back for the rest
of the pipeline to optimize.  Implementation lands in the preprocessing
refactor PR.
"""

import logging
import shlex
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
from minisweagent.run.utils.gpu_ids import parse_gpu_ids
from minisweagent.agents.parallel_agent import BestPatchResult
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment_class
from minisweagent.models import get_model
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.run.preprocess.orchestrator import (
    run_preprocessor_via_orchestrator as run_preprocessor,
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

_ROOT_TYPER_COMMANDS = frozenset({"translate", "main"})


def _typer_insert_main_subcommand(argv: list[str]) -> None:
    """Make ``geak -t …`` behave like GEAK_main (``mini.py``).

    Typer registers the default optimize flow as the ``main`` subcommand.
    Without inserting ``main``, users must type ``geak main -t``.  Mirror
    GEAK_main parity drivers by rewriting ``argv`` so root-level flags dispatch
    to ``main``.
    """
    if len(argv) < 2:
        return
    first = argv[1]
    if first in _ROOT_TYPER_COMMANDS:
        return
    argv.insert(1, "main")


def cli_entry() -> None:
    """ setuptools / ``pip install`` console_scripts entry — same as GEAK_main ``geak -t``. """
    _typer_insert_main_subcommand(sys.argv)
    app()


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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


def _derive_output_dir(output: Path | None, kernel_name: str | None) -> Path:
    """Derive the output directory from ``-o``/``--output``.

    - If output is a file path: output_dir = output.parent
    - If output is a directory: output_dir = output
    - If output is not provided: use ./optimization_logs/<kernel_name>_<timestamp>
    """
    if output is None:
        from minisweagent.run.utils.task_parser import generate_patch_output_dir

        return (Path.cwd() / Path(generate_patch_output_dir(kernel_name))).resolve()

    if output.suffix:
        return output.parent

    return output


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
    gpu_ids: str | None = typer.Option(None, "--gpu-ids", help="Comma-separated GPU IDs."),
    test_command: str | None = typer.Option(None, "--test_command", "--test-command", help="Test command"),
    target_language: str | None = typer.Option(
        None,
        "--target-language",
        help=(
            "Target kernel language for translation (e.g. 'triton', 'hip'). "
            "When set and different from the detected source language, a "
            "translation preprocess phase runs before optimization."
        ),
    ),
):
    # fmt: on
    del visual

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
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
        logger.info("Using model class: %s.", model_class)

    tools_cfg = config.get("tools") or {}
    disabled_tools: list[str] = []
    if tools_cfg.get("bash") is False:
        disabled_tools.append("bash")
    if tools_cfg.get("profiling") is False:
        disabled_tools.append("profiling")
        disabled_tools.append("profile_kernel")

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
    mode: str | None = None
    max_rounds = None
    if task_content:
        from minisweagent.run.utils.task_parser import parse_pipeline_params

        logger.info("[bold cyan]Checking task for pipeline parameters...[/bold cyan]")
        pipeline_params = parse_pipeline_params(task_content, model)
        logger.debug("pipeline_params: %s", pipeline_params)

        # Apply non-None extracted values (CLI flags still take priority).
        extracted_mode = pipeline_params.get("mode")
        if extracted_mode is not None and mode is None:
            mode = str(extracted_mode).strip().lower()
            logger.info("Using mode=%s from task content.", mode)
        if pipeline_params.get("max_rounds") is not None:
            max_rounds = pipeline_params["max_rounds"]
            logger.info("Using max rounds: %s.", max_rounds)

        # target_language: CLI flag takes priority over LLM extraction
        if target_language is None and pipeline_params.get("target_language"):
            target_language = str(pipeline_params["target_language"]).strip().lower()
            logger.info("Using target_language=%s from task content.", target_language)

        # Prompt for missing required params (kernel_url) — only if not already set
        if kernel_url is None:
            from minisweagent.run.utils.config_editor import prompt_missing_pipeline_params

            logger.info("Prompting for missing pipeline parameters.")
            pipeline_params, should_use_pipeline = prompt_missing_pipeline_params(
                pipeline_params, console, yolo
            )
            logger.info("pipeline_params: %s, should_use_pipeline: %s", pipeline_params, should_use_pipeline)

            if should_use_pipeline:
                if pipeline_params.get("kernel_url"):
                    kernel_url = pipeline_params["kernel_url"]
                    logger.info("Using kernel URL: %s.", kernel_url)

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

    # User-supplied harness path from natural-language prompt
    # ("Use the test harness at <path>" / "with harness <path>" / etc.).
    # This wires straight to ctx.harness so HarnessPhase Layer 2 fires —
    # no HarnessBuilder LLM call needed when the user already has a
    # working harness file.  CLI's ``patch.harness`` config key still
    # wins if explicitly set elsewhere.
    parsed_harness = parsed_config.get("test_harness")
    if parsed_harness and not config.get("patch", {}).get("harness"):
        config.setdefault("patch", {})["harness"] = parsed_harness
        logger.info("Using test harness from task content: %s", parsed_harness)
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

    parsed_gpu_ids = parse_gpu_ids(gpu_ids)

    kernel_name_for_output = parsed_config.get("kernel_name")
    if not kernel_name_for_output and kernel_url:
        kernel_name_for_output = Path(kernel_url).stem
    if not kernel_name_for_output and isinstance(kernel_target, str):
        kernel_name_for_output = Path(kernel_target).stem
    logger.info("Using kernel_name_for_output: %s", kernel_name_for_output)

    preprocess_output_dir = _derive_output_dir(output, kernel_name_for_output)
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
    if config.get("patch", {}).get("harness"):
        _display_cfg["test_harness"] = config["patch"]["harness"]
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
    _preprocess_kwargs = dict(
        kernel_url=kernel_target,
        repo=repo,
        output_dir=preprocess_output_dir,
        gpu_id=parsed_gpu_ids[0] if parsed_gpu_ids else 0,
        model=model,
        model_factory=lambda: get_model(model_name, config.get("model", {})),
        console=console,
        target_language=target_language,
    )
    logger.debug("Preprocess kwargs: %s", _preprocess_kwargs)

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
            logger.info("Correctness command: %s, Performance command: %s", correctness_command, performance_command)
            preprocess_ctx = run_preprocessor(
                **_preprocess_kwargs,
                correctness_command=correctness_command,
                performance_command=performance_command,
            )
        else:
            preprocess_ctx = run_preprocessor(**_preprocess_kwargs, eval_command=test_command)
    logger.debug("Preprocessor context: %s", preprocess_ctx)

    if preprocess_ctx.get("test_command") and not test_command:
        test_command = preprocess_ctx["test_command"]
    if preprocess_ctx.get("repo_root") and repo is None:
        repo = Path(preprocess_ctx["repo_root"])

    # Translation-phase notification:
    #
    # When ``--target-language`` is set and differs from the detected
    # source language, the preprocess ``TranslationPhase`` has already
    # run BEFORE we got here (it's the first phase in
    # PreprocessOrchestrator) and swapped ``preprocess_ctx["kernel_path"]``
    # to the translated kernel.  We log a confirmation here.  See
    # ``run/preprocess/phases/translation.py``.
    _detected_source = _normalize_kernel_type(
        (preprocess_ctx.get("discovery") or {}).get("kernel", {}).get("type")
        or preprocess_ctx.get("kernel_type")
    )
    if target_language and target_language != _detected_source and target_language != "other":
        logger.info(
            "Translation preprocess phase ran: source=%s -> target=%s (kernel now %s).",
            _detected_source,
            target_language,
            preprocess_ctx.get("kernel_path"),
        )
    elif target_language:
        logger.info(
            "target_language=%s matches detected source language; translation skipped.",
            target_language,
        )

    # Mode resolution: if the LLM / CLI did not specify a mode, default to ``mixed``.
    if mode is None:
        mode = "mixed"
        logger.info(
            "mode not specified; defaulting to mixed (half legacy-homogeneous + half planned). "
            "Explicit planned/fixed/mixed in the task (via parse_pipeline_params) overrides this.",
        )

    # Backfill discovery.kernel.type when preprocessor left it blank — saves
    # an extra file probe in the hot path (useful for mixed and other modes).
    if mode == "mixed":
        _discovery = preprocess_ctx.get("discovery") or {}
        _kernel_info = _discovery.get("kernel") or {}
        if (not _kernel_info.get("type") or _kernel_info.get("type") == "unknown") and preprocess_ctx.get("kernel_path"):
            from minisweagent.agents.heterogeneous.task_generator import _infer_kernel_type

            inferred = _infer_kernel_type(Path(preprocess_ctx["kernel_path"]))
            if inferred and inferred != "unknown":
                _kernel_info = dict(_kernel_info)
                _kernel_info["type"] = inferred
                _discovery = dict(_discovery)
                _discovery["kernel"] = _kernel_info
                preprocess_ctx = dict(preprocess_ctx)
                preprocess_ctx["discovery"] = _discovery
                logger.info("kernel_type backfilled from kernel path (mixed default path): %s", inferred)

    # Extract user constraints + directives regardless of mode.  Both
    # planned and fixed pipelines fold these into COMMANDMENT.md inside
    # ``run_pipeline`` (`_run_planned` / `_run_fixed`) so every worker sees them.
    # Preprocessor must still produce a base ``commandment`` for both modes.
    preprocess_ctx["user_instructions"] = task_content
    preprocess_ctx["rag_enabled"] = rag_enabled

    extra_addenda: list[str] = []
    extracted = extract_user_constraints(task_content, model)
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
    if extra_addenda:
        logger.info(
            "Extracted %d constraint(s) and %d directive(s) from task content.",
            len(extracted["constraints"]),
            len(extracted["directives"]),
        )

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

    from minisweagent.run.unified import PipelineContext, run_pipeline

    ctx = PipelineContext(
        preprocess_ctx=preprocess_ctx,
        user_prompt=task_content,
        kernel_language=_normalize_kernel_type(preprocess_ctx.get("kernel_type")),
        output_dir=preprocess_output_dir,
        gpu_ids=parsed_gpu_ids,
        model=model,
        model_factory=lambda: get_model(model_name, config.get("model", {})),
        config=config,
        max_rounds=max_rounds or config.get("orchestrator", {}).get("max_rounds"),
        env=env,
        env_class=env.__class__,
        env_kwargs=_env_kwargs,
        repo=repo_path,
        test_command=test_command or config.get("patch", {}).get("test_command"),
        metric=metric,
        rag_enabled=rag_enabled,
        extra_addenda=extra_addenda,
        model_name=model_name,
        console=console,
    )

    # ``fixed`` / ``planned`` / ``mixed`` all use the orchestrator.  Planned/mixed may return a FinalReport converted
    # to BestPatchResult; fixed returns BestPatchResult directly when applicable.
    result_or_report = run_pipeline(ctx, mode=mode)  # type: ignore[arg-type]
    logger.info("Run completed in %.0fs.", time.monotonic() - _run_t0)

    # Distinguish by return shape instead of carrying a mode flag here:
    # FinalReport comes from the planned path.
    from minisweagent.run.pipeline_types import FinalReport

    if isinstance(result_or_report, FinalReport) or (
        isinstance(result_or_report, dict) and "status" in result_or_report
    ):
        return _final_report_to_bestpatchresult(result_or_report)
    return result_or_report


# ──────────────────────────────────────────────────────────────────────
# ``geak translate`` — standalone kernel language porting
# ──────────────────────────────────────────────────────────────────────
#
# Unlike ``geak -t "<prompt>" --target-language Y`` (which translates
# AND optimizes in-place), ``geak translate`` runs the translation
# preprocess phase only and exits.  Useful when the user wants the
# translated kernel as a deliverable without the full optimization
# loop.


_TRANSLATE_HELP = """Translate a kernel from one language to another.

Runs only the translation preprocess phase (``TranslationAgent``) and
exits.  The translated kernel is written next to the source file with
the target language's extension.

Examples:

  geak translate --source kernel.py --target-language hip
  geak translate --source kernel.hip --target-language triton --output kernel_triton.py
"""


@app.command(name="translate", help=_TRANSLATE_HELP)
def translate(
    source: Path = typer.Option(..., "--source", "-s", help="Source kernel file (local path)."),
    target_language: str = typer.Option(..., "--target-language", "-t", help="Target kernel language (e.g. triton, hip, cuda)."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output path for the translated kernel.  Defaults to ``source.stem + target_suffix`` next to the source."),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use for translation."),
    max_attempts: int = typer.Option(3, "--max-attempts", help="Maximum verify-retry attempts before failing."),
):
    """Run the translation preprocess phase standalone."""
    configure_if_first_time()

    from minisweagent.models import get_model
    from minisweagent.run.preprocess.phases.base import PhaseContext
    from minisweagent.run.preprocess.phases.translation import TranslationPhase

    if not source.exists() or not source.is_file():
        console.print(f"[bold red]Error:[/bold red] source file not found: {source}")
        raise typer.Exit(1)

    # Load config for the model (same pattern as main()).
    base_config_path = builtin_config_dir / "mini_kernel_strategy_list.yaml"
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    config_path = builtin_config_dir / "geak.yaml"
    user_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = _deep_merge(base_config, user_config)

    model = get_model(model_name, config.get("model", {}))
    logger.info(
        "[bold cyan]geak translate[/bold cyan] source=%s -> target_language=%s (max_attempts=%d)",
        source,
        target_language,
        max_attempts,
    )

    ctx = PhaseContext(
        kernel_url=str(source.resolve()),
        output_dir=source.parent.resolve(),
        target_language=target_language.strip().lower(),
        translate_only=True,
        model=model,
    )

    phase = TranslationPhase()
    # Pass the model through to the agent builder.
    phase._injected_model = model  # type: ignore[attr-defined]

    try:
        phase.run(ctx)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Translation failed:[/bold red] {exc}")
        raise typer.Exit(1)
    except RuntimeError as exc:
        console.print(f"[bold red]Translation failed:[/bold red] {exc}")
        raise typer.Exit(2)

    translated_path = Path(ctx.kernel_path)
    if output is not None:
        # Caller requested a specific output path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(translated_path.read_bytes())
        translated_path = output

    console.print(
        f"[bold green]Translation complete.[/bold green] "
        f"Translated kernel: [cyan]{translated_path}[/cyan]"
    )
    logger.info("geak translate: wrote translated kernel to %s", translated_path)


if __name__ == "__main__":
    cli_entry()
