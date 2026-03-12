"""Helper functions extracted from ``mini.py``'s ``main()`` to keep the CLI
entry-point thin and the business logic testable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.agents.interactive_textual import TextualAgent
from minisweagent.agents.parallel_agent import ParallelAgent
from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent
from minisweagent.agents.unit_test_agent import format_discovery_for_agent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.config import deep_merge
from minisweagent.run.config.editor import load_and_merge_configs
from minisweagent.run.config.task_parser import _resolve_path_case
from minisweagent.run.pipeline.helpers import (
    _REPO_ROOT,
    DEFAULT_EVAL_BENCHMARK_ITERATIONS,
    create_validated_harness,
    extract_harness_path,
    inject_resolved_kernel,
    run_baseline_profile,
    run_discovery,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskInput:
    """Resolved task context produced by :func:`resolve_task_input`."""

    task_content: str | None = None
    tf_meta: dict | None = None
    task_worktree: Path | None = None
    codebase_ctx_text: str | None = None
    repo: Path | None = None
    test_command: str | None = None
    patch_output: Path | None = None
    yolo: bool = False
    resolved_kernel_path: str | None = None
    resolved_kernel_name: str | None = None
    discovery_result: Any | None = None


@dataclass
class RuntimeContext:
    """Aggregated runtime state produced by :func:`apply_runtime_settings`."""

    config: dict = field(default_factory=dict)
    model: Any = None
    env: Any = None
    env_class: type | None = None
    env_kwargs: dict = field(default_factory=dict)
    repo: Path | None = None
    test_command: str | None = None
    metric: str | None = None
    num_parallel: int | None = None
    parsed_gpu_ids: list[int] = field(default_factory=lambda: [0])
    patch_output: Path | None = None
    kernel_name: str | None = None


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------

def load_config(
    enable_strategies: bool,
    config_spec: Path | None,
    console: Console,
) -> dict:
    """Load, layer, and return the merged YAML configuration."""
    base_config_path = builtin_config_dir / "mini.yaml"
    console.print(f"Loading base config: [bold green]'{base_config_path.name}'[/bold green]")
    config = yaml.safe_load(base_config_path.read_text())

    template_name = (
        "mini_kernel_strategy_list.yaml" if enable_strategies
        else "mini_system_prompt.yaml"
    )
    template_path = builtin_config_dir / template_name
    console.print(f"Applying template: [bold green]'{template_name}'[/bold green] (save_patch always enabled)")
    config = deep_merge(config, yaml.safe_load(template_path.read_text()))

    if config_spec:
        config_path = get_config_path(config_spec)
        console.print(f"[dim]Applying user config from '{config_path}' (final override)[/dim]")
        config = deep_merge(config, yaml.safe_load(config_path.read_text()))

    _propagate_tools_config(config)
    return config


def _propagate_tools_config(config: dict) -> None:
    """Push ``tools.*`` keys into ``model.*`` / ``agent.*`` for backward compat."""
    tools_cfg = config.get("tools") or {}
    if tools_cfg:
        if "bash" in tools_cfg:
            config.setdefault("model", {}).setdefault("bash_tool", tools_cfg["bash"])
        if "profiling" in tools_cfg:
            config.setdefault("model", {}).setdefault("profiling", tools_cfg["profiling"])
        if "profiling_type" in tools_cfg:
            config.setdefault("agent", {}).setdefault("profiling_type", tools_cfg["profiling_type"])
        if tools_cfg.get("profiling") and "profiling_type" not in tools_cfg:
            config.setdefault("agent", {}).setdefault("profiling_type", "profiling")
        if "strategy_manager" in tools_cfg:
            config.setdefault("agent", {}).setdefault("use_strategy_manager", tools_cfg["strategy_manager"])
            config.setdefault("model", {}).setdefault("use_strategy_manager", tools_cfg["strategy_manager"])

    if "profiling" in config:
        config.setdefault("model", {}).setdefault("profiling", config["profiling"])
    if "profiling_type" in config:
        config.setdefault("agent", {}).setdefault("profiling_type", config["profiling_type"])
    if config.get("model", {}).get("profiling") and not config.get("agent", {}).get("profiling_type"):
        config.setdefault("agent", {})["profiling_type"] = "profiling"


# ---------------------------------------------------------------------------
# 2. Task resolution
# ---------------------------------------------------------------------------

def resolve_task_input(
    task: str | None,
    repo: Path | None,
    test_command: str | None,
    patch_output: Path | None,
    gpu_ids: str | None,
    kernel_url: str | None,
    workspace: Path | None,
    console: Console,
) -> TaskInput:
    """Parse the ``--task`` argument, resolve kernel URLs, run discovery.

    Returns a :class:`TaskInput`.  When the caller should fall back to an
    interactive prompt, ``task_content`` will be ``None``.

    May raise ``SystemExit`` for openevolve redirects.
    """
    ti = TaskInput(repo=repo, test_command=test_command, patch_output=patch_output)
    task_content = task

    if task:
        task_content = _parse_task_arg(task, ti, gpu_ids, console)

    # Default task when a kernel URL is provided but no explicit task
    if not task_content and kernel_url:
        task_content = (
            "Optimize this kernel for maximum speedup.\n"
            "Follow the workflow described in the pipeline instructions (INSTRUCTIONS.md).\n"
            "Use the discovered tests and benchmarks listed above for correctness and performance.\n"
            "Report final speedup when done."
        )
        console.print("[bold green]Using default kernel optimization task[/bold green]")

    # Resolve kernel URL -> local path + inject into task
    if kernel_url and task_content:
        task_content, ti.resolved_kernel_name = inject_resolved_kernel(
            kernel_url, str(workspace) if workspace else None, task_content,
        )
        m = re.search(r"Kernel path: (\S+)", task_content)
        if m:
            ti.resolved_kernel_path = m.group(1)

    # Test discovery
    if ti.resolved_kernel_path:
        discovery_block = run_discovery(ti.resolved_kernel_path, ti.resolved_kernel_name)
        if discovery_block:
            task_content = task_content + discovery_block
        ti.discovery_result = getattr(run_discovery, "_last_result", None)
    elif task and ".md" in task:
        try:
            with open(task, encoding="utf-8") as f:
                task_content = f.read()
        except OSError:
            pass

    ti.task_content = task_content
    return ti


def _parse_task_arg(
    task: str,
    ti: TaskInput,
    gpu_ids: str | None,
    console: Console,
) -> str | None:
    """Handle --task: plain text, plain file, or YAML-frontmatter task file."""
    task_path = Path(task)
    try:
        if not (task_path.exists() and task_path.is_file()):
            return None if not task.strip() else task

        raw = task_path.read_text(encoding="utf-8")
        if not raw.lstrip().startswith("---"):
            console.print(f"[bold green]Read task from file: {task_path}[/bold green]")
            return raw

        # --- Structured YAML-frontmatter task file ---
        return _handle_structured_task(task_path, raw, ti, gpu_ids, console)
    except OSError:
        return task


def _handle_structured_task(
    task_path: Path,
    _raw: str,
    ti: TaskInput,
    gpu_ids: str | None,
    console: Console,
) -> str | None:
    """Process a YAML-frontmatter ``.md`` task file."""
    from minisweagent.run.pipeline.task_file import (
        create_worktree,
        is_git_repo,
        read_task_file,
    )

    meta, body = read_task_file(task_path)
    ti.tf_meta = meta
    task_content = body
    console.print(f"[bold cyan]Loading structured task file: {task_path}[/bold cyan]")

    # Openevolve redirect
    if meta.get("agent_type", "strategy_agent") == "openevolve":
        console.print(
            "[bold yellow]Task has agent_type=openevolve. "
            "Redirecting to openevolve-worker...[/bold yellow]"
        )
        oe_args = ["openevolve-worker", "--from-task", str(task_path.resolve())]
        if gpu_ids:
            oe_args += ["--gpu", gpu_ids.split(",")[0]]
        raise SystemExit(subprocess.call(oe_args))

    # Derive repo / test_command / patch_output from metadata
    if not ti.repo and meta.get("kernel_path"):
        ti.repo = Path(meta["kernel_path"]).resolve().parent
    if not ti.repo and meta.get("repo_root"):
        ti.repo = Path(meta["repo_root"]).resolve()
    if not ti.test_command and meta.get("test_command"):
        ti.test_command = meta["test_command"]

    if not ti.patch_output:
        resolved = task_path.resolve()
        round_dir = resolved.parent.name
        pipeline_root = resolved.parent.parent.parent
        ti.patch_output = pipeline_root / "results" / round_dir / resolved.stem
        console.print(f"[dim]Derived --patch-output: {ti.patch_output}[/dim]")

    # Worktree
    tf_repo = (
        Path(meta["repo_root"]).resolve() if meta.get("repo_root")
        else (ti.repo.resolve() if ti.repo else None)
    )
    if tf_repo and tf_repo.is_dir() and is_git_repo(tf_repo):
        wt_dest = Path(ti.patch_output) / "_worktree"
        console.print(f"[bold cyan]Creating isolated worktree at {wt_dest}...[/bold cyan]")
        ti.task_worktree = create_worktree(tf_repo, wt_dest)
        ti.repo = ti.task_worktree

    ti.yolo = True

    # Assemble the skip-note preamble
    return _build_skip_note(meta, ti.test_command, ti, console) + (task_content or "")


def _build_skip_note(
    meta: dict,
    test_command: str | None,
    ti: TaskInput,
    console: Console,
) -> str:
    """Build the preamble injected before the task body for pipeline tasks."""
    lines: list[str] = [
        "NOTE: This task was generated by the task-generator pipeline. "
        "Baseline profiling and performance metrics are already available "
        "in the files listed below. Do NOT re-run baseline profiling or "
        "establish baseline performance -- skip directly to analyzing "
        "the provided data and implementing the optimization.",
        "",
    ]
    if meta.get("kernel_path"):
        lines.append(f"KERNEL FILE TO EDIT: {meta['kernel_path']}")
    if test_command:
        lines.append(f"TEST COMMAND: {test_command}")
    if meta.get("repo_root"):
        lines.append(f"REPO ROOT: {meta['repo_root']}")

    cmd_path = meta.get("commandment")
    if cmd_path and Path(cmd_path).exists():
        lines += ["", "## COMMANDMENT (evaluation contract -- you MUST follow these rules)",
                   Path(cmd_path).read_text().strip()]

    bm_path = meta.get("baseline_metrics")
    if bm_path and Path(bm_path).exists():
        bm = json.loads(Path(bm_path).read_text())
        lines += [
            "", "## Baseline Performance (your optimization must improve on these)",
            f"Total duration: {bm.get('duration_us', 'unknown')} us",
            f"Bottleneck: {bm.get('bottleneck', 'unknown')}",
        ]
        for k in bm.get("top_kernels", [])[:5]:
            bn_tag = f" [{k['bottleneck']}]" if k.get("bottleneck") else ""
            lines.append(
                f"  - {k.get('name', '?')}: {k.get('duration_us', '?')} us "
                f"({k.get('pct_of_total', '?')}%){bn_tag}"
            )

    prof_path = meta.get("profiling")
    if prof_path and Path(prof_path).exists():
        lines += ["", f"PROFILING DATA: {prof_path}",
                   "(Read this file for detailed per-kernel profiling metrics)"]

    ctx_path = meta.get("codebase_context")
    if ctx_path and Path(ctx_path).exists():
        ti.codebase_ctx_text = Path(ctx_path).read_text().strip()
        lines += ["", "## Codebase Context (repo structure and key files)",
                   ti.codebase_ctx_text]

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 3. Runtime settings
# ---------------------------------------------------------------------------

def apply_runtime_settings(
    config: dict,
    ti: TaskInput,
    *,
    yolo: bool,
    cost_limit: float | None,
    exit_immediately: bool,
    model_name: str | None,
    model_class: str | None,
    enable_strategies: bool,
    rag: bool,
    debug: bool,
    console: Console,
    metric: str | None = None,
    num_parallel: int | None = None,
    gpu_ids: str | None = None,
) -> RuntimeContext:
    """Apply CLI flags to config, create model + env, merge configs."""
    if yolo or ti.yolo:
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
    config.setdefault("model", {})["use_strategy_manager"] = enable_strategies

    model = get_model(model_name, config.get("model", {}))

    env_kwargs = config.get("env", {})
    env_class: type
    if rag:
        try:
            from minisweagent.mcp_integration.mcp_environment import MCPEnabledEnvironment
            from minisweagent.mcp_integration.prompts import INSTANCE_TEMPLATE, SYSTEM_TEMPLATE
            from minisweagent.mcp_integration.run_agent import DebugMCPEnvironment
        except ImportError as e:
            console.print("[red]Error: RAG retrieval requires langchain dependencies. Run: pip install -e '.[langchain]'[/red]")
            console.print(f"[red]Import error: {e}[/red]")
            raise typer.Exit(1)

        env_class = DebugMCPEnvironment if debug else MCPEnabledEnvironment
        env = env_class(**env_kwargs)
        if debug:
            console.print("[bold yellow]Debug mode enabled[/bold yellow]")
        config.setdefault("agent", {})["system_template"] = SYSTEM_TEMPLATE
        config.setdefault("agent", {})["instance_template"] = INSTANCE_TEMPLATE
        console.print("[bold green]RAG knowledge retrieval enabled[/bold green]")
    else:
        env_class = LocalEnvironment
        env = LocalEnvironment(**env_kwargs)

    detect_content = None if ti.tf_meta else ti.task_content
    result = load_and_merge_configs(
        config, ti.repo, ti.test_command, metric, num_parallel, gpu_ids, ti.patch_output,
        detect_content, yolo or ti.yolo, model, console,
    )

    rt = RuntimeContext(config=config, model=model, env=env,
                        env_class=env_class, env_kwargs=env_kwargs)

    if result == (None, None, None, None, None, None, None):
        console.print("[bold yellow]Continuing without automatic patch saving. You can still interact with the agent.[/bold yellow]")
        rt.parsed_gpu_ids = [0]
    else:
        rt.repo, rt.test_command, rt.metric, rt.num_parallel, rt.parsed_gpu_ids, rt.patch_output, rt.kernel_name = result

    if rt.repo:
        env.config.cwd = str(Path(rt.repo).resolve())

    return rt


# ---------------------------------------------------------------------------
# 4. Full pipeline mode
# ---------------------------------------------------------------------------

def run_full_pipeline(
    kernel_url: str,
    config: dict,
    model: Any,
    model_name: str | None,
    parsed_gpu_ids: list[int],
    patch_output: Path | None,
    max_rounds: int | None,
    heterogeneous: bool,
    console: Console,
) -> dict | None:
    """Run preprocessor -> orchestrator pipeline and return the report."""
    from minisweagent.run.pipeline.orchestrator import run_orchestrator
    from minisweagent.run.pipeline.preprocessor import run_preprocessor

    pipeline_output = patch_output or Path("geak_output")
    console.print("[bold cyan]--- GEAK Full Pipeline Mode ---[/bold cyan]")
    console.print(f"[dim]Kernel URL: {kernel_url}[/dim]")
    console.print(f"[dim]Output dir: {pipeline_output}[/dim]")

    preprocess_ctx = run_preprocessor(
        kernel_url,
        output_dir=pipeline_output,
        gpu_id=parsed_gpu_ids[0] if parsed_gpu_ids else 0,
        model=model,
        model_factory=lambda: get_model(model_name, config.get("model", {})),
        console=console,
    )

    model_name_resolved = model_name or config.get("model", {}).get("model_name")
    model_cfg = config.get("model", {})

    report = run_orchestrator(
        preprocess_ctx=preprocess_ctx,
        gpu_ids=parsed_gpu_ids or [0],
        model=model,
        model_factory=lambda: get_model(model_name_resolved, model_cfg),
        output_dir=pipeline_output,
        max_rounds=max_rounds,
        heterogeneous=heterogeneous,
        console=console,
    )

    console.print("\n[bold green]Pipeline complete.[/bold green]")
    if report:
        console.print(f"[dim]{json.dumps(report, indent=2, default=str)[:500]}[/dim]")
    return report


# ---------------------------------------------------------------------------
# 5. Agent construction
# ---------------------------------------------------------------------------

def build_agent(
    config: dict,
    model: Any,
    model_name: str | None,
    env: Any,
    ti: TaskInput,
    rt: RuntimeContext,
    *,
    visual: bool,
    enable_strategies: bool,
    strategy_file: str,
    create_test: bool,
    heterogeneous: bool,
    console: Console,
) -> tuple[Any, Path]:
    """Build and return ``(agent, agent_log_file)``."""
    repo = rt.repo
    test_command = rt.test_command
    num_parallel = rt.num_parallel
    parsed_gpu_ids = rt.parsed_gpu_ids
    patch_output = rt.patch_output
    kernel_name = rt.kernel_name
    metric = rt.metric

    # -- Test discovery / harness creation --
    if not ti.tf_meta and (create_test or not test_command):
        if not repo:
            raise ValueError("repo is required for --create-test or when test_command is missing. Please pass --repo.")
        test_command = _create_test_harness(
            ti, repo, kernel_name, patch_output, parsed_gpu_ids,
            model_name, config, console,
        )
        rt.test_command = test_command

    # -- Pre-agent profiling / commandment --
    pre_profiling, pre_baseline, pre_commandment = _run_pre_agent_profiling(
        ti, test_command, repo, num_parallel, parsed_gpu_ids, config, console,
    )

    # -- Agent class selection --
    if enable_strategies:
        base_agent_class = StrategyInteractiveAgent
        console.print(f"[bold cyan]Using Strategy Agent with strategy file: {strategy_file}[/bold cyan]")
    else:
        if visual == (os.getenv("MSWEA_VISUAL_MODE_DEFAULT", "false") == "false"):
            base_agent_class = TextualAgent
        else:
            base_agent_class = InteractiveAgent
        console.print(f"[bold cyan]Using Interactive Agent (visual={'on' if base_agent_class == TextualAgent else 'off'})[/bold cyan]")

    # -- Agent config --
    agent_config = config.get("agent", {})
    agent_config["use_strategy_manager"] = enable_strategies
    if enable_strategies:
        agent_config["strategy_file_path"] = strategy_file
    agent_config["save_patch"] = True
    agent_config["test_command"] = test_command or config.get("patch", {}).get("test_command")
    patch_dir = patch_output or config.get("patch", {}).get("patch_output_dir") or (global_config_dir / "patches")
    agent_config["patch_output_dir"] = str(patch_dir)
    agent_config["metric"] = metric or config.get("patch", {}).get("metric")
    if ti.codebase_ctx_text:
        agent_config["codebase_context"] = ti.codebase_ctx_text

    log_dir = Path(patch_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    agent_log_file = log_dir / "mini_agent.log"

    agent_class = ParallelAgent
    agent_config["agent_class"] = base_agent_class
    agent_config["num_parallel"] = num_parallel or 1
    agent_config["gpu_ids"] = parsed_gpu_ids

    bench_extra = f"--iterations {DEFAULT_EVAL_BENCHMARK_ITERATIONS}"
    config.setdefault("env", {}).setdefault("env", {}).setdefault(
        "GEAK_BENCHMARK_EXTRA_ARGS", bench_extra,
    )

    if num_parallel and num_parallel > 1:
        _configure_parallel_mode(
            agent_config, config, ti, rt, model, base_agent_class,
            pre_profiling, pre_baseline, pre_commandment,
            heterogeneous, console,
        )
    else:
        console.print("[bold cyan]Using Single Agent Mode[/bold cyan]")
        console.print(f"[dim]Using GPU: {parsed_gpu_ids[0]}[/dim]")
        env.config.env = env.config.env or {}
        env.config.env["HIP_VISIBLE_DEVICES"] = str(parsed_gpu_ids[0])
        env.config.env.setdefault("GEAK_BENCHMARK_EXTRA_ARGS", bench_extra)
        if repo:
            env.config.cwd = str(Path(repo).resolve())
            console.print(f"[dim]Working directory: {env.config.cwd}[/dim]")

    agent = agent_class(model, env, **agent_config)
    agent.log_file = agent_log_file

    # Set base_repo_path for patch generation (used for path substitution in test commands)
    if ti.tf_meta and ti.tf_meta.get("repo_root"):
        agent.base_repo_path = Path(ti.tf_meta["repo_root"]).resolve()
    console.print(f"[bold cyan]Agent log: {agent_log_file}[/bold cyan]")
    console.print(f"[dim]Tip: tail -f {agent_log_file}[/dim]")

    return agent, agent_log_file


def _create_test_harness(
    ti: TaskInput,
    repo: Path,
    kernel_name: str | None,
    patch_output: Path | None,
    parsed_gpu_ids: list[int],
    model_name: str | None,
    config: dict,
    console: Console,
) -> str:
    """Run UnitTestAgent to discover/create a test harness, return test_command."""
    discovery_context = ""
    if ti.discovery_result:
        console.print("[bold cyan]Formatting stashed discovery results for UnitTestAgent...[/bold cyan]")
        discovery_context = format_discovery_for_agent(ti.discovery_result)
    else:
        kernel_path = None
        if ti.resolved_kernel_path:
            kernel_path = Path(ti.resolved_kernel_path)
        elif ti.task_content and not ti.tf_meta:
            try:
                p = Path(ti.task_content)
                if p.is_file():
                    kernel_path = p
            except OSError:
                pass
        if kernel_path or repo:
            console.print("[bold cyan]Running content-based test discovery...[/bold cyan]")
            discovery_context = run_discovery(kernel_path=str(kernel_path or repo))

    if discovery_context:
        console.print("[dim]Discovery results ready -- feeding into UnitTestAgent.[/dim]")
    else:
        console.print("[dim]No discovery results -- UnitTestAgent will search/create from scratch.[/dim]")

    console.print("[bold yellow]Running UnitTestAgent to create test harness...[/bold yellow]")
    test_command, harness_results = create_validated_harness(
        model=get_model(model_name, config.get("model", {})),
        repo=repo,
        kernel_name=kernel_name or "unknown",
        log_dir=patch_output,
        discovery_context=discovery_context,
        gpu_id=parsed_gpu_ids[0] if parsed_gpu_ids else 0,
    )
    console.print(f"[bold green]Using UnitTestAgent test_command:[/bold green] {test_command}")
    for hr in (harness_results or []):
        s = "PASS" if hr["success"] else "FAIL"
        console.print(f"  [dim]--{hr['mode']}: {s} ({hr['duration_s']}s)[/dim]")
    return test_command


def _run_pre_agent_profiling(
    ti: TaskInput,
    test_command: str | None,
    repo: Path | None,
    num_parallel: int | None,
    parsed_gpu_ids: list[int],
    config: dict,
    console: Console,
) -> tuple[dict | None, dict | None, str | None]:
    """Run baseline profiling + commandment generation. Returns (profiling, baseline_metrics, commandment)."""
    profiling: dict | None = None
    baseline_metrics: dict | None = None
    commandment: str | None = None

    if not (ti.resolved_kernel_path and test_command and num_parallel and num_parallel > 1):
        return profiling, baseline_metrics, commandment

    try:
        console.print("[bold cyan]--- Pre-agent Baseline Profiling ---[/bold cyan]")
        profiling = run_baseline_profile(test_command, gpu_id=parsed_gpu_ids[0])
    except Exception as e:
        console.print(f"[yellow]Pre-agent profiling failed ({e}); tasks will use discovery only[/yellow]")

    if profiling:
        try:
            from minisweagent.baseline_metrics import build_baseline_metrics
            baseline_metrics = build_baseline_metrics(profiling, include_all=True)
            console.print(
                f"[bold green]Baseline: {baseline_metrics.get('duration_us', '?')} us, "
                f"bottleneck={baseline_metrics.get('bottleneck', '?')}[/bold green]"
            )
        except Exception as e:
            console.print(f"[yellow]Baseline metrics extraction failed ({e})[/yellow]")

    try:
        from minisweagent.tools.commandment import generate_commandment
        from minisweagent.tools.discovery_types import _infer_kernel_language
        harness_path = extract_harness_path(test_command)
        kl = _infer_kernel_language(Path(ti.resolved_kernel_path), "")
        commandment = generate_commandment(
            kernel_path=ti.resolved_kernel_path,
            harness_path=harness_path,
            repo_root=repo,
            kernel_language=kl,
        )
        console.print("[bold green]COMMANDMENT.md generated (pre-agent)[/bold green]")
    except Exception as e:
        console.print(f"[yellow]Commandment generation failed ({e})[/yellow]")

    return profiling, baseline_metrics, commandment


def _configure_parallel_mode(
    agent_config: dict,
    config: dict,
    ti: TaskInput,
    rt: RuntimeContext,
    model: Any,
    base_agent_class: type,
    pre_profiling: dict | None,
    pre_baseline: dict | None,
    pre_commandment: str | None,
    heterogeneous: bool,
    console: Console,
) -> None:
    """Configure agent_config for parallel (multi-GPU) execution."""
    console.print(f"[bold cyan]Using Parallel Mode: {rt.num_parallel} agents on GPUs {rt.parsed_gpu_ids}[/bold cyan]")

    repo_path = rt.repo or config.get("patch", {}).get("repo")
    if repo_path:
        p = Path(repo_path)
        if not p.exists():
            resolved = _resolve_path_case(p)
            if resolved is not None:
                p = resolved
        agent_config["repo"] = str(p.resolve())
        console.print(f"[dim]Repository: {agent_config['repo']}[/dim]")
    else:
        console.print("[bold yellow]Warning: No repo path specified for parallel execution[/bold yellow]")

    discovery_result = ti.discovery_result
    if heterogeneous and discovery_result and discovery_result.kernels and model:
        try:
            from minisweagent.run.pipeline.helpers import inject_pipeline_context
            from minisweagent.run.pipeline.task_generator import generate_tasks_from_content

            tasks = generate_tasks_from_content(
                discovery_result=discovery_result,
                base_task_context=ti.task_content,
                agent_class=base_agent_class,
                model=model,
                profiling_result=pre_profiling,
                commandment_content=pre_commandment,
                baseline_metrics=pre_baseline,
            )
            if tasks:
                for t in tasks:
                    t.task, t.config = inject_pipeline_context(
                        t.task, t.config,
                        commandment_text=pre_commandment,
                        baseline_metrics=pre_baseline,
                        kernel_path=ti.resolved_kernel_path,
                        repo_root=str(rt.repo) if rt.repo else None,
                        test_command=rt.test_command,
                        codebase_context=ti.codebase_ctx_text,
                    )
                agent_config["tasks"] = tasks
                console.print(f"[bold cyan]Task generator: {len(tasks)} optimization tasks (pool mode)[/bold cyan]")
                for t in tasks[:6]:
                    console.print(f"  [dim]- [{t.priority:2d}] {t.label} ({t.kernel_language})[/dim]")
                if len(tasks) > 6:
                    console.print(f"  [dim]  ... and {len(tasks) - 6} more[/dim]")
        except Exception as e:
            console.print(f"[yellow]Task generator failed ({e}), falling back to homogeneous mode[/yellow]")
    elif heterogeneous:
        console.print("[yellow]--heterogeneous requires discovery results (run preprocessing first). Falling back to homogeneous.[/yellow]")


# ---------------------------------------------------------------------------
# 6. INSTRUCTIONS.md loading
# ---------------------------------------------------------------------------

def load_instructions(workspace: Path | None) -> str:
    """Find and return the contents of ``INSTRUCTIONS.md``, or empty string."""
    candidates = [
        Path(workspace) / "INSTRUCTIONS.md" if workspace else None,
        Path.cwd() / "INSTRUCTIONS.md",
        _REPO_ROOT / "INSTRUCTIONS.md",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.read_text()
    return ""
