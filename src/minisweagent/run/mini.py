#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import copy
import os
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
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.agents.interactive_textual import TextualAgent
from minisweagent.agents.parallel_agent import ParallelAgent
from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent
from minisweagent.agents.unit_test_agent import run_discovery_pipeline, run_unit_test_agent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.run.utils.config_editor import load_and_merge_configs
from minisweagent.run.utils.save import save_traj
from minisweagent.run.utils.task_parser import _resolve_path_case
from minisweagent.utils.log import logger


def _run_discovery(kernel_path: str, kernel_name: str | None = None) -> str | tuple[str, object]:
    """Run test discovery on the resolved kernel and return formatted results for the task prompt.

    Returns:
        A formatted string for the task prompt.  The raw DiscoveryResult is
        stashed on the function object as ``_run_discovery._last_result`` so the
        caller can pass it to the task planner without a second discovery pass.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from minisweagent.tools.discovery import discover
        from minisweagent.tools.resolve_kernel_url_impl import find_resolved_clone_root
    except ImportError:
        return ""

    console = Console(highlight=False)
    console.print("\n[bold cyan]--- Test Discovery ---[/bold cyan]")
    if kernel_name:
        console.print(f"[dim]Kernel function: {kernel_name}[/dim]")
    try:
        kp = Path(kernel_path)
        # If the kernel lives inside a resolved clone, scope discovery to
        # that clone root so we don't accidentally scan the whole workspace.
        clone_root = find_resolved_clone_root(kp)
        if clone_root is not None:
            ws = clone_root
        else:
            ws = kp.parent
            for p in kp.parents:
                if (p / ".git").exists():
                    ws = p
                    break
        result = discover(workspace=ws, kernel_path=kp, interactive=False)
        _run_discovery._last_result = result  # stash for task planner
        lines = []
        kernel_stem = kp.stem.lower()
        match_terms = [kernel_stem]
        if kernel_name:
            match_terms.extend([p for p in kernel_name.lower().split("_") if len(p) > 2])

        def _is_relevant(path_str):
            return any(term in path_str.lower() for term in match_terms)

        relevant_tests = [t for t in result.tests if _is_relevant(str(t.file_path))]
        other_tests = [t for t in result.tests if not _is_relevant(str(t.file_path))][:3]
        all_display = relevant_tests + other_tests
        if all_display:
            console.print(
                f"[bold green]Found {len(result.tests)} test(s) ({len(relevant_tests)} matching '{kernel_stem}'):[/bold green]"
            )
            for t in all_display[:5]:
                marker = "+" if kernel_stem in str(t.file_path).lower() else "-"
                console.print(f"  [green]{marker}[/green] {t.file_path} [dim](confidence: {t.confidence:.1f})[/dim]")
                lines.append(f"  - {t.file_path} (confidence: {t.confidence:.1f}, command: {t.command})")
        else:
            console.print("[yellow]No existing tests found.[/yellow]")
        relevant_bench = [b for b in result.benchmarks if kernel_stem in str(b.file_path).lower()]
        if relevant_bench:
            console.print(f"[bold green]Found {len(relevant_bench)} matching benchmark(s):[/bold green]")
            for b in relevant_bench[:3]:
                console.print(f"  [green]+[/green] {b.file_path} [dim](confidence: {b.confidence:.1f})[/dim]")
                lines.append(f"  - Benchmark: {b.file_path} (confidence: {b.confidence:.1f})")
        console.print("[bold cyan]---------------------[/bold cyan]\n")
        # Include dependency graph and kernel info if available
        dep_graph_lines = []
        if result.kernels:
            k = result.kernels[0]
            dep_graph_lines.append(f"\n--- Kernel Analysis ---")
            dep_graph_lines.append(f"Type: {k.kernel_type} | Language: {k.kernel_language}")
            if k.inner_kernel_path:
                dep_graph_lines.append(f"Inner kernel: {k.inner_kernel_path} ({k.inner_kernel_language or 'unknown'})")
            if k.fusion_opportunities:
                dep_graph_lines.append(f"Fusion opportunities: {len(k.fusion_opportunities)}")
                for opp in k.fusion_opportunities[:3]:
                    dep_graph_lines.append(f"  - {opp}")

        dep_graph = result.dependency_graphs.get(result.kernels[0].kernel_name) if result.kernels else None
        if dep_graph:
            dep_graph_lines.append(f"\n{dep_graph.summary()}")

        dep_block = "\n".join(dep_graph_lines) + "\n---\n" if dep_graph_lines else ""

        if lines or dep_block:
            return (
                "\n--- Discovered Tests ---\n"
                + "\n".join(lines)
                + "\nRead these test files and reuse their reference implementations, input patterns, and tolerances.\n---\n"
                + dep_block
            )
    except Exception as e:
        console.print(f"[yellow]Discovery failed: {e}[/yellow]")
    return ""


def _inject_resolved_kernel(kernel_url: str, workspace: str | None, task: str) -> tuple[str, str | None]:
    """Resolve kernel URL to local path/line/kernel name and append to task."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from minisweagent.tools.resolve_kernel_url_impl import get_kernel_name_at_line, resolve_kernel_url
    except ImportError as e:
        raise SystemExit(f"Cannot resolve --kernel-url: geak_agent not found ({e}).") from e
    clone_into = Path(workspace) if workspace else Path.cwd()
    resolved = resolve_kernel_url(kernel_url, clone_into=clone_into)
    if resolved.get("error"):
        raise SystemExit(f"Kernel URL resolve failed: {resolved['error']}")
    path = resolved["local_file_path"]
    line_num = resolved.get("line_number")
    kernel_name = get_kernel_name_at_line(path, line_num) if line_num else None
    if line_num:
        line_info = f" Line: {line_num}"
        kernel_info = f", kernel name: {kernel_name!r}" if kernel_name else ""
        profile_hint = "When profiling, all kernels are reported; the agent can choose which to use."
    else:
        line_info = ""
        kernel_info = ""
        profile_hint = "Line number was not specified; discovery should identify the kernel(s) in the file."
    kernel_dir = str(Path(path).parent)
    output_dir = f"{kernel_dir}/optimization_output"
    oe_script = "${GEAK_OE_ROOT:-/opt/geak-oe}/examples/geak_eval/run_openevolve.py"
    block = f"""\n
--- Resolved kernel (from --kernel-url) ---
Kernel path: {path}{(" |" + line_info + kernel_info) if line_info else ""}
---

--- Workflow ---
Follow these steps IN ORDER. Do one step per response.

Step 1 - DISCOVER: Read and analyse the kernel file. Identify the kernel function, its inputs/outputs, dependencies, and any existing tests in the repo.

Step 2 - TEST GEN: Create a standalone test harness that can (a) verify correctness and (b) benchmark performance. Save it next to the kernel (e.g. {kernel_dir}/test_harness.py).

Step 3 - BENCHMARK & COMMANDMENT: Profile the baseline kernel with kernel-profile and create two artifacts:
  a) baseline_metrics.json -- latency/bandwidth numbers from profiling.
  b) COMMANDMENT.md -- the evaluation contract for OpenEvolve.
  {profile_hint}
  Profile command example: kernel-profile 'python3 {path} --profile'

Step 4 - OPTIMIZE: Do NOT edit the kernel by hand. Run the OpenEvolve optimizer:
  python3 {oe_script} {path} --iterations 10 --gpu 0 --output {output_dir}
  If you created COMMANDMENT.md and baseline_metrics.json, add:
  --commandment <path_to_COMMANDMENT.md> --baseline-metrics <path_to_baseline_metrics.json>

Step 5 - REPORT: Summarise results (speedup, best score, any errors) and submit:
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
---
"""
    return task + block, kernel_name


DEFAULT_CONFIG = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT = global_config_dir / "last_mini_run.traj.json"

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")
prompt_session = PromptSession(history=FileHistory(global_config_dir / "mini_task_history.txt"))


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path | None = typer.Option(None, "-c", "--config", help="Path to config file (overrides template selection)"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
    exit_immediately: bool = typer.Option( False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
    # Strategy mode configuration
    enable_strategies: bool = typer.Option(True, "--enable-strategies/--no-enable-strategies", help="Enable optimization strategy management (optool command). Auto-selects appropriate template.", rich_help_panel="Advanced"),
    strategy_file: str = typer.Option(".optimization_strategies.md", "--strategy-file", help="Path to strategy file (relative to workspace)", rich_help_panel="Advanced"),
    # Patch mode configuration (always enabled)
    test_command: str | None = typer.Option(None, "--test_command", "--test-command", help="Test command to run for patch validation"),
    create_test: bool = typer.Option(
        False,
        "--create-test",
        "--create_test",
        help="Auto-create/search unit tests and infer test_command when missing (or to override it).",
        rich_help_panel="Advanced",
    ),
    patch_output: Path | None = typer.Option(None, "--patch-output", help="Output directory for patch files and test results"),
    metric: str | None = typer.Option(None, "--metric", help="Metric extraction task description for LLM"),
    num_parallel: int | None = typer.Option(None, "--num-parallel", help="Number of parallel patch agents to run (only effective with --save-patch). If not specified, reads from config file."),
    repo: Path | None = typer.Option(None, "--repo", help="Repository path for parallel execution. Required when num_parallel > 1. Each agent will get an isolated workdir using git worktree."),
    gpu_ids: str | None = typer.Option(None, "--gpu-ids", help="Comma-separated GPU IDs for agents (e.g., '0,1,2,3'). For single agent, uses first GPU. Defaults to '0'."),
    # Runtime environment configuration (ported from MSA branch)
    runtime: str = typer.Option("local", "--runtime", help="Runtime environment: local, docker, or auto (auto-detects GPU availability).", rich_help_panel="Advanced"),
    docker_image: str | None = typer.Option(None, "--docker-image", help="Docker image to use when --runtime=docker.", rich_help_panel="Advanced"),
    workspace: Path | None = typer.Option(None, "--workspace", help="Workspace directory to mount in Docker.", rich_help_panel="Advanced"),
    kernel_url: str | None = typer.Option(None, "--kernel-url", help="Kernel as URL (e.g. https://github.com/.../file.py#L106). Resolved path/line/kernel name are injected into the task.", rich_help_panel="Kernel"),
) -> Any:
    # fmt: on
    configure_if_first_time()

    # 0. Runtime environment check (ported from MSA branch)
    if runtime in ("auto", "docker"):
        try:
            from minisweagent.runtime_env import RuntimeType, detect_runtime_environment
            rt_env = detect_runtime_environment()
            if runtime == "auto" and not rt_env.has_gpu:
                console.print(
                    "[bold yellow]No GPU detected locally. Consider --runtime docker.[/bold yellow]"
                )
            elif runtime == "docker":
                console.print(
                    f"[bold cyan]Docker runtime selected. Image: {docker_image or 'default'}[/bold cyan]"
                )
        except ImportError:
            if runtime == "docker":
                console.print("[bold yellow]runtime_env module not available; proceeding with local.[/bold yellow]")

    # 1. Load base config (mini.yaml - always loaded as foundation)
    base_config_path = builtin_config_dir / "mini.yaml"
    console.print(f"Loading base config: [bold green]'{base_config_path.name}'[/bold green]")
    config = yaml.safe_load(base_config_path.read_text())
    
    # 2. Select and merge template based on enable_strategies flag
    if enable_strategies:
        template_name = "mini_kernel_strategy_list.yaml"
    else:
        template_name = "mini_system_prompt.yaml"
    
    template_path = builtin_config_dir / template_name
    console.print(f"Applying template: [bold green]'{template_name}'[/bold green] (save_patch always enabled)")
    template_config = yaml.safe_load(template_path.read_text())
    config = _deep_merge(config, template_config)
    
    # 3. Load user config if explicitly specified (final override)
    if config_spec:
        config_path = get_config_path(config_spec)
        console.print(f"[dim]Applying user config from '{config_path}' (final override)[/dim]")
        user_config = yaml.safe_load(config_path.read_text())
        config = _deep_merge(config, user_config)

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

    # Backward compatibility: legacy top-level tool flags
    if "profiling" in config:
        config.setdefault("model", {}).setdefault("profiling", config["profiling"])
    if "profiling_type" in config:
        config.setdefault("agent", {}).setdefault("profiling_type", config["profiling_type"])
    if config.get("model", {}).get("profiling") and not config.get("agent", {}).get("profiling_type"):
        config.setdefault("agent", {})["profiling_type"] = "profiling"

    # Read task content - if task is a file path, read its content; otherwise use task as-is
    task_content = task
    if task:
        task_path = Path(task)
        if task_path.exists() and task_path.is_file():
            # Read file content regardless of extension (txt, md, etc.)
            task_content = task_path.read_text(encoding="utf-8")
            console.print(f"[bold green]Read task from file: {task_path}[/bold green]")
        elif not task.strip():
            # Empty task, prompt user
            task_content = None
    
    if not task_content:
        if kernel_url:
            task_content = (
                "Optimize this kernel for maximum speedup.\n"
                "Follow the workflow described in the pipeline instructions (INSTRUCTIONS.md).\n"
                "Use the discovered tests and benchmarks listed above for correctness and performance.\n"
                "Report final speedup when done."
            )
            console.print("[bold green]Using default kernel optimization task[/bold green]")
        else:
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

    # Resolve --kernel-url to local path, line, and kernel name; inject into task
    _resolved_kernel_path = None
    _resolved_kernel_name = None
    if kernel_url:
        task_content, _resolved_kernel_name = _inject_resolved_kernel(kernel_url, str(workspace) if workspace else None, task_content)
        import re as _re
        _m = _re.search(r"Kernel path: (\S+)", task_content)
        if _m:
            _resolved_kernel_path = _m.group(1)
    # Run test discovery and inject results into task
    if _resolved_kernel_path:
        discovery_block = _run_discovery(_resolved_kernel_path, _resolved_kernel_name)
        if discovery_block:
            task_content = task_content + discovery_block
    elif task and '.md' in task:
        with open(task, encoding="utf-8") as f:
            task = f.read()

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
    # Set use_strategy_manager in model config based on enable_strategies flag
    config.setdefault("model", {})["use_strategy_manager"] = enable_strategies
    model = get_model(model_name, config.get("model", {}))
    env = LocalEnvironment(**config.get("env", {}))

    # Load and merge configurations: Command-line > extra_config from yaml > auto-detect
    result = load_and_merge_configs(
        config, repo, test_command, metric, num_parallel, gpu_ids, patch_output,
        task_content, yolo, model, console
    )
    if result == (None, None, None, None, None, None, None):
        console.print("[bold yellow]Continuing without automatic patch saving. You can still interact with the agent.[/bold yellow]")
        # Keep original None values since user aborted
        repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output, kernel_name = None, None, None, None, [0], None, None
    else:
        repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output, kernel_name = result

    if create_test or not test_command:
        if not repo:
            raise ValueError("repo is required for --create-test or when test_command is missing. Please pass --repo.")

        # Step 0a: Run content-based discovery (fast, free, no LLM)
        discovery_context = ""
        _kernel_path = (
            Path(_resolved_kernel_path)
            if _resolved_kernel_path
            else Path(task)
            if task and Path(task).is_file()
            else None
        )
        if _kernel_path or repo:
            console.print("[bold cyan]Running content-based test discovery...[/bold cyan]")
            discovery_context = run_discovery_pipeline(
                kernel_path=_kernel_path or repo,
                repo=repo,
            )
            if discovery_context:
                console.print("[dim]Discovery found candidates — feeding into UnitTestAgent.[/dim]")
            else:
                console.print("[dim]Discovery found nothing — UnitTestAgent will search/create from scratch.[/dim]")

        # Step 0b: Run UnitTestAgent with discovery context
        console.print(
            "[bold yellow]Running UnitTestAgent to find or create test command...[/bold yellow]"
        )
        test_command = run_unit_test_agent(
            model=get_model(model_name, config.get("model", {})),
            repo=repo,
            kernel_name=kernel_name or "unknown",
            log_dir=patch_output,
            discovery_context=discovery_context,
        )
        console.print(f"[bold green]Using UnitTestAgent test_command:[/bold green] {test_command}")
    
    # ============ Step 1: Choose base agent class ============
    # Based on enable_strategies flag, select appropriate agent and template
    if enable_strategies:
        # Use strategy agent with mini_kernel_strategy_list.yaml template
        base_agent_class = StrategyInteractiveAgent
        console.print(f"[bold cyan]Using Strategy Agent with strategy file: {strategy_file}[/bold cyan]")
    else:
        # Use interactive agent with mini_system_prompt.yaml template
        # Choose between visual (Textual) and non-visual (Interactive) mode
        if visual == (os.getenv("MSWEA_VISUAL_MODE_DEFAULT", "false") == "false"):
            base_agent_class = TextualAgent
        else:
            base_agent_class = InteractiveAgent
        console.print(f"[bold cyan]Using Interactive Agent (visual={'on' if base_agent_class == TextualAgent else 'off'})[/bold cyan]")
    
    # Mode (yolo/confirm/human) is set via config and applies to all InteractiveAgent subclasses
    
    # ============ Step 2: Configure agent settings ============
    agent_config = config.get("agent", {})
    
    # Add strategy manager settings
    agent_config["use_strategy_manager"] = enable_strategies
    if enable_strategies:
        agent_config["strategy_file_path"] = strategy_file
    
    # Configure save_patch settings (always enabled)
    agent_config["save_patch"] = True
    agent_config["test_command"] = test_command or config.get("patch", {}).get("test_command")
    patch_dir = patch_output or config.get("patch", {}).get("patch_output_dir") or (global_config_dir / "patches")
    agent_config["patch_output_dir"] = str(patch_dir)
    agent_config["metric"] = metric or config.get("patch", {}).get("metric")
    
    # Create log directory and prepare log file path
    log_dir = Path(patch_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    agent_log_file = log_dir / "mini_agent.log"
    
    # ============ Step 3: Use ParallelAgent (supports both single and parallel execution) ============
    agent_class = ParallelAgent
    agent_config["agent_class"] = base_agent_class
    agent_config["num_parallel"] = num_parallel or 1
    agent_config["gpu_ids"] = parsed_gpu_ids
    
    if num_parallel and num_parallel > 1:
        console.print(f"[bold cyan]Using Parallel Mode: {num_parallel} agents on GPUs {parsed_gpu_ids}[/bold cyan]")
        
        # Configure repo path for parallel execution (preserve filesystem case)
        repo_path = repo or config.get("patch", {}).get("repo")
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

        # Generate dynamic optimization tasks from discovery results (if available)
        discovery_result = getattr(_run_discovery, "_last_result", None)
        if discovery_result and discovery_result.kernels:
            try:
                from minisweagent.run.task_planner import build_optimization_tasks
                tasks = build_optimization_tasks(
                    discovery_result=discovery_result,
                    base_task_context=task_content,
                    agent_class=base_agent_class,
                )
                if tasks:
                    agent_config["tasks"] = tasks
                    console.print(f"[bold cyan]Task planner: {len(tasks)} optimization tasks generated (pool mode)[/bold cyan]")
                    for t in tasks[:6]:
                        console.print(f"  [dim]- [{t.priority:2d}] {t.label} ({t.kernel_language})[/dim]")
                    if len(tasks) > 6:
                        console.print(f"  [dim]  ... and {len(tasks) - 6} more[/dim]")
            except Exception as e:
                console.print(f"[yellow]Task planner failed ({e}), falling back to homogeneous mode[/yellow]")
    else:
        console.print("[bold cyan]Using Single Agent Mode[/bold cyan]")
        console.print(f"[dim]Using GPU: {parsed_gpu_ids[0]}[/dim]")
        # Set HIP_VISIBLE_DEVICES for single agent GPU isolation
        env.config.env = env.config.env or {}
        env.config.env["HIP_VISIBLE_DEVICES"] = str(parsed_gpu_ids[0])
    
    # Create and run agent
    agent = agent_class(model, env, **agent_config)
    agent.log_file = agent_log_file
    console.print(f"[dim]Agent log: {agent_log_file}[/dim]")

    # Load INSTRUCTIONS.md if available (pipeline reference for the agent)
    instructions_content = ""
    for instructions_candidate in [
        Path(workspace) / "INSTRUCTIONS.md" if workspace else None,
        Path.cwd() / "INSTRUCTIONS.md",
        Path(__file__).resolve().parent.parent.parent.parent / "INSTRUCTIONS.md",
    ]:
        if instructions_candidate and instructions_candidate.is_file():
            instructions_content = instructions_candidate.read_text()
            console.print(f"Loaded pipeline instructions from [bold green]'{instructions_candidate}'[/bold green]")
            break

    try:
        exit_status, result = agent.run(
            task_content, instructions=instructions_content,
            output=output,
            save_traj_fn=save_traj,
            console=console,
            model_factory=lambda: get_model(model_name, config.get("model", {})),
            env_factory=lambda: LocalEnvironment(**copy.deepcopy(config.get("env", {}))),
        )
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        _exit_status, result = type(e).__name__, str(e)
    
    return agent


if __name__ == "__main__":
    app()
