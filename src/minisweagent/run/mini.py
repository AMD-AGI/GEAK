# Modifications Copyright(C)[2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import copy
import json
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
from minisweagent.agents.unit_test_agent import format_discovery_for_agent
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
    try:
        from automated_test_discovery.server import discover as atd_discover

        from minisweagent.tools.discovery_types import DiscoveryResult
    except ImportError:
        return ""

    console = Console(highlight=False)
    console.print("\n[bold cyan]--- Test Discovery ---[/bold cyan]")
    if kernel_name:
        console.print(f"[dim]Kernel function: {kernel_name}[/dim]")
    try:
        kp = Path(kernel_path)
        _discover_fn = getattr(atd_discover, "fn", atd_discover)
        disc_dict = _discover_fn(
            kernel_path=str(kp),
            output_dir=str(kp.parent),
        )

        result = DiscoveryResult.from_dict(disc_dict, kp)
        _run_discovery._last_result = result

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

        if lines:
            return (
                "\n--- Discovered Tests ---\n"
                + "\n".join(lines)
                + "\nRead these test files and reuse their reference implementations, input patterns, and tolerances.\n---\n"
            )
    except Exception as e:
        console.print(f"[yellow]Discovery failed: {e}[/yellow]")
    return ""


from minisweagent.run.pipeline_helpers import (
    create_validated_harness,
    extract_harness_path,
    run_baseline_profile,
)


def _inject_resolved_kernel(kernel_url: str, workspace: str | None, task: str) -> tuple[str, str | None]:
    """Resolve kernel URL to local path/line/kernel name and append to task."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from minisweagent.tools.resolve_kernel_url_impl import get_kernel_name_at_line, resolve_kernel_url
    except ImportError as e:
        raise SystemExit(f"Cannot resolve --kernel-url: resolve_kernel_url_impl not found ({e}).") from e
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
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic_model.AnthropicModel')", rich_help_panel="Advanced"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path | None = typer.Option(None, "-c", "--config", help="Path to config file (overrides template selection)"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT, "--traj-output", "--output", help="Output trajectory file (legacy; trajectories are also saved inside --patch-output)"),
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
    patch_output: Path | None = typer.Option(None, "-o", "--patch-output", help="Output directory for pipeline results, patches, and test artifacts"),
    metric: str | None = typer.Option(None, "--metric", help="Metric extraction task description for LLM"),
    num_parallel: int | None = typer.Option(None, "--num-parallel", help="Number of parallel patch agents to run (only effective with --save-patch). If not specified, reads from config file."),
    repo: Path | None = typer.Option(None, "--repo", help="Repository path for parallel execution. Required when num_parallel > 1. Each agent will get an isolated workdir using git worktree."),
    gpu_ids: str | None = typer.Option(None, "--gpu-ids", help="Comma-separated GPU IDs for agents (e.g., '0,1,2,3'). For single agent, uses first GPU. Defaults to '0'."),
    # Runtime environment configuration (ported from MSA branch)
    runtime: str = typer.Option("local", "--runtime", help="Runtime environment: local, docker, or auto (auto-detects GPU availability).", rich_help_panel="Advanced"),
    docker_image: str | None = typer.Option(None, "--docker-image", help="Docker image to use when --runtime=docker.", rich_help_panel="Advanced"),
    workspace: Path | None = typer.Option(None, "--workspace", help="Workspace directory to mount in Docker.", rich_help_panel="Advanced"),
    kernel_url: str | None = typer.Option(None, "--kernel-url", help="Kernel as URL (e.g. https://github.com/.../file.py#L106). Resolved path/line/kernel name are injected into the task.", rich_help_panel="Kernel"),
    max_rounds: int | None = typer.Option(None, "--max-rounds", help="Maximum optimisation rounds for the orchestrator (default: GEAK_MAX_ROUNDS env or 5).", rich_help_panel="Advanced"),
    allowed_agents: str | None = typer.Option(None, "--allowed-agents", help="Comma-separated list of allowed agent types (e.g. swe_agent,strategy_agent). Sets GEAK_ALLOWED_AGENTS.", rich_help_panel="Advanced"),
    excluded_agents: str | None = typer.Option(None, "--excluded-agents", help="Comma-separated list of excluded agent types (e.g. openevolve). Sets GEAK_EXCLUDED_AGENTS.", rich_help_panel="Advanced"),
    heterogeneous: bool = typer.Option(False, "--heterogeneous", help="Use LLM-generated diverse optimization tasks (requires preprocessing/discovery). Default: homogeneous.", rich_help_panel="Advanced"),
    from_task: Path | None = typer.Option(None, "--from-task", help="Deprecated: use --task with a YAML-frontmatter .md file instead.", hidden=True),
    rag: bool = typer.Option(False, "--rag", help="Enable RAG retrieval from AMD/NVIDIA knowledge base"),
    debug: bool = typer.Option(False, "-d", "--debug", help="Enable debug output (only with --rag)"),
) -> Any:
    # fmt: on
    configure_if_first_time()

    if allowed_agents:
        os.environ["GEAK_ALLOWED_AGENTS"] = allowed_agents
    if excluded_agents:
        os.environ["GEAK_EXCLUDED_AGENTS"] = excluded_agents

    # Deprecated --from-task: map to --task for backward compatibility
    _task_worktree: Path | None = None
    _codebase_ctx_text: str | None = None
    if from_task:
        if not task:
            task = str(from_task)
        console.print("[dim]Note: --from-task is deprecated. Use --task with a YAML-frontmatter .md file instead.[/dim]")

    # 0. Runtime environment check (ported from MSA branch)
    if runtime in ("auto", "docker"):
        try:
            from minisweagent.runtime_env import detect_runtime_environment
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

    # Read task content - auto-detect structured task files (YAML frontmatter)
    _tf_meta: dict | None = None
    task_content = task
    if task:
        task_path = Path(task)
        try:
            if task_path.exists() and task_path.is_file():
                raw = task_path.read_text(encoding="utf-8")
                if raw.lstrip().startswith("---"):
                    # Structured task file with YAML frontmatter
                    from minisweagent.run.task_file import (
                        create_worktree as _create_wt,
                    )
                    from minisweagent.run.task_file import (
                        is_git_repo as _is_git,
                    )
                    from minisweagent.run.task_file import (
                        read_task_file as _read_tf,
                    )
                    _tf_meta, _tf_body = _read_tf(task_path)
                    task_content = _tf_body
                    console.print(f"[bold cyan]Loading structured task file: {task_path}[/bold cyan]")

                    _tf_agent_type = _tf_meta.get("agent_type", "strategy_agent")
                    if _tf_agent_type == "openevolve":
                        console.print(
                            "[bold yellow]Task has agent_type=openevolve. "
                            "Redirecting to openevolve-worker...[/bold yellow]"
                        )
                        _oe_args = ["openevolve-worker", "--from-task", str(task_path.resolve())]
                        if gpu_ids:
                            _oe_args += ["--gpu", gpu_ids.split(",")[0]]
                        import subprocess
                        raise SystemExit(subprocess.call(_oe_args))

                    if not repo and _tf_meta.get("kernel_path"):
                        repo = Path(_tf_meta["kernel_path"]).resolve().parent
                    if not repo and _tf_meta.get("repo_root"):
                        repo = Path(_tf_meta["repo_root"]).resolve()
                    if not test_command and _tf_meta.get("test_command"):
                        test_command = _tf_meta["test_command"]

                    if not patch_output:
                        _task_path = task_path.resolve()
                        _round_dir = _task_path.parent.name
                        _pipeline_root = _task_path.parent.parent.parent
                        patch_output = _pipeline_root / "results" / _round_dir / _task_path.stem
                        console.print(f"[dim]Derived --patch-output: {patch_output}[/dim]")

                    _tf_repo = Path(_tf_meta["repo_root"]).resolve() if _tf_meta.get("repo_root") else (repo.resolve() if repo else None)
                    if _tf_repo and _tf_repo.is_dir() and _is_git(_tf_repo):
                        _wt_dest = Path(patch_output) / "_worktree"
                        console.print(f"[bold cyan]Creating isolated worktree at {_wt_dest}...[/bold cyan]")
                        _task_worktree = _create_wt(_tf_repo, _wt_dest)
                        repo = _task_worktree

                    yolo = True

                    _skip_lines = [
                        "NOTE: This task was generated by the task-generator pipeline. "
                        "Baseline profiling and performance metrics are already available "
                        "in the files listed below. Do NOT re-run baseline profiling or "
                        "establish baseline performance -- skip directly to analyzing "
                        "the provided data and implementing the optimization.",
                        "",
                    ]
                    if _tf_meta.get("kernel_path"):
                        _skip_lines.append(f"KERNEL FILE TO EDIT: {_tf_meta['kernel_path']}")
                    if test_command:
                        _skip_lines.append(f"TEST COMMAND: {test_command}")
                    if _tf_meta.get("repo_root"):
                        _skip_lines.append(f"REPO ROOT: {_tf_meta['repo_root']}")

                    _commandment_path = _tf_meta.get("commandment")
                    if _commandment_path and Path(_commandment_path).exists():
                        _cmd_text = Path(_commandment_path).read_text().strip()
                        _skip_lines.append("")
                        _skip_lines.append("## COMMANDMENT (evaluation contract -- you MUST follow these rules)")
                        _skip_lines.append(_cmd_text)

                    _baseline_path = _tf_meta.get("baseline_metrics")
                    if _baseline_path and Path(_baseline_path).exists():
                        _bm = json.loads(Path(_baseline_path).read_text())
                        _dur = _bm.get("duration_us", "unknown")
                        _bn = _bm.get("bottleneck", "unknown")
                        _skip_lines.append("")
                        _skip_lines.append("## Baseline Performance (your optimization must improve on these)")
                        _skip_lines.append(f"Total duration: {_dur} us")
                        _skip_lines.append(f"Bottleneck: {_bn}")
                        _top = _bm.get("top_kernels", [])
                        if _top:
                            _skip_lines.append("Top kernels by duration:")
                            for _k in _top[:5]:
                                _bn_tag = f" [{_k['bottleneck']}]" if _k.get("bottleneck") else ""
                                _skip_lines.append(
                                    f"  - {_k.get('name', '?')}: {_k.get('duration_us', '?')} us "
                                    f"({_k.get('pct_of_total', '?')}%){_bn_tag}"
                                )

                    _prof_path = _tf_meta.get("profiling")
                    if _prof_path and Path(_prof_path).exists():
                        _skip_lines.append("")
                        _skip_lines.append(f"PROFILING DATA: {_prof_path}")
                        _skip_lines.append("(Read this file for detailed per-kernel profiling metrics)")

                    _codebase_ctx_path = _tf_meta.get("codebase_context")
                    if _codebase_ctx_path and Path(_codebase_ctx_path).exists():
                        _codebase_ctx_text = Path(_codebase_ctx_path).read_text().strip()
                        _skip_lines.append("")
                        _skip_lines.append("## Codebase Context (repo structure and key files)")
                        _skip_lines.append(_codebase_ctx_text)

                    _skip_lines.append("")
                    _skip_note = "\n".join(_skip_lines) + "\n"
                    if task_content and not task_content.startswith("NOTE: This task was generated"):
                        task_content = _skip_note + task_content
                else:
                    task_content = raw
                    console.print(f"[bold green]Read task from file: {task_path}[/bold green]")
            elif not task.strip():
                task_content = None
        except OSError:
            pass
    
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

    _env_kwargs = config.get("env", {})
    if rag:
        try:
            from minisweagent.mcp_integration.mcp_environment import MCPEnabledEnvironment
            from minisweagent.mcp_integration.prompts import INSTANCE_TEMPLATE, SYSTEM_TEMPLATE
            from minisweagent.mcp_integration.run_agent import DebugMCPEnvironment
        except ImportError as e:
            console.print("[red]Error: RAG retrieval requires langchain dependencies. Run: pip install -e '.[langchain]'[/red]")
            console.print(f"[red]Import error: {e}[/red]")
            raise typer.Exit(1)

        if debug:
            env = DebugMCPEnvironment(**_env_kwargs)
            console.print("[bold yellow]Debug mode enabled[/bold yellow]")
        else:
            env = MCPEnabledEnvironment(**_env_kwargs)

        config.setdefault("agent", {})["system_template"] = SYSTEM_TEMPLATE
        config.setdefault("agent", {})["instance_template"] = INSTANCE_TEMPLATE
        console.print("[bold green]RAG knowledge retrieval enabled[/bold green]")
    else:
        env = LocalEnvironment(**_env_kwargs)

    # Load and merge configurations: Command-line > extra_config from yaml > auto-detect
    # When running from a task file, skip auto-detection from the task body:
    # all config is already extracted from task metadata (repo, test_command, etc.)
    _detect_content = None if _tf_meta else task_content
    result = load_and_merge_configs(
        config, repo, test_command, metric, num_parallel, gpu_ids, patch_output,
        _detect_content, yolo, model, console
    )
    if result == (None, None, None, None, None, None, None):
        console.print("[bold yellow]Continuing without automatic patch saving. You can still interact with the agent.[/bold yellow]")
        # Keep original None values since user aborted
        repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output, kernel_name = None, None, None, None, [0], None, None
    else:
        repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output, kernel_name = result

    # Set the agent's working directory to the repo/worktree so bash commands
    # and save_and_test run in the correct location (not the container root).
    if repo:
        env.config.cwd = str(Path(repo).resolve())

    # ============ Full pipeline mode: geak <url> ============
    # When a kernel URL is provided (and we're not in a structured-task sub-agent mode),
    # route through the preprocessor -> orchestrator pipeline instead of the
    # legacy monolithic flow.
    if kernel_url and not _tf_meta:
        from minisweagent.run.orchestrator import run_orchestrator
        from minisweagent.run.preprocessor import run_preprocessor

        _pipeline_output = patch_output or Path("geak_output")
        console.print("[bold cyan]--- GEAK Full Pipeline Mode ---[/bold cyan]")
        console.print(f"[dim]Kernel URL: {kernel_url}[/dim]")
        console.print(f"[dim]Output dir: {_pipeline_output}[/dim]")

        preprocess_ctx = run_preprocessor(
            kernel_url,
            output_dir=_pipeline_output,
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
            output_dir=_pipeline_output,
            max_rounds=max_rounds,
            heterogeneous=heterogeneous,
            console=console,
            agent_config=config.get("agent"),
        )

        console.print("\n[bold green]Pipeline complete.[/bold green]")
        if report:
            console.print(f"[dim]{json.dumps(report, indent=2, default=str)[:500]}[/dim]")
        return None

    if not _tf_meta and (create_test or not test_command):
        if not repo:
            raise ValueError("repo is required for --create-test or when test_command is missing. Please pass --repo.")

        # Step 0a: Format discovery context for UnitTestAgent
        # Reuse the stashed result from _run_discovery() if available so we
        # don't run a second redundant scan of the same codebase.
        discovery_context = ""
        _stashed_result = getattr(_run_discovery, "_last_result", None)
        if _stashed_result:
            console.print("[bold cyan]Formatting stashed discovery results for UnitTestAgent...[/bold cyan]")
            discovery_context = format_discovery_for_agent(_stashed_result)
        else:
            # No stashed result (e.g. no --kernel-url) — run discovery once
            _kernel_path = None
            if _resolved_kernel_path:
                _kernel_path = Path(_resolved_kernel_path)
            elif task and not _tf_meta:
                try:
                    _p = Path(task)
                    if _p.is_file():
                        _kernel_path = _p
                except OSError:
                    pass
            if _kernel_path or repo:
                console.print("[bold cyan]Running content-based test discovery...[/bold cyan]")
                discovery_context = _run_discovery(
                    kernel_path=str(_kernel_path or repo),
                )

        if discovery_context:
            console.print("[dim]Discovery results ready — feeding into UnitTestAgent.[/dim]")
        else:
            console.print("[dim]No discovery results — UnitTestAgent will search/create from scratch.[/dim]")

        # Step 0b: Detect PyTorch→FlyDSL translation task
        from minisweagent.agents.unit_test_agent import detect_pytorch_translation_task

        _is_pytorch_translation = False
        _effective_kernel_path: Path | None = None
        if _resolved_kernel_path:
            _effective_kernel_path = Path(_resolved_kernel_path)
        elif repo and kernel_name and kernel_name != "unknown":
            _candidate = repo / f"{kernel_name}.py"
            if _candidate.exists():
                _effective_kernel_path = _candidate

        if _effective_kernel_path and detect_pytorch_translation_task(_effective_kernel_path):
            _is_pytorch_translation = True
            console.print("[bold magenta]Detected PyTorch-to-FlyDSL translation task[/bold magenta]")

        # Step 0c: Run UnitTestAgent with harness validation + retry
        console.print(
            "[bold yellow]Running UnitTestAgent to create test harness...[/bold yellow]"
        )
        test_command, _harness_results = create_validated_harness(
            model=get_model(model_name, config.get("model", {})),
            repo=repo,
            kernel_name=kernel_name or "unknown",
            log_dir=patch_output,
            discovery_context=discovery_context,
            gpu_id=parsed_gpu_ids[0] if parsed_gpu_ids else 0,
            pytorch_translation=_is_pytorch_translation,
            kernel_path=_effective_kernel_path if _is_pytorch_translation else None,
        )
        console.print(f"[bold green]Using UnitTestAgent test_command:[/bold green] {test_command}")
        for _hr in (_harness_results or []):
            _s = "PASS" if _hr["success"] else "FAIL"
            console.print(f"  [dim]--{_hr['mode']}: {_s} ({_hr['duration_s']}s)[/dim]")

    # ============ Step 0c: Pre-agent baseline profiling + commandment generation ============
    # These run once up front so the task generator has richer context.
    # Results are stored in _pre_agent_* variables and passed to the task generator.
    _pre_agent_profiling: dict | None = None
    _pre_agent_baseline_metrics: dict | None = None
    _pre_agent_commandment: str | None = None

    if _resolved_kernel_path and test_command and (num_parallel and num_parallel > 1):
        # -- Baseline profiling --
        try:
            console.print("[bold cyan]--- Pre-agent Baseline Profiling ---[/bold cyan]")
            _pre_agent_profiling = run_baseline_profile(
                test_command, gpu_id=parsed_gpu_ids[0],
            )
        except Exception as e:
            console.print(f"[yellow]Pre-agent profiling failed ({e}); tasks will use discovery only[/yellow]")

        # -- Baseline metrics (from profiling) --
        if _pre_agent_profiling:
            try:
                from minisweagent.baseline_metrics import build_baseline_metrics
                _pre_agent_baseline_metrics = build_baseline_metrics(
                    _pre_agent_profiling, include_all=True,
                )
                dur = _pre_agent_baseline_metrics.get("duration_us", "?")
                bn = _pre_agent_baseline_metrics.get("bottleneck", "?")
                console.print(f"[bold green]Baseline: {dur} us, bottleneck={bn}[/bold green]")
            except Exception as e:
                console.print(f"[yellow]Baseline metrics extraction failed ({e})[/yellow]")

        # -- Commandment generation --
        if _resolved_kernel_path:
            try:
                from minisweagent.tools.commandment import generate_commandment
                from minisweagent.tools.discovery_types import _infer_kernel_language
                _harness_path = extract_harness_path(test_command)
                _kl = _infer_kernel_language(Path(_resolved_kernel_path), "")
                _pre_agent_commandment = generate_commandment(
                    kernel_path=_resolved_kernel_path,
                    harness_path=_harness_path,
                    repo_root=repo,
                    kernel_language=_kl,
                )
                console.print("[bold green]COMMANDMENT.md generated (pre-agent)[/bold green]")
            except Exception as e:
                console.print(f"[yellow]Commandment generation failed ({e})[/yellow]")

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

    # Pass codebase context to agent config so SubAgentTool children receive it
    if _codebase_ctx_text:
        agent_config["codebase_context"] = _codebase_ctx_text

    # Create log directory and prepare log file path
    log_dir = Path(patch_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    agent_log_file = log_dir / "mini_agent.log"
    
    # ============ Step 3: Use ParallelAgent (supports both single and parallel execution) ============
    agent_class = ParallelAgent
    agent_config["agent_class"] = base_agent_class
    agent_config["num_parallel"] = num_parallel or 1
    agent_config["gpu_ids"] = parsed_gpu_ids

    # Ensure all agents (single and parallel) use the same benchmark
    # iteration count as the orchestrator evaluation.
    from minisweagent.run.pipeline_helpers import DEFAULT_EVAL_BENCHMARK_ITERATIONS
    _bench_extra = f"--iterations {DEFAULT_EVAL_BENCHMARK_ITERATIONS}"
    config.setdefault("env", {}).setdefault("env", {}).setdefault(
        "GEAK_BENCHMARK_EXTRA_ARGS", _bench_extra,
    )

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

        # Generate dynamic optimization tasks (LLM-assisted, heterogeneous mode only)
        discovery_result = getattr(_run_discovery, "_last_result", None)
        if heterogeneous and discovery_result and discovery_result.kernels and model:
            try:
                from minisweagent.run.pipeline_helpers import inject_pipeline_context
                from minisweagent.run.task_generator import generate_tasks_from_content

                tasks = generate_tasks_from_content(
                    discovery_result=discovery_result,
                    base_task_context=task_content,
                    agent_class=base_agent_class,
                    model=model,
                    profiling_result=_pre_agent_profiling,
                    commandment_content=_pre_agent_commandment,
                    baseline_metrics=_pre_agent_baseline_metrics,
                )
                if tasks:
                    for t in tasks:
                        t.task, t.config = inject_pipeline_context(
                            t.task,
                            t.config,
                            commandment_text=_pre_agent_commandment,
                            baseline_metrics=_pre_agent_baseline_metrics,
                            kernel_path=_resolved_kernel_path,
                            repo_root=str(repo) if repo else None,
                            test_command=test_command,
                            codebase_context=_codebase_ctx_text,
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
    else:
        console.print("[bold cyan]Using Single Agent Mode[/bold cyan]")
        console.print(f"[dim]Using GPU: {parsed_gpu_ids[0]}[/dim]")
        env.config.env = env.config.env or {}
        env.config.env["HIP_VISIBLE_DEVICES"] = str(parsed_gpu_ids[0])
        env.config.env.setdefault("GEAK_BENCHMARK_EXTRA_ARGS", _bench_extra)
    
    # Create and run agent
    agent = agent_class(model, env, **agent_config)
    agent.log_file = agent_log_file
    if _tf_meta and _tf_meta.get("repo_root"):
        agent.base_repo_path = Path(_tf_meta["repo_root"]).resolve()
    console.print(f"[bold cyan]Agent log: {agent_log_file}[/bold cyan]")
    console.print(f"[dim]Tip: tail -f {agent_log_file}[/dim]")

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
            env_factory=lambda _repo=repo: (MCPEnabledEnvironment if rag else LocalEnvironment)(
                **{**copy.deepcopy(_env_kwargs), **({"cwd": str(Path(_repo).resolve())} if _repo else {})}
            ),
        )
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        _exit_status, result = type(e).__name__, str(e)
    
    return agent


if __name__ == "__main__":
    app()
