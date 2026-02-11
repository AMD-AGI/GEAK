#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import os
import sys
import traceback
from pathlib import Path
from typing import Any

import typer
import yaml
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from geakagent import global_config_dir
from geakagent.agents.interactive import InteractiveAgent
from geakagent.agents.interactive_textual import TextualAgent
from geakagent.agents.patch_agent import PatchAgent
from geakagent.config import builtin_config_dir, get_config_path
from geakagent.environments.local import LocalEnvironment
from geakagent.environments.docker import DockerEnvironment
from geakagent.models import get_model
from geakagent.run.extra.config import configure_if_first_time
from geakagent.run.utils.save import save_traj
from geakagent.runtime_env import (
    prompt_runtime_environment,
    get_runtime_config_for_agent,
    RuntimeType,
    display_runtime_info,
)
from geakagent.utils.log import logger

def _run_discovery(kernel_path: str, kernel_name: str | None = None) -> str:
    """Run test discovery on the resolved kernel and return formatted results for the task prompt."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from geak_agent.mcp_tools.discovery import discover
    except ImportError:
        return ""

    console = Console(highlight=False)
    console.print("\n[bold cyan]━━━ Test Discovery ━━━[/bold cyan]")
    if kernel_name:
        console.print(f"[dim]Kernel function: {kernel_name}[/dim]")
    try:
        kp = Path(kernel_path)
        # Find repo root (parent with .git) for workspace scope
        ws = kp.parent
        for p in kp.parents:
            if (p / ".git").exists():
                ws = p
                break
        result = discover(workspace=ws, kernel_path=kp, interactive=False)
        lines = []
        # Match by both filename stem and kernel function name
        kernel_stem = kp.stem.lower()  # e.g. "rope" from "rope.py"
        match_terms = [kernel_stem]
        if kernel_name:
            # Add kernel function name parts (e.g. "rope_fwd" -> ["rope", "fwd"])
            match_terms.extend([p for p in kernel_name.lower().split("_") if len(p) > 2])
        def _is_relevant(path_str):
            path_lower = path_str.lower()
            return any(term in path_lower for term in match_terms)
        # Show kernel-relevant tests first (name match), then top others
        relevant_tests = [t for t in result.tests if _is_relevant(str(t.file_path))]
        other_tests = [t for t in result.tests if not _is_relevant(str(t.file_path))][:3]
        all_display = relevant_tests + other_tests
        if all_display:
            console.print(f"[bold green]Found {len(result.tests)} test(s) ({len(relevant_tests)} matching '{kernel_stem}'):[/bold green]")
            for t in all_display[:5]:
                marker = "★" if kernel_stem in str(t.file_path).lower() else "·"
                console.print(f"  [green]{marker}[/green] {t.file_path} [dim](confidence: {t.confidence:.1f})[/dim]")
                lines.append(f"  - {t.file_path} (confidence: {t.confidence:.1f}, command: {t.command})")
        else:
            console.print("[yellow]No existing tests found.[/yellow]")
        relevant_bench = [b for b in result.benchmarks if kernel_stem in str(b.file_path).lower()]
        if relevant_bench:
            console.print(f"[bold green]Found {len(relevant_bench)} matching benchmark(s):[/bold green]")
            for b in relevant_bench[:3]:
                console.print(f"  [green]★[/green] {b.file_path} [dim](confidence: {b.confidence:.1f})[/dim]")
                lines.append(f"  - Benchmark: {b.file_path} (confidence: {b.confidence:.1f})")
        console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━[/bold cyan]\n")

        if lines:
            return "\n--- Discovered Tests ---\n" + "\n".join(lines) + "\nRead these test files and reuse their reference implementations, input patterns, and tolerances.\n---\n"
    except Exception as e:
        console.print(f"[yellow]Discovery failed: {e}[/yellow]")
    return ""


def _inject_resolved_kernel(kernel_url: str, workspace: str | None, task: str) -> tuple[str, str | None]:
    """Resolve kernel URL to local path/line/kernel name and append to task. Returns (task, kernel_name or None). Raises on resolve error."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from geak_agent.resolve_kernel_url import resolve_kernel_url, get_kernel_name_at_line
    except ImportError as e:
        raise SystemExit(f"Cannot resolve --kernel-url: geak_agent not found ({e}). Run from repo root or install geak_agent.") from e
    clone_into = (Path(workspace) if workspace else Path.cwd())
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
    oe_script = f"${{GEAK_OE_ROOT:-/opt/geak-oe}}/examples/geak_eval/run_openevolve.py"

    block = f"""\n
--- Resolved kernel (from --kernel-url) ---
Kernel path: {path}{(' |' + line_info + kernel_info) if line_info else ''}
---

--- Workflow ---
Follow these steps IN ORDER. Do one step per response.

Step 1 - DISCOVER: Read and analyse the kernel file. Identify the kernel function, its inputs/outputs, dependencies, and any existing tests in the repo.

Step 2 - TEST GEN: Create a standalone test harness that can (a) verify correctness and (b) benchmark performance. Save it next to the kernel (e.g. {kernel_dir}/test_harness.py).

Step 3 - BENCHMARK & COMMANDMENT: Profile the baseline kernel with kernel-profile and create two artifacts:
  a) baseline_metrics.json — latency/bandwidth numbers from profiling.
  b) COMMANDMENT.md — the evaluation contract for OpenEvolve.
  {profile_hint}
  Profile command example: kernel-profile 'python3 {path} --profile'

  <critical>
  COMMANDMENT.md FORMAT RULES — every section body must contain ONLY executable shell commands, one per line.
  No markdown, no comments, no descriptions, no blank lines, no headers inside the body.
  Example of a CORRECT COMMANDMENT.md:

  ## SETUP
  export HIP_VISIBLE_DEVICES=0
  export PYTHONPATH=/workspace/myrepo:$PYTHONPATH

  ## CORRECTNESS
  python3 /workspace/test_harness.py --correctness

  ## PROFILE
  python3 /workspace/test_harness.py --baseline --output /tmp/profile_result.json

  WRONG (will break OpenEvolve — do NOT do this):
  ## SETUP
  Environment setup commands (run once before evaluation):
  export HIP_VISIBLE_DEVICES=0
  </critical>

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
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'geakagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
    exit_immediately: bool = typer.Option( False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
    save_patch: bool = typer.Option(False, "--save-patch", help="Save git patches and test results"),
    test_command: str | None = typer.Option(None, "--test-command", help="Test command to run for patch validation"),
    patch_output: Path | None = typer.Option(None, "--patch-output", help="Output directory for patch files and test results"),
    metric: str | None = typer.Option(None, "--metric", help="Metric extraction task description for LLM"),
    runtime: str | None = typer.Option(None, "--runtime", help="Runtime environment: 'local', 'docker', or 'auto' (default: auto)", rich_help_panel="Runtime"),
    docker_image: str | None = typer.Option(None, "--docker-image", help="Docker image to use (default: lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x)", rich_help_panel="Runtime"),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace directory to mount in Docker", rich_help_panel="Runtime"),
    no_runtime_check: bool = typer.Option(False, "--no-runtime-check", help="Skip runtime environment detection", rich_help_panel="Runtime"),
    kernel_url: str | None = typer.Option(None, "--kernel-url", help="Kernel as URL (e.g. https://github.com/.../file.py#L106). Resolved path/line/kernel name are injected into the task.", rich_help_panel="Kernel"),
) -> Any:
    # fmt: on
    configure_if_first_time()
    config_path = get_config_path(config_spec)
    console.print(f"Loading agent config from [bold green]'{config_path}'[/bold green]")
    config = yaml.safe_load(config_path.read_text())

    if not task:
        if kernel_url:
            # Default task for kernel optimization when --kernel-url is provided without -t
            # We'll inject the resolved kernel path later via _inject_resolved_kernel,
            # but we need to set the task now before that runs.
            task = (
                "Optimize this kernel for maximum speedup. Do NOT use OpenEvolve. Instead:\n"
                "1. DISCOVER: Read the wrapper file AND trace imports to find the inner "
                "@triton.jit kernel "
                "Read the discovered test files shown above.\n"
                "2. Profile the baseline with kernel-profile.\n"
                "3. OPTIMIZE: Edit ONLY the inner kernel file "
                "Do NOT change the wrapper file (BLOCK_S, num_warps etc give negligible gains). "
                "Analyze the profiling metrics to identify bottlenecks and fix them. "
                "Revert any change that makes performance worse before trying something new.\n"
                "4. After EACH edit, run correctness check and re-profile. Iterate until you "
                "achieve significant speedup.\n"
                "5. Report final speedup."
            )
            console.print(f"[bold green]Using default kernel optimization task[/bold green]")
        else:
            console.print("[bold yellow]What do you want to do?")
            task = prompt_session.prompt(
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
        task, _resolved_kernel_name = _inject_resolved_kernel(kernel_url, workspace, task)
        # Extract the resolved path from the task (it's in the injected block)
        import re as _re
        _m = _re.search(r"Kernel path: (\S+)", task)
        if _m:
            _resolved_kernel_path = _m.group(1)
    # Run test discovery and inject results into task
    if _resolved_kernel_path:
        discovery_block = _run_discovery(_resolved_kernel_path, _resolved_kernel_name)
        if discovery_block:
            task = task + discovery_block

    # Runtime environment detection and configuration
    runtime_env = None
    if not no_runtime_check:
        if runtime == "local":
            # Force local environment
            from geakagent.runtime_env import RuntimeEnvironment
            runtime_env = RuntimeEnvironment(runtime_type=RuntimeType.LOCAL)
            display_runtime_info(runtime_env)
        elif runtime == "docker":
            # Force Docker environment
            from geakagent.runtime_env import RuntimeEnvironment, DEFAULT_DOCKER_IMAGE
            image = docker_image or DEFAULT_DOCKER_IMAGE
            runtime_env = RuntimeEnvironment(
                runtime_type=RuntimeType.DOCKER,
                docker_image=image,
                docker_devices=["/dev/kfd", "/dev/dri"],
                has_gpu=True,
                has_triton=True,
                has_torch=True
            )
            display_runtime_info(runtime_env)
        else:
            # Auto-detect (default)
            runtime_env = prompt_runtime_environment(auto_confirm=yolo)
        
        # Update config based on runtime environment
        if runtime_env and hasattr(runtime_env, 'runtime_type') and runtime_env.runtime_type == RuntimeType.DOCKER:
            workspace_path = workspace or os.getcwd()
            runtime_config = get_runtime_config_for_agent(runtime_env, workspace_path)
            config["env"] = runtime_config

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
    model = get_model(model_name, config.get("model", {}))
    
    # Create environment based on runtime configuration
    env_config = config.get("env", {})
    if runtime_env and runtime_env.runtime_type == RuntimeType.DOCKER:
        env = DockerEnvironment(**env_config)
    else:
        env = LocalEnvironment(**env_config)

    # Both visual flag and the MSWEA_VISUAL_MODE_DEFAULT flip the mode, so it's essentially a XOR
    agent_class = InteractiveAgent
    if visual == (os.getenv("MSWEA_VISUAL_MODE_DEFAULT", "false") == "false"):
        agent_class = TextualAgent
    agent_config = config.get("agent", {})
    if save_patch:
        agent_class = PatchAgent
        agent_config["save_patch"] = True
        agent_config["test_command"] = test_command
        patch_dir = patch_output or (global_config_dir / "patches")
        agent_config["patch_output_dir"] = str(patch_dir)
        agent_config["metric"] = metric

    agent = agent_class(model, env, **agent_config)
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

    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run(task, instructions=instructions_content)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        if output:
            save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
    return agent


if __name__ == "__main__":
    app()
