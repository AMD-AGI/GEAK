"""Agent-based task generator -- produces optimization tasks by running a
read-only planning agent that inspects profiling data and kernel metadata.

The agent reads files via ``str_replace_editor view`` and submits a JSON
task list via the ``submit`` tool.  No rule-based fallback: an LLM model
is required.

Priority scheme (lower = higher priority, runs first):
  0  -- OpenEvolve on the inner kernel (highest impact, automated)
  5  -- Kernel fusion / advanced tuning
  10 -- Targeted optimization (autotune, memory, launch config)
  15 -- Profile-guided (generic fallback)

Usage (Python):
    from minisweagent.run.task_generator import generate_tasks
    tasks = generate_tasks(
        discovery_result=result,
        base_task_context=task_text,
        agent_class=StrategyAgent,
        model=model,
        profiling_path=Path("profile.json"),
        commandment_path=Path("COMMANDMENT.md"),
    )

Usage (CLI):
    python -m minisweagent.run.task_generator \\
        --kernel-path /path/to/kernel.py \\
        --profiling profiler_output.json \\
        --commandment COMMANDMENT.md \\
        --baseline-metrics baseline_metrics.json
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from minisweagent.agents.agent_spec import AgentTask
from minisweagent.run.task_planner import _GPU_AND_PROFILER_RULES
from minisweagent.tools.discovery import DiscoveryResult

logger = logging.getLogger(__name__)

_KNOWLEDGE_BASE_REL = "knowledge_base/optimization_strategies.py"

# ============================================================================
# Prompt templates for the planning agent
# ============================================================================

_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert GPU kernel optimization planner for AMD GPUs. You have
access to profiling data, kernel metadata, and a knowledge base of
optimization strategies via file paths. Read the files you need using
the `str_replace_editor` tool (command: "view"), reason about the best
optimization approach, then submit your task list as JSON via the
`submit` tool.

## Available Agents and Tools

### Agents (task execution)

1. **strategy_agent** (default) -- An LLM-guided agent with bash, editor,
   and profiling tools. It reads code, reasons about bottlenecks, and edits
   the kernel directly. Best for targeted optimizations: autotune config,
   memory access patterns, launch configuration, kernel fusion. Has access
   to the kernel-evolve and kernel-ercs MCP tools described below.

2. **openevolve** -- An evolutionary optimizer that mutates the kernel and
   evaluates candidates automatically using COMMANDMENT.md and
   baseline_metrics.json. No LLM reasoning during the optimization loop.
   Best for inner kernels where small code changes yield large speedups.

### Tools (available to strategy_agent via MCP)

3. **kernel-evolve** -- LLM-guided mutation/crossover of kernel code.
   Commands: `generate` (optimized variant from bottleneck + strategy),
   `mutate` (improve existing kernel), `crossover` (combine two kernels),
   `strategies` (list strategies for a bottleneck), `params` (suggest
   parameters for a kernel type). Instruct the strategy_agent to use
   kernel-evolve when the task involves generating new kernel variants.

4. **kernel-ercs** -- LLM-based evaluation, reflection, compatibility
   checking, and AMD GPU specs. Commands: `evaluate` (score kernel quality),
   `reflect` (analyze test results, suggest next steps), `compat` (check
   AMD compatibility), `specs` (AMD MI350X specs). Instruct the
   strategy_agent to use kernel-ercs for quality assessment and reflection.

## Task priority scheme (lower number = higher priority = runs first)

- 0: OpenEvolve on inner kernel (highest impact, automated)
- 5: Kernel fusion, advanced language-specific tuning
- 10: Targeted optimizations (autotune, memory, launch config)
- 15: Profile-guided generic optimization (fallback)

## Your analysis process

1. Use `str_replace_editor` with command "view" to read the profiling file
   first. Identify which sub-kernels are real optimization targets vs.
   framework noise (e.g., PyTorch ATen elementwise ops, ROCm runtime
   kernels, hipMemcpy internals).
2. Read the discovery file for kernel metadata (language, inner kernel, etc.).
3. Read the knowledge base for applicable optimization strategies.
4. Optionally read baseline metrics, COMMANDMENT.md, deep search findings,
   or prior results if the paths are provided.
5. Group related kernels (e.g., multiple Tensile GEMMs with different tile
   sizes are one target; CK GEMM variants are another).
6. For each group, propose a specific optimization task naming:
   - The target sub-kernels
   - The backend/language (CK, Tensile, Triton, HIP, PyTorch)
   - Concrete strategies from the knowledge base
   - Which agent/tool to use (and specific tool commands if applicable)
   - Expected impact
7. Consider: OpenEvolve for parameter tuning, kernel fusion for reducing
   launch overhead, elimination of unnecessary framework kernels.
8. If prior round results are provided, do NOT re-generate tasks for
   strategies that already succeeded. Focus on what failed or was not tried.

## Output format

When you are done analyzing, call the `submit` tool with the `summary`
parameter containing a JSON array of task objects. Each task has:
- "label": short kebab-case identifier (e.g. "openevolve-inner", "ck-tile-tuning")
- "priority": integer 0-15
- "agent_type": "strategy_agent" or "openevolve"
- "kernel_language": "python", "cpp", or "asm"
- "task_prompt": detailed instructions for the sub-agent (specific
  optimization focus, which tools to use, what to measure). This is
  the FULL prompt the agent will see.

## Rules for task_prompt content

{gpu_rules}

**FORBIDDEN tasks**: NEVER generate tasks that modify the test harness,
test file, or test command. The test harness is the evaluation contract --
it defines correctness and must remain unchanged. Tasks like "test harness
optimization", "test improvement", or "benchmark refactoring" are INVALID.
Only generate tasks that optimize the *kernel* code itself.

**Path deduplication**: The task file metadata already stores kernel_path,
commandment, baseline_metrics, and profiling paths. Do NOT repeat these
file paths in the task_prompt body. Instead, reference them generically
(e.g. "the kernel file", "the COMMANDMENT", "baseline metrics"). The
sub-agent receives these paths automatically from the task metadata.

**Baseline comparison**: Each task_prompt MUST instruct the sub-agent to
compare its results against the baseline metrics provided in the task
metadata. The sub-agent should report the specific metric improvement
(e.g. duration reduction, bandwidth improvement) relative to baseline.

**COMMANDMENT adherence**: Each task_prompt MUST instruct the sub-agent
to read and follow the COMMANDMENT file. The COMMANDMENT defines the
correctness criteria and constraints. Any changes that violate the
COMMANDMENT must be rejected by the sub-agent itself.

**Verification for strategy_agent tasks**: Each task_prompt for
strategy_agent tasks MUST include instructions to:
1. Read the COMMANDMENT and follow its constraints
2. Verify correctness after making changes (use the `test_perf` tool)
3. Profile the result to measure improvement (use the `profile_kernel` tool)
4. Compare results against baseline metrics and report before/after numbers
5. If correctness tests fail, revert changes and report failure

OpenEvolve tasks handle verification automatically via COMMANDMENT.md.

Submit ONLY the JSON array via the submit tool. No markdown fences, no explanation.
""").format(gpu_rules=_GPU_AND_PROFILER_RULES.strip())


_INSTANCE_TEMPLATE = textwrap.dedent("""\
Generate optimization tasks for the kernel at {{ kernel_path }}.

## Kernel Metadata
- Name: {{ kernel_name }}
- Type: {{ kernel_type }}
- Language: {{ kernel_language }}
{% if inner_kernel_path %}- Inner kernel: {{ inner_kernel_path }}
- Inner kernel language: {{ inner_kernel_language }}
{% endif %}{% if has_autotune %}- Has autotune: yes
{% endif %}{% if function_names %}- Functions: {{ function_names }}
{% endif %}
## Files to read (use `str_replace_editor` with command "view")
{% if discovery_path %}- **Discovery** (kernel info, tests, benchmarks): {{ discovery_path }}
{% endif %}{% if profiling_path %}- **Profiling** (sub-kernels, bottlenecks, metrics): {{ profiling_path }}
{% endif %}{% if baseline_metrics_path %}- **Baseline metrics**: {{ baseline_metrics_path }}
{% endif %}{% if commandment_path %}- **COMMANDMENT.md** (evaluation contract): {{ commandment_path }}
{% endif %}{% if knowledge_base_path %}- **Knowledge base** (optimization strategies): {{ knowledge_base_path }}
{% endif %}{% if deep_search_path %}- **Deep search findings**: {{ deep_search_path }}
{% endif %}{% if previous_results_path %}- **Prior round results**: {{ previous_results_path }}
{% endif %}
## Instructions

Read the profiling file first to understand the sub-kernel landscape, then
the discovery file for context. Consult the knowledge base for applicable
strategies. Finally, submit your task list as JSON via the `submit` tool.

{{ base_task_context }}
""")


# ============================================================================
# Public API
# ============================================================================


def generate_tasks(
    discovery_result: DiscoveryResult,
    base_task_context: str,
    agent_class: type,
    model: Any,
    *,
    profiling_path: Path | None = None,
    commandment_path: Path | None = None,
    baseline_metrics_path: Path | None = None,
    deep_search_path: Path | None = None,
    previous_results_dir: Path | None = None,
    discovery_path: Path | None = None,
) -> list[AgentTask]:
    """Generate optimization tasks using an LLM planning agent.

    Args:
        discovery_result: Output of DiscoveryPipeline.run().
        base_task_context: Common context prepended to each task prompt.
        agent_class: Default agent class for tasks (typically StrategyAgent).
        model: LLM model instance (required).
        profiling_path: Path to kernel-profile JSON output.
        commandment_path: Path to COMMANDMENT.md.
        baseline_metrics_path: Path to baseline_metrics.json.
        deep_search_path: Path to deep search findings file.
        previous_results_dir: Path to previous round results directory.
        discovery_path: Path to the discovery.json file.

    Returns:
        List of AgentTask sorted by priority.

    Raises:
        RuntimeError: If the agent fails to submit results.
    """
    if not discovery_result.kernels:
        return []

    submitted_text = _run_task_agent(
        discovery_result=discovery_result,
        base_task_context=base_task_context,
        model=model,
        profiling_path=profiling_path,
        commandment_path=commandment_path,
        baseline_metrics_path=baseline_metrics_path,
        deep_search_path=deep_search_path,
        previous_results_dir=previous_results_dir,
        discovery_path=discovery_path,
    )

    kernel = discovery_result.kernels[0]
    return _parse_llm_response(
        submitted_text,
        agent_class,
        kernel_path=str(kernel.file_path),
        commandment_path=str(commandment_path) if commandment_path else None,
        baseline_metrics_path=str(baseline_metrics_path) if baseline_metrics_path else None,
    )


def generate_tasks_from_content(
    discovery_result: DiscoveryResult,
    base_task_context: str,
    agent_class: type,
    model: Any,
    *,
    profiling_result: dict | None = None,
    commandment_content: str | None = None,
    baseline_metrics: dict | None = None,
    deep_search_content: str | None = None,
    previous_results_dir: Path | None = None,
    discovery_path: Path | None = None,
) -> list[AgentTask]:
    """Convenience wrapper that materializes in-memory content to temp files.

    Use this when the caller has data in memory (dicts/strings) rather than
    on disk.  Each non-None content argument is written to a temporary file
    whose path is then forwarded to :func:`generate_tasks`.
    """
    tmp_files: list[Path] = []
    try:
        profiling_path = _write_temp(json.dumps(profiling_result, indent=2), ".json") if profiling_result else None
        if profiling_path:
            tmp_files.append(profiling_path)

        commandment_path = _write_temp(commandment_content, ".md") if commandment_content else None
        if commandment_path:
            tmp_files.append(commandment_path)

        baseline_metrics_path = (
            _write_temp(json.dumps(baseline_metrics, indent=2), ".json") if baseline_metrics else None
        )
        if baseline_metrics_path:
            tmp_files.append(baseline_metrics_path)

        deep_search_path = _write_temp(deep_search_content, ".md") if deep_search_content else None
        if deep_search_path:
            tmp_files.append(deep_search_path)

        return generate_tasks(
            discovery_result=discovery_result,
            base_task_context=base_task_context,
            agent_class=agent_class,
            model=model,
            profiling_path=profiling_path,
            commandment_path=commandment_path,
            baseline_metrics_path=baseline_metrics_path,
            deep_search_path=deep_search_path,
            previous_results_dir=previous_results_dir,
            discovery_path=discovery_path,
        )
    finally:
        for f in tmp_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass


# ============================================================================
# Agent execution
# ============================================================================


def _write_temp(content: str, suffix: str) -> Path:
    """Write content to a temporary file and return its path."""
    fd, name = tempfile.mkstemp(suffix=suffix, prefix=".task_gen_")
    os.close(fd)
    Path(name).write_text(content)
    return Path(name)


def _find_knowledge_base(workspace: Path) -> Path | None:
    """Locate the optimization strategies knowledge base file."""
    for root in [workspace, workspace.parent, workspace.parent.parent]:
        p = root / _KNOWLEDGE_BASE_REL
        if p.exists():
            return p
    return None


def _run_task_agent(
    discovery_result: DiscoveryResult,
    base_task_context: str,
    model: Any,
    profiling_path: Path | None,
    commandment_path: Path | None,
    baseline_metrics_path: Path | None,
    deep_search_path: Path | None,
    previous_results_dir: Path | None,
    discovery_path: Path | None,
) -> str:
    """Run a read-only planning agent and return the submitted JSON text."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.tools.tools_runtime import get_tools_list

    kernel = discovery_result.kernels[0]
    workspace = discovery_result.workspace_path or kernel.file_path.parent

    read_only_tools = [t for t in get_tools_list() if t["name"] in ("str_replace_editor", "submit")]
    model_impl = getattr(model, "_impl", model)
    original_tools = list(model_impl.tools) if hasattr(model_impl, "tools") else None
    model_impl.tools = read_only_tools

    tmp_files: list[Path] = []
    try:
        env = LocalEnvironment(cwd=str(workspace))
        kb_path = _find_knowledge_base(Path(workspace))

        prev_results_path: Path | None = None
        if previous_results_dir and Path(previous_results_dir).is_dir():
            summary = _scan_previous_results(Path(previous_results_dir))
            if summary:
                prev_results_path = _write_temp(summary, "_prev_results.md")
                tmp_files.append(prev_results_path)

        template_vars = {
            "kernel_path": str(kernel.file_path),
            "kernel_name": kernel.kernel_name,
            "kernel_type": kernel.kernel_type,
            "kernel_language": kernel.kernel_language,
            "inner_kernel_path": str(kernel.inner_kernel_path) if kernel.inner_kernel_path else "",
            "inner_kernel_language": kernel.inner_kernel_language or "",
            "has_autotune": kernel.has_autotune,
            "function_names": ", ".join(kernel.function_names) if kernel.function_names else "",
            "discovery_path": str(discovery_path) if discovery_path else "",
            "profiling_path": str(profiling_path) if profiling_path else "",
            "commandment_path": str(commandment_path) if commandment_path else "",
            "baseline_metrics_path": str(baseline_metrics_path) if baseline_metrics_path else "",
            "knowledge_base_path": str(kb_path) if kb_path else "",
            "deep_search_path": str(deep_search_path) if deep_search_path else "",
            "previous_results_path": str(prev_results_path) if prev_results_path else "",
            "base_task_context": base_task_context,
        }

        tg_step_limit = int(os.getenv("GEAK_TASKGEN_STEP_LIMIT", "75"))
        tg_cost_limit = float(os.getenv("GEAK_TASKGEN_COST_LIMIT", "10.0"))

        agent = DefaultAgent(
            model,
            env,
            system_template=_SYSTEM_PROMPT,
            instance_template=_INSTANCE_TEMPLATE,
            step_limit=tg_step_limit,
            cost_limit=tg_cost_limit,
        )

        logger.info(
            "Starting task-generation agent (step_limit=%d, cost_limit=%.1f)",
            tg_step_limit,
            tg_cost_limit,
        )

        exit_type, exit_msg = agent.run(
            task="generate optimization tasks",
            **template_vars,
        )

        if exit_type == "Submitted":
            return exit_msg

        raise RuntimeError(f"Task-generation agent did not submit results (exit: {exit_type}): {exit_msg[:500]}")
    finally:
        if original_tools is not None:
            model_impl.tools = original_tools
        for f in tmp_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass


def _parse_llm_response(
    content: str,
    agent_class: type,
    *,
    kernel_path: str | None = None,
    commandment_path: str | None = None,
    baseline_metrics_path: str | None = None,
) -> list[AgentTask]:
    """Parse JSON response into AgentTask objects.

    When the LLM sets ``agent_type`` to ``"openevolve"``, the task is
    assigned to :class:`OpenEvolveWorker` and the required config
    (kernel_path, commandment_path, baseline_metrics_path) is populated
    from the caller-supplied paths.
    """
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    raw_tasks = json.loads(content)
    if not isinstance(raw_tasks, list):
        raise TypeError(f"Expected JSON array, got {type(raw_tasks).__name__}")

    from minisweagent.agents.openevolve_worker import OpenEvolveWorker

    tasks: list[AgentTask] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label", "unknown"))
        priority = int(item.get("priority", 10))
        priority = max(0, min(15, priority))
        agent_type = str(item.get("agent_type", "strategy_agent"))
        kernel_language = str(item.get("kernel_language", "python"))
        task_prompt = str(item.get("task_prompt", ""))

        if not task_prompt:
            continue

        if agent_type == "openevolve":
            oe_config: dict[str, Any] = {}
            if kernel_path:
                oe_config["kernel_path"] = kernel_path
            if commandment_path:
                oe_config["commandment_path"] = commandment_path
            if baseline_metrics_path:
                oe_config["baseline_metrics_path"] = baseline_metrics_path
            tasks.append(
                AgentTask(
                    agent_class=OpenEvolveWorker,
                    task=task_prompt,
                    label=label,
                    priority=priority,
                    kernel_language=kernel_language,
                    config=oe_config,
                )
            )
        else:
            tasks.append(
                AgentTask(
                    agent_class=agent_class,
                    task=task_prompt,
                    label=label,
                    priority=priority,
                    kernel_language=kernel_language,
                )
            )

    if not tasks:
        raise ValueError("LLM response contained no valid tasks")

    return sorted(tasks, key=lambda t: t.priority)


# ============================================================================
# CLI helpers
# ============================================================================


def _scan_previous_results(results_dir: Path) -> str:
    """Scan a previous round's results directory and build a summary.

    Reads task_*/patch_*_test.txt and task_*/*.log to understand what
    each task achieved. Returns a Markdown summary for the LLM prompt.
    """
    import re as _re

    sections: list[str] = []
    # Scan all subdirectories that contain results (task labels or task_N).
    # Skip 'worktrees' and hidden directories.
    task_dirs = sorted(
        d for d in results_dir.iterdir() if d.is_dir() and d.name not in ("worktrees",) and not d.name.startswith(".")
    )
    if not task_dirs:
        return ""

    for td in task_dirs:
        label = td.name
        patches = sorted(td.glob("patch_*.patch"))
        test_outputs = sorted(td.glob("patch_*_test.txt"))
        log_files = sorted(td.glob("*.log"))

        section = [f"### {label}"]
        section.append(f"- Patches produced: {len(patches)}")

        for tf in test_outputs[:3]:
            try:
                content = tf.read_text(errors="replace")[-2000:]
                speedups = _re.findall(r"speedup[:\s]+([0-9.]+)x?", content, _re.IGNORECASE)
                durations = _re.findall(r"duration[:\s]+([0-9.]+)\s*(?:us|µs|ms)", content, _re.IGNORECASE)
                if speedups:
                    section.append(f"- {tf.name}: speedup = {speedups[-1]}")
                elif durations:
                    section.append(f"- {tf.name}: duration = {durations[-1]}")
                else:
                    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                    tail = lines[-3:] if len(lines) >= 3 else lines
                    section.append(f"- {tf.name} (tail): {' | '.join(tail)}")
            except Exception:
                section.append(f"- {tf.name}: (unreadable)")

        for lf in log_files[:1]:
            try:
                content = lf.read_text(errors="replace")[-1000:]
                if "ERROR" in content or "Traceback" in content:
                    section.append(f"- Log ({lf.name}): contains errors")
                else:
                    section.append(f"- Log ({lf.name}): completed")
            except Exception:
                pass

        sections.append("\n".join(section))

    return "## Previous Round Results\n\n" + "\n\n".join(sections) + "\n"


# ============================================================================
# CLI
# ============================================================================


def main():
    """Generate optimization tasks from the command line."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate optimization tasks using an LLM planning agent",
    )
    parser.add_argument("--kernel-path", default=None, help="Path to the kernel file")
    parser.add_argument(
        "--from-discovery",
        default=None,
        metavar="FILE",
        help="Read discovery.json and extract kernel-path and repo-root",
    )
    parser.add_argument("--profiling", default=None, help="Path to kernel-profile JSON output")
    parser.add_argument("--commandment", default=None, help="Path to COMMANDMENT.md")
    parser.add_argument("--baseline-metrics", default=None, help="Path to baseline_metrics.json")
    parser.add_argument("--model", default=None, help="Model name (default: from config/env)")
    parser.add_argument("--repo-root", default=None, help="Repository root (for discovery)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="DIR",
        help="Write task files to this directory (one .md per task) instead of JSON to stdout",
    )
    parser.add_argument(
        "--from-results",
        default=None,
        metavar="DIR",
        help="Previous round results directory (for iterative refinement)",
    )
    parser.add_argument(
        "--deep-search",
        default=None,
        metavar="FILE",
        help="Path to deep search findings (JSON or Markdown file)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Round number for task file frontmatter (default: 1)",
    )

    args = parser.parse_args()

    # Populate from discovery JSON if provided (explicit flags override)
    disc_json = None
    test_command = None
    if args.from_discovery:
        disc_json = json.loads(Path(args.from_discovery).read_text())
        if not args.kernel_path:
            args.kernel_path = (disc_json.get("kernel") or {}).get("file")
        if not args.repo_root:
            args.repo_root = disc_json.get("workspace")
        focused = disc_json.get("focused_test") or {}
        if focused.get("focused_command"):
            test_command = focused["focused_command"]
        else:
            for t in disc_json.get("tests") or []:
                if t.get("command"):
                    test_command = t["command"]
                    break

    if not args.kernel_path:
        parser.error("--kernel-path is required (or provide --from-discovery)")

    kernel_path = Path(args.kernel_path).resolve()
    if not kernel_path.exists():
        print(f"ERROR: kernel path not found: {args.kernel_path}", file=sys.stderr)
        sys.exit(1)

    # Build DiscoveryResult -- from pre-computed JSON if available, else run pipeline
    from minisweagent.tools.discovery import (
        BenchmarkInfo,
        DiscoveryResult,
        KernelInfo,
        TestInfo,
    )

    if disc_json:
        print(f"[task-generator] Loading discovery from {args.from_discovery}...", file=sys.stderr)
        kernel_info = disc_json.get("kernel") or {}
        kernels = []
        if kernel_info.get("file"):
            kernels.append(
                KernelInfo(
                    file_path=Path(kernel_info["file"]),
                    kernel_name=kernel_info.get("name", kernel_path.stem),
                    kernel_type=kernel_info.get("type", "unknown"),
                    kernel_language="python" if kernel_path.suffix == ".py" else "cpp",
                    function_names=kernel_info.get("functions", []),
                )
            )
        tests = [
            TestInfo(
                file_path=Path(t["file"]),
                test_type=t.get("type", "script"),
                command=t.get("command", ""),
                confidence=t.get("confidence", 0.5),
            )
            for t in (disc_json.get("tests") or [])
        ]
        benchmarks = [
            BenchmarkInfo(
                file_path=Path(b["file"]),
                bench_type=b.get("type", "script"),
                command=b.get("command", ""),
                confidence=b.get("confidence", 0.5),
            )
            for b in (disc_json.get("benchmarks") or [])
        ]
        workspace = Path(disc_json.get("workspace", kernel_path.parent))
        discovery_result = DiscoveryResult(
            kernels=kernels,
            tests=tests,
            benchmarks=benchmarks,
            workspace_path=workspace,
        )
    else:
        print(f"[task-generator] Running discovery on {kernel_path}...", file=sys.stderr)
        try:
            from minisweagent.tools.discovery import DiscoveryPipeline

            repo_root = Path(args.repo_root).resolve() if args.repo_root else None
            workspace = repo_root or kernel_path.parent
            pipeline = DiscoveryPipeline(workspace_path=workspace)
            discovery_result = pipeline.run(kernel_path=kernel_path)
        except Exception as e:
            print(f"ERROR: discovery failed: {e}", file=sys.stderr)
            sys.exit(1)

    if not discovery_result.kernels:
        print("ERROR: no kernels found by discovery", file=sys.stderr)
        sys.exit(1)

    # Create model (REQUIRED)
    model_name = args.model or os.environ.get("GEAK_MODEL")
    try:
        from minisweagent.config import get_config_path
        from minisweagent.models import get_model

        geak_cfg = get_config_path("geak")
        model_config = {}
        if geak_cfg.exists():
            import yaml

            full_cfg = yaml.safe_load(geak_cfg.read_text()) or {}
            model_config = full_cfg.get("model", {})
        model = get_model(model_name, config=model_config)
        print(f"[task-generator] Using model: {model.config.model_name}", file=sys.stderr)
    except Exception as e:
        print(
            f"ERROR: task-generator requires an LLM model. Set GEAK_MODEL or use --model. ({e})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Placeholder agent class for CLI output
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

    agent_class = StrategyInteractiveAgent

    base_task_context = f"Optimize the kernel at {kernel_path} for maximum performance."

    # Resolve file paths (pass through to the agent, not loaded into memory)
    profiling_path = Path(args.profiling).resolve() if args.profiling else None
    commandment_path = Path(args.commandment).resolve() if args.commandment else None
    baseline_metrics_path = Path(args.baseline_metrics).resolve() if args.baseline_metrics else None
    deep_search_path = Path(args.deep_search).resolve() if args.deep_search else None
    previous_results_dir = Path(args.from_results).resolve() if args.from_results else None
    discovery_path = Path(args.from_discovery).resolve() if args.from_discovery else None

    # Generate tasks
    tasks = generate_tasks(
        discovery_result=discovery_result,
        base_task_context=base_task_context,
        agent_class=agent_class,
        model=model,
        profiling_path=profiling_path,
        commandment_path=commandment_path,
        baseline_metrics_path=baseline_metrics_path,
        deep_search_path=deep_search_path,
        previous_results_dir=previous_results_dir,
        discovery_path=discovery_path,
    )

    # Print summary to stderr
    print(f"\n[task-generator] Generated {len(tasks)} task(s):\n", file=sys.stderr)
    for t in tasks:
        print(f"  [{t.priority:2d}] {t.label} ({t.kernel_language})", file=sys.stderr)

    # Output: directory of task files or JSON to stdout
    if args.output:
        from minisweagent.run.task_file import write_task_file

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        from minisweagent.agents.openevolve_worker import OpenEvolveWorker as _OEW

        manifest = []
        for i, t in enumerate(tasks):
            filename = f"{t.priority:02d}_{t.label}.md"
            task_path = out_dir / filename

            metadata = {
                "label": t.label,
                "priority": t.priority,
                "agent_type": "openevolve" if t.agent_class is _OEW else "strategy_agent",
                "kernel_language": t.kernel_language,
                "kernel_path": str(kernel_path),
                "repo_root": args.repo_root,
                "commandment": args.commandment,
                "baseline_metrics": args.baseline_metrics,
                "profiling": args.profiling,
                "test_command": test_command,
                "round": args.round,
            }

            body = f"# {t.label}\n\n{t.task}\n"
            write_task_file(task_path, metadata, body)

            manifest.append(
                {
                    "index": i,
                    "label": t.label,
                    "priority": t.priority,
                    "kernel_language": t.kernel_language,
                    "file": str(task_path),
                }
            )

        print(f"\n[task-generator] Wrote {len(tasks)} task file(s) to {out_dir}/", file=sys.stderr)
        print(json.dumps(manifest, indent=2))
    else:
        output = []
        for i, t in enumerate(tasks):
            output.append(
                {
                    "index": i,
                    "label": t.label,
                    "priority": t.priority,
                    "kernel_language": t.kernel_language,
                    "task_prompt_preview": t.task[:300] + ("..." if len(t.task) > 300 else ""),
                }
            )
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
