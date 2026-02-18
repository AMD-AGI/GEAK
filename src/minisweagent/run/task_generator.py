"""LLM-assisted task generator -- produces optimization tasks using LLM reasoning.

Given rich context (discovery results, profiling output, COMMANDMENT.md,
baseline metrics), asks an LLM to generate a prioritized list of optimization
tasks with appropriate sub-agent assignments.

Falls back to the rule-based ``task_planner.build_optimization_tasks()`` when:
- The LLM call fails (network error, rate limit, malformed output)
- No model is available (``--no-llm-planning`` or missing API key)

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
        profiling_result=profiling,
        commandment_content=commandment,
        baseline_metrics=metrics,
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
import textwrap
from pathlib import Path
from typing import Any

from minisweagent.agents.agent_spec import AgentTask
from minisweagent.run.task_planner import (
    _GPU_AND_PROFILER_RULES,
    build_optimization_tasks,
)
from minisweagent.tools.discovery import DiscoveryResult

logger = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 1

# ============================================================================
# System prompt for the LLM task generator
# ============================================================================

_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert GPU kernel optimization planner. Given information about a
kernel (its type, structure, profiling bottlenecks, and available tests), you
generate a prioritized list of optimization tasks.

## Available sub-agent types

1. **strategy_agent** (default) -- An LLM-guided agent that reads code,
   profiles, reasons about bottlenecks, and edits the kernel directly.
   Best for targeted optimizations: autotune config, memory access patterns,
   launch configuration, kernel fusion.

2. **openevolve** -- An evolutionary optimizer that mutates the kernel and
   evaluates candidates automatically (no LLM reasoning during optimization).
   Best for inner kernels where small code changes can yield large speedups.
   Requires a COMMANDMENT.md and baseline_metrics.json.

## Task priority scheme (lower number = higher priority = runs first)

- 0: OpenEvolve on inner kernel (highest impact, automated)
- 5: Kernel fusion, advanced language-specific tuning
- 10: Targeted optimizations (autotune, memory, launch config)
- 15: Profile-guided generic optimization (fallback)

## Output format

Return a JSON array of task objects. Each task has:
- "label": short kebab-case identifier (e.g. "openevolve-inner", "triton-autotune")
- "priority": integer 0-15
- "agent_type": "strategy_agent" or "openevolve"
- "kernel_language": "python", "cpp", or "asm"
- "task_prompt": detailed instructions for the sub-agent (include file paths,
  specific optimization focus, what to measure). This is the FULL prompt the
  agent will see.

## Rules for task_prompt content

{gpu_rules}

Return ONLY the JSON array. No markdown fences, no explanation.
""").format(gpu_rules=_GPU_AND_PROFILER_RULES.strip())


# ============================================================================
# Public API
# ============================================================================

def generate_tasks(
    discovery_result: DiscoveryResult,
    base_task_context: str,
    agent_class: type,
    *,
    model: Any | None = None,
    profiling_result: dict | None = None,
    commandment_content: str | None = None,
    baseline_metrics: dict | None = None,
) -> list[AgentTask]:
    """Generate optimization tasks, using LLM if available.

    Args:
        discovery_result: Output of DiscoveryPipeline.run().
        base_task_context: Common context prepended to each task prompt.
        agent_class: Default agent class for tasks (typically StrategyAgent).
        model: LLM model instance (if None, falls back to rule-based planner).
        profiling_result: Output from kernel-profile (dict with "results" key).
        commandment_content: Content of the generated COMMANDMENT.md.
        baseline_metrics: Content of baseline_metrics.json (parsed dict).

    Returns:
        List of AgentTask sorted by priority.
    """
    if not discovery_result.kernels:
        return []

    if model is None:
        logger.info("No model provided; falling back to rule-based task planner")
        return build_optimization_tasks(discovery_result, base_task_context, agent_class)

    try:
        tasks = _generate_with_llm(
            discovery_result=discovery_result,
            base_task_context=base_task_context,
            agent_class=agent_class,
            model=model,
            profiling_result=profiling_result,
            commandment_content=commandment_content,
            baseline_metrics=baseline_metrics,
        )
        if tasks:
            return tasks
        logger.warning("LLM returned empty task list; falling back to rule-based planner")
    except Exception as e:
        logger.warning("LLM task generation failed (%s); falling back to rule-based planner", e)

    return build_optimization_tasks(discovery_result, base_task_context, agent_class)


# ============================================================================
# LLM interaction
# ============================================================================

def _generate_with_llm(
    discovery_result: DiscoveryResult,
    base_task_context: str,
    agent_class: type,
    model: Any,
    profiling_result: dict | None,
    commandment_content: str | None,
    baseline_metrics: dict | None,
) -> list[AgentTask]:
    """Call the LLM and parse the response into AgentTask objects."""
    user_prompt = _build_user_prompt(
        discovery_result=discovery_result,
        base_task_context=base_task_context,
        profiling_result=profiling_result,
        commandment_content=commandment_content,
        baseline_metrics=baseline_metrics,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(_MAX_LLM_RETRIES + 1):
        response = model.query(messages)
        content = response.get("content", "").strip()

        try:
            tasks = _parse_llm_response(content, agent_class, base_task_context)
            return tasks
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            if attempt < _MAX_LLM_RETRIES:
                logger.info("LLM output parse failed (attempt %d): %s; retrying", attempt + 1, e)
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response was not valid JSON: {e}\n"
                        "Please return ONLY a JSON array of task objects with keys: "
                        "label, priority, agent_type, kernel_language, task_prompt. "
                        "No markdown fences, no explanation."
                    ),
                })
            else:
                raise

    return []  # unreachable but satisfies type checker


def _build_user_prompt(
    discovery_result: DiscoveryResult,
    base_task_context: str,
    profiling_result: dict | None,
    commandment_content: str | None,
    baseline_metrics: dict | None,
) -> str:
    """Assemble all available context into a structured prompt."""
    sections: list[str] = []

    # -- Kernel info --
    kernel = discovery_result.kernels[0]
    sections.append(textwrap.dedent(f"""\
    ## Kernel Information
    - File: {kernel.file_path}
    - Name: {kernel.kernel_name}
    - Type: {kernel.kernel_type}
    - Language: {kernel.kernel_language}
    - Inner kernel: {kernel.inner_kernel_path or 'N/A'}
    - Inner kernel language: {kernel.inner_kernel_language or 'N/A'}
    - Has autotune: {kernel.has_autotune}
    - Function names: {', '.join(kernel.function_names) if kernel.function_names else 'N/A'}
    """))

    # -- Fusion opportunities --
    if kernel.fusion_opportunities:
        sections.append("## Fusion Opportunities")
        for opp in kernel.fusion_opportunities:
            sections.append(f"- {opp}")
        sections.append("")

    # -- Dependency graph --
    dep_graph = discovery_result.dependency_graphs.get(kernel.kernel_name)
    if dep_graph:
        sections.append(f"## Dependency Graph\n{dep_graph.summary()}\n")

    # -- Discovered tests --
    if discovery_result.tests:
        sections.append("## Discovered Tests")
        for t in discovery_result.tests[:5]:
            sections.append(f"- {t.file_path} (confidence: {t.confidence:.2f}, command: {t.command})")
        sections.append("")

    # -- Discovered benchmarks --
    if discovery_result.benchmarks:
        sections.append("## Discovered Benchmarks")
        for b in discovery_result.benchmarks[:3]:
            sections.append(f"- {b.file_path} (confidence: {b.confidence:.2f})")
        sections.append("")

    # -- Profiling result --
    if profiling_result:
        sections.append("## Profiling Results (baseline)")
        results = profiling_result.get("results", [])
        if results:
            gpu_result = results[0]
            kernels = gpu_result.get("kernels", [])
            for k in kernels[:10]:
                dur = k.get("duration_us", k.get("metrics", {}).get("duration_us", "?"))
                bn = k.get("bottleneck", "?")
                sections.append(f"- {k.get('name', '?')}: {dur} us, bottleneck={bn}")
            observations = gpu_result.get("observations", [])
            if observations:
                sections.append("Observations:")
                for obs in observations[:5]:
                    sections.append(f"  - {obs}")
        sections.append("")

    # -- Baseline metrics --
    if baseline_metrics:
        sections.append("## Baseline Metrics")
        sections.append(f"- Duration: {baseline_metrics.get('duration_us', '?')} us")
        sections.append(f"- Kernel: {baseline_metrics.get('kernel_name', '?')}")
        sections.append(f"- Bottleneck: {baseline_metrics.get('bottleneck', '?')}")
        metrics = baseline_metrics.get("metrics", {})
        for k, v in list(metrics.items())[:10]:
            sections.append(f"- {k}: {v}")
        sections.append("")

    # -- COMMANDMENT.md --
    if commandment_content:
        sections.append(f"## COMMANDMENT.md (evaluation contract)\n```\n{commandment_content}\n```\n")

    # -- Base task context (truncated) --
    ctx_preview = base_task_context[:2000]
    if len(base_task_context) > 2000:
        ctx_preview += "\n... (truncated)"
    sections.append(f"## Base Task Context\n{ctx_preview}\n")

    # -- Instructions --
    sections.append(textwrap.dedent("""\
    ## Your Task
    Based on all the information above, generate a prioritized list of
    optimization tasks as a JSON array. Each task should target a specific
    optimization opportunity. Include the full base task context in each
    task_prompt so the sub-agent has all the information it needs.

    Consider:
    - If an inner kernel exists, OpenEvolve on it is usually the highest-impact task (priority 0).
    - Profiling bottlenecks should inform what targeted optimizations to suggest.
    - Fusion opportunities from the dependency graph are high-value tasks.
    - Include at least one profile-guided fallback task (priority 15).
    - Each task_prompt MUST include absolute file paths and specific instructions.
    """))

    return "\n".join(sections)


def _parse_llm_response(
    content: str,
    agent_class: type,
    base_task_context: str,
) -> list[AgentTask]:
    """Parse JSON response into AgentTask objects."""
    # Strip markdown code fences if the LLM wrapped the JSON
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    raw_tasks = json.loads(content)
    if not isinstance(raw_tasks, list):
        raise TypeError(f"Expected JSON array, got {type(raw_tasks).__name__}")

    tasks: list[AgentTask] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label", "unknown"))
        priority = int(item.get("priority", 10))
        priority = max(0, min(15, priority))  # clamp to valid range
        kernel_language = str(item.get("kernel_language", "python"))
        task_prompt = str(item.get("task_prompt", ""))

        if not task_prompt:
            continue

        tasks.append(AgentTask(
            agent_class=agent_class,
            task=task_prompt,
            label=label,
            priority=priority,
            kernel_language=kernel_language,
        ))

    if not tasks:
        raise ValueError("LLM response contained no valid tasks")

    return sorted(tasks, key=lambda t: t.priority)


# ============================================================================
# CLI
# ============================================================================

def _scan_previous_results(results_dir: Path) -> str:
    """Scan a previous round's results directory and build a summary.

    Reads task_*/patch_*_test.txt and task_*/*.log to understand what
    each task achieved. Returns a Markdown summary for the LLM prompt.
    """
    import re as _re

    sections: list[str] = []
    task_dirs = sorted(results_dir.glob("task_*"))
    if not task_dirs:
        return ""

    for td in task_dirs:
        label = td.name
        patches = sorted(td.glob("patch_*.patch"))
        test_outputs = sorted(td.glob("patch_*_test.txt"))
        log_files = sorted(td.glob("*.log"))

        section = [f"### {label}"]
        section.append(f"- Patches produced: {len(patches)}")

        # Extract key metrics from test outputs
        for tf in test_outputs[:3]:
            try:
                content = tf.read_text(errors="replace")[-2000:]
                # Look for common metric patterns
                speedups = _re.findall(r"speedup[:\s]+([0-9.]+)x?", content, _re.IGNORECASE)
                durations = _re.findall(r"duration[:\s]+([0-9.]+)\s*(?:us|µs|ms)", content, _re.IGNORECASE)
                if speedups:
                    section.append(f"- {tf.name}: speedup = {speedups[-1]}")
                elif durations:
                    section.append(f"- {tf.name}: duration = {durations[-1]}")
                else:
                    # Last 3 non-empty lines as fallback
                    lines = [l.strip() for l in content.splitlines() if l.strip()]
                    tail = lines[-3:] if len(lines) >= 3 else lines
                    section.append(f"- {tf.name} (tail): {' | '.join(tail)}")
            except Exception:
                section.append(f"- {tf.name}: (unreadable)")

        # Check for errors in logs
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


def main():
    """Generate optimization tasks from the command line."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate optimization tasks using LLM reasoning (or rule-based fallback)",
    )
    parser.add_argument("--kernel-path", default=None, help="Path to the kernel file")
    parser.add_argument(
        "--from-discovery", default=None, metavar="FILE",
        help="Read discovery.json and extract kernel-path and repo-root",
    )
    parser.add_argument("--profiling", default=None, help="Path to kernel-profile JSON output")
    parser.add_argument("--commandment", default=None, help="Path to COMMANDMENT.md")
    parser.add_argument("--baseline-metrics", default=None, help="Path to baseline_metrics.json")
    parser.add_argument("--model", default=None, help="Model name (default: from config/env)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM, use rule-based planner only")
    parser.add_argument("--repo-root", default=None, help="Repository root (for discovery)")
    parser.add_argument(
        "-o", "--output", default=None, metavar="DIR",
        help="Write task files to this directory (one .md per task) instead of JSON to stdout",
    )
    parser.add_argument(
        "--from-results", default=None, metavar="DIR",
        help="Previous round results directory (for iterative refinement)",
    )
    parser.add_argument(
        "--round", type=int, default=1,
        help="Round number for task file frontmatter (default: 1)",
    )

    args = parser.parse_args()

    # Populate from discovery JSON if provided (explicit flags override)
    disc_json = None
    if args.from_discovery:
        disc_json = json.loads(Path(args.from_discovery).read_text())
        if not args.kernel_path:
            args.kernel_path = (disc_json.get("kernel") or {}).get("file")
        if not args.repo_root:
            args.repo_root = disc_json.get("workspace")

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
        # Reconstruct DiscoveryResult from the test-discovery JSON output
        print(f"[task-generator] Loading discovery from {args.from_discovery}...", file=sys.stderr)
        kernel_info = disc_json.get("kernel") or {}
        kernels = []
        if kernel_info.get("file"):
            kernels.append(KernelInfo(
                file_path=Path(kernel_info["file"]),
                kernel_name=kernel_info.get("name", kernel_path.stem),
                kernel_type=kernel_info.get("type", "unknown"),
                kernel_language="python" if kernel_path.suffix == ".py" else "cpp",
                function_names=kernel_info.get("functions", []),
            ))
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

    # Load optional inputs
    profiling_result = None
    if args.profiling:
        profiling_result = json.loads(Path(args.profiling).read_text())

    commandment_content = None
    if args.commandment:
        commandment_content = Path(args.commandment).read_text()

    baseline_metrics = None
    if args.baseline_metrics:
        baseline_metrics = json.loads(Path(args.baseline_metrics).read_text())

    # Scan previous results for iterative refinement
    previous_results_summary = ""
    if args.from_results:
        results_dir = Path(args.from_results).resolve()
        if results_dir.is_dir():
            previous_results_summary = _scan_previous_results(results_dir)
            if previous_results_summary:
                print(f"[task-generator] Loaded results from {results_dir}", file=sys.stderr)
            else:
                print(f"[task-generator] No task results found in {results_dir}", file=sys.stderr)

    # Create model (unless --no-llm)
    model = None
    if not args.no_llm:
        try:
            from minisweagent.models import get_model
            model = get_model(args.model)
            print(f"[task-generator] Using model: {model.config.model_name}", file=sys.stderr)
        except Exception as e:
            print(f"[task-generator] Could not create model ({e}); using rule-based fallback", file=sys.stderr)

    # Placeholder agent class for CLI output
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent
    agent_class = StrategyInteractiveAgent

    base_task_context = f"Optimize the kernel at {kernel_path} for maximum performance."
    if previous_results_summary:
        base_task_context += f"\n\n{previous_results_summary}"

    # Generate tasks
    tasks = generate_tasks(
        discovery_result=discovery_result,
        base_task_context=base_task_context,
        agent_class=agent_class,
        model=model,
        profiling_result=profiling_result,
        commandment_content=commandment_content,
        baseline_metrics=baseline_metrics,
    )

    # Print summary to stderr
    print(f"\n[task-generator] Generated {len(tasks)} task(s):\n", file=sys.stderr)
    for i, t in enumerate(tasks):
        print(f"  [{t.priority:2d}] {t.label} ({t.kernel_language})", file=sys.stderr)

    # Output: directory of task files or JSON to stdout
    if args.output:
        from minisweagent.run.task_file import write_task_file

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = []
        for i, t in enumerate(tasks):
            filename = f"{t.priority:02d}_{t.label}.md"
            task_path = out_dir / filename

            metadata = {
                "label": t.label,
                "priority": t.priority,
                "agent_type": "openevolve" if "openevolve" in t.label else "strategy_agent",
                "kernel_language": t.kernel_language,
                "kernel_path": str(kernel_path),
                "repo_root": args.repo_root,
                "commandment": args.commandment,
                "baseline_metrics": args.baseline_metrics,
                "profiling": args.profiling,
                "round": args.round,
            }

            body = f"# {t.label}\n\n{t.task}\n"
            write_task_file(task_path, metadata, body, relative_to=out_dir)

            manifest.append({
                "index": i,
                "label": t.label,
                "priority": t.priority,
                "kernel_language": t.kernel_language,
                "file": str(task_path),
            })

        print(f"\n[task-generator] Wrote {len(tasks)} task file(s) to {out_dir}/", file=sys.stderr)
        print(json.dumps(manifest, indent=2))
    else:
        output = []
        for i, t in enumerate(tasks):
            output.append({
                "index": i,
                "label": t.label,
                "priority": t.priority,
                "kernel_language": t.kernel_language,
                "task_prompt_preview": t.task[:300] + ("..." if len(t.task) > 300 else ""),
            })
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
