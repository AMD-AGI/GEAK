"""Compatibility task generator for the heterogeneous pipeline.

This restores the missing ``task-generator`` entrypoint and provides a
deterministic, rule-based fallback that builds tasks from discovery results
without requiring the larger reorg-only planning stack.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from minisweagent.agents.agent_spec import AgentTask
from minisweagent.agents.heterogeneous.task_planner import build_optimization_tasks
from minisweagent.run.preprocess.discovery_types import DiscoveryResult


def _scan_single_round_results(results_dir: Path) -> list[str]:
    """Scan a single round's results directory and return section strings."""
    import re as _re

    sections: list[str] = []
    task_dirs = sorted(
        d for d in results_dir.iterdir() if d.is_dir() and d.name not in ("worktrees",) and not d.name.startswith(".")
    )
    if not task_dirs:
        return sections

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

    return sections


def _scan_previous_results(results_dir: Path) -> str:
    """Scan prior results directories and build a Markdown summary."""
    sections: list[str] = []

    round_subdirs = sorted(
        d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("round_")
    ) if results_dir.is_dir() else []

    if round_subdirs:
        for rd in round_subdirs:
            round_sections = _scan_single_round_results(rd)
            if round_sections:
                sections.append(f"## {rd.name.replace('_', ' ').title()} Results\n")
                sections.extend(round_sections)
    else:
        single_sections = _scan_single_round_results(results_dir)
        if single_sections:
            sections.append("## Previous Round Results\n")
            sections.extend(single_sections)

    if not sections:
        return ""

    return "\n\n".join(sections) + "\n"


def _task_to_manifest_entry(task: AgentTask, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "label": task.label,
        "priority": task.priority,
        "kernel_language": task.kernel_language,
        "task_prompt_preview": task.task[:300] + ("..." if len(task.task) > 300 else ""),
    }


def _write_task_file(path: Path, metadata: dict[str, Any], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: value
        for key, value in metadata.items()
        if value is not None
    }
    path.write_text(
        "---\n"
        + yaml.dump(payload, default_flow_style=False, sort_keys=False)
        + "---\n\n"
        + body
        + ("" if body.endswith("\n") else "\n")
    )


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
    codebase_context_path: Path | None = None,
    previous_tasks_dir: Path | None = None,
    round_evaluations: list[dict[str, Any]] | None = None,
    current_round: int = 1,
    num_gpus: int = 1,
) -> list[AgentTask]:
    """Generate tasks from discovery using the deterministic planner.

    The extra arguments are accepted for compatibility with the reorg API.
    This fallback currently uses the existing rule-based planner instead of an
    LLM planning pass so callers retain a working task-generation surface.
    """
    _ = (
        model,
        profiling_path,
        commandment_path,
        baseline_metrics_path,
        deep_search_path,
        previous_results_dir,
        discovery_path,
        codebase_context_path,
        previous_tasks_dir,
        round_evaluations,
        current_round,
        num_gpus,
    )
    return build_optimization_tasks(discovery_result, base_task_context, agent_class)


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
    codebase_context_path: Path | None = None,
    previous_tasks_dir: Path | None = None,
    round_evaluations: list[dict[str, Any]] | None = None,
    current_round: int = 1,
    num_gpus: int = 1,
) -> list[AgentTask]:
    """Compatibility wrapper that mirrors the reorg public API."""
    _ = (
        profiling_result,
        commandment_content,
        baseline_metrics,
        deep_search_content,
    )
    return generate_tasks(
        discovery_result=discovery_result,
        base_task_context=base_task_context,
        agent_class=agent_class,
        model=model,
        previous_results_dir=previous_results_dir,
        discovery_path=discovery_path,
        codebase_context_path=codebase_context_path,
        previous_tasks_dir=previous_tasks_dir,
        round_evaluations=round_evaluations,
        current_round=current_round,
        num_gpus=num_gpus,
    )


def main():
    """Generate optimization tasks from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate optimization tasks using the compatibility planner",
    )
    parser.add_argument("--kernel-path", default=None, help="Path to the kernel file")
    parser.add_argument(
        "--from-discovery",
        default=None,
        metavar="FILE",
        help="Read discovery.json and extract kernel-path and repo-root",
    )
    parser.add_argument("--repo-root", default=None, help="Repository root (for metadata)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="DIR",
        help="Write task files to this directory instead of JSON to stdout",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Round number for task file frontmatter (default: 1)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of available GPUs (default: 1)",
    )
    from minisweagent.run.pipeline_helpers import add_agent_filter_args, apply_agent_filter_env

    add_agent_filter_args(parser)
    args = parser.parse_args()
    apply_agent_filter_env(args)

    disc_json: dict[str, Any] | None = None
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

    if disc_json is None:
        disc_json = {
            "kernel": {
                "file": str(kernel_path),
                "name": kernel_path.stem,
                "type": "triton" if kernel_path.suffix == ".py" else "unknown",
                "functions": [kernel_path.stem],
            },
            "workspace": str(Path(args.repo_root).resolve()) if args.repo_root else str(kernel_path.parent),
            "tests": [],
            "benchmarks": [],
        }

    discovery_result = DiscoveryResult.from_dict(disc_json, kernel_path)
    if not discovery_result.kernels:
        print("ERROR: no kernels found by discovery", file=sys.stderr)
        sys.exit(1)

    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

    tasks = generate_tasks(
        discovery_result=discovery_result,
        base_task_context=f"Optimize the kernel at {kernel_path} for maximum performance.",
        agent_class=StrategyInteractiveAgent,
        model=None,
        num_gpus=args.num_gpus,
    )

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for i, task in enumerate(tasks):
            task_path = out_dir / f"{task.priority:02d}_{task.label}.md"
            metadata = {
                "label": task.label,
                "priority": task.priority,
                "agent_type": "strategy_agent",
                "kernel_language": task.kernel_language,
                "kernel_path": str(kernel_path),
                "repo_root": args.repo_root,
                "round": args.round,
                "num_gpus": task.num_gpus,
            }
            _write_task_file(task_path, metadata, f"# {task.label}\n\n{task.task}\n")
            entry = _task_to_manifest_entry(task, i)
            entry["file"] = str(task_path)
            manifest.append(entry)
        print(json.dumps(manifest, indent=2))
        return

    print(json.dumps([_task_to_manifest_entry(task, i) for i, task in enumerate(tasks)], indent=2))


__all__ = [
    "generate_tasks",
    "generate_tasks_from_content",
    "_scan_previous_results",
    "main",
]


if __name__ == "__main__":
    main()
