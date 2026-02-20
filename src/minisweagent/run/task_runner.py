"""Batch task runner -- reads task .md files from a directory and runs them
in parallel across available GPUs using the pool scheduler.

Usage:
    run-tasks --task-dir tasks/round1/ --gpu-ids 0,1,2,3
    run-tasks --task-dir tasks/round1/ --gpu-ids 0  # serial on one GPU
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _build_tasks_from_dir(task_dir: Path) -> list[Any]:
    """Read all .md task files and build AgentTask objects."""
    from minisweagent.agents.agent_spec import AgentTask, _agent_type_to_class
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent
    from minisweagent.run.task_file import read_task_file

    task_files = sorted(task_dir.glob("*.md"))
    if not task_files:
        print(f"ERROR: no .md task files found in {task_dir}", file=sys.stderr)
        sys.exit(1)

    _AGENT_TYPE_TO_CLASS = _agent_type_to_class()

    tasks: list[AgentTask] = []
    for tf in task_files:
        meta, body = read_task_file(tf)

        agent_type = meta.get("agent_type", "strategy_agent")
        label = meta.get("label", tf.stem)
        priority = int(meta.get("priority", 10))
        kernel_language = meta.get("kernel_language", "python")

        agent_class = _AGENT_TYPE_TO_CLASS.get(agent_type, StrategyInteractiveAgent)

        cfg: dict[str, Any] = {}
        if agent_type == "openevolve":
            if meta.get("kernel_path"):
                cfg["kernel_path"] = meta["kernel_path"]
            if meta.get("commandment"):
                cfg["commandment_path"] = meta["commandment"]
            if meta.get("baseline_metrics"):
                cfg["baseline_metrics_path"] = meta["baseline_metrics"]

        if meta.get("test_command"):
            cfg["test_command"] = meta["test_command"]
        elif meta.get("commandment"):
            from minisweagent.run.dispatch import _derive_test_command_from_commandment

            derived = _derive_test_command_from_commandment(meta["commandment"])
            if derived:
                cfg["test_command"] = derived

        tasks.append(
            AgentTask(
                agent_class=agent_class,
                task=body,
                label=label,
                priority=priority,
                kernel_language=kernel_language,
                config=cfg,
            )
        )

    return tasks


def main():
    """Run tasks from a directory in parallel across GPUs."""
    import argparse
    import copy

    parser = argparse.ArgumentParser(
        description="Run optimization tasks from a directory in parallel across GPUs",
    )
    parser.add_argument(
        "--task-dir",
        required=True,
        help="Directory containing .md task files (from task-generator -o)",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU device IDs (default: auto-detect or 0)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: from GEAK_MODEL env or config)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (override task file metadata)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for patches and results (default: <task-dir>/../results)",
    )

    args = parser.parse_args()

    task_dir = Path(args.task_dir).resolve()
    if not task_dir.is_dir():
        print(f"ERROR: task directory not found: {args.task_dir}", file=sys.stderr)
        sys.exit(1)

    # Build tasks from directory
    tasks = _build_tasks_from_dir(task_dir)
    print(f"[run-tasks] Loaded {len(tasks)} task(s) from {task_dir}:", file=sys.stderr)
    for t in tasks:
        agent_label = "openevolve" if "OpenEvolve" in t.agent_class.__name__ else "strategy"
        print(f"  [{t.priority:2d}] {t.label} ({t.kernel_language}, {agent_label})", file=sys.stderr)

    # Resolve GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]
    else:
        from minisweagent.agents.agent_spec import detect_available_gpus

        gpu_ids = detect_available_gpus()
    print(f"[run-tasks] Using GPU(s): {gpu_ids}", file=sys.stderr)

    # Determine repo root from first task file metadata
    repo_root: Path | None = None
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        from minisweagent.run.task_file import read_task_file

        first_task = sorted(task_dir.glob("*.md"))[0]
        meta, _ = read_task_file(first_task)
        if meta.get("repo_root"):
            repo_root = Path(meta["repo_root"]).resolve()
        elif meta.get("kernel_path"):
            repo_root = Path(meta["kernel_path"]).resolve().parent

    if not repo_root or not repo_root.is_dir():
        print("ERROR: could not determine repo root. Use --repo-root.", file=sys.stderr)
        sys.exit(1)

    # Output directory: default mirrors the task dir structure under results/
    # e.g. tasks/round_1/ -> results/round_1/
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        round_name = task_dir.name
        output_dir = task_dir.parent.parent / "results" / round_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run-tasks] Output directory: {output_dir}", file=sys.stderr)

    # Check if repo is a git repo
    from minisweagent.run.task_file import is_git_repo

    is_git = is_git_repo(repo_root)

    # Create model
    model_name = args.model or os.environ.get("GEAK_MODEL")
    try:
        from minisweagent.config import get_config_path
        from minisweagent.models import get_model

        geak_cfg = get_config_path("geak")
        model_config: dict[str, Any] = {}
        if geak_cfg.exists():
            import yaml

            full_cfg = yaml.safe_load(geak_cfg.read_text()) or {}
            model_config = full_cfg.get("model", {})
        model = get_model(model_name, config=model_config)
        print(f"[run-tasks] Using model: {model.config.model_name}", file=sys.stderr)
    except Exception as e:
        print(
            f"ERROR: run-tasks requires an LLM model. Set GEAK_MODEL or use --model. ({e})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build agent config
    agent_config: dict[str, Any] = {
        "patch_output_dir": str(output_dir),
        "save_patch": True,
    }

    # Load test_command from first task file if available
    meta_first, _ = read_task_file(sorted(task_dir.glob("*.md"))[0])
    if meta_first.get("test_command"):
        agent_config["test_command"] = meta_first["test_command"]

    # Run tasks in parallel
    from minisweagent.agents.parallel_agent import ParallelAgent
    from minisweagent.environments.local import LocalEnvironment

    print(f"\n[run-tasks] Starting pool execution: {len(tasks)} tasks on {len(gpu_ids)} GPU(s)\n", file=sys.stderr)

    results = ParallelAgent.run_parallel(
        num_parallel=len(gpu_ids),
        repo_path=repo_root,
        is_git_repo=is_git,
        task_content="",
        agent_class=tasks[0].agent_class,
        agent_config=agent_config,
        model_factory=lambda: get_model(model_name, config=copy.deepcopy(model_config)),
        env_factory=lambda: LocalEnvironment(cwd=str(repo_root)),
        base_patch_dir=output_dir,
        output=None,
        gpu_ids=gpu_ids,
        tasks=tasks,
    )

    # Report results
    print(f"\n[run-tasks] Completed {len(results)} task(s):\n", file=sys.stderr)
    summary = []
    for agent_id, exit_status, result, _extra in results:
        task = tasks[agent_id] if agent_id < len(tasks) else None
        label = task.label if task else f"task_{agent_id}"
        print(f"  [{agent_id}] {label}: {exit_status}", file=sys.stderr)
        summary.append(
            {
                "task_id": agent_id,
                "label": label,
                "exit_status": str(exit_status),
                "result_preview": str(result)[:300] if result else "",
            }
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
