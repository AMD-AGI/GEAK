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
    """Read all .md task files and build AgentTask objects.

    Delegates to ``dispatch.task_file_to_agent_task`` so that the
    standalone CLI produces the same context-enriched tasks as the
    orchestrator pipeline.
    """
    from minisweagent.run.dispatch import task_file_to_agent_task

    task_files = sorted(task_dir.glob("*.md"))
    if not task_files:
        print(f"ERROR: no .md task files found in {task_dir}", file=sys.stderr)
        sys.exit(1)

    return [task_file_to_agent_task(tf) for tf in task_files]


def main():
    """Run tasks from a directory in parallel across GPUs.

    Uses the same ``run_task_batch`` path as the orchestrator, ensuring
    identical task construction, context injection, and agent-type
    filtering.
    """
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
        "--output-dir",
        default=None,
        help="Output directory for patches and results (default: <task-dir>/../results)",
    )
    parser.add_argument(
        "--allowed-agents",
        default=None,
        help="Comma-separated list of allowed agent types. Sets GEAK_ALLOWED_AGENTS.",
    )
    parser.add_argument(
        "--excluded-agents",
        default=None,
        help="Comma-separated list of excluded agent types. Sets GEAK_EXCLUDED_AGENTS.",
    )

    args = parser.parse_args()

    if args.allowed_agents:
        os.environ["GEAK_ALLOWED_AGENTS"] = args.allowed_agents
    if args.excluded_agents:
        os.environ["GEAK_EXCLUDED_AGENTS"] = args.excluded_agents

    task_dir = Path(args.task_dir).resolve()
    if not task_dir.is_dir():
        print(f"ERROR: task directory not found: {args.task_dir}", file=sys.stderr)
        sys.exit(1)

    task_files = sorted(task_dir.glob("*.md"))
    if not task_files:
        print(f"ERROR: no .md task files found in {task_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]
    else:
        from minisweagent.agents.agent_spec import detect_available_gpus

        gpu_ids = detect_available_gpus()
    print(f"[run-tasks] Using GPU(s): {gpu_ids}", file=sys.stderr)

    # Output directory: default mirrors the task dir structure under results/
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        round_name = task_dir.name
        output_dir = task_dir.parent.parent / "results" / round_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run-tasks] Output directory: {output_dir}", file=sys.stderr)

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

        def model_factory():
            return get_model(model_name, config=copy.deepcopy(model_config))

        print(f"[run-tasks] Using model: {model_factory().config.model_name}", file=sys.stderr)
    except Exception as e:
        print(
            f"ERROR: run-tasks requires an LLM model. Set GEAK_MODEL or use --model. ({e})",
            file=sys.stderr,
        )
        sys.exit(1)

    from minisweagent.run.dispatch import run_task_batch

    print(
        f"\n[run-tasks] Starting pool execution: {len(task_files)} tasks on {len(gpu_ids)} GPU(s)\n",
        file=sys.stderr,
    )

    results = run_task_batch(
        task_files=[Path(f) for f in task_files],
        gpu_ids=gpu_ids,
        output_dir=output_dir,
        model_factory=model_factory,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
