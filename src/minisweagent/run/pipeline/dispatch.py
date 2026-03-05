"""Dispatch helpers: run task files via ParallelAgent pool mode.

This module provides ``run_task_batch()`` which converts a list of task
file paths into ``AgentTask`` objects and feeds them into the existing
``ParallelAgent.run_parallel(tasks=...)`` pool mode.  The orchestrator
calls this; so does the ``run-tasks`` CLI indirectly.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── model ensemble support ───────────────────────────────────────────

def _build_ensemble_factory(base_factory):
    """Wrap *base_factory* to rotate through models in GEAK_MODEL_ENSEMBLE.

    When ``GEAK_MODEL_ENSEMBLE`` is set (comma-separated model names), each
    call to the returned factory creates an entirely new model instance with
    the next name in the round-robin list.  Because ``AmdLlmModel`` selects
    its vendor backend (OpenAI / Claude / Gemini) at construction time based
    on the model name, we must create a fresh instance per call rather than
    mutating an existing one.

    If the env var is unset or empty, returns *base_factory* unchanged.
    """
    ensemble_str = os.environ.get("GEAK_MODEL_ENSEMBLE", "").strip()
    if not ensemble_str:
        return base_factory

    model_names = [n.strip() for n in ensemble_str.split(",") if n.strip()]
    if len(model_names) < 2:
        return base_factory

    logger.info("Model ensemble enabled: %s", model_names)
    name_cycle = itertools.cycle(model_names)

    def _ensemble_factory():
        next_name = next(name_cycle)
        try:
            from minisweagent.models.amd_llm import AmdLlmModel
            model = AmdLlmModel(model_name=next_name)
            logger.info("Ensemble: created AmdLlmModel(%s)", next_name)
            return model
        except Exception:
            logger.warning(
                "Ensemble: failed to create model %s, falling back to base",
                next_name,
                exc_info=True,
            )
            return base_factory()

    return _ensemble_factory


def _read_commandment_section(commandment_path: str, section: str) -> str | None:
    """Read a section from a COMMANDMENT.md file verbatim.

    Returns the raw command lines for the given section (e.g. ``"SETUP"``,
    ``"CORRECTNESS"``, ``"PROFILE"``, ``"BENCHMARK"``,
    ``"FULL_BENCHMARK"``), exactly as written.  No parsing, no extraction,
    no transformation.
    
    Fenced code blocks (```bash, ```, etc.) are stripped automatically.
    """
    try:
        text = Path(commandment_path).read_text()
    except OSError:
        logger.debug("Could not read commandment at %s", commandment_path)
        return None

    lines: list[str] = []
    in_section = False
    # Pattern to match fenced code block markers (```bash, ```sh, ```, etc.)
    fence_pattern = re.compile(r"^```\w*$")
    
    for raw_line in text.splitlines():
        header = re.match(r"^##\s+(\w+)", raw_line.strip())
        if header:
            if header.group(1) == section:
                in_section = True
                continue
            elif in_section:
                break
            continue
        if in_section:
            stripped = raw_line.strip()
            # Skip fenced code block markers
            if fence_pattern.match(stripped):
                continue
            if stripped:
                lines.append(stripped)

    return "\n".join(lines) if lines else None


def _commandment_test_command(commandment_path: str) -> str | None:
    """Build a test command that executes SETUP then CORRECTNESS then BENCHMARK.

    The COMMANDMENT is the single source of truth.  Commands are executed
    *as-is* -- no parsing, no unwrapping, no modification.  The runtime
    must set ``GEAK_WORK_DIR`` and ``GEAK_GPU_DEVICE`` so that variable
    references in the commands resolve correctly.

    We write a temporary shell script instead of ``bash -c '...'`` because
    COMMANDMENT sections often contain single-quoted strings (e.g.
    ``printf '...'``) that cannot be nested inside a single-quoted
    ``bash -c`` wrapper.

    Note: This script is an internal implementation detail for agent tools
    (save_and_test).  OpenEvolve reads COMMANDMENT.md directly and does
    not use this script.
    """
    setup = _read_commandment_section(commandment_path, "SETUP")
    correctness = _read_commandment_section(commandment_path, "CORRECTNESS")
    benchmark = _read_commandment_section(commandment_path, "BENCHMARK")

    if not correctness:
        return None

    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    if setup:
        lines.append(setup)
    lines.append(correctness)
    if benchmark:
        lines.append(benchmark)
    script_body = "\n".join(lines) + "\n"

    cmd_dir = Path(commandment_path).parent
    # Use unique filename to avoid race conditions when multiple agents
    # run concurrently with the same COMMANDMENT directory
    import tempfile
    fd, script_path = tempfile.mkstemp(
        prefix="_geak_test_cmd_",
        suffix=".sh",
        dir=str(cmd_dir),
    )
    os.close(fd)
    script_path = Path(script_path)
    script_path.write_text(script_body)
    script_path.chmod(0o755)

    return str(script_path)


def task_file_to_agent_task(task_file: Path):
    """Read a task markdown file and convert it to an AgentTask.

    This is the canonical task-construction path used by both the
    orchestrator (``run_task_batch``) and the standalone ``run-tasks``
    CLI.  It:
      - Applies agent-type filtering (``filter_agent_type``)
      - Sets per-agent-type config (mode, strategy_manager, etc.)
      - Injects full pipeline context (COMMANDMENT, baseline metrics,
        profiling data, codebase context) into the task body
    """
    from minisweagent.agents.agent_spec import AgentTask
    from minisweagent.run.pipeline.task_file import read_task_file

    meta, body = read_task_file(task_file)

    from minisweagent.agents.agent_spec import _agent_type_to_class, filter_agent_type
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

    agent_type = filter_agent_type(meta.get("agent_type", "strategy_agent"))
    agent_class = _agent_type_to_class().get(agent_type, StrategyInteractiveAgent)

    if agent_type == "openevolve":
        # OpenEvolveWorker extends DefaultAgent (not InteractiveAgent),
        # so it does not accept 'mode' or 'use_strategy_manager'.
        cfg: dict = {
            "save_patch": True,
            "step_limit": 0,
            "cost_limit": 0.0,
        }
        if meta.get("kernel_path"):
            cfg["kernel_path"] = meta["kernel_path"]
        if meta.get("commandment"):
            cfg["commandment_path"] = meta["commandment"]
        if meta.get("baseline_metrics"):
            cfg["baseline_metrics_path"] = meta["baseline_metrics"]
    else:
        cfg = {
            "save_patch": True,
            "step_limit": 0,
            "cost_limit": 0.0,
            "mode": "yolo",
            "use_strategy_manager": True,
        }

    # COMMANDMENT is the single source of truth for test commands.
    # Its SETUP + CORRECTNESS + BENCHMARK sections are executed verbatim.
    if meta.get("commandment") and Path(meta["commandment"]).exists():
        derived = _commandment_test_command(meta["commandment"])
        if derived:
            cfg["test_command"] = derived
            logger.info("test_command from COMMANDMENT (verbatim): %s", derived)
    if not cfg.get("test_command") and meta.get("test_command"):
        cfg["test_command"] = meta["test_command"]
        logger.warning(
            "No COMMANDMENT available; falling back to raw test_command: %s",
            meta["test_command"],
        )

    # Prepend pipeline context so the sub-agent has all necessary information.
    # IMPORTANT: Paths from metadata use the ORIGINAL repo root.  The parallel
    # agent's _replace_paths() rewrites them to the worktree path before the
    # agent sees the task text.  We must include these paths verbatim here.
    from minisweagent.run.pipeline.helpers import inject_pipeline_context

    commandment_text: str | None = None
    _cmd_path = meta.get("commandment")
    if _cmd_path and Path(_cmd_path).exists():
        commandment_text = Path(_cmd_path).read_text().strip()

    baseline_metrics: dict | None = None
    _bm_path = meta.get("baseline_metrics")
    if _bm_path and Path(_bm_path).exists():
        import json as _json

        baseline_metrics = _json.loads(Path(_bm_path).read_text())

    codebase_ctx_text: str | None = None
    _cb_path = meta.get("codebase_context")
    if _cb_path and Path(_cb_path).exists():
        codebase_ctx_text = Path(_cb_path).read_text().strip()

    benchmark_baseline_text: str | None = None
    _bb_path = meta.get("benchmark_baseline")
    if _bb_path and Path(_bb_path).exists():
        benchmark_baseline_text = Path(_bb_path).read_text().strip()

    body, cfg = inject_pipeline_context(
        body,
        cfg,
        commandment_text=commandment_text,
        baseline_metrics=baseline_metrics,
        profiling_path=meta.get("profiling"),
        kernel_path=meta.get("kernel_path"),
        repo_root=meta.get("repo_root"),
        test_command=cfg.get("test_command"),
        codebase_context=codebase_ctx_text,
        benchmark_baseline=benchmark_baseline_text,
    )

    if meta.get("starting_patch"):
        cfg["starting_patch"] = meta["starting_patch"]

    return AgentTask(
        agent_class=agent_class,
        task=body,
        label=meta.get("label", task_file.stem),
        priority=int(meta.get("priority", 10)),
        kernel_language=meta.get("kernel_language", "python"),
        config=cfg,
        num_gpus=int(meta.get("num_gpus", 1)),
    )


def run_task_batch(
    task_files: list[Path],
    gpu_ids: list[int],
    output_dir: Path,
    model_factory,
    *,
    console=None,
) -> dict[str, Any]:
    """Run a batch of task files via ParallelAgent pool mode.

    Parameters
    ----------
    task_files:
        List of task markdown file paths.
    gpu_ids:
        GPU device IDs to use.
    output_dir:
        Base output directory for results.
    model_factory:
        Callable returning a new model instance.
    console:
        Optional Rich console.

    Returns
    -------
    dict with 'completed', 'failed', and 'results' keys.
    """
    from minisweagent.agents.parallel_agent import ParallelAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.run.pipeline.task_file import read_task_file

    if not task_files:
        return {"completed": 0, "failed": 0, "results": []}

    tasks = [task_file_to_agent_task(f) for f in task_files]

    # Determine repo_path and harness_path from first task's metadata
    meta_0, _ = read_task_file(task_files[0])
    repo_root = meta_0.get("repo_root")
    repo_path = Path(repo_root).resolve() if repo_root else Path.cwd()
    harness_path = meta_0.get("harness_path", "")

    is_git = False
    if repo_path.is_dir():
        is_git = (repo_path / ".git").exists() or (repo_path / ".git").is_file()

    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    agent_config: dict[str, Any] = {
        "save_patch": True,
    }

    # Pre-seed GEAK_REPO_ROOT and GEAK_HARNESS so COMMANDMENT commands
    # can reference them as variables (no hardcoded paths).
    from minisweagent.run.pipeline.helpers import DEFAULT_AGENT_BENCHMARK_ITERATIONS

    base_env_vars: dict[str, str] = {
        "GEAK_REPO_ROOT": str(repo_path.resolve()),
        "GEAK_BENCHMARK_EXTRA_ARGS": f"--iterations {DEFAULT_AGENT_BENCHMARK_ITERATIONS}",
    }
    if harness_path:
        base_env_vars["GEAK_HARNESS"] = harness_path

    def env_factory():
        return LocalEnvironment(
            **{"cwd": str(repo_path.resolve()), "timeout": 3600, "env": base_env_vars}
        )

    effective_model_factory = _build_ensemble_factory(model_factory)

    try:
        raw_results = ParallelAgent.run_parallel(
            num_parallel=len(gpu_ids),
            repo_path=repo_path,
            is_git_repo=is_git,
            task_content="",
            agent_class=tasks[0].agent_class if tasks else type(None),
            agent_config=agent_config,
            model_factory=effective_model_factory,
            env_factory=env_factory,
            base_patch_dir=results_dir,
            output=None,
            gpu_ids=gpu_ids,
            console=console,
            tasks=tasks,
        )
    except Exception as exc:
        logger.error("Task batch execution failed: %s", exc, exc_info=True)
        return {
            "completed": 0,
            "failed": len(tasks),
            "error": str(exc),
            "results": [],
        }

    completed = 0
    failed = 0
    summaries = []

    for entry in raw_results:
        agent_idx, _agent, exit_status, result = entry
        label = tasks[agent_idx].label if agent_idx < len(tasks) else f"task_{agent_idx}"
        success = exit_status not in ("error", "Error", None)
        if success:
            completed += 1
        else:
            failed += 1

        # Count patches written to the task's result directory
        task_result_dir = results_dir / label
        patch_count = len(list(task_result_dir.glob("*.patch"))) if task_result_dir.is_dir() else 0

        summaries.append(
            {
                "index": agent_idx,
                "label": label,
                "exit": str(exit_status),
                "patches": patch_count,
            }
        )

    return {
        "completed": completed,
        "failed": failed,
        "results": summaries,
        "results_dir": str(results_dir),
    }


def run_from_task(
    task_file: Path,
    gpu_id: int = 0,
    output_dir: Path | None = None,
    model_factory=None,
    *,
    console=None,
) -> dict[str, Any]:
    """Run a single task file. Python-callable wrapper around geak --from-task.

    Shares the same underlying code as the CLI ``--from-task`` path but
    returns a results dict instead of printing to console.
    """
    # Default: tasks/round_N/00_label.md -> results/round_N/
    if output_dir:
        out = output_dir
    else:
        round_name = task_file.parent.name
        out = task_file.parent.parent.parent / "results" / round_name
    return run_task_batch(
        task_files=[task_file],
        gpu_ids=[gpu_id],
        output_dir=out,
        model_factory=model_factory,
        console=console,
    )
