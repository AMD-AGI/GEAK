"""Agent-based task generator -- produces optimization tasks by running a
read-only planning agent that inspects profiling data and kernel metadata.

The agent reads files via ``str_replace_editor view`` and submits a JSON
task list via the ``submit`` tool.  No rule-based fallback: an LLM model
is required.

Priority scheme (lower = higher priority, runs first):
  0  -- Algorithmic kernel-body rewrites (highest impact)
  5  -- Kernel fusion / advanced tuning
  10 -- Targeted optimization (autotune, memory, launch config)
  15 -- Profile-guided (generic fallback)

Usage (Python):
    from minisweagent.agents.heterogeneous.task_generator import generate_tasks
    tasks = generate_tasks(
        base_task_context=task_text,
        agent_class=StrategyAgent,
        model=model,
        kernel_path="/path/to/kernel.py",
        kernel_name="my_kernel",
        kernel_type="triton",
        profiling_path=Path("profile.json"),
        commandment_path=Path("COMMANDMENT.md"),
    )

Usage (CLI):
    python -m minisweagent.agents.heterogeneous.task_generator \\
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
import time
from pathlib import Path
from typing import Any

from minisweagent.agents.agent_spec import AgentTask
from minisweagent.agents.heterogeneous.prompts import (
    TASKGEN_INSTANCE_TEMPLATE as _INSTANCE_TEMPLATE,
)
from minisweagent.agents.heterogeneous.prompts import (
    TASKGEN_SYSTEM_PROMPT as _SYSTEM_PROMPT,
)
from minisweagent.agents.heterogeneous.prompts import (
    build_agent_restriction_addendum as _build_agent_restriction_addendum,
)
from minisweagent.agents.heterogeneous.result_scanning import (  # noqa: F401
    scan_previous_results as _scan_previous_results,
)
from minisweagent.agents.heterogeneous.result_scanning import (
    scan_previous_tasks as _scan_previous_tasks,
)
from minisweagent.agents.heterogeneous.workload_guidance import _build_workload_guidance  # noqa: F401
from minisweagent.debug_runtime import emit_debug_log, model_tools_snapshot, tool_names

logger = logging.getLogger(__name__)

_KNOWLEDGE_BASE_REL = "knowledge_base/optimization_strategies.py"


# ============================================================================
# Kernel metadata extraction
# ============================================================================


def _infer_kernel_type(kernel_path: Path) -> str:
    """Infer kernel_type from file content/extension when discovery.json is absent.

    For Python files, checks for direct Triton markers first, then follows
    ``from X import ...`` statements (up to 2 levels) to detect wrapper files
    that import Triton kernels from other modules.
    """
    ext = kernel_path.suffix.lower()
    if ext == ".py":
        try:
            text = kernel_path.read_text(errors="ignore")
            if "@triton" in text or "tl." in text:
                logger.debug("_infer_kernel_type: triton markers found in %s", kernel_path.name)
                return "triton"
            if "import triton" in text:
                if _check_imported_triton(text, kernel_path):
                    logger.debug("_infer_kernel_type: triton detected via import-follow in %s", kernel_path.name)
                    return "triton"
                logger.debug("_infer_kernel_type: bare 'import triton' in %s; classifying as triton.", kernel_path.name)
                return "triton"
        except OSError as exc:
            logger.debug("_infer_kernel_type: could not read %s: %s", kernel_path, exc)
        logger.debug("_infer_kernel_type: no triton markers in %s; returning 'unknown'.", kernel_path.name)
        return "unknown"
    if ext in (".cu", ".hip", ".hpp", ".cpp"):
        path_lower = str(kernel_path).lower()
        if "composable_kernel" in path_lower or "/ck_" in path_lower or "/ck/" in path_lower:
            logger.debug("_infer_kernel_type: CK path pattern in %s", kernel_path.name)
            return "ck"
        logger.debug("_infer_kernel_type: native extension %s → hip", ext)
        return "hip"
    logger.debug("_infer_kernel_type: unrecognised extension %s; returning 'unknown'.", ext)
    return "unknown"


def _check_imported_triton(content: str, file_path: Path, _depth: int = 0) -> bool:
    """Follow imports to check if any imported module contains @triton.jit."""
    if _depth > 2:
        return False

    import re
    import sys

    import_re = re.compile(r"^\s*from\s+([\w.]+)\s+import\s", re.MULTILINE)
    search_dirs = [file_path.parent]
    for sp in sys.path:
        p = Path(sp)
        if p.is_dir():
            search_dirs.append(p)

    for m in import_re.finditer(content):
        module_path = m.group(1).replace(".", "/")
        for base in search_dirs:
            candidate = base / f"{module_path}.py"
            if not candidate.is_file():
                candidate = base / module_path / "__init__.py"
            if not candidate.is_file():
                continue
            try:
                imported = candidate.read_text(errors="ignore")[:8192]
            except OSError:
                continue
            if "@triton.jit" in imported or "@triton.autotune" in imported:
                return True
            if _depth < 2 and "import triton" in imported:
                if _check_imported_triton(imported, candidate, _depth + 1):
                    return True
            break
    return False


def _extract_kernel_meta(
    discovery: dict | None,
    kernel_path: str,
) -> dict[str, Any]:
    """Build flat kernel metadata from a discovery.json dict and kernel path.

    When discovery.json is available, reads kernel_type from it directly.
    When absent, infers kernel_type from file extension and content.
    Other fields use simple defaults -- the LLM reads CODEBASE_CONTEXT.md
    for the full dependency tree, function names, and import relationships.
    """
    kp = Path(kernel_path) if kernel_path else Path("unknown.py")
    kernel_info = (discovery or {}).get("kernel") or {}
    ktype = kernel_info.get("type") or _infer_kernel_type(kp)
    from minisweagent.run.preprocess.discovery_types import _infer_kernel_language

    return {
        "kernel_path": str(kp),
        "kernel_name": kernel_info.get("name", kp.stem),
        "kernel_type": ktype,
        "kernel_language": _infer_kernel_language(kp, ktype),
        "function_names": kernel_info.get("functions", []),
        "workspace_path": (discovery or {}).get("workspace", str(kp.parent)),
    }


# ============================================================================
# Public API
# ============================================================================


def generate_tasks(
    base_task_context: str,
    agent_class: type,
    model: Any,
    *,
    kernel_path: str = "",
    kernel_name: str = "",
    kernel_type: str = "unknown",
    kernel_language: str = "python",
    function_names: list[str] | None = None,
    workspace_path: str = "",
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
    """Generate optimization tasks using an LLM planning agent.

    Args:
        base_task_context: Common context prepended to each task prompt.
        agent_class: Default agent class for tasks (typically StrategyAgent).
        model: LLM model instance (required).
        kernel_path: Absolute path to the kernel file.
        kernel_name: Human-readable kernel name.
        kernel_type: Backend type (triton, hip, cuda, ck, asm, unknown).
        kernel_language: Source language (python, cpp, asm).
        function_names: Key function names within the kernel file.
        workspace_path: Working directory for the planning agent.
        profiling_path: Path to kernel-profile JSON output.
        commandment_path: Path to COMMANDMENT.md.
        baseline_metrics_path: Path to baseline_metrics.json.
        deep_search_path: Path to deep search findings file.
        previous_results_dir: Path to previous round results directory.
        discovery_path: Path to the discovery.json file.
        codebase_context_path: Path to CODEBASE_CONTEXT.md file.
        previous_tasks_dir: Path to the parent tasks/ directory.
        round_evaluations: List of orchestrator round evaluation dicts.
        current_round: Current round number (for scanning prior tasks).

    Returns:
        List of AgentTask sorted by priority.

    Raises:
        RuntimeError: If the agent fails to submit results.
    """
    if not kernel_path:
        logger.warning("generate_tasks: kernel_path is empty; returning no tasks.")
        return []

    submitted_text = _run_task_agent(
        kernel_path=kernel_path,
        kernel_name=kernel_name,
        kernel_type=kernel_type,
        kernel_language=kernel_language,
        function_names=function_names or [],
        workspace_path=workspace_path,
        base_task_context=base_task_context,
        model=model,
        profiling_path=profiling_path,
        commandment_path=commandment_path,
        baseline_metrics_path=baseline_metrics_path,
        deep_search_path=deep_search_path,
        previous_results_dir=previous_results_dir,
        discovery_path=discovery_path,
        codebase_context_path=codebase_context_path,
        previous_tasks_dir=previous_tasks_dir,
        round_evaluations=round_evaluations,
        current_round=current_round,
        num_gpus=num_gpus,
    )

    return _parse_llm_response(
        submitted_text,
        agent_class,
        kernel_path=kernel_path,
        commandment_path=str(commandment_path) if commandment_path else None,
        baseline_metrics_path=str(baseline_metrics_path) if baseline_metrics_path else None,
    )


def generate_tasks_from_content(
    base_task_context: str,
    agent_class: type,
    model: Any,
    *,
    kernel_path: str = "",
    kernel_name: str = "",
    kernel_type: str = "unknown",
    kernel_language: str = "python",
    function_names: list[str] | None = None,
    workspace_path: str = "",
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
            base_task_context=base_task_context,
            agent_class=agent_class,
            model=model,
            kernel_path=kernel_path,
            kernel_name=kernel_name,
            kernel_type=kernel_type,
            kernel_language=kernel_language,
            function_names=function_names,
            workspace_path=workspace_path,
            profiling_path=profiling_path,
            commandment_path=commandment_path,
            baseline_metrics_path=baseline_metrics_path,
            deep_search_path=deep_search_path,
            previous_results_dir=previous_results_dir,
            discovery_path=discovery_path,
            codebase_context_path=codebase_context_path,
            previous_tasks_dir=previous_tasks_dir,
            round_evaluations=round_evaluations,
            current_round=current_round,
            num_gpus=num_gpus,
        )
    finally:
        for f in tmp_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove temp file %s", f)


def write_task_files(
    tasks: list[AgentTask],
    output_dir: Path,
    *,
    kernel_path: str = "",
    repo_root: str = "",
    commandment: str = "",
    baseline_metrics: str = "",
    profiling: str = "",
    codebase_context: str = "",
    benchmark_baseline: str = "",
    test_command: str = "",
    starting_patch: str = "",
    harness_path: str = "",
    round_num: int = 1,
) -> list[Path]:
    """Write AgentTask objects to .md task files on disk.

    Returns the list of written file paths.  Used by both the orchestrator
    tool (``tool_generate_tasks``) and the CLI (``main``).
    """
    from minisweagent.agents.agent_spec import _agent_class_to_type
    from minisweagent.run.task_file import write_task_file

    class_to_type = _agent_class_to_type()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for t in tasks:
        filename = f"{t.priority:02d}_{t.label}.md"
        task_path = output_dir / filename
        metadata = {
            "label": t.label,
            "priority": t.priority,
            "agent_type": class_to_type.get(t.agent_class, "strategy_agent"),
            "kernel_language": t.kernel_language,
            "kernel_path": kernel_path,
            "repo_root": repo_root,
            "commandment": commandment,
            "baseline_metrics": baseline_metrics,
            "profiling": profiling,
            "codebase_context": codebase_context,
            "benchmark_baseline": benchmark_baseline,
            "starting_patch": starting_patch,
            "harness_path": harness_path,
            "num_gpus": t.num_gpus,
            "test_command": test_command,
            "round": round_num,
        }
        body = f"# {t.label}\n\n{t.task}\n"
        write_task_file(task_path, metadata, body)
        paths.append(task_path)

    return paths


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
    *,
    kernel_path: str,
    kernel_name: str,
    kernel_type: str,
    kernel_language: str,
    function_names: list[str],
    workspace_path: str,
    base_task_context: str,
    model: Any,
    profiling_path: Path | None,
    commandment_path: Path | None,
    baseline_metrics_path: Path | None,
    deep_search_path: Path | None,
    previous_results_dir: Path | None,
    discovery_path: Path | None,
    codebase_context_path: Path | None = None,
    previous_tasks_dir: Path | None = None,
    round_evaluations: list[dict[str, Any]] | None = None,
    current_round: int = 1,
    num_gpus: int = 1,
) -> str:
    """Run a read-only planning agent and return the submitted JSON text."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.tools.tools_runtime import get_tools_list

    workspace = Path(workspace_path) if workspace_path else Path(kernel_path).parent

    read_only_tools = [t for t in get_tools_list() if t["name"] in ("str_replace_editor", "submit")]
    # AmdLlmModel forwards set_tools() to its _impl; snapshot the actual target.
    _model_target = getattr(model, "_impl", model)
    original_tools = list(_model_target.tools) if hasattr(_model_target, "tools") else None
    # region agent log
    emit_debug_log(
        "task_generator.py:_run_task_agent:before_override",
        "Replacing model tools for task-planning sub-agent",
        {
            "workspace": str(workspace),
            "read_only_tools": tool_names(read_only_tools),
            "original_tools": tool_names(original_tools),
            "model_target_type": type(_model_target).__name__,
            "model_target_id": id(_model_target),
            "model_before": model_tools_snapshot(model),
        },
        hypothesis_id="H1",
    )
    # endregion
    if hasattr(model, "set_tools"):
        model.set_tools(read_only_tools)
    else:
        _model_target.tools = read_only_tools

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

        prev_tasks_path: Path | None = None
        if previous_tasks_dir and Path(previous_tasks_dir).is_dir() and current_round > 1:
            tasks_summary = _scan_previous_tasks(Path(previous_tasks_dir), current_round)
            if tasks_summary:
                prev_tasks_path = _write_temp(tasks_summary, "_prev_tasks.md")
                tmp_files.append(prev_tasks_path)

        round_evals_path: Path | None = None
        if round_evaluations:
            evals_text = "## Orchestrator Round Evaluations\n\n"
            for rev in round_evaluations:
                r_num = rev.get("round", "?")
                evals_text += f"### Round {r_num}\n"
                evals_text += f"- Best task: {rev.get('best_task', 'N/A')}\n"
                fb = rev.get("full_benchmark", {})
                canonical_speedup = (
                    fb.get("verified_speedup", "N/A")
                    if isinstance(fb, dict) and fb
                    else rev.get("benchmark_speedup", "N/A")
                )
                evals_text += f"- Canonical benchmark speedup: {canonical_speedup}x\n"
                if fb:
                    evals_text += f"- Verified kernel time: {fb.get('kernel_time_ms', 'N/A')}ms\n"
                profile = rev.get("profile_comparison", {})
                if profile:
                    evals_text += f"- Profile comparison: {json.dumps(profile, default=str)[:500]}\n"
                evals_text += f"- Best patch: {rev.get('best_patch', 'N/A')}\n\n"
            round_evals_path = _write_temp(evals_text, "_round_evals.md")
            tmp_files.append(round_evals_path)

        template_vars = {
            "kernel_path": kernel_path,
            "kernel_name": kernel_name,
            "kernel_type": kernel_type,
            "kernel_language": kernel_language,
            "function_names": ", ".join(function_names) if function_names else "",
            "codebase_context_path": str(codebase_context_path) if codebase_context_path else "",
            "discovery_path": str(discovery_path) if discovery_path else "",
            "profiling_path": str(profiling_path) if profiling_path else "",
            "commandment_path": str(commandment_path) if commandment_path else "",
            "baseline_metrics_path": str(baseline_metrics_path) if baseline_metrics_path else "",
            "knowledge_base_path": str(kb_path) if kb_path else "",
            "deep_search_path": str(deep_search_path) if deep_search_path else "",
            "previous_results_path": str(prev_results_path) if prev_results_path else "",
            "previous_tasks_path": str(prev_tasks_path) if prev_tasks_path else "",
            "round_evaluations_path": str(round_evals_path) if round_evals_path else "",
            "base_task_context": base_task_context,
            "num_gpus": num_gpus,
            "memory_context": "",
            "workload_guidance": "",
        }

        _bm_dict: dict[str, Any] = {}
        try:
            from minisweagent.memory.integration import (  # pylint: disable=import-error,no-name-in-module
                assemble_memory_context,
            )
            from minisweagent.memory.working_notebook import (  # pylint: disable=import-error,no-name-in-module
                summarize_working_notebook,
            )

            if baseline_metrics_path and Path(baseline_metrics_path).exists():
                _bm_dict = json.loads(Path(baseline_metrics_path).read_text())
            _notebook_dir = None
            if baseline_metrics_path and Path(baseline_metrics_path).exists():
                _notebook_dir = Path(baseline_metrics_path).resolve().parent / "_working_memory"
            elif previous_results_dir and Path(previous_results_dir).is_dir():
                _notebook_dir = Path(previous_results_dir).resolve().parent.parent / "_working_memory"
            _wm_ctx = summarize_working_notebook(_notebook_dir)
            _mem = assemble_memory_context(
                kernel_path=kernel_path,
                bottleneck_type=_bm_dict.get("bottleneck"),
                profiling_metrics=_bm_dict,
            )
            combined_memory = "\n\n".join(part.strip() for part in (_wm_ctx, _mem or "") if part and str(part).strip())
            if combined_memory:
                template_vars["memory_context"] = combined_memory
        except Exception as exc:
            logger.warning("Memory assembly failed in task generator: %s", exc)

        _kernel_meta = {
            "file_path": kernel_path,
            "kernel_name": kernel_name,
            "kernel_type": kernel_type,
        }
        template_vars["workload_guidance"] = _build_workload_guidance(_kernel_meta, _bm_dict)

        tg_step_limit = int(os.getenv("GEAK_TASKGEN_STEP_LIMIT", "200"))
        tg_cost_limit = float(os.getenv("GEAK_TASKGEN_COST_LIMIT", "50.0"))

        system_prompt = _SYSTEM_PROMPT + _build_agent_restriction_addendum()

        agent = DefaultAgent(
            model,
            env,
            system_template=system_prompt,
            instance_template=_INSTANCE_TEMPLATE,
            step_limit=tg_step_limit,
            cost_limit=tg_cost_limit,
        )

        _context_files = [
            k
            for k in (
                "profiling_path",
                "commandment_path",
                "baseline_metrics_path",
                "codebase_context_path",
                "previous_results_path",
                "round_evaluations_path",
            )
            if template_vars.get(k)
        ]
        logger.info(
            "Starting task-generation agent "
            "(step_limit=%d, cost=%.1f, context=%s) — this may take a few minutes",
            tg_step_limit,
            tg_cost_limit,
            ", ".join(k.replace("_path", "") for k in _context_files) or "minimal",
        )

        _t0 = time.monotonic()
        exit_type, exit_msg = agent.run(
            task="generate optimization tasks",
            **template_vars,
        )
        _elapsed = time.monotonic() - _t0

        if exit_type == "Submitted":
            logger.info(
                "Task-generation agent completed in %.1fs (%d chars).",
                _elapsed,
                len(exit_msg),
            )
            return exit_msg

        logger.warning("Task-generation agent did not submit (exit_type=%s).", exit_type)
        raise RuntimeError(f"Task-generation agent did not submit results (exit: {exit_type}): {exit_msg[:500]}")
    finally:
        if original_tools is not None:
            if hasattr(model, "set_tools"):
                model.set_tools(original_tools)
            else:
                _model_target.tools = original_tools
        # region agent log
        emit_debug_log(
            "task_generator.py:_run_task_agent:after_restore",
            "Finished task-planning tool restore",
            {
                "restored_tools": tool_names(original_tools),
                "model_target_type": type(_model_target).__name__,
                "model_target_id": id(_model_target),
                "model_after": model_tools_snapshot(model),
            },
            hypothesis_id="H1",
        )
        # endregion
        for f in tmp_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove temp file %s", f)


def _parse_llm_response(
    content: str,
    agent_class: type,
    *,
    kernel_path: str | None = None,
    commandment_path: str | None = None,
    baseline_metrics_path: str | None = None,
) -> list[AgentTask]:
    """Parse JSON response into AgentTask objects."""
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

    from minisweagent.agents.agent_spec import _agent_type_to_class, filter_agent_type

    type_to_class = _agent_type_to_class()

    tasks: list[AgentTask] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            logger.debug("_parse_llm_response: skipping non-dict item: %s", type(item).__name__)
            continue

        label = str(item.get("label", "unknown"))
        try:
            priority = int(item.get("priority", 10))
        except (ValueError, TypeError):
            logger.debug(
                "_parse_llm_response: invalid priority %r for '%s'; defaulting to 10.", item.get("priority"), label
            )
            priority = 10
        priority = max(0, min(15, priority))
        agent_type = filter_agent_type(str(item.get("agent_type", "strategy_agent")))
        kernel_language = str(item.get("kernel_language", "python"))
        task_prompt = str(item.get("task_prompt", ""))
        try:
            task_num_gpus = max(1, int(item.get("num_gpus", 1)))
        except (ValueError, TypeError):
            task_num_gpus = 1

        if not task_prompt:
            logger.debug("_parse_llm_response: skipping task '%s' with empty prompt.", label)
            continue

        resolved_class = type_to_class.get(agent_type, agent_class)
        if agent_type not in type_to_class:
            logger.debug("_parse_llm_response: unknown agent_type %r for '%s'; using default class.", agent_type, label)

        cfg: dict[str, Any] = {}

        tasks.append(
            AgentTask(
                agent_class=resolved_class,
                task=task_prompt,
                label=label,
                priority=priority,
                kernel_language=kernel_language,
                config=cfg,
                num_gpus=task_num_gpus,
            )
        )

    if not tasks:
        raise ValueError("LLM response contained no valid tasks")

    return sorted(tasks, key=lambda t: t.priority)


# ============================================================================
# CLI helpers
# ============================================================================


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
        "--codebase-context",
        default=None,
        metavar="FILE",
        help="Path to CODEBASE_CONTEXT.md (auto-detected from --from-discovery directory if not set)",
    )
    parser.add_argument(
        "--benchmark-baseline",
        default=None,
        metavar="FILE",
        help="Path to benchmark_baseline.txt (canonical benchmark output from preprocessing)",
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
        help="Number of available GPUs (guides task count and GPU allocation, default: 1)",
    )
    from minisweagent.run.pipeline_helpers import add_agent_filter_args, apply_agent_filter_env

    add_agent_filter_args(parser)

    args = parser.parse_args()
    apply_agent_filter_env(args)

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

    # Auto-detect codebase context from --from-discovery directory
    if not args.codebase_context and args.from_discovery:
        _ctx_sibling = Path(args.from_discovery).parent / "CODEBASE_CONTEXT.md"
        if _ctx_sibling.exists():
            args.codebase_context = str(_ctx_sibling)

    if not args.kernel_path:
        parser.error("--kernel-path is required (or provide --from-discovery)")

    kernel_path = Path(args.kernel_path).resolve()
    if not kernel_path.exists():
        print(f"ERROR: kernel path not found: {args.kernel_path}", file=sys.stderr)
        sys.exit(1)

    if not disc_json:
        # No pre-computed discovery JSON -- run automated-test-discovery
        print(f"[task-generator] Running discovery on {kernel_path}...", file=sys.stderr)
        try:
            from automated_test_discovery.server import discover as atd_discover

            _discover_fn = getattr(atd_discover, "fn", atd_discover)
            disc_json = _discover_fn(
                kernel_path=str(kernel_path),
                output_dir=str(kernel_path.parent),
            )
        except Exception as e:
            print(f"ERROR: discovery failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[task-generator] Loading discovery from {args.from_discovery}...", file=sys.stderr)

    kernel_meta = _extract_kernel_meta(disc_json, str(kernel_path))

    if not kernel_meta["kernel_path"] or kernel_meta["kernel_path"] == "unknown.py":
        print("ERROR: no kernel found in discovery", file=sys.stderr)
        sys.exit(1)

    # Create model (REQUIRED)
    try:
        from minisweagent.run.pipeline_helpers import load_geak_model

        model = load_geak_model(args.model or os.environ.get("GEAK_MODEL"))
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
    codebase_context_path = Path(args.codebase_context).resolve() if args.codebase_context else None

    # Generate tasks
    tasks = generate_tasks(
        base_task_context=base_task_context,
        agent_class=agent_class,
        model=model,
        kernel_path=kernel_meta["kernel_path"],
        kernel_name=kernel_meta["kernel_name"],
        kernel_type=kernel_meta["kernel_type"],
        kernel_language=kernel_meta["kernel_language"],
        function_names=kernel_meta["function_names"],
        workspace_path=kernel_meta["workspace_path"],
        profiling_path=profiling_path,
        commandment_path=commandment_path,
        baseline_metrics_path=baseline_metrics_path,
        deep_search_path=deep_search_path,
        previous_results_dir=previous_results_dir,
        discovery_path=discovery_path,
        codebase_context_path=codebase_context_path,
        num_gpus=args.num_gpus,
    )

    # Print summary to stderr
    print(f"\n[task-generator] Generated {len(tasks)} task(s):\n", file=sys.stderr)
    for t in tasks:
        print(f"  [{t.priority:2d}] {t.label} ({t.kernel_language})", file=sys.stderr)

    # Output: directory of task files or JSON to stdout
    if args.output:
        out_dir = Path(args.output)
        task_paths = write_task_files(
            tasks,
            out_dir,
            kernel_path=str(kernel_path),
            repo_root=args.repo_root or "",
            commandment=args.commandment or "",
            baseline_metrics=args.baseline_metrics or "",
            profiling=args.profiling or "",
            codebase_context=args.codebase_context or "",
            benchmark_baseline=args.benchmark_baseline or "",
            test_command=test_command or "",
            round_num=args.round,
        )

        manifest = [
            {
                "index": i,
                "label": tasks[i].label,
                "priority": tasks[i].priority,
                "kernel_language": tasks[i].kernel_language,
                "file": str(f),
            }
            for i, f in enumerate(task_paths)
        ]

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
