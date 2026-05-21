"""Task planner — produces a ``CandidatePool`` for each optimization round.

The planner wraps the existing ``task_generator.generate_tasks`` LLM call
and augments its output with a canonical ``kind="fixed"`` entry so the
dispatcher always has something to fill non-planned slots with.

In pure ``fixed`` mode, the LLM call is skipped entirely and the pool
contains only the canonical fixed entry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from minisweagent.run.planner.candidate_pool import CandidatePool, CandidateTask

logger = logging.getLogger(__name__)


class TaskPlanner:
    """Produces a ``CandidatePool`` of M tasks each round.

    M is independent of the number of parallel workers N — the dispatcher
    does the selection.
    """

    def __init__(
        self,
        *,
        model: Any,
        subagent_registry: Any | None = None,
        preprocess_ctx: dict[str, Any],
        kernel_meta: dict[str, Any],
    ) -> None:
        self._model = model
        self._subagent_registry = subagent_registry
        self._preprocess_ctx = preprocess_ctx
        self._kernel_meta = kernel_meta

    def build_pool(
        self,
        *,
        round_num: int,
        user_prompt: str,
        round_evals: list[dict[str, Any]],
        mode: str,
        agent_class: type,
        output_dir: Path | None = None,
        num_gpus: int = 1,
        rag_enabled: bool = False,
    ) -> CandidatePool:
        """Produce a ``CandidatePool`` of M tasks for the current round.

        - ``mode="fixed"``: skip LLM, return a single ``kind="fixed"`` entry
        - ``mode="planned"`` or ``"mixed"``: call the LLM planner and include
          a canonical fixed entry alongside the planned ones
        """
        from minisweagent.run.compose import ComposeInputs, compose_task_body

        kernel_language = str(self._kernel_meta.get("kernel_language") or "python")
        composed_body = compose_task_body(ComposeInputs(
            user_prompt=user_prompt,
            mode="fixed",
            preprocess_ctx=self._preprocess_ctx,
            kernel_language=kernel_language,
        ))
        canonical_fixed = CandidateTask(
            label="fixed-canonical",
            body=composed_body,
            kind="fixed",
            agent_name="general-kernel-optimization",
            priority=5,
            kernel_language=kernel_language,
        )

        if mode == "fixed" or num_gpus <= 1:
            logger.info("TaskPlanner: %s — skipping LLM planner, single canonical entry", "fixed mode" if mode == "fixed" else f"single GPU (num_gpus={num_gpus})")
            return CandidatePool(round_num=round_num, items=(canonical_fixed,))

        planned_tasks = self._call_llm_planner(
            round_num=round_num,
            user_prompt=user_prompt,
            round_evals=round_evals,
            agent_class=agent_class,
            output_dir=output_dir,
            num_gpus=num_gpus,
            rag_enabled=rag_enabled,
        )

        candidates: list[CandidateTask] = []
        for task in planned_tasks:
            candidates.append(
                CandidateTask(
                    label=task.label,
                    body=task.task,
                    kind="planned",
                    agent_name=task.config.get("agent_name", ""),
                    priority=task.priority,
                    kernel_language=task.kernel_language,
                    num_gpus=task.num_gpus,
                )
            )

        # Only inject fixed-canonical when planned tasks don't fill all GPU slots
        planned_gpu_total = sum(c.num_gpus for c in candidates)
        if planned_gpu_total < num_gpus:
            candidates.append(canonical_fixed)
            logger.info(
                "TaskPlanner: round %d pool has %d candidates (%d planned, 1 fixed fallback; planned %d/%d GPUs)",
                round_num,
                len(candidates),
                len(planned_tasks),
                planned_gpu_total,
                num_gpus,
            )
        else:
            logger.info(
                "TaskPlanner: round %d pool has %d candidates (%d planned, no fixed needed; planned %d/%d GPUs)",
                round_num,
                len(candidates),
                len(planned_tasks),
                planned_gpu_total,
                num_gpus,
            )
        return CandidatePool(round_num=round_num, items=tuple(candidates))

    def _call_llm_planner(
        self,
        *,
        round_num: int,
        user_prompt: str,
        round_evals: list[dict[str, Any]],
        agent_class: type,
        output_dir: Path | None = None,
        num_gpus: int = 1,
        rag_enabled: bool = False,
    ) -> list[Any]:
        """Delegate to the existing ``task_generator.generate_tasks``."""
        from minisweagent.agents.heterogeneous.task_generator import generate_tasks

        km = self._kernel_meta
        pp = self._preprocess_ctx
        preprocess_dir = Path(pp.get("preprocess_dir") or ".")

        return generate_tasks(
            base_task_context=user_prompt,
            agent_class=agent_class,
            model=self._model,
            kernel_path=str(km.get("kernel_path") or pp.get("kernel_path") or ""),
            kernel_name=str(km.get("kernel_name") or ""),
            kernel_type=str(km.get("kernel_type") or ""),
            kernel_language=str(km.get("kernel_language") or "python"),
            function_names=km.get("function_names") or [],
            workspace_path=str(km.get("workspace_path") or pp.get("repo_root") or ""),
            profiling_path=preprocess_dir / "profile.json" if preprocess_dir else None,
            commandment_path=preprocess_dir / "COMMANDMENT.md" if preprocess_dir else None,
            baseline_metrics_path=preprocess_dir / "baseline_metrics.json" if preprocess_dir else None,
            previous_results_dir=Path(output_dir) / "results" if output_dir else None,
            discovery_path=preprocess_dir / "discovery.json" if preprocess_dir else None,
            codebase_context_path=preprocess_dir / "CODEBASE_CONTEXT.md" if preprocess_dir else None,
            previous_tasks_dir=Path(output_dir) / "tasks" if output_dir else None,
            round_evaluations=round_evals,
            current_round=round_num,
            num_gpus=num_gpus,
            output_dir=Path(output_dir) / "tasks" / f"round_{round_num}" if output_dir else None,
            rag_enabled=rag_enabled,
            subagent_registry=self._subagent_registry,
        )
