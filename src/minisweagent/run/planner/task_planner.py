"""Task planner — produces a ``CandidatePool`` for each optimization round.

The planner wraps the existing ``task_generator.generate_tasks`` LLM call
and augments its output with a canonical ``kind="fixed"`` entry so the
dispatcher always has something to fill non-planned slots with.

In pure ``fixed`` mode, the LLM call is skipped entirely and the pool
contains only the canonical fixed entry.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from minisweagent.run.planner.candidate_pool import CandidatePool, CandidateTask

logger = logging.getLogger(__name__)

# Optimized backend-rewrite targets (FlyDSL/TileLang are always >= CK/Triton/HIP
# on gfx942). When ``GEAK_FIXED_REWRITE_TARGET`` is set to one of these, the
# canonical fixed-mode task is routed through the matching ``<source>-to-<target>``
# rewrite subagent instead of the default in-place optimizer — so in mixed mode the
# fixed half attempts a backend rewrite while the planned half does in-place tuning.
_REWRITE_TARGETS = ("flydsl", "tilelang")
# Map the detected kernel_language to the rewrite subagent's source token.
_LANG_TO_REWRITE_SOURCE = {
    "triton": "triton",
    "hip": "hip",
    "hip_cpp": "hip",
    "cuda": "hip",
    "ck": "ck",
    "python": "pytorch",
    "pytorch": "pytorch",
    "tilelang": "tilelang",
    "flydsl": "flydsl",
}


def _fixed_rewrite_agent_name(kernel_language: str, kernel_type: str = "") -> str | None:
    """Return the rewrite subagent name for the fixed slot, or None for default.

    Controlled by ``GEAK_FIXED_REWRITE_TARGET`` (``flydsl``|``tilelang``). Returns
    ``None`` when unset, the target is invalid, or the source already equals the
    target (no-op rewrite) — in which case the caller keeps the default in-place
    ``general-kernel-optimization`` agent.

    Source selection prefers ``kernel_type`` (the precise triton/hip/ck/flydsl
    classifier from ``_infer_kernel_type``) over ``kernel_language`` (which coarsely
    reports ``python`` for Triton ``.py`` files) so a Triton kernel routes to
    ``triton-to-<target>`` rather than ``pytorch-to-<target>``.
    """
    target = (os.environ.get("GEAK_FIXED_REWRITE_TARGET") or "").strip().lower()
    if target not in _REWRITE_TARGETS:
        return None
    source = (
        _LANG_TO_REWRITE_SOURCE.get(str(kernel_type or "").strip().lower())
        or _LANG_TO_REWRITE_SOURCE.get(str(kernel_language or "").strip().lower())
    )
    if not source or source == target:
        return None
    return f"{source}-to-{target}"


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
        num_parallel: int = 1,
        rag_enabled: bool = False,
    ) -> CandidatePool:
        """Produce a ``CandidatePool`` for the current round.

        ``num_parallel`` is the number of subagent slots to plan for —
        which may exceed the physical GPU count under the gwiab-scheduler.
        It is NOT ``len(gpu_ids)``.

        - ``mode="fixed"``: skip LLM, return a single ``kind="fixed"`` entry.
        - ``mode="planned"`` or ``"mixed"``: call the LLM planner and ALWAYS
          include the canonical fixed entry alongside the planned ones, so
          downstream fill/pad has a guaranteed source for the canonical body.
        """
        from minisweagent.run.compose import ComposeInputs, compose_task_body

        kernel_language = str(self._kernel_meta.get("kernel_language") or "python")
        composed_body = compose_task_body(
            ComposeInputs(
                user_prompt=user_prompt,
                mode="fixed",
                preprocess_ctx=self._preprocess_ctx,
                kernel_language=kernel_language,
            )
        )
        # Default fixed-mode agent is the in-place optimizer. When the operator
        # opts into backend rewrites (GEAK_FIXED_REWRITE_TARGET=flydsl|tilelang),
        # route the fixed slot through the matching rewrite subagent instead.
        _kernel_type = str(self._kernel_meta.get("kernel_type") or "")
        _fixed_agent = _fixed_rewrite_agent_name(kernel_language, _kernel_type) or "general-kernel-optimization"
        if _fixed_agent != "general-kernel-optimization":
            logger.info(
                "TaskPlanner: fixed slot routed to rewrite subagent %r "
                "(GEAK_FIXED_REWRITE_TARGET) for kernel_language=%s",
                _fixed_agent,
                kernel_language,
            )
        canonical_fixed = CandidateTask(
            label="fixed-canonical",
            body=composed_body,
            kind="fixed",
            agent_name=_fixed_agent,
            priority=5,
            kernel_language=kernel_language,
            num_gpus=num_gpus,
        )

        if mode == "fixed" or num_parallel <= 1:
            logger.info(
                "TaskPlanner: %s — skipping LLM planner, single canonical entry",
                "fixed mode" if mode == "fixed" else f"single worker (num_parallel={num_parallel})",
            )
            return CandidatePool(round_num=round_num, items=(canonical_fixed,))

        planned_tasks = self._call_llm_planner(
            round_num=round_num,
            user_prompt=user_prompt,
            round_evals=round_evals,
            agent_class=agent_class,
            output_dir=output_dir,
            num_slots=num_parallel,
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

        # Always inject canonical_fixed so pool.fixed is never empty.
        # The dispatcher's fill/pad paths rely on this body to top up
        # subagent slots the planner did not produce a task for.
        candidates.append(canonical_fixed)
        planned_slot_total = sum(c.num_gpus for c in candidates if c.kind == "planned")
        logger.info(
            "TaskPlanner: round %d pool has %d candidates (%d planned + 1 canonical fixed; planned slots %d/%d)",
            round_num,
            len(candidates),
            len(planned_tasks),
            planned_slot_total,
            num_parallel,
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
        num_slots: int = 1,
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
            num_gpus=num_slots,
            output_dir=Path(output_dir) / "tasks" / f"round_{round_num}" if output_dir else None,
            rag_enabled=rag_enabled,
            subagent_registry=self._subagent_registry,
        )
