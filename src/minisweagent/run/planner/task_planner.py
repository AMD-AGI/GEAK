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

# Dedicated, source-agnostic kernel-AUTHORING rewrite subagents.
_REWRITE_FLYDSL = "flydsl-kernel-rewrite"
_REWRITE_TILELANG = "tilelang-kernel-rewrite"
_REWRITE_ASM = "asm-kernel-rewrite"  # highest-perf tier (the ceiling under the DSLs)
_REWRITE_CK = "ck-kernel-rewrite"    # CK 2-stage / ck_tile codegen (shipped ck_moe / batched_gemm_*_CK)

# Adaptive op-type → best-fit rewrite backend(s). Picks the backend(s) with the real edge
# on gfx942 for that op class, so we don't waste a slot on a rewrite that can't win:
#   attention / FlashAttention / MLA  -> TileLang (FA ~1.5x Triton, MLA ~parity w/ asm)
#   MoE / grouped-expert GEMM         -> FlyDSL + ASM (fused-MoE; asm recovers last 10-20%)
#   linear-attn / gated-delta decode  -> FlyDSL (aiter flydsl_gdr_decode)
#   norm / rmsnorm / elementwise      -> FlyDSL + TileLang + ASM (DSLs were regressing vs baseline
#                                        norm; add asm as a 3rd parallel attempt so best-of-N has a
#                                        hand-tuned fallback when both DSLs lose)
#   plain GEMM                        -> FlyDSL + ASM (DSL edge small on blockscale; asm wins ceiling)
# ASM is added to compute-heavy classes (GEMM/MoE) where it has historically beaten the DSLs, and now
# also to norm/elementwise because the DSL-only pool produced sub-1.0x (regressing) rewrites there —
# the asm tier gives best-of-N a hand-scheduled candidate. Still NOT stacked on attention
# (TileLang ≈ asm there) — keeping that pool focused.
#   plain GEMM / MoE                 -> + CK (shipped ck_moe_stage1/2, batched_gemm_*_CK, instance tuning)
# CK is added to the compute-heavy GEMM/MoE classes only (its codegen 2-stage instances target exactly
# those); best-of-N + the parity gate discard a losing CK attempt, so adding the lane cannot regress
# results. CK is not added to attention/norm (no shipped CK edge there vs TileLang/asm).
_OPTYPE_REWRITE_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("attention", "attn", "flash", "mla", "sdpa", "fmha"), (_REWRITE_TILELANG,)),
    (("moe", "expert", "fused_moe", "grouped"), (_REWRITE_FLYDSL, _REWRITE_ASM, _REWRITE_CK, _REWRITE_TILELANG)),
    (("gated_delta", "linear_attn", "recurrent", "gdn", "gdr"), (_REWRITE_FLYDSL,)),
    (("gemm", "matmul", "mm_", "_mm", "linear"), (_REWRITE_FLYDSL, _REWRITE_ASM, _REWRITE_CK)),
    (("norm", "rmsnorm", "layernorm", "elementwise", "quant"), (_REWRITE_FLYDSL, _REWRITE_TILELANG, _REWRITE_ASM)),
)


def _select_rewrite_subagents(kernel_name: str, function_names: list[str] | None) -> tuple[str, ...]:
    """Pick the op-appropriate authoring rewrite subagent(s) by name signal.

    Matches the kernel name + function names against op-class keywords and returns the
    backend(s) with the real edge for that op on gfx942. ASM (the highest-perf tier) is added
    only for compute-heavy GEMM/MoE classes. Defaults to both DSLs for norm/unknown ops where
    the edge is ambiguous and hand-asm isn't worth it — best-of-N decides.
    """
    hay = " ".join([str(kernel_name or "")] + [str(f) for f in (function_names or [])]).lower()
    for needles, agents in _OPTYPE_REWRITE_RULES:
        if any(n in hay for n in needles):
            return agents
    return (_REWRITE_FLYDSL, _REWRITE_TILELANG)


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
        canonical_fixed = CandidateTask(
            label="fixed-canonical",
            body=composed_body,
            kind="fixed",
            agent_name="general-kernel-optimization",
            priority=5,
            kernel_language=kernel_language,
            num_gpus=num_gpus,
        )

        # Adaptive authoring-rewrite candidates: pick the DSL(s) that actually win for
        # this kernel's op-type (attention->TileLang, MoE/linear-attn->FlyDSL, GEMM/norm->
        # both). Added as extra fixed candidates so the round tries the right rewrite
        # alongside the in-place optimizer + planned strategies; best-of-N selects the winner.
        rewrite_candidates: list[CandidateTask] = []
        registered = set(self._subagent_registry.list_names()) if self._subagent_registry else set()
        _chosen = _select_rewrite_subagents(
            str(self._kernel_meta.get("kernel_name") or ""),
            self._kernel_meta.get("function_names"),
        )
        for sa in _chosen:
            if sa in registered:
                rewrite_candidates.append(
                    CandidateTask(
                        label=sa,
                        body=composed_body,
                        kind="fixed",
                        agent_name=sa,
                        priority=5,
                        kernel_language=kernel_language,
                        num_gpus=num_gpus,
                    )
                )
        if rewrite_candidates:
            logger.info(
                "TaskPlanner: added %d op-adaptive authoring-rewrite candidate(s): %s",
                len(rewrite_candidates),
                [c.agent_name for c in rewrite_candidates],
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

        # Add the FlyDSL + TileLang authoring rewrite candidates so they compete every round.
        candidates.extend(rewrite_candidates)

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
