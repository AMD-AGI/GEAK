"""Dispatcher — selects N tasks from a CandidatePool based on mode.

The dispatcher is the **only** component that knows about mode semantics.
Its job is purely selection: from a pool of M candidates, pick N to run
on the available parallel workers.  In mixed mode, K (the number of
planned slots) is determined adaptively based on per-task outcomes from
prior rounds.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from minisweagent.run.dispatch_plan import DispatchPlan, DispatchPlanItem
from minisweagent.run.planner.candidate_pool import CandidatePool, CandidateTask

logger = logging.getLogger(__name__)

# Speedups at or above this are treated as timing-saturation / measurement
# garbage (e.g. a divide-by-near-zero "10000x" artifact) and are excluded from
# adaptive-K scoring so they cannot hijack slot allocation.
MAX_PLAUSIBLE_SPEEDUP = 100.0


class Dispatcher:
    """Select N tasks from a CandidatePool based on mode."""

    def select(
        self,
        pool: CandidatePool,
        mode: str,
        n: int,
        round_evals: list[dict[str, Any]] | None = None,
    ) -> DispatchPlan:
        """Pick exactly N items from *pool* according to *mode*.

        Returns a ``DispatchPlan`` with exactly N ``DispatchPlanItem``s.
        """
        if n < 1:
            n = 1

        k = self._k_for_mode(mode, n, round_evals)
        planned = sorted(pool.planned, key=lambda c: c.priority)
        fixed = pool.fixed
        registry = pool.registry

        picked_planned = planned[: min(k, len(planned))]
        fill_needed = n - len(picked_planned)
        picked_fill = self._fill(fixed, registry, fill_needed)

        items = [self._to_plan_item(c) for c in picked_planned + picked_fill]

        if len(items) != n:
            logger.warning(
                "Dispatcher.select: expected %d items but got %d; padding with fixed",
                n,
                len(items),
            )
            # Invariant (post-TaskPlanner refactor): when the planner has run,
            # ``pool.fixed`` always contains the canonical fixed entry. We rely
            # on it here so pad tasks dispatch the same body as fixed-mode would,
            # never an empty string. A violation means TaskPlanner.build_pool
            # stopped injecting canonical_fixed — fail loudly so we notice.
            assert fixed, (
                "selector pad branch reached with empty pool.fixed — "
                "TaskPlanner.build_pool must always inject canonical_fixed"
            )
            while len(items) < n:
                idx = len(items)
                base_fixed = fixed[0] if fixed else None
                items.append(
                    DispatchPlanItem(
                        label=f"fixed-pad-{idx}",
                        task=base_fixed.body if base_fixed else "",
                        kind="fixed",
                        agent_name=base_fixed.agent_name if base_fixed else "general-kernel-optimization",
                        priority=5,
                        kernel_language=base_fixed.kernel_language if base_fixed else "python",
                        num_gpus=base_fixed.num_gpus if base_fixed else 1,
                    )
                )
            items = items[:n]

        logger.info(
            "Dispatcher: mode=%s N=%d K=%d → %d planned + %d fill",
            mode,
            n,
            k,
            len(picked_planned),
            fill_needed,
        )
        return DispatchPlan(
            round_num=pool.round_num,
            mode=mode,
            items=tuple(items),
        )

    @staticmethod
    def _k_for_mode(
        mode: str, n: int, round_evals: list[dict[str, Any]] | None = None
    ) -> int:
        """Adaptive K allocation for mixed mode.

        Fixed/planned modes are unchanged. Mixed mode splits the N slots
        between the planned and fixed sources using a **max-seeking** signal:
        kernel optimization cares about the *best* result a source produced,
        not its mean — a single strong planned candidate should attract slots
        even if most planned attempts were mediocre, and mean-averaging
        otherwise collapses both sources to parity. Each source's peak speedup
        is:

          * **clamped** to the plausible open range ``(0, MAX_PLAUSIBLE_SPEEDUP)``
            so timing-saturation / garbage speedups can't hijack allocation, and
          * **weighted by the source's success rate** (plausible results /
            dispatched count) so failure-prone sources lose slots.

        K is allocated proportionally to the two scores and clamped to
        ``[1, n - 1]`` to keep an exploration floor of one slot per source.
        """
        if mode == "fixed":
            return 0
        if mode == "planned":
            return n
        if not round_evals or n <= 2:
            return max(1, n // 2)

        planned_speeds: list[float] = []
        planned_total = 0
        fixed_speeds: list[float] = []
        fixed_total = 0
        for rev in round_evals:
            for pt in rev.get("per_task", []):
                kind = pt.get("kind")
                if kind == "planned":
                    planned_total += 1
                elif kind == "fixed":
                    fixed_total += 1
                else:
                    continue
                spd = pt.get("speedup")
                # plausibility clamp: drop non-positive, None, and
                # saturation/garbage artifacts at or above MAX_PLAUSIBLE_SPEEDUP
                if spd is None or spd <= 0 or spd >= MAX_PLAUSIBLE_SPEEDUP:
                    continue
                (planned_speeds if kind == "planned" else fixed_speeds).append(spd)

        if not planned_speeds and not fixed_speeds:
            return max(1, n // 2)

        def _source_score(speeds: list[float], total: int) -> float:
            # max-seeking (optimization cares about the BEST result a source
            # produced), discounted by the source's success rate (failure
            # penalty). A source with no plausible result scores a neutral 1.0.
            if not speeds:
                return 1.0
            best = max(speeds)
            success_rate = len(speeds) / max(1, total)
            return best * success_rate

        planned_score = _source_score(planned_speeds, planned_total)
        fixed_score = _source_score(fixed_speeds, fixed_total)
        if planned_score + fixed_score <= 0:
            return max(1, n // 2)
        k = round(n * planned_score / (planned_score + fixed_score))
        return max(1, min(k, n - 1))

    @staticmethod
    def _fill(
        fixed: list[CandidateTask],
        registry: list[CandidateTask],
        n: int,
    ) -> list[CandidateTask]:
        """Fixed-first ordering, with replication when the pool is short.

        Order: existing fixed entries -> registry entries -> replicated copies
        of the canonical fixed body (labelled fixed-fill-0, fixed-fill-1, ...).
        """
        out: list[CandidateTask] = []
        for c in fixed:
            if len(out) >= n:
                break
            out.append(c)
        for c in registry:
            if len(out) >= n:
                break
            out.append(c)
        while len(out) < n and fixed:
            base = fixed[0]
            out.append(replace(base, label=f"{base.label}-fill-{len(out)}"))
        return out

    @staticmethod
    def _to_plan_item(c: CandidateTask) -> DispatchPlanItem:
        """Convert a ``CandidateTask`` to a ``DispatchPlanItem``."""
        return DispatchPlanItem(
            label=c.label,
            task=c.body,
            agent_name=c.agent_name,
            kind=c.kind,
            priority=c.priority,
            kernel_language=c.kernel_language,
            num_gpus=c.num_gpus,
        )
