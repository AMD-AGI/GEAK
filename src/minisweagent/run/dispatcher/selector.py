"""Dispatcher — selects N tasks from a CandidatePool based on mode.

The dispatcher is the **only** component that knows about mode semantics.
Its job is purely selection: from a pool of M candidates, pick N to run
on the available parallel workers.  The selection rule is determined by
``_k_for_mode``, the single seam that the adaptive future replaces.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from minisweagent.run.dispatch_plan import DispatchPlan, DispatchPlanItem
from minisweagent.run.planner.candidate_pool import CandidatePool, CandidateTask

logger = logging.getLogger(__name__)


class Dispatcher:
    """Select N tasks from a CandidatePool based on mode."""

    def select(
        self,
        pool: CandidatePool,
        mode: str,
        n: int,
    ) -> DispatchPlan:
        """Pick exactly N items from *pool* according to *mode*.

        Returns a ``DispatchPlan`` with exactly N ``DispatchPlanItem``s.
        """
        if n < 1:
            n = 1

        k = self._k_for_mode(mode, n)
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
            while len(items) < n:
                idx = len(items)
                items.append(
                    DispatchPlanItem(
                        label=f"fixed-pad-{idx}",
                        task=fixed[0].body if fixed else "",
                        kind="fixed",
                        priority=5,
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
    def _k_for_mode(mode: str, n: int) -> int:
        """THE ONLY PLACE the mode-to-K mapping lives.

        Adaptive K replaces this function; signature stays the same.

        Returns the number of slots to fill with planned candidates.
        """
        return {"fixed": 0, "planned": n, "mixed": n // 2}.get(mode, n // 2)

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
