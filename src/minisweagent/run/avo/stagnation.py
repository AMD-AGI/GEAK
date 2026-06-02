"""StagnationDetector — deterministic stall / cycle detection for AVO.

This is the "alarm clock" half of the two-layer supervisor (see
``docs/developer/avo_design.md`` §8). It contains **no LLM logic**: it tracks a
handful of counters across variation steps and returns the highest triggered
:class:`StagnationLevel`. The deterministic layer guarantees intervention even
if the LLM supervisor would itself stall.

Thresholds are loaded from ``avo.stagnation`` in the merged config.
"""

from __future__ import annotations

import enum
import logging
from collections import deque
from dataclasses import dataclass, field

from minisweagent.run.avo.result import VariationResult

logger = logging.getLogger(__name__)


class StagnationLevel(enum.IntEnum):
    """Escalating intervention levels. The controller acts on ``>= REDIRECT``."""

    NONE = 0  # continue current direction
    NUDGE = 1  # inject a reminder into the next step's prompt
    INTERRUPT = 2  # force-end the current step early (handled by step_limit)
    REDIRECT = 3  # call avo-supervisor, switch direction, reset worktree
    ESCALATE = 4  # run one GEAK parallel rescue round (mode="planned")


@dataclass(frozen=True)
class StagnationSignal:
    level: StagnationLevel
    reason: str
    counters: dict


# Default thresholds; mirror ``avo.stagnation`` in geak_avo.yaml.
_DEFAULTS: dict = {
    "steps_without_commit": 80,
    "wall_time_without_commit_s": 2700,
    "consecutive_correctness_failures": 5,
    "consecutive_no_improvement": 8,
    "patch_hash_repeat": 3,
    "supervisor_cycles_without_commit": 3,
}


@dataclass
class StagnationDetector:
    """Tracks counters across steps and emits the highest triggered level."""

    config: dict = field(default_factory=dict)

    # ── live counters ──
    steps_without_commit: int = 0
    wall_time_without_commit_s: float = 0.0
    consecutive_correctness_failures: int = 0
    consecutive_no_improvement: int = 0
    supervisor_cycles_without_commit: int = 0
    _recent_hashes: deque = field(default_factory=lambda: deque(maxlen=16))

    def __post_init__(self) -> None:
        merged = dict(_DEFAULTS)
        merged.update(self.config or {})
        self.config = merged

    # ------------------------------------------------------------------

    def _threshold(self, key: str) -> float:
        return float(self.config.get(key, _DEFAULTS[key]))

    def evaluate(self, result: VariationResult, committed: bool) -> StagnationSignal:
        """Update counters from one step's result and return the signal.

        ``committed`` is whether :meth:`LineageStore.maybe_commit` accepted a
        new version this step (the single source of "progress").
        """
        if committed:
            self._reset_progress_counters()
            return StagnationSignal(StagnationLevel.NONE, "committed new best", self._snapshot())

        # No commit this step → advance stall counters.
        self.steps_without_commit += 1
        self.wall_time_without_commit_s += float(result.wall_time_s or 0.0)

        any_correct = any(a.correctness_passed for a in result.attempts)
        if result.attempts and not any_correct:
            self.consecutive_correctness_failures += 1
        else:
            self.consecutive_correctness_failures = 0

        improved = result.best_speedup is not None and result.best_speedup > 1.001
        if not improved:
            self.consecutive_no_improvement += 1
        else:
            self.consecutive_no_improvement = 0

        repeat_hits = self._update_patch_cycle(result)

        return self._classify(repeat_hits)

    def _classify(self, repeat_hits: int) -> StagnationSignal:
        """Return the highest triggered level given current counters."""
        # ESCALATE first — supervisor already tried repeatedly with no commit.
        if self.supervisor_cycles_without_commit >= self._threshold("supervisor_cycles_without_commit"):
            return StagnationSignal(
                StagnationLevel.ESCALATE,
                f"supervisor intervened {self.supervisor_cycles_without_commit}x without a commit",
                self._snapshot(),
            )

        redirect_reasons = []
        if self.steps_without_commit >= self._threshold("steps_without_commit"):
            redirect_reasons.append(f"{self.steps_without_commit} steps without commit")
        if self.wall_time_without_commit_s >= self._threshold("wall_time_without_commit_s"):
            redirect_reasons.append(f"{self.wall_time_without_commit_s:.0f}s without commit")
        if self.consecutive_correctness_failures >= self._threshold("consecutive_correctness_failures"):
            redirect_reasons.append(f"{self.consecutive_correctness_failures} consecutive correctness failures")
        if self.consecutive_no_improvement >= self._threshold("consecutive_no_improvement"):
            redirect_reasons.append(f"{self.consecutive_no_improvement} consecutive no-improvement steps")
        if repeat_hits >= self._threshold("patch_hash_repeat"):
            redirect_reasons.append(f"identical patch seen {repeat_hits}x (cycle)")

        if redirect_reasons:
            return StagnationSignal(StagnationLevel.REDIRECT, "; ".join(redirect_reasons), self._snapshot())

        # Soft NUDGE once we are halfway to the redirect thresholds.
        if (
            self.consecutive_no_improvement >= max(1, self._threshold("consecutive_no_improvement") // 2)
            or self.consecutive_correctness_failures >= max(1, self._threshold("consecutive_correctness_failures") // 2)
        ):
            return StagnationSignal(StagnationLevel.NUDGE, "approaching stall thresholds", self._snapshot())

        return StagnationSignal(StagnationLevel.NONE, "progressing", self._snapshot())

    # ------------------------------------------------------------------

    def _update_patch_cycle(self, result: VariationResult) -> int:
        """Track repeated identical patches; return max repeat count this window."""
        hits = 0
        for attempt in result.attempts:
            if not attempt.patch_hash:
                continue
            self._recent_hashes.append(attempt.patch_hash)
            count = sum(1 for h in self._recent_hashes if h == attempt.patch_hash)
            hits = max(hits, count)
        return hits

    def note_supervisor_cycle(self) -> None:
        """Record that the LLM supervisor was invoked (called by the controller)."""
        self.supervisor_cycles_without_commit += 1

    def reset(self, *, partial: bool = False) -> None:
        """Reset counters after an intervention.

        ``partial=True`` keeps ``supervisor_cycles_without_commit`` (so repeated
        supervisor calls still eventually ESCALATE) but clears the per-direction
        stall counters that a redirect is expected to fix.
        """
        self.steps_without_commit = 0
        self.wall_time_without_commit_s = 0.0
        self.consecutive_correctness_failures = 0
        self.consecutive_no_improvement = 0
        self._recent_hashes.clear()
        if not partial:
            self.supervisor_cycles_without_commit = 0

    def _reset_progress_counters(self) -> None:
        """A commit means real progress — clear everything, including supervisor cycles."""
        self.reset(partial=False)

    def _snapshot(self) -> dict:
        return {
            "steps_without_commit": self.steps_without_commit,
            "wall_time_without_commit_s": round(self.wall_time_without_commit_s, 1),
            "consecutive_correctness_failures": self.consecutive_correctness_failures,
            "consecutive_no_improvement": self.consecutive_no_improvement,
            "supervisor_cycles_without_commit": self.supervisor_cycles_without_commit,
        }
