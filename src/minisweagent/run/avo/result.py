"""Shared result dataclasses for an AVO variation step.

These are deliberately plain dataclasses (no GEAK imports) so that
``lineage_store`` and ``stagnation`` stay unit-testable without a GPU, a
model, or any heavy GEAK dependency.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AttemptRecord:
    """One ``save_and_test`` attempt within a variation step.

    ``verified_speedup`` is the independently parsed benchmark speedup (the same
    value GEAK's ``post_round_evaluate`` trusts). ``None`` means the attempt did
    not produce a parseable verified benchmark — it must never be promoted into
    the committed lineage.
    """

    strategy: str | None = None
    returncode: int = 1
    correctness_passed: bool = False
    verified_speedup: float | None = None
    patch_hash: str | None = None
    ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "returncode": self.returncode,
            "correctness_passed": self.correctness_passed,
            "verified_speedup": self.verified_speedup,
            "patch_hash": self.patch_hash,
            "ts": self.ts,
        }


@dataclass
class VariationResult:
    """Structured outcome of one ``OptimizationAgent`` run (one variation step)."""

    step_index: int
    step_dir: Path
    strategy: str | None = None
    exit_status: str = ""  # "Submitted" | "LimitsExceeded" | exception class name
    attempts: list[AttemptRecord] = field(default_factory=list)
    best_patch_path: Path | None = None
    best_speedup: float | None = None  # verified; None if nothing verified
    best_correct: bool = False
    wall_time_s: float = 0.0
    # Per-shape verified speedups (B2 regression guard). Empty when single-shape
    # or unavailable. Maps shape label -> speedup (candidate/baseline).
    per_shape_speedups: dict[str, float] = field(default_factory=dict)
    # Continuous-memory signal (P-mem-3 / option C): the agent's own rationale
    # for this step, a short verbatim tail of its reasoning + tool output, and a
    # small dict of profiler metrics — captured per step and replayed (bounded)
    # into later steps as a causal evolution log.
    rationale: str = ""
    raw_tail: str = ""
    profiling: dict[str, float] = field(default_factory=dict)

    @property
    def produced_verified_improvement_candidate(self) -> bool:
        """True if this step yielded a correct, verified-benchmark patch."""
        return self.best_correct and self.best_speedup is not None and self.best_patch_path is not None

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "step_dir": str(self.step_dir),
            "strategy": self.strategy,
            "exit_status": self.exit_status,
            "attempts": [a.to_dict() for a in self.attempts],
            "best_patch_path": str(self.best_patch_path) if self.best_patch_path else None,
            "best_speedup": self.best_speedup,
            "best_correct": self.best_correct,
            "wall_time_s": self.wall_time_s,
        }


def patch_hash(patch_text: str) -> str:
    """Stable short hash of a patch body, used for cycle detection."""
    return hashlib.sha1(patch_text.encode("utf-8", errors="ignore")).hexdigest()[:12]
