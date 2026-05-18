"""Typed objects for the task planner's output.

A ``CandidatePool`` is the planner's output — a flat list of ``M`` candidate
optimization tasks whose size is independent of the number of parallel workers
``N``.  The dispatcher selects which ``N`` of these ``M`` candidates to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class CandidateTask:
    """One LLM-emitted (or Python-emitted) optimization candidate."""

    label: str
    body: str
    kind: Literal["planned", "fixed", "registry"]
    agent_name: str = ""
    priority: int = 10
    kernel_language: str = "python"
    num_gpus: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "body": self.body,
            "kind": self.kind,
            "agent_name": self.agent_name,
            "priority": self.priority,
            "kernel_language": self.kernel_language,
            "num_gpus": self.num_gpus,
        }


@dataclass(frozen=True)
class CandidatePool:
    """The output of ``TaskPlanner.build_pool`` — size independent of N."""

    round_num: int
    items: tuple[CandidateTask, ...]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def planned(self) -> list[CandidateTask]:
        return [c for c in self.items if c.kind == "planned"]

    @property
    def fixed(self) -> list[CandidateTask]:
        return [c for c in self.items if c.kind == "fixed"]

    @property
    def registry(self) -> list[CandidateTask]:
        return [c for c in self.items if c.kind == "registry"]
