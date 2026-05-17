"""Typed optimization-round dispatch plan.

The planner ultimately decides a set of workers to run in a round. Historically
that decision was represented as an untyped ``list[AgentTask]``. This module
adds a small typed wrapper so future orchestration can reason about the plan
before execution, including which registry subagent should handle each task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from minisweagent.agents.agent_spec import AgentTask


@dataclass(frozen=True)
class DispatchPlanItem:
    """One worker entry in a round-level dispatch plan."""

    label: str
    task: str
    agent_type: str = "strategy_agent"
    agent_name: str = ""
    kind: str = "planned"
    priority: int = 10
    kernel_language: str = "python"
    num_gpus: int = 1
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_agent_task(cls, task: AgentTask) -> "DispatchPlanItem":
        return cls(
            label=task.label,
            task=task.task,
            agent_name=str(task.config.get("agent_name", "") or ""),
            kind=str(task.config.get("kind", "planned") or "planned"),
            priority=task.priority,
            kernel_language=task.kernel_language,
            num_gpus=task.num_gpus,
            config=dict(task.config),
        )

    def to_agent_task(self, agent_class: type) -> AgentTask:
        config = dict(self.config)
        if self.agent_name:
            config["agent_name"] = self.agent_name
        config["kind"] = self.kind
        return AgentTask(
            agent_class=agent_class,
            task=self.task,
            label=self.label,
            priority=self.priority,
            kernel_language=self.kernel_language,
            config=config,
            num_gpus=self.num_gpus,
        )


@dataclass(frozen=True)
class DispatchPlan:
    """A planner decision for one optimization round."""

    round_num: int
    mode: str
    items: tuple[DispatchPlanItem, ...]

    @classmethod
    def from_agent_tasks(
        cls,
        *,
        round_num: int,
        mode: str,
        tasks: list[AgentTask],
    ) -> "DispatchPlan":
        return cls(
            round_num=round_num,
            mode=mode,
            items=tuple(DispatchPlanItem.from_agent_task(t) for t in tasks),
        )

    def to_agent_tasks(self, agent_class: type) -> list[AgentTask]:
        return [item.to_agent_task(agent_class) for item in self.items]

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_num,
            "mode": self.mode,
            "tasks": [
                {
                    "label": item.label,
                    "priority": item.priority,
                    "agent_type": item.agent_type,
                    "agent_name": item.agent_name,
                    "kind": item.kind,
                    "kernel_language": item.kernel_language,
                    "num_gpus": item.num_gpus,
                    "task_prompt": item.task,
                }
                for item in self.items
            ],
        }


__all__ = ["DispatchPlan", "DispatchPlanItem"]
