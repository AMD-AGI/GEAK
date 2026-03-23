"""AgentSpec and AgentTask -- describe sub-agents for parallel execution.

AgentSpec: Legacy fixed-GPU-assignment model (one spec per GPU).
AgentTask: Decoupled model -- tasks are independent of GPU assignment.
           The GPU pool scheduler assigns GPUs dynamically at runtime.

Used by ParallelAgent.run_parallel() to spawn agents.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _agent_type_to_class() -> dict[str, type]:
    """Canonical mapping from task-file ``agent_type`` string to class.

    Lazy import to avoid circular dependencies at module level.
    """
    return {}


def _agent_class_to_type() -> dict[type, str]:
    """Reverse mapping: agent class -> agent_type string."""
    return {cls: name for name, cls in _agent_type_to_class().items()}


ALL_AGENT_TYPES: frozenset[str] = frozenset({"strategy_agent"})

_DEFAULT_FALLBACK_AGENT = "strategy_agent"


def get_allowed_agent_types() -> set[str] | None:
    """Return the effective set of allowed agent types, or *None* if unrestricted."""
    allowed_raw = os.environ.get("GEAK_ALLOWED_AGENTS", "").strip()
    excluded_raw = os.environ.get("GEAK_EXCLUDED_AGENTS", "").strip()

    if not allowed_raw and not excluded_raw:
        return None

    if allowed_raw:
        if excluded_raw:
            logger.warning(
                "Both GEAK_ALLOWED_AGENTS and GEAK_EXCLUDED_AGENTS are set; "
                "GEAK_ALLOWED_AGENTS takes precedence."
            )
        allowed = {t.strip() for t in allowed_raw.split(",") if t.strip()}
        return allowed & ALL_AGENT_TYPES

    excluded = {t.strip() for t in excluded_raw.split(",") if t.strip()}
    return ALL_AGENT_TYPES - excluded


def filter_agent_type(agent_type: str) -> str:
    """Remap *agent_type* to the fallback if it is not allowed."""
    allowed = get_allowed_agent_types()
    if allowed is None:
        return agent_type

    if agent_type in allowed:
        return agent_type

    fallback = os.environ.get("GEAK_FALLBACK_AGENT", "").strip() or _DEFAULT_FALLBACK_AGENT
    if fallback not in allowed:
        fallback = next(iter(sorted(allowed)), _DEFAULT_FALLBACK_AGENT)

    logger.warning(
        "Agent type %r is not allowed (allowed=%s); remapping to %r",
        agent_type,
        sorted(allowed),
        fallback,
    )
    return fallback


@dataclass
class AgentTask:
    """A single optimization task, independent of GPU assignment."""

    agent_class: type
    task: str = ""
    label: str = ""
    priority: int = 10
    kernel_language: str = "python"
    config: dict[str, Any] = field(default_factory=dict)
    step_limit: int = 0
    cost_limit: float = 0.0
    num_gpus: int = 1


@dataclass
class AgentSpec:
    """Specification for a single sub-agent in a heterogeneous parallel run."""

    agent_class: type
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    config: dict[str, Any] = field(default_factory=dict)
    step_limit: int = 0
    cost_limit: float = 0.0
    label: str = ""

    @property
    def hip_visible_devices(self) -> str:
        """HIP_VISIBLE_DEVICES value for this agent."""
        return ",".join(str(g) for g in self.gpu_ids)

    @property
    def num_gpus(self) -> int:
        return len(self.gpu_ids)


def detect_available_gpus() -> list[int]:
    """Detect available AMD GPU device IDs via rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid", "--csv"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return [0]

        gpu_ids = []
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.split(",")
            if parts:
                try:
                    gpu_ids.append(int(parts[0].strip()))
                except ValueError:
                    continue
        return gpu_ids if gpu_ids else [0]

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return [0]
