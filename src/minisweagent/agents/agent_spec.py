"""AgentSpec -- describes one sub-agent for heterogeneous parallel execution.

Used by ParallelAgent.run_parallel() to spawn different agent types with
different GPU allocations and configuration overrides.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSpec:
    """Specification for a single sub-agent in a heterogeneous parallel run.

    Attributes:
        agent_class: The agent class to instantiate (e.g. StrategyAgent, OpenEvolveWorker).
        gpu_ids: List of GPU device IDs assigned to this agent.
        config: Config overrides merged into the base agent_config.
        step_limit: Per-agent step limit (0 = inherit from parent).
        cost_limit: Per-agent cost limit (0.0 = inherit from parent).
        label: Human-readable label for logging (e.g. "algorithmic", "memory").
    """

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
    """Detect available AMD GPU device IDs via rocm-smi.

    Returns a list of integer device IDs, or [0] as fallback.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid", "--csv"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return [0]

        # Parse CSV output: header + rows with device IDs
        gpu_ids = []
        for line in result.stdout.strip().splitlines()[1:]:  # skip header
            parts = line.split(",")
            if parts:
                try:
                    gpu_ids.append(int(parts[0].strip()))
                except ValueError:
                    continue
        return gpu_ids if gpu_ids else [0]

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return [0]
