"""AgentSpec and AgentTask -- describe sub-agents for parallel execution.

AgentSpec: Legacy fixed-GPU-assignment model (one spec per GPU).
AgentTask: Decoupled model -- tasks are independent of GPU assignment.
           The GPU pool scheduler assigns GPUs dynamically at runtime.

Used by ParallelAgent.run_parallel() to spawn agents.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _agent_type_to_class() -> dict[str, type]:
    """Canonical mapping from task-file ``agent_type`` string to class.

    Lazy import to avoid circular dependencies at module level. The YAML
    subagent registry contributes names that resolve to the same
    OptimizationAgent class; descriptor-specific prompts/config are merged
    later when task files are dispatched.
    """
    from minisweagent.agents.optimization_agent import OptimizationAgent

    mapping: dict[str, type] = {"strategy_agent": OptimizationAgent}
    try:
        from minisweagent.subagents import SubAgentRegistry

        for name in SubAgentRegistry().list_names():
            mapping.setdefault(name, OptimizationAgent)
    except Exception:
        logger.debug("SubAgentRegistry unavailable; using built-in agent types only.")
    return mapping


def _agent_class_to_type() -> dict[type, str]:
    """Reverse mapping: agent class -> agent_type string."""
    return {cls: name for name, cls in _agent_type_to_class().items()}


def _all_agent_types() -> frozenset[str]:
    return frozenset(_agent_type_to_class())


ALL_AGENT_TYPES: frozenset[str] = _all_agent_types()

_DEFAULT_FALLBACK_AGENT = "strategy_agent"


def get_allowed_agent_types() -> set[str] | None:
    """Return the effective set of allowed agent types, or *None* if unrestricted.

    Reads ``GEAK_ALLOWED_AGENTS`` (allowlist) and ``GEAK_EXCLUDED_AGENTS``
    (blocklist) from the environment.  When *both* are set the allowlist
    wins and the blocklist is ignored (a warning is logged).
    """
    allowed_raw = os.environ.get("GEAK_ALLOWED_AGENTS", "").strip()
    excluded_raw = os.environ.get("GEAK_EXCLUDED_AGENTS", "").strip()

    if not allowed_raw and not excluded_raw:
        return None

    if allowed_raw:
        if excluded_raw:
            logger.warning(
                "Both GEAK_ALLOWED_AGENTS and GEAK_EXCLUDED_AGENTS are set; GEAK_ALLOWED_AGENTS takes precedence."
            )
        allowed = {t.strip() for t in allowed_raw.split(",") if t.strip()}
        return allowed & _all_agent_types()

    excluded = {t.strip() for t in excluded_raw.split(",") if t.strip()}
    return _all_agent_types() - excluded


def filter_agent_type(agent_type: str) -> str:
    """Safety-net filter: remap *agent_type* to the fallback if it is not allowed.

    When no filtering env vars are set this is a no-op.
    """
    allowed = get_allowed_agent_types()
    if allowed is None:
        return agent_type

    if agent_type in allowed:
        return agent_type

    fallback = os.environ.get("GEAK_FALLBACK_AGENT", "").strip() or _DEFAULT_FALLBACK_AGENT

    # Validate fallback is in allowed set; if not, pick first allowed type
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
    """A single optimization task, independent of GPU assignment.

    The GPU pool scheduler in ParallelAgent._run_pool() assigns GPUs
    dynamically at execution time. If there are more tasks than GPUs,
    tasks queue and run as GPU slots free up (like ProcessPoolExecutor).

    Attributes:
        agent_class: The agent class to instantiate.
        task: Specific instructions for this agent (overrides the base task_content).
        label: Human-readable label for logging (e.g. "fusion-rope-cos-sin").
        priority: Lower number = higher priority. OpenEvolve=0, fusion=5, tuning=10, etc.
        kernel_language: Language context ("python", "cpp", "asm") for task prompt context.
        config: Config overrides merged into the base agent_config.
        step_limit: Per-task step limit (0 = inherit from parent).
        cost_limit: Per-task cost limit (0.0 = inherit from parent).
    """

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
    """Specification for a single sub-agent in a heterogeneous parallel run.

    Legacy model: each spec is hard-wired to specific GPU IDs.
    Prefer AgentTask + _run_pool() for new code.

    Attributes:
        agent_class: The agent class to instantiate (e.g. StrategyAgent).
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


