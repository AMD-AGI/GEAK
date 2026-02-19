"""Tool: sub_agent -- spawn a child DefaultAgent for focused sub-tasks.

The main agent can delegate specific work (e.g. algorithm rewrite, cross-file
edits) to a child agent with a targeted prompt and bounded step/cost budget.
The child shares the parent's model and environment but runs independently.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SubAgentTool:
    """ToolRuntime-compatible callable that spawns a child DefaultAgent.

    The child agent runs a full step loop with its own budget, then returns
    its final result to the parent.
    """

    def __init__(self, model=None, env=None):
        self._model = model
        self._env = env
        self._codebase_context: str | None = None

    def set_context(self, model, env, codebase_context: str | None = None):
        """Set model, env, and optional codebase context."""
        self._model = model
        self._env = env
        if codebase_context is not None:
            self._codebase_context = codebase_context

    def __call__(
        self,
        task: str,
        step_limit: int = 10,
        cost_limit: float = 1.0,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Spawn a child agent to perform a focused sub-task.

        Args:
            task: The sub-task description for the child agent.
            step_limit: Max steps for the child (default 10).
            cost_limit: Max cost in dollars for the child (default 1.0).
            system_prompt: Override the child's system prompt (optional).

        Returns:
            {output: str, returncode: int}
        """
        if not self._model or not self._env:
            return {"output": "sub_agent not initialized (no model/env)", "returncode": 1}

        try:
            from minisweagent.agents.default import DefaultAgent
        except ImportError as e:
            return {"output": f"Cannot import DefaultAgent: {e}", "returncode": 1}

        # Build child config
        child_config: dict[str, Any] = {
            "step_limit": step_limit,
            "cost_limit": cost_limit,
        }
        if system_prompt:
            child_config["system_template"] = system_prompt

        if self._codebase_context:
            task = (
                "## Codebase Context (repo structure and key files)\n"
                + self._codebase_context
                + "\n\n"
                + task
            )

        logger.info(f"[sub_agent] Spawning child agent: steps={step_limit}, cost=${cost_limit}")

        try:
            child = DefaultAgent(self._model, self._env, **child_config)
            exit_status, result = child.run(task)
            logger.info(f"[sub_agent] Child finished: {exit_status}")
            return {
                "output": f"Sub-agent completed ({exit_status}): {result}",
                "returncode": 0 if exit_status == "Submitted" else 1,
            }
        except Exception as e:
            logger.error(f"[sub_agent] Child agent failed: {e}")
            return {"output": f"Sub-agent error: {e}", "returncode": 1}
