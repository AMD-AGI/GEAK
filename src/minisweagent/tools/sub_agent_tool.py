"""Tool: sub_agent -- spawn a child DefaultAgent for focused sub-tasks.

The main agent can delegate specific work (e.g. algorithm rewrite, cross-file
edits) to a child agent with a targeted prompt and bounded step/cost budget.
The child shares the parent's model and environment but runs independently.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

logger = logging.getLogger(__name__)

MIN_CHILD_STEP_LIMIT = 150
DEFAULT_CHILD_STEP_LIMIT = MIN_CHILD_STEP_LIMIT
DEFAULT_CHILD_COST_LIMIT = 0.0


def _normalize_child_budget(step_limit: int, cost_limit: float) -> tuple[int, float]:
    """Promote undersized child budgets so sub-agents can finish substantive rewrites."""
    effective_step_limit = step_limit if step_limit > 0 else DEFAULT_CHILD_STEP_LIMIT
    if effective_step_limit < MIN_CHILD_STEP_LIMIT:
        logger.info(
            "[sub_agent] Promoting child step_limit from %s to %s",
            step_limit,
            MIN_CHILD_STEP_LIMIT,
        )
        effective_step_limit = MIN_CHILD_STEP_LIMIT

    if cost_limit <= 0:
        return effective_step_limit, DEFAULT_CHILD_COST_LIMIT

    if effective_step_limit <= MIN_CHILD_STEP_LIMIT:
        logger.info(
            "[sub_agent] Disabling child cost_limit=%s so it does not interfere with "
            "the %s-step budget",
            cost_limit,
            effective_step_limit,
        )
        return effective_step_limit, DEFAULT_CHILD_COST_LIMIT

    return effective_step_limit, cost_limit


class SubAgentTool:
    """ToolRuntime-compatible callable that spawns a child DefaultAgent.

    The child agent runs a full step loop with its own budget, then returns
    its final result to the parent.
    """

    def __init__(self, model=None, env=None):
        self._model = model
        self._env = env
        self._codebase_context: str | None = None
        self._inherited_config: dict[str, Any] = {}
        self._save_and_test_context = None

    def set_context(
        self,
        model,
        env,
        codebase_context: str | None = None,
        inherited_config: dict[str, Any] | None = None,
        save_and_test_context=None,
    ):
        """Set model, env, and parent execution context for child agents."""
        self._model = model
        self._env = env
        if codebase_context is not None:
            self._codebase_context = codebase_context
        self._inherited_config = dict(inherited_config or {})
        self._save_and_test_context = save_and_test_context

    def __call__(
        self,
        task: str,
        step_limit: int = DEFAULT_CHILD_STEP_LIMIT,
        cost_limit: float = DEFAULT_CHILD_COST_LIMIT,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Spawn a child agent to perform a focused sub-task.

        Args:
            task: The sub-task description for the child agent.
            step_limit: Max steps for the child (default 150, minimum enforced 150).
            cost_limit: Max cost in dollars for the child (default 0.0, disabled).
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

        effective_step_limit, effective_cost_limit = _normalize_child_budget(step_limit, cost_limit)

        # Build child config
        inherited_config = {
            key: value
            for key, value in self._inherited_config.items()
            if value is not None
        }
        child_config: dict[str, Any] = {
            **inherited_config,
            "step_limit": effective_step_limit,
            "cost_limit": effective_cost_limit,
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

        logger.info(
            "[sub_agent] Spawning child agent: requested_steps=%s, requested_cost=$%s, "
            "effective_steps=%s, effective_cost=$%s",
            step_limit,
            cost_limit,
            effective_step_limit,
            effective_cost_limit,
        )

        try:
            child = DefaultAgent(self._model, self._env, **child_config)
            self._inherit_save_and_test_context(child)
            exit_status, result = child.run(task)
            self._sync_patch_counter_from_child(child)
            logger.info(f"[sub_agent] Child finished: {exit_status}")
            return {
                "output": f"Sub-agent completed ({exit_status}): {result}",
                "returncode": 0 if exit_status == "Submitted" else 1,
            }
        except Exception as e:
            logger.error(f"[sub_agent] Child agent failed: {e}")
            return {"output": f"Sub-agent error: {e}", "returncode": 1}

    def _inherit_save_and_test_context(self, child) -> None:
        """Give child agents the same benchmark contract and patch numbering."""
        parent_ctx = self._save_and_test_context
        child_ctx = getattr(child, "_save_and_test_context", None)
        if not parent_ctx or not child_ctx:
            return

        inherited_ctx = replace(
            child_ctx,
            test_command=parent_ctx.test_command,
            timeout=parent_ctx.timeout,
            patch_output_dir=parent_ctx.patch_output_dir,
            env_vars=parent_ctx.env_vars,
            base_repo_path=parent_ctx.base_repo_path,
            log_fn=parent_ctx.log_fn,
            patch_counter=parent_ctx.patch_counter,
        )
        save_and_test_tool = child.toolruntime._tool_table.get("save_and_test")
        if save_and_test_tool:
            save_and_test_tool.set_context(inherited_ctx)
        child._save_and_test_context = inherited_ctx
        child.patch_counter = inherited_ctx.patch_counter

    def _sync_patch_counter_from_child(self, child) -> None:
        """Advance the parent patch counter after a child save_and_test run."""
        parent_ctx = self._save_and_test_context
        child_ctx = getattr(child, "_save_and_test_context", None)
        if not parent_ctx or not child_ctx:
            return
        parent_ctx.patch_counter = child_ctx.patch_counter
