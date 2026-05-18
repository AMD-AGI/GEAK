"""GEMM tuning agent.

Thin :class:`~minisweagent.agents.default.DefaultAgent` subclass configured via
``mini_gemm_tuning.yaml``.  Intended for FP8 block-scaled GEMM / aiter / CK tuner
workflows; no workspace bootstrap or harness orchestration here—only the agent
loop over a local shell environment and the configured LLM backend (e.g.
LiteLLM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.run.utils.parallel_helpers import redirect_output_to_file


@dataclass
class GemmTuningAgentConfig(AgentConfig):
    """Config loaded from ``mini_gemm_tuning.yaml`` (or provided via kwargs)."""


class GemmTuningAgent(DefaultAgent):
    """Agent focused on GEMM tuning tasks (shell + LLM)."""

    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=GemmTuningAgentConfig, **kwargs)


def run_gemm_tuning_agent(
    *,
    model: Model,
    cwd: Path,
    agent_config: dict[str, Any],
    task: str,
    env_overrides: dict[str, str] | None = None,
    local_env: dict[str, Any] | None = None,
    log_dir: Path | None = None,
    log_name: str = "task_0.log",
    **run_template_vars: Any,
) -> tuple[str, str]:
    """Run one agent session and return ``(exit_status, message)`` from :meth:`DefaultAgent.run`.

    When *log_dir* is set, conversation is appended to ``log_dir / log_name`` (default
    ``task_0.log``, same basename as :class:`~minisweagent.agents.parallel_agent.ParallelAgent`),
    and this thread's stdout/stderr are redirected there for the duration of the run.
    ``patch_output_dir`` is set to *log_dir* so ``traj.json`` is written alongside.
    """
    kwargs = dict(agent_config)
    if log_dir:
        kwargs["patch_output_dir"] = str(log_dir)

    le = dict(local_env or {})
    le["cwd"] = str(Path(cwd).resolve())
    if env_overrides:
        merged_env = dict(le.get("env") or {})
        merged_env.update(env_overrides)
        le["env"] = merged_env

    env = LocalEnvironment(**le)
    agent = GemmTuningAgent(model, env, **kwargs)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_name
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("GEMM tuning agent conversation log\n")
            f.write("=" * 60 + "\n\n")
        agent.log_file = log_path
        with redirect_output_to_file(log_path):
            return agent.run(task, **run_template_vars)

    return agent.run(task, **run_template_vars)
