"""Translation subagent.

This agent translates a kernel from one language to another (e.g.
PyTorch -> FlyDSL).  It is configured via a translation-specific YAML
(e.g. ``mini_kernel_pytorch_to_flydsl.yaml``) and receives FlyDSL
knowledge base content as a template variable.

The layered design mirrors :mod:`~minisweagent.agents.unit_test_agent`:

- ``TranslationAgent``        — minimal ``DefaultAgent`` subclass
- ``run_translation_agent()`` — single-round wrapper (build task, call
  ``agent.run()``, return result)
- ``run_translation()`` in :mod:`~minisweagent.run.preprocess.translate`
  — multi-round orchestration with retry loop, self-review, and perf
  measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig


@dataclass
class TranslationAgentConfig(AgentConfig):
    """Config loaded from a translation YAML (or provided via kwargs)."""


class TranslationAgent(DefaultAgent):
    """Agent that translates a kernel from one language to another."""

    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=TranslationAgentConfig, **kwargs)


def run_translation_agent(
    *,
    model: Model,
    repo_root: Path,
    agent_config: dict[str, Any],
    task: str,
    kb_content: str,
    env_overrides: dict[str, str] | None = None,
    test_command: str | None = None,
    log_dir: Path | None = None,
    log_name: str = "translation_agent.log",
) -> tuple[str, str]:
    """Run a single translation round and return ``(exit_status, result)``.

    Parameters
    ----------
    model:
        LLM model instance.
    repo_root:
        Working directory for the agent environment.
    agent_config:
        Agent config dict loaded from the translation YAML.
    task:
        Task prompt describing the translation to perform.
    kb_content:
        Knowledge base content injected as ``{{knowledge_base}}`` in the
        instance template.
    env_overrides:
        Extra environment variables (e.g. PYTHONPATH for FlyDSL).
    test_command:
        Harness command for ``save_and_test`` tool.
    log_dir:
        Directory for the agent log file.
    log_name:
        Filename for the agent log.

    Returns
    -------
    Tuple of ``(exit_status, result_text)`` from the agent's ``run()``.
    """
    kwargs = dict(agent_config)
    if test_command:
        kwargs["test_command"] = test_command
    if log_dir:
        kwargs["patch_output_dir"] = str(log_dir)

    if env_overrides:
        env_config = LocalEnvironmentConfig(cwd=str(repo_root), env=env_overrides)
    else:
        env_config = LocalEnvironmentConfig(cwd=str(repo_root))
    env = LocalEnvironment(**env_config.__dict__)

    agent = TranslationAgent(model, env, **kwargs)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        agent.log_file = log_dir / log_name

    return agent.run(task, knowledge_base=kb_content)
