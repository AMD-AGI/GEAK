"""Unit test subagent.

This agent searches for (or creates) unit/benchmark tests for a kernel and returns
one test command string to be used by the main agent.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.config import get_config_path
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig


@dataclass
class UnitTestAgentConfig(AgentConfig):
    """Config loaded from mini_unit_test_agent.yaml (or provided via kwargs)."""


class UnitTestAgent(DefaultAgent):
    """Agent that returns a single TEST_COMMAND line via MINI_SWE_AGENT_FINAL_OUTPUT."""

    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=UnitTestAgentConfig, **kwargs)


def _extract_test_command(text: str) -> str:
    match = re.search(r"TEST_COMMAND:\s*(.+)\s*$", text.strip(), re.MULTILINE)
    if not match:
        raise ValueError(f"UnitTestAgent did not return TEST_COMMAND. Output was:\n{text}")
    return match.group(1).strip()


def run_unit_test_agent(*, model: Model, repo: Path, kernel_name: str, log_dir: Path | None = None) -> str:
    """Run UnitTestAgent in `repo` and return the extracted test command string."""
    config_path = get_config_path("mini_unit_test_agent")
    config = yaml.safe_load(config_path.read_text())
    agent_config = config.get("agent", {})

    env = LocalEnvironment(**LocalEnvironmentConfig(cwd=str(repo)).__dict__)
    agent = UnitTestAgent(model, env, **agent_config)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        agent.log_file = log_dir / "unit_test_agent.log"

    task = f"Find or create unit/benchmark tests for kernel: {kernel_name}\nRepository: {repo}"
    exit_status, result = agent.run(task)
    if exit_status != "Submitted":
        raise RuntimeError(f"UnitTestAgent did not finish successfully: {exit_status}\n{result}")

    return _extract_test_command(result)
