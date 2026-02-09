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


def run_discovery_pipeline(kernel_path: Path, repo: Path) -> str:
    """Run MSA's content-based discovery pipeline and format results for the agent.

    Returns a string block describing discovered tests/benchmarks (may be empty
    if nothing was found or the pipeline is unavailable).
    """
    try:
        from minisweagent.tools.discovery import DiscoveryPipeline
    except ImportError:
        return ""

    try:
        pipeline = DiscoveryPipeline(workspace_path=repo)
        result = pipeline.run(kernel_path=kernel_path, interactive=False)
    except Exception:
        return ""

    lines: list[str] = []
    lines.append("## Pre-Discovery Results (automated content-based scan)")
    lines.append("")

    if result.tests:
        lines.append("### Discovered Test Files (ranked by confidence):")
        for i, t in enumerate(result.tests[:5], 1):
            conf_pct = min(int(t.confidence * 100), 100)
            lines.append(f"  {i}. `{t.file_path}` — {t.test_type}, {conf_pct}% confidence")
            lines.append(f"     Suggested command: `{t.command}`")
        lines.append("")

    if result.benchmarks:
        lines.append("### Discovered Benchmark Files (ranked by confidence):")
        for i, b in enumerate(result.benchmarks[:5], 1):
            conf_pct = min(int(b.confidence * 100), 100)
            lines.append(f"  {i}. `{b.file_path}` — {b.bench_type}, {conf_pct}% confidence")
            lines.append(f"     Suggested command: `{b.command}`")
        lines.append("")

    if not result.tests and not result.benchmarks:
        lines.append("No existing tests or benchmarks were found by the automated scan.")
        lines.append("You will need to create them from scratch.")
        lines.append("")

    lines.append("Use these results as a starting point. Validate any discovered")
    lines.append("tests/benchmarks before using them. Create new ones if none are suitable.")

    return "\n".join(lines)


def run_unit_test_agent(
    *,
    model: Model,
    repo: Path,
    kernel_name: str,
    log_dir: Path | None = None,
    discovery_context: str = "",
) -> str:
    """Run UnitTestAgent in ``repo`` and return the extracted test command string.

    If *discovery_context* is provided (e.g. from :func:`run_discovery_pipeline`),
    it is appended to the task prompt so the agent starts with pre-scanned results
    instead of exploring from scratch.
    """
    config_path = get_config_path("mini_unit_test_agent")
    config = yaml.safe_load(config_path.read_text())
    agent_config = config.get("agent", {})

    env = LocalEnvironment(**LocalEnvironmentConfig(cwd=str(repo)).__dict__)
    agent = UnitTestAgent(model, env, **agent_config)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        agent.log_file = log_dir / "unit_test_agent.log"

    task = f"Find or create unit/benchmark tests for kernel: {kernel_name}\nRepository: {repo}"
    if discovery_context:
        task += f"\n\n{discovery_context}"

    exit_status, result = agent.run(task)
    if exit_status != "Submitted":
        raise RuntimeError(f"UnitTestAgent did not finish successfully: {exit_status}\n{result}")

    return _extract_test_command(result)
