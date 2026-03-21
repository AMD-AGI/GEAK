"""Shape fixer agent.

Lightweight post-UTA agent that verifies the generated harness uses the
correct shapes from the benchmark file.  Runs after the UnitTestAgent
produces a harness and BEFORE runtime validation.

Input: harness path + benchmark file path
Task: compare shapes, fix if wrong
Output: fixed harness (in-place edit)
"""

import re
from dataclasses import dataclass
from pathlib import Path

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.config import load_agent_config
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig


@dataclass
class ShapeFixerConfig(AgentConfig):
    pass


class ShapeFixerAgent(DefaultAgent):
    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=ShapeFixerConfig, **kwargs)


SYSTEM_PROMPT = """\
You are a meta-checking agent. Read two files, compare shapes, done.

Step 1: Read the SHAPE SOURCE FILE. What configs does it benchmark?
  (Look for the config variables, loops, products that feed the timing.)

Step 2: Read the HARNESS FILE. What configs does it use?

Step 3: Same configs? Same count, same values?
  - YES: print SHAPES_VERIFIED. Stop.
  - NO: fix the harness configs to match the source file. Print SHAPES_FIXED. Stop.

That is all. Do not run anything. Do not explore. Just read and compare.
"""


def run_shape_fixer(
    *,
    model: Model,
    repo: Path,
    harness_path: Path,
    benchmark_file: Path,
    kernel_path: Path | None = None,
    log_dir: Path | None = None,
    gpu_id: int = 0,
) -> bool:
    """Run the shape fixer agent. Returns True if shapes were verified or fixed."""
    try:
        agent_config, _ = load_agent_config("mini_shape_fixer")
    except Exception:
        agent_config = {}

    env = LocalEnvironment(**LocalEnvironmentConfig(cwd=str(repo)).__dict__)

    if not agent_config:
        agent_config = {
            "system_template": SYSTEM_PROMPT,
            "step_limit": 0.0,
            "cost_limit": 0.0,
            "instance_template": "Your task is: {{task}}",
            "action_observation_template": (
                "<returncode>{{output.returncode}}</returncode>\n"
                "<output>\n{{ output.output -}}\n</output>"
            ),
            "format_error_template": (
                "Please always provide EXACTLY ONE action in triple backticks, "
                "found {{actions|length}} actions."
            ),
        }

    agent = ShapeFixerAgent(model, env, **agent_config)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        agent.log_file = log_dir / "shape_fixer_agent.log"

    task = (
        f"SHAPE SOURCE FILE: {benchmark_file}\n"
        f"HARNESS FILE: {harness_path}\n"
        f"\nRead both files. Does the harness use the same benchmark configs? Fix if not.\n"
    )

    exit_status, result = agent.run(task)

    if "SHAPES_VERIFIED" in (result or ""):
        return True
    if "SHAPES_FIXED" in (result or ""):
        return True
    return False
