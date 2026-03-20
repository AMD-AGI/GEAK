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
You are ShapeFixerAgent. You have exactly TWO files to work with:
  1. SHAPE SOURCE FILE (given in task)
  2. HARNESS FILE (given in task)

You must NOT read, explore, or use ANY other file. Only these two.

Do these steps IN ORDER. Stop immediately after step 3 or 4:

Step 1: Read the SHAPE SOURCE FILE. Find:
  - The shape/config values (x_vals_list, for-loops, product calls, etc.)
  - Helper functions that construct inputs (input_helper, setup_inputs, etc.)

Step 2: Read the HARNESS FILE. Find its shape definitions and how it
  sets up kernel inputs.

Step 3: Check TWO things:
  a) Do the harness shapes match the SHAPE SOURCE FILE's shapes?
  b) Does the harness reuse the SHAPE SOURCE FILE's helper functions
     for input construction (if they exist), or does it reinvent them?

Step 4: If both are correct: print SHAPES_OK. STOP.

Step 5: If shapes are wrong or the harness reinvents input construction
  that the SHAPE SOURCE FILE already provides: fix the harness to import
  and use the source file's helpers/shapes. Print SHAPES_FIXED. STOP.

FORBIDDEN:
- Do NOT read any file other than the two given.
- Do NOT explore the repository.
- Do NOT change anything except shape/config definitions in the harness.
- Do NOT add shapes from other files.
- Do NOT continue after printing SHAPES_OK or SHAPES_FIXED.
"""


def run_shape_fixer(
    *,
    model: Model,
    repo: Path,
    harness_path: Path,
    benchmark_file: Path,
    kernel_path: Path | None = None,
    log_dir: Path | None = None,
) -> bool:
    """Run the shape fixer agent. Returns True if shapes were OK or fixed."""
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
        f"\nYou may ONLY read these two files. No other files.\n"
        f"Step 1: cat {benchmark_file}\n"
        f"Step 2: cat {harness_path}\n"
        f"Step 3: Do shapes match? SHAPES_OK or fix and SHAPES_FIXED.\n"
    )

    exit_status, result = agent.run(task)

    if "SHAPES_OK" in (result or ""):
        return True
    if "SHAPES_FIXED" in (result or ""):
        return True
    return False
