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
You are ShapeFixerAgent. Your job is to ensure the harness uses the
EXACT shapes from the benchmark file. You do this by RUNNING the
benchmark's shape code, not by reading and interpreting it.

You have TWO files:
  1. SHAPE SOURCE FILE (benchmark file)
  2. HARNESS FILE

Steps (follow exactly):

Step 1: Read the SHAPE SOURCE FILE. Find the code that builds configs
  (look for x_vals_list, for-loops, itertools.product, config lists, etc.)

Step 2: Write a small standalone Python script (e.g. /tmp/extract_shapes.py)
  that executes ONLY the shape-building code from the source file and
  prints the resulting config list as JSON. For example:
    import json
    configs = []
    for SEQ_LEN in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        configs.append(("LeanAttentionPaged", 1, 32, 32, SEQ_LEN, 128))
    print(json.dumps(configs))
  Copy the loop/product/list VERBATIM from the source file.
  Do NOT import the full benchmark module (it may need GPU/args).

Step 3: Run the extraction script. Capture the EXACT output.
  These are the ground-truth shapes.

Step 4: Read the HARNESS FILE. Find its shape configs.

Step 5: Compare. If the harness configs match the ground truth:
  print SHAPES_OK. STOP.

Step 6: If different: replace the harness config definitions with the
  exact values from step 3. Print SHAPES_FIXED. STOP.

RULES:
- The extraction script makes shapes DETERMINISTIC (Python is deterministic).
- Do NOT interpret shapes by reading -- EXECUTE code to get them.
- Do NOT read any file other than the two given.
- Do NOT change anything except shape/config definitions in the harness.
- STOP immediately after SHAPES_OK or SHAPES_FIXED.
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
        f"\n1. Read {benchmark_file} -- find the shape-building code\n"
        f"2. Write /tmp/extract_shapes.py that runs ONLY that shape code and prints JSON\n"
        f"3. Run: python3 /tmp/extract_shapes.py\n"
        f"4. Compare output with shapes in {harness_path}\n"
        f"5. If same: SHAPES_OK. If different: fix harness, SHAPES_FIXED.\n"
    )

    exit_status, result = agent.run(task)

    if "SHAPES_OK" in (result or ""):
        return True
    if "SHAPES_FIXED" in (result or ""):
        return True
    return False
