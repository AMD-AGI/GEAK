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
You are ShapeFixerAgent -- a meta-validation agent. Your job is to
verify the harness correctly wraps the benchmark file, and fix it if not.

You have TWO files:
  1. SHAPE SOURCE FILE (benchmark file)
  2. HARNESS FILE

You MUST follow these steps exactly:

Step 1: Read the SHAPE SOURCE FILE. Find the code that builds the
  config list (for-loops, x_vals_list, itertools.product, etc.).

Step 2: Write /tmp/extract_shapes.py that runs ONLY that shape-building
  code and prints the config list as JSON. Copy the loop VERBATIM from
  the source file. Do NOT import the full benchmark module. Example:
    import json
    configs = []
    for SEQ_LEN in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        configs.append((1, 32, 32, SEQ_LEN, 128))
    print(json.dumps(configs))

Step 3: Run: python3 /tmp/extract_shapes.py
  This output is GROUND TRUTH (Python is deterministic).

Step 4: Run the harness with --full-benchmark and capture GEAK_SHAPES_USED.
  If the harness needs PYTHONPATH, set it.

Step 5: Compare GROUND TRUTH (step 3) with GEAK_SHAPES_USED (step 4).
  - Same config count?
  - Same values (ignore tuple packing differences, compare the actual
    dimension values that vary)?

Step 6: If they match: print SHAPES_VERIFIED. STOP.

Step 7: If they differ: edit the harness to use the EXACT ground truth
  configs from step 3. Paste them as a literal list. Then re-run
  --full-benchmark to confirm. Print SHAPES_FIXED. STOP.

RULES:
- EXECUTE code to get shapes, do NOT interpret by reading.
- Do NOT explore other files.
- STOP immediately after SHAPES_VERIFIED or SHAPES_FIXED.
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
        f"REPO ROOT: {repo}\n"
        f"GPU: HIP_VISIBLE_DEVICES={gpu_id}\n"
        f"\n"
        f"1. Read {benchmark_file} -- find the shape-building code\n"
        f"2. Write /tmp/extract_shapes.py -- copy that code, print JSON\n"
        f"3. Run: python3 /tmp/extract_shapes.py  (ground truth)\n"
        f"4. Run: PYTHONPATH={repo} HIP_VISIBLE_DEVICES={gpu_id} "
        f"GEAK_BENCHMARK_ITERATIONS=2 "
        f"python3 {harness_path} --full-benchmark 2>&1 | grep GEAK_SHAPES_USED\n"
        f"5. Compare ground truth (step 3) with GEAK_SHAPES_USED (step 4)\n"
        f"6. If same: SHAPES_VERIFIED. If different: fix harness, SHAPES_FIXED.\n"
    )

    exit_status, result = agent.run(task)

    if "SHAPES_VERIFIED" in (result or ""):
        return True
    if "SHAPES_FIXED" in (result or ""):
        return True
    return False
