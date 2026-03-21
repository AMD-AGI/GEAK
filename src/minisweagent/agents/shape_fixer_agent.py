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
verify the harness benchmarks the SAME configs as the benchmark file.

You have TWO files:
  1. SHAPE SOURCE FILE (benchmark file)
  2. HARNESS FILE

Steps:

Step 1: Read the SHAPE SOURCE FILE. Find:
  a) The config VARIABLES (e.g. BATCH_SIZES, DIM2S, KS, SEQ_LENS, etc.)
  b) How the benchmark COMBINES them (itertools.product, for-loops, x_vals_list)
  c) Which function is TIMED (look for do_bench, perf_report, Event timing)
  d) What configs feed that timing function -- THESE are the benchmark shapes

  IMPORTANT: Do NOT run main(). main() may have arg parsing and code paths
  that produce different configs. Instead, find the config variables and
  the product/loop that builds the benchmarked config list.

Step 2: Write /tmp/extract_shapes.py that imports or copies ONLY the
  config variables and product/loop, and prints the config list as JSON.
  Example for a file with BATCH_SIZES=[1,2,4], DIM2S=[128,256], KS=[2,8]:
    import json, itertools
    BATCH_SIZES = [1, 2, 4]
    DIM2S = [128, 256]
    KS = [2, 8]
    configs = list(itertools.product(BATCH_SIZES, DIM2S, KS))
    print(json.dumps(configs))
  Copy the variable values and product VERBATIM from the source file.

Step 3: Run: python3 /tmp/extract_shapes.py
  This output is GROUND TRUTH.

Step 4: Run the harness --full-benchmark and capture GEAK_SHAPES_USED.

Step 5: Compare config COUNT and VALUES.
  - The harness should benchmark ALL configs from the source file.
  - If the harness has MORE configs than ground truth (e.g. added test
    shapes), that is also WRONG.

Step 6: If they match: print SHAPES_VERIFIED. STOP.

Step 7: If different: fix the harness to use the exact ground truth
  configs. Print SHAPES_FIXED. STOP.

RULES:
- Find the BENCHMARKED configs, not arbitrary code path configs.
- Do NOT run main() or the full benchmark module.
- Do NOT explore other files.
- STOP after SHAPES_VERIFIED or SHAPES_FIXED.
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
