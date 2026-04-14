"""Preprocess-owned shape fixer agent.

Lightweight post-UTA agent that verifies the generated harness uses the
correct shapes from the benchmark file. Runs after the UnitTestAgent
produces a harness and BEFORE runtime validation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.run.preprocess.config_loader import load_preprocess_agent_config

logger = logging.getLogger(__name__)


@dataclass
class ShapeFixerConfig(AgentConfig):
    pass


class ShapeFixerAgent(DefaultAgent):
    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=ShapeFixerConfig, **kwargs)


SYSTEM_PROMPT = """\
You are a meta-checking agent. Read two files, compare shapes, done.

Step 1: Read the SHAPE SOURCE FILE. What configs does it benchmark?
  (Look for the config variables, loops, products, helper functions, and
  ordering that feed the timing.)
  Determine the SOURCE-ORDERED full case stream.
  Then determine the exact sampled cases that `--benchmark`,
  `--correctness`, and `--profile` should use.
  If the source file is a task-runner or task config, use it only as an
  adapter unless it clearly mirrors the repo-native benchmark/test layer.
  For rocPRIM-style repos, prefer repo-native `benchmark/benchmark_*.cpp`,
  `benchmark/*.parallel.hpp`, `test/rocprim/test_*.cpp`, target names like
  `benchmark_device_*` / `test_device_*`, and their emitted case IDs as the
  source of truth.

Step 2: Read the HARNESS FILE. What configs does it use?
  Determine the HARNESS-ORDERED full case stream.
  Then determine the exact sampled cases that the harness will use for
  `--benchmark`, `--correctness`, and `--profile`.

Step 3: Compare EXACT sampled semantics, not just the config universe.
  - SAME VALUES / SAME COUNT is NOT sufficient if a different order causes
    `_pick()` (or equivalent logic) to choose different sampled cases.
  - If the source file already defines a case list/helper, that ordering is
    authoritative.
  - If a task-runner exists, preserve it only insofar as it faithfully
    wraps the same repo-native benchmark/test semantics. Do NOT approve a
    wrapper that invents new commands, unsupported flags, or a different
    case set than the real benchmark/test sources.
  - If the source file uses nested loops, preserve that loop nesting order.
  - Tuple vs list syntax is fine only when the resulting sampled cases are
    exactly the same.
  - YES: exact sampled cases for `--benchmark`, `--correctness`, and
    `--profile` all match the source file. Print SHAPES_VERIFIED. Stop.
  - NO: fix the harness so it preserves the source ordering and yields the
    exact same sampled cases. Print SHAPES_FIXED. Stop.

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
        agent_config, _ = load_preprocess_agent_config("mini_shape_fixer")
    except Exception:
        logger.debug("Failed to load preprocess agent config for mini_shape_fixer", exc_info=True)
        agent_config = {}

    env = LocalEnvironment(**LocalEnvironmentConfig(cwd=str(repo)).__dict__)

    if not agent_config:
        agent_config = {
            "system_template": SYSTEM_PROMPT,
            "step_limit": 0.0,
            "cost_limit": 0.0,
            "instance_template": "Your task is: {{task}}",
            "action_observation_template": (
                "<returncode>{{output.returncode}}</returncode>\n<output>\n{{ output.output -}}\n</output>"
            ),
            "format_error_template": (
                "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
            ),
        }

    agent = ShapeFixerAgent(model, env, **agent_config)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        agent.log_file = log_dir / "shape_fixer_agent.log"

    task = (
        f"SHAPE SOURCE FILE: {benchmark_file}\n"
        f"HARNESS FILE: {harness_path}\n"
        f"\nRead both files. Preserve the source file's ordered full case stream.\n"
        f"Check whether the harness would choose the exact same sampled cases for\n"
        f"`--benchmark`, `--correctness`, and `--profile`. Full-benchmark being\n"
        f"correct is NOT sufficient if sampled modes drift. Fix if needed.\n"
    )

    exit_status, result = agent.run(task)

    if "SHAPES_VERIFIED" in (result or ""):
        return True
    if "SHAPES_FIXED" in (result or ""):
        return True
    return False
