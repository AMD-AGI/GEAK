"""Preprocess-owned shape fixer agent.

Lightweight post-UTA harness-repair agent that verifies the generated
harness preserves the benchmark/test sampling contract and can apply
minimal fixes before the preprocess stage performs authoritative
re-validation.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent, Submitted
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.run.preprocess.config_loader import load_preprocess_agent_config

logger = logging.getLogger(__name__)


@dataclass
class ShapeFixerConfig(AgentConfig):
    pass


class ShapeFixerAgent(DefaultAgent):
    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=ShapeFixerConfig, **kwargs)

    def query(self) -> dict:
        """Terminate immediately when the model emits a final shape verdict."""
        response = super().query()
        content = (response.get("content") or "").strip()
        if not response.get("tools") and not self._will_use_bash(response):
            for verdict in ("SHAPES_VERIFIED", "SHAPES_FIXED"):
                if verdict in content:
                    raise Submitted(verdict)
        return response


SYSTEM_PROMPT = """\
You are a harness repair agent. Start by comparing the authoritative
source file against the generated harness, then apply the smallest
source-faithful fix needed to make the sampled modes work.

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

Step 3: Compare full sampled semantics, not just the config universe.
  - SAME VALUES / SAME COUNT is NOT sufficient if a different order causes
    `_pick()` (or equivalent logic) to choose different sampled cases.
  - Preserve the source file's helper semantics and per-tensor contracts:
    dtype, device, layout, contiguity, auxiliary buffers/caches/scales,
    index dtypes, and helper-side preprocessing.
  - Do NOT normalize all tensors to a single dtype just because the main
    activations use that dtype.
  - If a task-runner exists, preserve it only insofar as it faithfully
    wraps the same repo-native benchmark/test semantics. Do NOT approve a
    wrapper that invents new commands, unsupported flags, or a different
    case set than the real benchmark/test sources.
  - If the source file uses nested loops, preserve that loop nesting order.
  - Tuple vs list syntax is fine only when the resulting sampled cases are
    exactly the same.

Step 4: Validate and repair minimally.
  - If validation commands are provided, you MAY run them from the repo
    root to understand why a mode fails.
  - Understand failures from stderr before editing.
  - Re-run only the relevant failing mode(s) after each edit.
  - Prefer the smallest source-faithful fix over rewriting working harness
    logic.
  - If no edit is needed and the relevant validation commands pass, print
    SHAPES_VERIFIED. Stop.
  - If you edited the harness and the relevant validation commands pass,
    print SHAPES_FIXED. Stop.

Stay focused: read the source file, the harness, and directly imported
helper files when needed. Do not broad-explore the repo.
"""


class ShapeFixerTimeoutError(RuntimeError):
    """Raised when the shape fixer exceeds its wall-clock timeout."""


def _shape_fixer_timeout_handler(_signum, _frame) -> None:
    raise ShapeFixerTimeoutError("Shape fixer timed out")


@contextmanager
def _shape_fixer_timeout(timeout_s: int):
    if timeout_s <= 0 or os.name == "nt" or not hasattr(signal, "SIGALRM"):
        yield
        return

    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _shape_fixer_timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    except (AttributeError, ValueError):
        yield
        return

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_shape_fixer_task(
    *,
    benchmark_file: Path,
    harness_path: Path,
    kernel_path: Path | None,
    gpu_id: int,
    validation_feedback: list[str] | None,
) -> str:
    quoted_harness = shlex.quote(str(harness_path))
    validation_prefix = f"HIP_VISIBLE_DEVICES={gpu_id} GEAK_BENCHMARK_ITERATIONS=5"
    validation_commands = [
        f"{validation_prefix} python {quoted_harness} --correctness",
        f"{validation_prefix} python {quoted_harness} --profile",
        f"{validation_prefix} python {quoted_harness} --benchmark",
    ]

    lines = [
        f"SHAPE SOURCE FILE: {benchmark_file}",
        f"HARNESS FILE: {harness_path}",
    ]
    if kernel_path is not None:
        lines.append(f"KERNEL FILE: {kernel_path}")

    lines.extend(
        [
            "",
            "Read the source file and harness first. The source file is authoritative for the full harness contract, not just shapes.",
            "Preserve the source file's ordered full case stream and the per-tensor execution contract: dtype, device, layout, contiguity, auxiliary buffers/caches/scales, index dtypes, helper-side preprocessing, and supported flags.",
            "Check whether the harness chooses the exact same sampled cases for `--benchmark`, `--correctness`, and `--profile`.",
            "Full-benchmark being correct is NOT sufficient if sampled modes drift.",
            "Do NOT normalize all tensors to a single dtype just because the main activations use that dtype.",
            "Prefer the smallest source-faithful fix; do not rewrite working parts of the harness.",
            "",
            "Suggested validation commands (run from the repo root only if you need them):",
            *[f"- {cmd}" for cmd in validation_commands],
            "",
            "If a validation command fails, understand the reason from stderr before editing. After each edit, rerun only the relevant failing command(s) until the harness is source-faithful and the edited modes pass.",
        ]
    )

    if validation_feedback:
        lines.extend(
            [
                "",
                "PREVIOUS REVALIDATION FAILURES:",
                *[f"- {err}" for err in validation_feedback],
                "",
                "Use the failure feedback above to guide the minimal repair. Do not make unrelated changes.",
            ]
        )

    return "\n".join(lines) + "\n"


def run_shape_fixer(
    *,
    model: Model,
    repo: Path,
    harness_path: Path,
    benchmark_file: Path,
    kernel_path: Path | None = None,
    log_dir: Path | None = None,
    gpu_id: int = 0,
    validation_feedback: list[str] | None = None,
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

    task = _build_shape_fixer_task(
        benchmark_file=benchmark_file,
        harness_path=harness_path,
        kernel_path=kernel_path,
        gpu_id=gpu_id,
        validation_feedback=validation_feedback,
    )

    timeout_s = int(os.environ.get("GEAK_SHAPE_FIXER_TIMEOUT", "300"))
    with _shape_fixer_timeout(timeout_s):
        exit_status, result = agent.run(task)

    if "SHAPES_VERIFIED" in (result or ""):
        return True
    if "SHAPES_FIXED" in (result or ""):
        return True
    return False
