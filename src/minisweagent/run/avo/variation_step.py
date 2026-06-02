"""One AVO variation step = one ``OptimizationAgent`` run in the repo worktree.

This module wraps GEAK's existing ``OptimizationAgent`` without subclassing it.
The AVO contract is delivered through (a) a task-body prefix and (b) forced
skill injection. The agent's own ``save_and_test`` outputs (``patch_*.patch`` /
``patch_*_test.txt`` under ``patch_output_dir``) are parsed back into a
:class:`VariationResult`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from minisweagent.run.avo.result import AttemptRecord, VariationResult, patch_hash

logger = logging.getLogger(__name__)

# AVO contract prepended to every variation step's task body (mirrors
# skills/avo-evolution/docs/variation_step_contract.md).
_AVO_CONTRACT = """\
## AVO Variation Step Contract
- You are executing ONE variation step in a continuous evolution run.
- Lineage best so far: {best_summary}
- Active direction (assigned by supervisor): "{direction}"
- You MUST call save_and_test after each meaningful edit.
- If this direction shows no improvement after 3 attempts, mark it failed via
  strategy_manager and request the next strategy.
- DO NOT declare the overall optimization complete — only THIS step ends.
- Only submit when you have a verified improvement OR you have exhausted this
  direction and documented why.
"""


def select_skills(kernel_language: str, task: str) -> list[str]:
    """Pick which GEAK skills to force-inject for this step (Controller-driven)."""
    skills = ["avo-evolution"]
    lang = (kernel_language or "").lower()
    if lang == "flydsl":
        skills.append("flydsl")
    if "gemm" in task.lower():
        skills.append("fp8-gemm-tuning-sglang-aiter")
    if "attention" in task.lower() or "flash" in task.lower():
        skills.append("attention-microarch-optimization")
    return skills


def inject_skill_bodies(task_body: str, skills: list[str]) -> str:
    """Force-inject SKILL.md bodies via the existing SkillRuntime discovery.

    In a multi-day run the model cannot be relied upon to emit a ``use_skill``
    action every step, so the controller prepends the resolved skill bodies.
    No change to ``SkillRuntime`` itself — this is a read-only reuse.
    """
    from minisweagent.skills.skill_runtime import SkillRuntime

    rt = SkillRuntime()
    bodies: list[str] = []
    for name in skills:
        desc = rt.skills.get(name)
        if desc is None:
            continue
        try:
            bodies.append((desc.path / "SKILL.md").read_text(encoding="utf-8"))
        except OSError:
            logger.debug("inject_skill_bodies: could not read SKILL.md for %s", name)
    if not bodies:
        return task_body
    return "\n\n---\n\n".join(bodies + [task_body])


def compose_task(base_task: str, lineage, direction: dict[str, Any]) -> str:
    """Prefix the AVO contract onto the base task body."""
    contract = _AVO_CONTRACT.format(
        best_summary=lineage.summary(last_n=5),
        direction=direction.get("strategy") or "(none assigned — pick the highest-priority pending strategy)",
    )
    return f"{contract}\n\n{base_task}"


def run_variation_step(
    *,
    repo: Path,
    base_task: str,
    step_dir: Path,
    lineage,
    direction: dict[str, Any],
    output_dir: Path,
    avo_config: dict,
    model_factory,
    deadline=None,
    nudge: str | None = None,
) -> VariationResult:
    """Build + run one OptimizationAgent and return a structured result."""
    step_dir.mkdir(parents=True, exist_ok=True)
    step_index = _step_index_from_dir(step_dir)
    t0 = time.monotonic()

    task_body = compose_task(base_task, lineage, direction)
    if nudge:
        task_body += f"\n\n## Supervisor nudge\n{nudge}\n"
    task_body = inject_skill_bodies(task_body, select_skills(lineage.language, base_task))

    exit_status = "NotRun"
    try:
        agent = _build_agent(repo, step_dir, output_dir, avo_config, model_factory)
        exit_status, _msg = agent.run(task_body)
    except Exception as exc:  # noqa: BLE001 — a failed step must not kill the loop
        logger.exception("variation step %d crashed: %s", step_index, exc)
        exit_status = type(exc).__name__

    result = _collect_result(step_dir, step_index, direction.get("strategy"))
    result.exit_status = exit_status
    result.wall_time_s = time.monotonic() - t0
    logger.info(
        "variation step %d done: exit=%s attempts=%d best_speedup=%s",
        step_index,
        exit_status,
        len(result.attempts),
        result.best_speedup,
    )
    return result


# ---------------------------------------------------------------------------
# Agent construction (mirrors GEAK's task_file_to_agent_task essentials)
# ---------------------------------------------------------------------------


def _build_agent(repo: Path, step_dir: Path, output_dir: Path, avo_config: dict, model_factory):
    from minisweagent.agents.optimization_agent import OptimizationAgent
    from minisweagent.environments import get_environment_class

    model = model_factory()

    env_kwargs: dict[str, Any] = {
        "cwd": str(repo),
        "env": {"PAGER": "cat", "MANPAGER": "cat", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"},
        "timeout": int(avo_config.get("env_timeout_s", 3600)),
    }
    env = get_environment_class("local")(**env_kwargs)

    cfg: dict[str, Any] = {
        "save_patch": True,
        "step_limit": int(avo_config.get("variation_step_limit", 200)),
        "cost_limit": float(avo_config.get("variation_cost_limit", 0.0)),
        "mode": "yolo",
        "use_strategy_manager": True,
        "use_skills": True,
        "tool_profile": "full",
        "patch_output_dir": str(step_dir),
    }

    test_command = _derive_test_command(output_dir)
    if test_command:
        cfg["test_command"] = test_command

    system_prompt = _load_optimizer_system_prompt()
    if system_prompt:
        cfg["system_template"] = system_prompt

    agent = OptimizationAgent(model, env, **cfg)
    agent.base_repo_path = repo
    agent.log_file = step_dir / "agent.log"
    return agent


def _derive_test_command(output_dir: Path) -> str | None:
    """Reuse GEAK's COMMANDMENT → verbatim test command derivation."""
    commandment = Path(output_dir) / "COMMANDMENT.md"
    if not commandment.exists():
        return None
    try:
        from minisweagent.run.dispatch import _commandment_test_command

        return _commandment_test_command(str(commandment))
    except Exception as exc:  # noqa: BLE001
        logger.warning("variation step: failed to derive test command from COMMANDMENT: %s", exc)
        return None


def _load_optimizer_system_prompt() -> str | None:
    """Reuse the general-kernel-optimization subagent's system prompt."""
    try:
        from minisweagent.subagents import SubAgentRegistry

        registry = SubAgentRegistry()
        descriptor = registry.get("general-kernel-optimization")
        if descriptor is None:
            return None
        return registry.load_system_prompt(descriptor)
    except Exception as exc:  # noqa: BLE001
        logger.debug("variation step: could not load optimizer system prompt: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------


def _collect_result(step_dir: Path, step_index: int, strategy: str | None) -> VariationResult:
    """Scan ``patch_*_test.txt`` files for parsed speedups and correctness."""
    from minisweagent.memory.working_notebook import parse_speedup_report

    result = VariationResult(step_index=step_index, step_dir=step_dir, strategy=strategy)

    test_files = sorted(step_dir.glob("patch_*_test.txt"))
    for test_file in test_files:
        # patch_0 is the unmodified baseline; never a commit candidate.
        if test_file.name.startswith("patch_0_"):
            continue
        try:
            text = test_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        parsed = parse_speedup_report(text)
        speedup = parsed.get("overall_speedup")
        correctness = _correctness_passed(text)
        patch_path = test_file.with_name(test_file.name.replace("_test.txt", ".patch"))
        ph = None
        if patch_path.exists():
            try:
                ph = patch_hash(patch_path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                ph = None

        result.attempts.append(
            AttemptRecord(
                strategy=strategy,
                returncode=0 if correctness else 1,
                correctness_passed=correctness,
                verified_speedup=speedup,
                patch_hash=ph,
                ts=time.time(),
            )
        )

        if correctness and speedup is not None and speedup > (result.best_speedup or 0.0):
            result.best_speedup = speedup
            result.best_correct = True
            result.best_patch_path = patch_path if patch_path.exists() else None

    return result


def _correctness_passed(text: str) -> bool:
    """Heuristic correctness detection from a save_and_test log."""
    low = text.lower()
    positive = any(m in low for m in ("correctness: pass", "correctness passed", "all tests passed", "✓ correctness"))
    negative = any(m in low for m in ("correctness: fail", "correctness failed", "assertionerror", "traceback (most recent"))
    if negative and not positive:
        return False
    return positive


def _step_index_from_dir(step_dir: Path) -> int:
    name = step_dir.name
    digits = "".join(ch for ch in name if ch.isdigit())
    try:
        return int(digits) if digits else 0
    except ValueError:
        return 0
