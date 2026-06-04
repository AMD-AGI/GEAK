"""One AVO variation step = one ``OptimizationAgent`` run in the repo worktree.

This module wraps GEAK's existing ``OptimizationAgent`` without subclassing it.
The AVO contract is delivered through (a) a task-body prefix and (b) forced
skill injection. The agent's own ``save_and_test`` outputs (``patch_*.patch`` /
``patch_*_test.txt`` under ``patch_output_dir``) are parsed back into a
:class:`VariationResult`.
"""

from __future__ import annotations

import functools
import logging
import re
import subprocess
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


# --- P-mem-3 (option C): continuous-memory capture + injection -------------

_RATIONALE_MAX = 800
_RAW_TAIL_MAX = 1800


def _capture_agent_memory(agent) -> tuple[str, str]:
    """Extract (rationale, raw_tail) from a finished agent's message history.

    rationale = the agent's last substantive assistant message (its own account
    of what it did / why). raw_tail = a short verbatim concatenation of the last
    few assistant/tool turns. Both truncated to keep later prompts bounded.
    """
    msgs = getattr(agent, "messages", None) or []
    rationale = ""
    for m in reversed(msgs):
        if m.get("role") == "assistant" and str(m.get("content") or "").strip():
            rationale = str(m["content"]).strip()[:_RATIONALE_MAX]
            break
    tail_parts: list[str] = []
    budget = _RAW_TAIL_MAX
    for m in reversed(msgs):
        role = m.get("role")
        if role not in ("assistant", "tool"):
            continue
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        chunk = f"[{role}] {content}"
        chunk = chunk[: max(0, budget)]
        tail_parts.append(chunk)
        budget -= len(chunk)
        if budget <= 0:
            break
    raw_tail = "\n".join(reversed(tail_parts))
    return rationale, raw_tail


def _fmt_profiling_delta(cur: dict, parent: dict) -> str:
    """One-line profiler delta vs the parent version (best-effort, cross-backend)."""
    if not cur:
        return ""
    bits = []
    for k, v in sorted(cur.items()):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if k in parent:
            try:
                pv = float(parent[k])
                bits.append(f"{k} {pv:.3g}→{fv:.3g}")
                continue
            except (TypeError, ValueError):
                pass
        bits.append(f"{k}={fv:.3g}")
    return ("bottleneck Δ: " + ", ".join(bits[:5])) if bits else ""


def build_evolution_log(output_dir: Path, *, k_recent: int = 2, max_versions: int = 8) -> str | None:
    """Build a bounded causal evolution log from per-step memory entries (P-mem-3).

    Recent ``k_recent`` steps are shown with their verbatim raw tail; older steps
    collapse to structured one-liners (strategy → speedup, committed?, profiling
    delta vs the prior entry, rationale/failure). This carries the *causal*
    signal of a continuous session (what changed, why, how the bottleneck moved)
    without keeping raw history unbounded.
    """
    import json as _json

    log_dir = Path(output_dir) / "avo_state" / "evolution_log"
    if not log_dir.is_dir():
        return None
    entries = []
    for p in sorted(log_dir.glob("step_*.json")):
        try:
            entries.append(_json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    if not entries:
        return None
    entries.sort(key=lambda e: e.get("step_index", 0))

    blocks = ["## Evolution log (causal history — what changed, why, and the effect)"]
    older = entries[:-k_recent] if k_recent > 0 else entries
    recent = entries[-k_recent:] if k_recent > 0 else []

    older = older[-max_versions:]
    prev_prof: dict = {}
    for e in older:
        sp = e.get("verified_speedup")
        sp_s = f"{sp:.4f}x" if isinstance(sp, (int, float)) else "n/a"
        flag = "committed" if e.get("committed") else "rejected"
        delta = _fmt_profiling_delta(e.get("profiling") or {}, prev_prof)
        prev_prof = e.get("profiling") or prev_prof
        note = (e.get("failure") or e.get("rationale") or "").strip().replace("\n", " ")[:140]
        line = f"- step {e.get('step_index')}: {e.get('strategy') or '?'} → {sp_s} [{flag}]"
        if delta:
            line += f" | {delta}"
        if note:
            line += f" | {note}"
        blocks.append(line)

    for e in recent:
        sp = e.get("verified_speedup")
        sp_s = f"{sp:.4f}x" if isinstance(sp, (int, float)) else "n/a"
        flag = "committed" if e.get("committed") else "rejected"
        blocks.append(
            f"### step {e.get('step_index')} (recent): {e.get('strategy') or '?'} → {sp_s} [{flag}]"
        )
        if e.get("raw_tail"):
            blocks.append(f"```\n{e['raw_tail']}\n```")
    return "\n".join(blocks)


def write_evolution_entry(output_dir: Path, result, committed: bool) -> None:
    """Persist one step's memory entry for the evolution log (called by controller)."""
    import json as _json

    log_dir = Path(output_dir) / "avo_state" / "evolution_log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "step_index": result.step_index,
            "strategy": result.strategy,
            "committed": bool(committed),
            "verified_speedup": result.best_speedup,
            "per_shape": result.per_shape_speedups,
            "profiling": result.profiling,
            "rationale": result.rationale,
            "failure": None if result.best_correct else f"no committable candidate ({result.exit_status})",
            "raw_tail": result.raw_tail,
        }
        (log_dir / f"step_{result.step_index:04d}.json").write_text(
            _json.dumps(entry, indent=2, default=str), encoding="utf-8"
        )
    except OSError as exc:
        logger.debug("evolution-log write failed (non-fatal): %s", exc)


@functools.lru_cache(maxsize=1)
def hardware_summary() -> str | None:
    """Best-effort one-line target-hardware summary, injected into every step (D1).

    Kernel decisions (tiling, occupancy, split-K, WMMA/MFMA) depend on the real
    GPU. Because AVO resets the agent each step, the hardware facts must be
    re-grounded in every prompt rather than relying on the agent to re-probe.
    Tries ROCm (rocminfo) then NVIDIA (nvidia-smi); returns ``None`` if neither
    is available, leaving the prompt unchanged.
    """
    # AMD / ROCm
    try:
        from minisweagent.run.utils.gpu_arch import detect_gpu_arch

        arch = detect_gpu_arch()
        if arch:
            return f"## Target hardware\n- GPU arch: {arch} (AMD ROCm). Ground tiling/occupancy/MFMA-WMMA choices in this arch."
    except Exception:  # noqa: BLE001
        pass
    # NVIDIA
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        line = out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 and out.stdout.strip() else ""
        if line:
            return f"## Target hardware\n- GPU: {line} (NVIDIA). Ground tiling/occupancy/tensor-core choices in this SM/arch."
    except Exception:  # noqa: BLE001
        pass
    return None


def _read_memory_summary(notebook_root: Path | None) -> str | None:
    """Compact summary of prior attempts in this run (cross-step memory, P-mem-2).

    Reuses GEAK's ``WorkingNotebook.summarize_dir``; returns ``None`` on the first
    step (no events yet) so the prompt is unchanged when there is no memory.
    """
    if notebook_root is None:
        return None
    try:
        from minisweagent.memory.working_notebook import WorkingNotebook

        summary = WorkingNotebook.summarize_dir(notebook_root)
        return summary or None
    except Exception as exc:  # noqa: BLE001 — memory is best-effort, never fatal
        logger.debug("variation step: memory summary unavailable: %s", exc)
        return None


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


# Delayed-profiling stage notes (CuTeGen): withhold profiler-driven micro-tuning
# until the kernel structure is sound, to avoid premature convergence to a poor
# local optimum.
_STRUCTURAL_NOTE = (
    "## Optimization stage: STRUCTURAL (profiling withheld)\n"
    "Establish a strong overall structure FIRST — tiling / work decomposition, memory-hierarchy use, "
    "and data movement / pipelining. Do NOT do profiler-driven micro-tuning (e.g. tile-size sweeps) this "
    "step, and do not lean on profile.json yet. Profiling feedback is introduced in later steps once the "
    "structure is sound (this avoids premature convergence to a poor local optimum)."
)
_PROFILING_NOTE = (
    "## Optimization stage: PROFILING-GUIDED\n"
    "The kernel structure is established. You may now use profile.json / the profile_kernel tool for "
    "targeted low-level tuning (occupancy, tile sizes, bank conflicts, memory fences)."
)


def compose_task(
    base_task: str,
    lineage,
    direction: dict[str, Any],
    memory_summary: str | None = None,
    exemplar: str | None = None,
    profiling_enabled: bool = True,
    hardware: str | None = None,
    lineage_context: str | None = None,
    evolution_log: str | None = None,
) -> str:
    """Prefix the AVO contract (+ hardware + stage + exemplar + lineage + evolution log + memory)."""
    contract = _AVO_CONTRACT.format(
        best_summary=lineage.summary(last_n=5),
        direction=direction.get("strategy") or "(none assigned — pick the highest-priority pending strategy)",
    )
    parts = [contract]
    if hardware:
        parts.append(hardware)
    parts.append(_PROFILING_NOTE if profiling_enabled else _STRUCTURAL_NOTE)
    if exemplar:
        parts.append(exemplar)
    if lineage_context:
        parts.append(lineage_context)
    if evolution_log:
        parts.append(evolution_log)
    if memory_summary:
        parts.append(f"## Cross-step memory (prior attempts in this run)\n{memory_summary}")
    parts.append(base_task)
    return "\n\n".join(parts)


# Max chars of the best-version diff injected as an exemplar (keeps the prompt
# bounded for long runs — see design doc §15.1).
_EXEMPLAR_DIFF_MAX = 4000
# Smaller cap per version when several prior implementations are shown together.
_LINEAGE_DIFF_MAX = 1500


def _fmt_per_shape(per_shape: dict, limit: int = 4) -> str:
    if not per_shape:
        return ""
    items = sorted(per_shape.items(), key=lambda kv: kv[1])[:limit]  # worst-first (where to improve)
    return " | per-shape: " + ", ".join(f"{s}={v:.3f}x" for s, v in items)


def build_lineage_context(lineage, k: int = 3) -> str | None:
    """Inject several prior implementations + their verified/per-shape scores (#2).

    AVO paper §3.2: the agent examines *multiple* prior implementations in P_t and
    compares their characteristics within a step. The current best is already
    applied in the worktree (see ``build_best_exemplar``); this surfaces the next
    best / most-diverse committed versions as *alternatives to compare against*,
    each with its verified speedup, per-shape profile, and a truncated diff. Full
    versions are retrievable via ``git show avo-v{id}``. Bounded for long runs.
    """
    if k <= 0:
        return None
    best_id = getattr(lineage, "best_id", None)
    nodes = lineage.top_k(k, exclude_baseline=True, exclude_id=best_id)
    if not nodes:
        return None
    blocks = [
        "## Prior lineage implementations (alternatives to compare — NOT your current base)",
        "These are other committed versions in P_t. Compare their approaches/per-shape behavior to decide your "
        "next move. Full source of any version: `git show avo-v{id}`.",
    ]
    for n in nodes:
        diff = ""
        if n.patch:
            p = Path(n.patch)
            if p.exists():
                try:
                    diff = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    diff = ""
        if len(diff) > _LINEAGE_DIFF_MAX:
            diff = diff[:_LINEAGE_DIFF_MAX] + "\n... [diff truncated; use `git show` for full] ...\n"
        header = f"### {n.id} — {n.speedup:.4f}x" + (f" via {n.strategy}" if n.strategy else "")
        header += _fmt_per_shape(n.per_shape)
        body = f"```diff\n{diff}\n```" if diff else "(patch unavailable; use `git show avo-" + str(n.id) + "`)"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def build_best_exemplar(lineage) -> str | None:
    """Build a 'current best implementation' exemplar from the lineage (P-B).

    Mirrors Kernel-Smith's practice of injecting the top-performing program +
    its structured metrics. The worktree is already reset to this version, so
    this shows the agent *what produced* the current best and its verified
    speedup, to build on rather than re-derive. Returns ``None`` for the
    baseline (v0, no patch).
    """
    node = getattr(lineage, "best_node", None)
    if node is None or not getattr(node, "patch", None):
        return None
    patch_path = Path(node.patch)
    if not patch_path.exists():
        return None
    try:
        diff = patch_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if len(diff) > _EXEMPLAR_DIFF_MAX:
        diff = diff[:_EXEMPLAR_DIFF_MAX] + "\n... [diff truncated] ...\n"
    return (
        f"## Current best implementation ({node.id}, {node.speedup:.4f}x"
        f"{f' via {node.strategy}' if node.strategy else ''})\n"
        "You start from this version (already applied in your worktree). It was reached by the diff below; "
        "build on it — do not re-derive from the baseline.\n"
        f"```diff\n{diff}\n```"
    )


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
    notebook_root: Path | None = None,
    profiling_enabled: bool = True,
) -> VariationResult:
    """Build + run one OptimizationAgent and return a structured result."""
    step_dir.mkdir(parents=True, exist_ok=True)
    step_index = _step_index_from_dir(step_dir)
    t0 = time.monotonic()

    # Patches + best_results.json land in GEAK's canonical results/round_N/<worker>/
    # layout so the controller can reuse ``evaluate_round_best`` for verified,
    # per-shape-geomean scoring (P0). Logs/strategies stay under ``step_dir``.
    # ESCALATE rescue workers pass an explicit worker dir via ``_escalate_patch_dir``.
    escalate_dir = avo_config.get("_escalate_patch_dir")
    if escalate_dir:
        worker_dir = Path(escalate_dir)
    else:
        worker_dir = output_dir / "results" / f"round_{step_index}" / "avo-worker"
    worker_dir.mkdir(parents=True, exist_ok=True)

    memory_summary = _read_memory_summary(notebook_root)
    exemplar = build_best_exemplar(lineage) if avo_config.get("inject_best_exemplar", True) else None
    lineage_context = build_lineage_context(lineage, int(avo_config.get("lineage_context_k", 3)))
    evolution_log = (
        build_evolution_log(
            output_dir,
            k_recent=int(avo_config.get("evolution_log_recent", 2)),
            max_versions=int(avo_config.get("evolution_log_max_versions", 8)),
        )
        if avo_config.get("evolution_log_enabled", True)
        else None
    )
    task_body = compose_task(
        base_task,
        lineage,
        direction,
        memory_summary=memory_summary,
        exemplar=exemplar,
        profiling_enabled=profiling_enabled,
        hardware=hardware_summary(),
        lineage_context=lineage_context,
        evolution_log=evolution_log,
    )
    if nudge:
        task_body += f"\n\n## Supervisor nudge\n{nudge}\n"
    task_body = inject_skill_bodies(task_body, select_skills(lineage.language, base_task))

    exit_status = "NotRun"
    agent = None
    try:
        agent = _build_agent(repo, step_dir, worker_dir, output_dir, avo_config, model_factory)
        exit_status, _msg = agent.run(task_body)
    except Exception as exc:  # noqa: BLE001 — a failed step must not kill the loop
        logger.exception("variation step %d crashed: %s", step_index, exc)
        exit_status = type(exc).__name__

    # Light parse of the worker dir for cycle/patch-hash signal. The controller
    # may overwrite best_speedup/best_correct with the independently-verified
    # FULL_BENCHMARK geomean (see controller._apply_verified_score).
    result = _collect_result(worker_dir, step_index, direction.get("strategy"))
    result.step_dir = step_dir
    result.exit_status = exit_status
    # Capture the agent's rationale + a short raw tail for the evolution log (P-mem-3).
    if agent is not None:
        result.rationale, result.raw_tail = _capture_agent_memory(agent)
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


def _build_agent(repo: Path, step_dir: Path, worker_dir: Path, output_dir: Path, avo_config: dict, model_factory):
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
        "patch_output_dir": str(worker_dir),
        # P-mem-1: pin the strategy file to a single run-wide location so the
        # "tried / failed" strategy memory persists across variation steps and is
        # shared with the supervisor. An absolute path makes OptimizationAgent's
        # _get_strategy_file ignore the per-step patch_output_dir. Kept outside the
        # repo worktree so it never leaks into kernel patches.
        "strategy_file_path": str((output_dir / ".optimization_strategies.md").resolve()),
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
    """Parse a step index from a step dir name.

    Uses the **first** contiguous digit group, not every digit concatenated, so
    multi-number names stay sane: ``variation_0001`` → 1, and an ESCALATE worker
    dir ``escalate_9001_0`` → 9001 (the rescue round) rather than ``90010``.
    """
    match = re.search(r"\d+", step_dir.name)
    try:
        return int(match.group(0)) if match else 0
    except ValueError:
        return 0
