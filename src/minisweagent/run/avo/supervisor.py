"""Supervisor — the LLM re-planning half of the two-layer AVO supervisor.

Responsibilities (see ``docs/developer/avo_design.md`` §8):

- :func:`build_bundle`     — assemble a read-only context bundle (lineage,
                             failures, strategy state, profile bottleneck) for
                             the supervisor to reason over.
- :func:`run_supervisor`   — obtain a JSON *directive*. Prefers the
                             ``avo-supervisor`` subagent prompt via a single LLM
                             query; falls back to a deterministic strategy
                             taxonomy when no model is available or the LLM
                             output cannot be parsed. This fallback is what
                             keeps a multi-day run progressing even if the
                             supervisor LLM misbehaves.
- :func:`apply_directive`  — execute the directive on GEAK's existing
                             ``strategy_manager`` state machine (mark failed,
                             add new strategies) and update ``direction.json``.

The supervisor never edits kernels or runs the GPU; it only proposes
directions. The controller executes its decisions.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from minisweagent import get_repo_root
from minisweagent.run.avo.lineage_store import LineageStore
from minisweagent.run.avo.stagnation import StagnationSignal

logger = logging.getLogger(__name__)

# Deterministic fallback rotation — generic, language-agnostic optimization
# directions used when the LLM supervisor is unavailable or unparseable.
_FALLBACK_TAXONOMY: list[dict[str, str]] = [
    {"name": "memory_coalescing", "expected": "improve global memory access pattern"},
    {"name": "vectorized_load_store", "expected": "reduce memory transactions"},
    {"name": "shared_memory_tiling", "expected": "reduce redundant global loads"},
    {"name": "loop_unrolling", "expected": "expose more ILP"},
    {"name": "occupancy_tuning", "expected": "rebalance registers / block size"},
    {"name": "pipeline_overlap", "expected": "hide latency by overlapping stages"},
    {"name": "warp_specialization", "expected": "split warp roles to overlap work"},
    {"name": "reduce_synchronization", "expected": "remove unnecessary barriers/fences"},
]


def build_bundle(signal: StagnationSignal, lineage: LineageStore, step_dir: Path, output_dir: Path) -> str:
    """Build the read-only stagnation bundle (JSON string) for the supervisor."""
    bundle: dict[str, Any] = {
        "stagnation": {"reason": signal.reason, "counters": signal.counters},
        "lineage_summary": lineage.summary(last_n=8),
        "current_direction": lineage.current_direction(),
        "best_speedup": lineage.best_speedup,
        "recent_attempts": _recent_attempts(lineage, limit=10),
        # P-mem-1: read the run-wide strategy file (same path the variation agents
        # write to), not a per-step dir, so the supervisor sees the real
        # tried/failed history.
        "strategy_state": _read_strategy_state(output_dir),
        "profile_bottleneck": _read_profile_bottleneck(output_dir),
    }
    return json.dumps(bundle, indent=2, default=str)


def run_supervisor(bundle: str, avo_config: dict, *, model: Any = None) -> dict:
    """Return a directive dict. LLM-first, deterministic-fallback."""
    if model is not None:
        directive = _run_llm_supervisor(bundle, model)
        if directive is not None:
            return directive
        logger.warning("supervisor: LLM path failed/unparseable; using fallback taxonomy.")
    return _fallback_directive(bundle)


def apply_directive(
    directive: dict,
    lineage: LineageStore,
    strategy_file: Path,
    *,
    supervisor_cycle: int,
    repo: Path | None = None,
) -> None:
    """Execute a supervisor directive against GEAK's strategy_manager + lineage."""
    from minisweagent.tools.strategy_manager import StrategyManager

    diagnosis = str(directive.get("diagnosis", "")).strip()
    mark_failed = directive.get("mark_failed") or []
    new_strategies = directive.get("new_strategies") or []
    backtrack_to = directive.get("backtrack_to_id")

    manager: StrategyManager | None = None
    if strategy_file.exists():
        manager = StrategyManager(str(strategy_file))

    if manager is not None:
        _mark_failed_by_name(manager, mark_failed, diagnosis)
        for strat in new_strategies:
            try:
                manager.add_strategy(
                    name=str(strat.get("name", "unnamed")),
                    description=str(strat.get("name", "unnamed")),
                    expected=str(strat.get("expected", "")),
                    target=strat.get("priority"),
                )
            except Exception as exc:  # noqa: BLE001 — strategy I/O is best-effort
                logger.warning("supervisor: add_strategy failed for %r: %s", strat, exc)
        try:
            manager.add_note(f"[avo-supervisor] {diagnosis}")
        except Exception:  # noqa: BLE001
            logger.debug("supervisor: add_note failed", exc_info=True)

    # Pick the next direction: first proposed new strategy, else current.
    next_strategy = ""
    if new_strategies:
        next_strategy = str(new_strategies[0].get("name", ""))
    lineage.set_direction(next_strategy, assigned_by="supervisor", supervisor_cycle=supervisor_cycle)

    # Optional backtrack to an earlier lineage node (P2): move the active-best
    # pointer and, if we have the repo, reset the worktree to that version.
    if backtrack_to:
        logger.info("supervisor: directive requests backtrack to %s", backtrack_to)
        if lineage.set_best_pointer(str(backtrack_to)) and repo is not None:
            lineage.reset_worktree_to(repo, str(backtrack_to))

    _log_supervisor(lineage, directive, supervisor_cycle)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------


def _run_llm_supervisor(bundle: str, model: Any) -> dict | None:
    """Single-shot LLM query using the avo-supervisor system prompt."""
    system_prompt = _load_supervisor_system_prompt()
    if not system_prompt:
        return None
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Stagnation bundle:\n{bundle}\n\nReturn ONLY the JSON directive."},
    ]
    try:
        response = model.query(messages)
        content = response.get("content", "") if isinstance(response, dict) else str(response)
    except Exception as exc:  # noqa: BLE001 — never let supervisor errors stop the loop
        logger.warning("supervisor: model.query failed: %s", exc)
        return None
    return _parse_directive(content)


def _load_supervisor_system_prompt() -> str | None:
    path = get_repo_root() / "subagents" / "avo-supervisor" / "SYSTEM_PROMPT.md"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_directive(content: str) -> dict | None:
    """Extract a JSON directive from an LLM response (tolerant of code fences)."""
    if not content:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    raw = fenced.group(1) if fenced else None
    if raw is None:
        brace = re.search(r"\{.*\}", content, re.DOTALL)
        raw = brace.group(0) if brace else None
    if raw is None:
        return None
    try:
        directive = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return directive if isinstance(directive, dict) else None


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


def _fallback_directive(bundle: str) -> dict:
    """Pick the next untried taxonomy direction based on the strategy state."""
    tried: set[str] = set()
    try:
        data = json.loads(bundle)
        for item in data.get("strategy_state", {}).get("failed", []):
            tried.add(str(item).lower())
        cur = data.get("current_direction", {}).get("strategy")
        if cur:
            tried.add(str(cur).lower())
    except (json.JSONDecodeError, AttributeError):
        pass

    nxt = next((t for t in _FALLBACK_TAXONOMY if t["name"] not in tried), _FALLBACK_TAXONOMY[0])
    return {
        "diagnosis": "fallback: LLM supervisor unavailable; rotating to next generic direction.",
        "mark_failed": [],
        "new_strategies": [{"name": nxt["name"], "priority": "high", "expected": nxt["expected"]}],
        "backtrack_to_id": None,
    }


# ---------------------------------------------------------------------------
# Bundle helpers
# ---------------------------------------------------------------------------


def _recent_attempts(lineage: LineageStore, limit: int) -> list[dict]:
    path = lineage.attempts_path
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[-limit:]:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        pass
    return rows


def _read_strategy_state(base_dir: Path) -> dict:
    strat_file = base_dir / ".optimization_strategies.md"
    if not strat_file.exists():
        return {"successful": [], "failed": [], "pending": []}
    try:
        from minisweagent.tools.strategy_manager import StrategyManager

        manager = StrategyManager(str(strat_file))
        buckets: dict[str, list[str]] = {"successful": [], "failed": [], "pending": [], "exploring": []}
        for _idx, strat in manager.list_strategies():
            buckets.setdefault(strat.status.value, []).append(strat.name)
        return buckets
    except Exception:  # noqa: BLE001
        return {"successful": [], "failed": [], "pending": []}


def _read_profile_bottleneck(output_dir: Path) -> str:
    profile = Path(output_dir) / "profile.json"
    if not profile.exists():
        return "unknown (no profile.json)"
    try:
        data = json.loads(profile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown (unreadable profile.json)"
    for key in ("bottleneck", "bottleneck_type", "summary", "limiter"):
        if key in data:
            return str(data[key])
    return "unknown (no bottleneck field)"


def _mark_failed_by_name(manager: Any, names: list, diagnosis: str) -> None:
    if not names:
        return
    lowered = {str(n).lower() for n in names}
    try:
        for idx, strat in manager.list_strategies():
            if strat.name.lower() in lowered:
                manager.mark_status(idx, "failed", result="superseded by supervisor", details=diagnosis[:200])
    except Exception as exc:  # noqa: BLE001
        logger.warning("supervisor: mark_failed_by_name failed: %s", exc)


def _log_supervisor(lineage: LineageStore, directive: dict, supervisor_cycle: int) -> None:
    log_path = lineage.state_dir / "supervisor_log.jsonl"
    try:
        import time

        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"ts": time.time(), "cycle": supervisor_cycle, "directive": directive},
                    default=str,
                )
                + "\n"
            )
    except OSError:
        logger.debug("supervisor: failed to write supervisor_log.jsonl", exc_info=True)
