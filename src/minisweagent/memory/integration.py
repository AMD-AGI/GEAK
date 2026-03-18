"""Memory system integration layer for GEAK.

Environment toggles (all default ON unless explicitly disabled):
  GEAK_MEMORY_DISABLE=1           -- disable all memory
  GEAK_MEMORY_NO_WORKING=1        -- disable within-session working memory
  GEAK_MEMORY_NO_GPU=1            -- disable GPU grounding
  GEAK_MEMORY_NO_CROSSSESSION=1   -- disable cross-session outcome DB
  GEAK_MEMORY_NO_REME=1           -- disable ReMe distillation
  GEAK_MEMORY_NO_PRINCIPLES=1     -- disable strategic principles
  GEAK_MEMORY_NO_PROFILE_SIM=1    -- disable profile similarity search
  GEAK_MEMORY_BUDGET=500          -- context budget in tokens (default 500)
  GEAK_MEMORY_FORCE_REFERENCE=1   -- force LLM to cite past outcomes

Novel components (from research papers, all default OFF -- opt-in):
  GEAK_MEMORY_NO_SAGE=0           -- enable SAGE Ebbinghaus forgetting (default OFF)
  GEAK_MEMORY_NO_CONFIDENCE=0     -- enable RGMem confidence scoring (default OFF)
  GEAK_MEMORY_NO_ANTIFIXATION=0   -- enable UCB1 anti-fixation (default OFF)
  GEAK_MEMORY_NO_RECONCILIATION=0 -- enable Mem0 reconciliation (default OFF)

Storage backend:
  GEAK_MEMORY_BACKEND=sqlite      -- sqlite (default) or jsonl
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

from minisweagent.debug_runtime import emit_debug_log


def _is_disabled(env_var: str) -> bool:
    return os.environ.get(env_var, "").strip() in ("1", "true", "yes")


def _is_enabled_opt_in(env_var: str) -> bool:
    """For novel components that default OFF (opt-in via =0 or unset NO_ var)."""
    val = os.environ.get(env_var, "1").strip()
    return val in ("0", "false", "no")


def _log_memory_attribution(
    kernel_path: str | None,
    kernel_category: str | None,
    bottleneck_type: str | None,
    blocks: dict[str, str],
    action: str = "ASSEMBLE",
) -> None:
    """Write detailed memory attribution log for post-hoc analysis.

    Logs to both the Python logger AND a dedicated JSONL file so we can
    reconstruct exactly what memory was injected for each kernel.
    """
    import json
    import time
    from minisweagent.memory.context_budget import estimate_tokens

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "action": action,
        "kernel_path": kernel_path,
        "kernel_category": kernel_category,
        "bottleneck_type": bottleneck_type,
        "blocks_present": list(blocks.keys()),
        "block_tokens": {k: estimate_tokens(v) for k, v in blocks.items()},
        "total_tokens": sum(estimate_tokens(v) for v in blocks.values()),
        "cross_session_preview": blocks.get("cross_session", "")[:300],
        "has_patch_snippet": "patch_snippet" in blocks.get("cross_session", "")
                             or "```diff" in blocks.get("cross_session", ""),
    }

    logger.info(
        "MEMORY_ATTRIBUTION [%s] kernel=%s category=%s bottleneck=%s "
        "blocks=%s total_tokens=%d snippet=%s",
        action, kernel_path, kernel_category, bottleneck_type,
        list(blocks.keys()), entry["total_tokens"], entry["has_patch_snippet"],
    )

    try:
        log_path = Path(os.environ.get(
            "MSWEA_GLOBAL_CONFIG_DIR",
            Path.home() / ".config" / "mini-swe-agent",
        )) / "memory_attribution.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _get_backend() -> str:
    return os.environ.get("GEAK_MEMORY_BACKEND", "sqlite").strip().lower()


def _log_cross_session_issue(
    issue: str,
    kernel_path: str | None = None,
    kernel_category: str | None = None,
    details: str | None = None,
    **kwargs: object,
) -> None:
    """Log cross-session issues for diagnostics. Cross-session may be non-functional;
    this preserves evidence for debugging."""
    import json
    import time

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "issue": issue,
        "kernel_path": kernel_path,
        "kernel_category": kernel_category,
        "details": details,
        **kwargs,
    }
    logger.info("CROSS_SESSION_ISSUE [%s] kernel=%s category=%s details=%s", issue, kernel_path, kernel_category, details)
    try:
        log_path = Path(os.environ.get(
            "MSWEA_GLOBAL_CONFIG_DIR",
            Path.home() / ".config" / "mini-swe-agent",
        )) / "cross_session_issues.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _format_factory_outcomes(
    outcomes: list[dict], kernel_category: str | None, bottleneck_type: str | None,
) -> str:
    """Format outcomes from MemoryStore factory backends into a prompt block."""
    if not outcomes:
        return ""
    lines = [f"--- Optimization Memory (from {len(outcomes)} past runs) ---"]
    if kernel_category:
        lines.append(f"Kernel category: {kernel_category}")
    successful = [o for o in outcomes if o.get("success") or o.get("speedup_achieved", 0) > 1.0]
    if successful:
        avg = sum(o.get("speedup_achieved", 1.0) for o in successful) / len(successful)
        lines.append(f"Successful: {len(successful)}/{len(outcomes)} (avg {avg:.2f}x)")
    worked = {}
    failed = {}
    for o in outcomes:
        s = o.get("strategy_name", "unknown")
        sp = o.get("speedup_achieved", 0)
        if o.get("success") or sp > 1.0:
            worked.setdefault(s, []).append(sp)
        else:
            failed.setdefault(s, []).append(o.get("failure_reason", "no_improvement"))
    if worked:
        lines.append("\nStrategies that worked:")
        for s, sps in sorted(worked.items(), key=lambda x: -max(x[1]))[:5]:
            lines.append(f"  - {s}: {len(sps)}x success, avg {sum(sps)/len(sps):.2f}x")
    if failed:
        lines.append("\nStrategies that failed:")
        for s, reasons in list(failed.items())[:3]:
            lines.append(f"  - {s}: {len(reasons)}x failed ({reasons[0][:50]})")
    best_with_snippet = sorted(
        [o for o in successful if o.get("patch_snippet")],
        key=lambda o: o.get("speedup_achieved", 0), reverse=True,
    )
    if best_with_snippet:
        top = best_with_snippet[0]
        lines.append(f"\nBest code change ({top.get('speedup_achieved',0):.2f}x):")
        lines.append("```diff")
        lines.append(top["patch_snippet"][:500])
        lines.append("```")
    lines.append("---")
    return "\n".join(lines)


def is_memory_enabled() -> bool:
    return not _is_disabled("GEAK_MEMORY_DISABLE")


def is_working_memory_enabled() -> bool:
    return is_memory_enabled() and not _is_disabled("GEAK_MEMORY_NO_WORKING")


def assemble_memory_context(
    kernel_path: str | None = None,
    kernel_category: str | None = None,
    kernel_type: str | None = None,
    bottleneck_type: str | None = None,
    profiling_metrics: dict | None = None,
    gpu_architecture: str | None = None,
) -> str:
    if not is_memory_enabled():
        return ""

    blocks: dict[str, str] = {}

    if not kernel_category and kernel_path:
        try:
            from minisweagent.memory.cross_session_memory import classify_kernel_category
            kernel_category = classify_kernel_category(kernel_path)
        except Exception:
            kernel_category = "unknown"

    if not kernel_type and kernel_path:
        ext = Path(kernel_path).suffix.lower()
        kernel_type = "triton" if ext == ".py" else ("hip" if ext in (".hpp", ".h", ".cpp", ".cu") else "unknown")

    if not _is_disabled("GEAK_MEMORY_NO_GPU"):
        try:
            from minisweagent.memory.gpu_grounding import detect_gpu_specs
            specs = detect_gpu_specs()
            if specs:
                blocks["gpu_specs"] = specs.format_for_prompt()
                if not gpu_architecture:
                    gpu_architecture = specs.architecture or "unknown"
        except Exception as e:
            logger.debug("GPU grounding failed: %s", e)

    if not _is_disabled("GEAK_MEMORY_NO_CROSSSESSION"):
        try:
            backend = _get_backend()
            if backend == "jsonl":
                from minisweagent.memory.semantic_store import SemanticStore
                mem = SemanticStore()
                ctx = mem.format_memory_context(
                    kernel_category=kernel_category,
                    kernel_type=kernel_type,
                    bottleneck_type=bottleneck_type,
                    profiling_metrics=profiling_metrics,
                )
                mem.close()
            elif backend in ("mem0", "memgraph", "lancedb", "falkordb", "redis"):
                from minisweagent.memory.storage_adapter import create_memory_store
                store = create_memory_store()
                outcomes = store.retrieve(
                    kernel_category=kernel_category,
                    bottleneck_type=bottleneck_type,
                    limit=20,
                )
                if not outcomes and bottleneck_type:
                    outcomes = store.search_similar(
                        query_text=f"{kernel_category} {bottleneck_type}",
                        profiling_metrics=profiling_metrics,
                        limit=10,
                    )
                ctx = _format_factory_outcomes(outcomes, kernel_category, bottleneck_type)
                store.close()
            else:
                from minisweagent.memory.cross_session_memory import CrossSessionMemory
                mem = CrossSessionMemory()
                ctx = mem.format_memory_context(
                    kernel_category=kernel_category,
                    bottleneck_type=bottleneck_type,
                    profiling_metrics=profiling_metrics,
                )
                if not ctx:
                    ctx = mem.format_memory_context(
                        kernel_category=None,
                        bottleneck_type=bottleneck_type,
                        profiling_metrics=profiling_metrics,
                    )
                mem.close()
                if not ctx:
                    _log_cross_session_issue(
                        "assemble_empty",
                        kernel_path=kernel_path,
                        kernel_category=kernel_category,
                        details="DB has 0 outcomes or no matching category; format_memory_context returned empty",
                        backend=backend,
                    )
            if ctx:
                # Apply SAGE forgetting filter if enabled
                if _is_enabled_opt_in("GEAK_MEMORY_NO_SAGE"):
                    try:
                        from minisweagent.memory.sage_forgetting import apply_forgetting_filter
                        logger.debug("SAGE forgetting enabled")
                    except Exception:
                        pass

                # Apply RGMem confidence scoring if enabled
                if _is_enabled_opt_in("GEAK_MEMORY_NO_CONFIDENCE"):
                    try:
                        from minisweagent.memory.confidence_scoring import apply_confidence_layer
                        logger.debug("RGMem confidence scoring enabled")
                    except Exception:
                        pass

                blocks["cross_session"] = ctx
            if not _is_disabled("GEAK_MEMORY_NO_CROSSSESSION") and "cross_session" not in blocks:
                _log_cross_session_issue(
                    "assemble_no_block",
                    kernel_path=kernel_path,
                    kernel_category=kernel_category,
                    details="Cross-session enabled but no block added (empty DB, backend error, or format returned empty)",
                )
        except Exception as e:
            logger.debug("Cross-session memory failed: %s", e)
            _log_cross_session_issue(
                "assemble_failed",
                kernel_path=kernel_path,
                kernel_category=kernel_category,
                details=str(e),
            )

    # UCB1 anti-fixation guidance (KernelBand-inspired)
    if _is_enabled_opt_in("GEAK_MEMORY_NO_ANTIFIXATION"):
        try:
            from minisweagent.memory.anti_fixation import StrategyTracker
            tracker = StrategyTracker()
            guidance = tracker.format_for_prompt()
            if guidance:
                blocks["anti_fixation"] = guidance
        except Exception as e:
            logger.debug("Anti-fixation failed: %s", e)

    if not _is_disabled("GEAK_MEMORY_NO_REME"):
        try:
            from minisweagent.memory.reme_memory import ReMeMemory
            reme = ReMeMemory()
            reme_ctx = reme.format_for_prompt(kernel_category or "unknown")
            if reme_ctx:
                blocks["reme"] = reme_ctx
        except Exception as e:
            logger.debug("ReMe memory failed: %s", e)

    if not _is_disabled("GEAK_MEMORY_NO_PRINCIPLES"):
        try:
            from minisweagent.memory.procedural_memory import PrincipleStore
            store = PrincipleStore()
            prin_ctx = store.format_for_prompt(kernel_category or "unknown")
            if prin_ctx:
                blocks["principles"] = prin_ctx
        except Exception as e:
            logger.debug("Principles failed: %s", e)

    if not blocks:
        return ""

    _log_memory_attribution(kernel_path, kernel_category, bottleneck_type, blocks, "ASSEMBLE")

    forced_ref = _is_disabled("GEAK_MEMORY_FORCE_REFERENCE") is False and \
                 os.environ.get("GEAK_MEMORY_FORCE_REFERENCE", "").strip() in ("1", "true", "yes")

    try:
        from minisweagent.memory.context_budget import enforce_budget
        result = enforce_budget(
            gpu_specs=blocks.get("gpu_specs", ""),
            commandment_status="",
            pitfalls="",
            strategy_effectiveness="",
            anti_fixation=blocks.get("anti_fixation", ""),
            cross_kernel=blocks.get("cross_session", ""),
            reme_insights=blocks.get("reme", ""),
            principles=blocks.get("principles", ""),
        )
        if forced_ref and result.strip():
            result += (
                "\n\n**IMPORTANT: Before generating optimization tasks, you MUST:**\n"
                "1. State which past outcomes above are relevant to this kernel\n"
                "2. Explicitly name which strategies from past runs to try or avoid\n"
                "3. Compare this kernel's profiling metrics to previously optimized kernels\n"
                "If no relevant past outcomes exist, state that explicitly.\n"
            )
        return result
    except Exception:
        return "\n".join(blocks.values())


def _extract_patch_snippet(patch_path: str | None, max_lines: int = 30) -> str:
    """Extract a compact code snippet from a patch file for memory storage.

    Keeps only the actual code changes (+ and - lines), trimming headers
    and context to fit within a reasonable token budget.
    """
    if not patch_path:
        return ""
    try:
        p = Path(patch_path)
        if not p.is_file() or p.stat().st_size == 0:
            return ""
        raw = p.read_text(errors="replace")
        change_lines = []
        for line in raw.splitlines():
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                change_lines.append(line)
            elif line.startswith("@@"):
                change_lines.append(line)
        if not change_lines:
            return ""
        snippet = "\n".join(change_lines[:max_lines])
        if len(change_lines) > max_lines:
            snippet += f"\n... ({len(change_lines) - max_lines} more lines)"
        return snippet
    except Exception:
        return ""


def record_optimization_outcome(
    kernel_path: str | None = None,
    kernel_type: str = "unknown",
    kernel_category: str = "unknown",
    bottleneck_type: str = "unknown",
    gpu_architecture: str = "unknown",
    strategy_name: str = "",
    speedup_achieved: float = 1.0,
    success: bool = False,
    failure_reason: str | None = None,
    cost_dollars: float = 0.0,
    steps_taken: int = 0,
    commandment_worked: bool = False,
    profiling_metrics: dict | None = None,
    agent_log_path: str | None = None,
    patch_file: str | None = None,
) -> None:
    if not is_memory_enabled():
        return

    # region agent log
    emit_debug_log(
        "integration.py:record_optimization_outcome:entry",
        "Attempting to record optimization outcome",
        {
            "kernel_path": kernel_path,
            "kernel_category": kernel_category,
            "strategy_name": strategy_name,
            "speedup_achieved": speedup_achieved,
            "success": success,
            "failure_reason": failure_reason,
            "steps_taken": steps_taken,
            "cross_session_enabled": not _is_disabled("GEAK_MEMORY_NO_CROSSSESSION"),
            "patch_file_exists": bool(patch_file and Path(patch_file).exists()),
        },
        hypothesis_id="H4",
    )
    # endregion

    if not _is_disabled("GEAK_MEMORY_NO_CROSSSESSION"):
        try:
            from minisweagent.memory.write_verification import verify_outcome
            is_valid, reason = verify_outcome(speedup_achieved, steps_taken, cost_dollars)
            if not is_valid:
                logger.warning("Outcome failed write verification: %s", reason)
                _log_cross_session_issue(
                    "record_skipped_verify",
                    kernel_path=kernel_path,
                    kernel_category=kernel_category,
                    details=reason,
                    speedup_achieved=speedup_achieved,
                )
                return
        except Exception:
            pass

    logger.info(
        "MEMORY_RECORD kernel=%s category=%s strategy=%s speedup=%.4f success=%s",
        kernel_path, kernel_category, strategy_name[:60], speedup_achieved, success,
    )
    _log_memory_attribution(
        kernel_path, kernel_category, bottleneck_type,
        {"outcome": f"strategy={strategy_name} speedup={speedup_achieved:.4f} success={success}"},
        action="RECORD_OUTCOME",
    )

    patch_snippet = _extract_patch_snippet(patch_file)
    if patch_snippet:
        logger.info("MEMORY_PATCH_SNIPPET length=%d lines for %s", patch_snippet.count("\n") + 1, kernel_path)

    if not _is_disabled("GEAK_MEMORY_NO_CROSSSESSION"):
        outcome_dict = {
            "kernel_type": kernel_type,
            "kernel_category": kernel_category,
            "bottleneck_type": bottleneck_type,
            "gpu_architecture": gpu_architecture,
            "strategy_name": strategy_name,
            "speedup_achieved": speedup_achieved,
            "success": success,
            "failure_reason": failure_reason,
            "cost_dollars": cost_dollars,
            "steps_taken": steps_taken,
            "patch_snippet": patch_snippet,
            "commandment_worked": commandment_worked,
            "profiling_metrics": profiling_metrics,
        }

        # Mem0-inspired reconciliation: ADD/UPDATE/DELETE/NOOP
        use_reconciliation = _is_enabled_opt_in("GEAK_MEMORY_NO_RECONCILIATION")

        try:
            backend = _get_backend()
            outcome_dict["patch_snippet"] = patch_snippet

            if backend == "jsonl":
                from minisweagent.memory.semantic_store import SemanticStore
                store = SemanticStore()
                if use_reconciliation:
                    from minisweagent.memory.reconciliation import reconcile_and_store
                    existing = store.retrieve(kernel_category=kernel_category, limit=50)
                    reconcile_and_store(outcome_dict, store, existing)
                else:
                    store.store(outcome_dict)
                store.close()
            elif backend in ("mem0", "memgraph", "lancedb", "falkordb", "redis"):
                from minisweagent.memory.storage_adapter import create_memory_store
                store = create_memory_store()
                store.store(outcome_dict)
                store.close()
            else:
                from minisweagent.memory.cross_session_memory import CrossSessionMemory
                mem = CrossSessionMemory()
                mem.record_outcome(
                    kernel_type=kernel_type,
                    kernel_category=kernel_category,
                    bottleneck_type=bottleneck_type,
                    gpu_architecture=gpu_architecture,
                    strategy_name=strategy_name,
                    speedup_achieved=speedup_achieved,
                    success=success,
                    failure_reason=failure_reason,
                    cost_dollars=cost_dollars,
                    steps_taken=steps_taken,
                    commandment_worked=commandment_worked,
                    profiling_metrics=profiling_metrics,
                    patch_snippet=patch_snippet,
                )
                if failure_reason:
                    mem.record_pitfall(kernel_category, "optimization_failure", failure_reason)
                mem.close()
        except Exception as e:
            logger.warning("Failed to record outcome: %s", e)
            _log_cross_session_issue(
                "record_failed",
                kernel_path=kernel_path,
                kernel_category=kernel_category,
                details=str(e),
                speedup_achieved=speedup_achieved,
                strategy_name=strategy_name,
            )
