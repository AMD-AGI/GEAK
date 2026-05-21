"""Single source of truth for composing the task body handed to the main agent.

Before this module existed the task body was assembled in two very different
places depending on mode:

  - **fixed**   — inside ``cli.py`` by concatenating the user prompt
    with ``assemble_memory_context`` output.  One prompt replicated
    across N parallel agents.  (Legacy name: "homogeneous".)
  - **planned** — inside the planned-mode orchestrator and task generator,
    by calling out to the planner LLM, which produced per-task bodies
    that already included the commandment and user constraints.  N
    planned strategies, one per worker.  (Legacy name: "heterogeneous".)

Both modes eventually fed identical downstream stages (``ParallelAgent`` ->
``OptimizationAgent``), so the divergence was purely presentational.  This
module centralizes the composition in one function so that:

  - new modes (``mixed`` splits fixed+planned; ``translate`` runs verify-retry)
    get a consistent entrypoint,
  - cross-session memory injection is applied identically,
  - the KernelLanguage-system-prompt binding (Triton / HIP / ...) is resolved
    in exactly one place.

The function deliberately does not call the planner LLM.  In ``planned``
mode it assembles the *base* task body shared by every planner sub-task;
the planner itself still lives in ``task_generator.py`` and is invoked
from ``run/unified.py``.

Mode vocabulary (matches the end-state execution plan):

  - ``fixed``   — one task body, replicated across ``num_parallel`` copies
  - ``planned`` — N planner-generated task bodies (one per strategy)
  - ``auto``    — controller picks ``fixed``-vs-``planned`` per round

Translation (source→target language) is NOT a ``run_pipeline`` mode.  It is
a **conditional preprocess phase** that runs before the optimization
loop when ``target_language ≠ source_language``; after it completes,
``ctx.kernel_path`` and ``ctx.language`` are swapped to the translated
kernel and the pipeline continues (or exits, if the user passed
``--translate-only``).  The phase owns its own narrow ``TranslationAgent``
subagent (a ``SubagentBase`` subclass with a verify-retry loop against
golden tensors); it does NOT reuse ``OptimizationAgent``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

Mode = Literal["fixed", "planned", "mixed"]


@dataclass
class ComposeInputs:
    """Inputs to ``compose_task_body``.

    Kept deliberately loose (``Any`` for ``preprocess_ctx``) so existing
    callers can pass the current dict-shaped context without a migration
    tax.  A stricter ``PipelineContext`` dataclass will take its place once
    every caller flows through ``run_pipeline``.
    """

    user_prompt: str
    mode: Mode
    preprocess_ctx: dict[str, Any]
    kernel_language: str | None = None
    extra_addenda: list[str] = field(default_factory=list)


_MEMORY_SECTION_HEADER = "\n\n### Optimization Memory (from past kernel optimization runs)\n"
_ANALYSIS_SECTION_HEADER = "\n\n### Kernel Analysis (produced by KernelAnalysisAgent)\n"


def _inject_kernel_analysis(user_prompt: str, preprocess_ctx: dict[str, Any]) -> tuple[str, int]:
    """Prepend the [A]-[D] rubric markdown to the task body when available.

    ``preprocess_ctx["kernel_analysis_md"]`` is populated by
    ``ExplorePhase`` via ``KernelAnalysisAgent``.  When absent / empty
    (no language detected, no model available, subagent failed —
    all best-effort paths), the original prompt is returned unchanged.

    Returns ``(augmented_prompt, chars_injected)``.
    """
    rubric = preprocess_ctx.get("kernel_analysis_md")
    if not rubric or not isinstance(rubric, str) or not rubric.strip():
        return user_prompt, 0
    return user_prompt + _ANALYSIS_SECTION_HEADER + rubric.strip() + "\n", len(rubric)


def _inject_memory(user_prompt: str, preprocess_ctx: dict[str, Any]) -> tuple[str, int]:
    """Append cross-session memory context to the user prompt, if available.

    Returns (augmented_prompt, chars_injected).  On any failure the original
    prompt is returned unchanged — memory is best-effort context, never a
    hard dependency.
    """
    try:
        from minisweagent.memory.integration import assemble_memory_context
    except Exception as exc:
        logger.warning("Cross-session memory import unavailable: %s", exc)
        return user_prompt, 0

    kernel_path = preprocess_ctx.get("kernel_path", "")
    baseline = preprocess_ctx.get("baseline_metrics") or {}
    if isinstance(baseline, str):
        import json

        try:
            baseline = json.loads(baseline)
        except Exception:
            baseline = {}

    try:
        mem = assemble_memory_context(
            kernel_path=kernel_path,
            bottleneck_type=(baseline or {}).get("bottleneck", ""),
            profiling_metrics=baseline or {},
        )
    except Exception as exc:
        logger.warning("Cross-session memory retrieval failed: %s", exc)
        return user_prompt, 0

    if not mem:
        logger.info("Cross-session memory: no relevant experiences found")
        return user_prompt, 0

    return user_prompt + _MEMORY_SECTION_HEADER + mem, len(mem)


def compose_task_body(inputs: ComposeInputs) -> str:
    """Return the final task body that will be handed to ``OptimizationAgent``.

    Applied in order:
      1. Mode-specific framing (currently identical across modes — the
         ``mode`` parameter exists so future modes can branch without
         callers changing shape).
      2. **Kernel analysis rubric** ([A]-[D] from KernelAnalysisAgent,
         when ``preprocess_ctx['kernel_analysis_md']`` is populated).
         Injected BEFORE memory so the rubric sets the optimization
         framing before any KB evidence lands.
      3. Cross-session memory injection (enabled for every mode that
         optimizes; retrieval itself is gated by
         ``GEAK_USE_CROSS_SESSION_MEMORY``).
      4. Extra addenda from the caller (e.g. the hetero orchestrator's
         user-constraints / directives block).
    """
    body = inputs.user_prompt

    body, analysis_chars = _inject_kernel_analysis(body, inputs.preprocess_ctx)
    if analysis_chars:
        logger.info(
            "compose_task_body: injected %d chars of kernel analysis rubric",
            analysis_chars,
        )

    body, memory_chars = _inject_memory(body, inputs.preprocess_ctx)
    if memory_chars:
        logger.info(
            "compose_task_body: injected %d chars of cross-session memory",
            memory_chars,
        )

    for extra in inputs.extra_addenda:
        if extra and extra.strip():
            body = body + "\n\n" + extra.strip()

    return body


__all__ = ["ComposeInputs", "Mode", "compose_task_body"]
