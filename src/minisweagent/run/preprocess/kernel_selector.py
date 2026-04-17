"""LLM-based kernel selection for baseline metrics.

Given profiler output with multiple kernels, asks an LLM to identify which
kernel(s) correspond to the optimization target, ordered by relevance.

Uses the same ``model.query([system, user])`` pattern as
``minisweagent.run.utils.task_parser`` for one-shot structured output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_JSON_ARRAY_FENCE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)

_KERNEL_SELECTION_SYSTEM_PROMPT = (
    "You are a GPU kernel profiling expert. You select which profiled GPU "
    "kernels correspond to a given optimization target. Always respond with "
    "a valid JSON array of kernel name strings. Do not use tools."
)


def _parse_json_array(content: str) -> list[str] | None:
    """Parse a JSON array from model response, stripping optional fences."""
    content = content.strip()
    m = _JSON_ARRAY_FENCE.search(content)
    if m:
        content = m.group(1)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and all(isinstance(n, str) for n in parsed):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def select_relevant_kernels(
    profiler_result: dict[str, Any],
    *,
    kernel_name: str,
    kernel_path: str,
    model_factory=None,
    gpu_index: int = 0,
    baseline_kernel_names: list[str] | None = None,
) -> list[str] | None:
    """Return names of profiled kernels relevant to the optimization target.

    Kernels are returned in relevance order (most relevant first).

    Args:
        profiler_result: Full profile.json dict (all kernels).
        kernel_name: Target kernel name from discovery.
        kernel_path: Target kernel file path.
        model_factory: Callable returning a model instance.
        gpu_index: Which GPU result to read.
        baseline_kernel_names: If provided, these were the baseline-selected
            kernels.  The LLM is hinted to prefer matching them if they
            still exist (reduces non-determinism in cross-round comparisons).

    Returns:
        List of kernel name strings ordered by relevance, or ``None`` if
        selection fails (caller should fall back to ``include_all=True``).
    """
    if not model_factory:
        logger.info("No model available for kernel selection; using all kernels.")
        return None

    results = profiler_result.get("results", [])
    if not results or gpu_index >= len(results):
        return None

    kernels = results[gpu_index].get("kernels", [])
    if not kernels:
        return None
    if len(kernels) == 1:
        return [kernels[0]["name"]]

    kernel_summary = "\n".join(
        f"  [{i}] {k['name']}  duration={k.get('duration_us', '?')}us  "
        f"bottleneck={k.get('bottleneck', '?')}  "
        f"observations={k.get('observations', [])}"
        for i, k in enumerate(kernels)
    )

    baseline_hint = ""
    if baseline_kernel_names:
        baseline_hint = (
            f"\n\nFor reference, the baseline profile selected these kernels: "
            f"{baseline_kernel_names}. Prefer matching them if they still exist "
            f"under the same or similar names, but also include any new kernels "
            f"that are clearly part of the optimization target."
        )

    prompt = (
        f"The optimization target is the kernel '{kernel_name}' "
        f"defined in file '{kernel_path}'.\n\n"
        f"The profiler captured these GPU kernels during execution:\n"
        f"{kernel_summary}\n\n"
        "Which of these profiled kernels are part of the optimization target? "
        "Return a JSON array of the exact kernel name strings, ordered from "
        "most relevant to least relevant. Include kernels that are directly "
        "part of the target operation. Exclude unrelated framework/runtime "
        f"kernels (e.g., distribution, elementwise helpers).{baseline_hint}"
    )

    try:
        model = model_factory()
        response = model.query([
            {"role": "system", "content": _KERNEL_SELECTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        names = _parse_json_array(response.get("content", ""))
        if names is None:
            logger.warning(
                "LLM kernel selection returned unparseable response: %s",
                response.get("content", "")[:200],
            )
            return None

        available = {k["name"] for k in kernels}
        valid = [n for n in names if n in available]
        if valid:
            logger.info("LLM selected %d kernel(s): %s", len(valid), valid)
            return valid
        logger.warning("LLM returned no valid kernel names from: %s", names)
    except Exception as exc:
        logger.warning("Kernel selection LLM call failed: %s", exc)

    return None
