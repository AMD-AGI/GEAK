"""HIP KernelLanguage instance.

Registered at import time.  Prompt / template paths resolve to files
under this folder when they exist; ``None`` otherwise so the base
class's lazy-load helpers return ``""`` without raising.
"""

from __future__ import annotations

from pathlib import Path

from minisweagent.kernel_languages import registry
from minisweagent.kernel_languages.base import KernelLanguage

_DIR = Path(__file__).parent
_TRANSLATION_DIR = _DIR.parent / "_translation"


def _p(name: str) -> Path | None:
    candidate = _DIR / name
    return candidate if candidate.exists() else None


HIP = KernelLanguage(
    name="hip",
    # HIP wrappers are typically .py (pybind11 bindings, torch.utils.cpp_extension
    # wrappers), or raw .cu / .hip / .cpp. Triton also claims .py — the
    # detect_hints disambiguate.
    file_extensions=frozenset({".py", ".cu", ".hip", ".cpp", ".cxx"}),
    detect_hints=(
        r"__global__\s+void\b",
        r"hipLaunchKernelGGL\b",
        r"\bhip[A-Z]\w*\(",  # hipMalloc, hipMemcpy, etc.
        r'#include\s*[<"]hip/hip_runtime\.h',
        r"torch\.utils\.cpp_extension",
        r"scripts/task_runner\.py.*(compile|correctness|performance)",
    ),
    kb_namespace="hip",
    # Prompts & templates
    system_prompt_path=_p("system_prompt.md"),
    orchestrator_system_prompt_path=_p("orchestrator_system_prompt.md"),
    optimization_prompt_path=_p("optimization_prompt.md"),
    planner_strategy_hints_path=_p("planner_strategy_hints.md"),
    optimizer_hints_path=_p("optimizer_hints.md"),
    builder_hints_path=_p("builder_hints.md"),
    memory_hints_path=_p("memory_hints.md"),
    idioms_path=_p("idioms.md"),
    harness_template_path=_p("harness.j2"),
    commandment_template_path=_p("commandment.j2"),
    translation_hints_dir=_TRANSLATION_DIR if _TRANSLATION_DIR.exists() else None,
    tool_set=frozenset(),
)

registry.register(HIP)


__all__ = ["HIP"]
