"""Triton KernelLanguage instance.

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
    """Return ``_DIR / name`` if the file exists, else ``None``."""
    candidate = _DIR / name
    return candidate if candidate.exists() else None


TRITON = KernelLanguage(
    name="triton",
    file_extensions=frozenset({".py"}),
    detect_hints=(
        r"@triton\.jit\b",
        r"^import\s+triton",
        r"\bfrom\s+triton\b",
        r"\btl\.load\s*\(",
        r"\btl\.store\s*\(",
    ),
    kb_namespace="triton",
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
    # Translation hint packs live at kernel_languages/_translation/ (shared
    # across languages so pair-specific packs like triton_to_hip.md sit
    # next to each other and can be edited together).
    translation_hints_dir=_TRANSLATION_DIR if _TRANSLATION_DIR.exists() else None,
    tool_set=frozenset(),  # populated in PR-3; empty = use tools_runtime defaults
)

registry.register(TRITON)


__all__ = ["TRITON"]
