"""GEAK subagents — narrow-purpose LLM "employees" extending `SubagentBase`.

See docs/refactor/EXECUTION_PLAN.md §7 Principle #9 + §16.2.

Three purpose-grouped subfolders:
- `subagents/preprocess/` — one-shot subagents run during preprocessing
  (HarnessBuilder, KernelAnalysisAgent, UnitTestAgent, ShapeFixerAgent).
- `subagents/memory/`     — per-round runtime subagents
  (CrossSessionMemoryAnalysisAgent).
- `subagents/translation/` — standalone multi-round subagent invoked by
  `geak translate` (TranslationLoop).

CI invariants:
- `scripts/refactor_ci/check_subagent_location.py` enforces that every class
  subclassing `SubagentBase` lives under this package.
- `scripts/refactor_ci/check_subagent_base_contract.py` enforces that every
  `SubagentBase` subclass overrides EXACTLY ONE of `run()` / `loop()`.
"""

from minisweagent.subagents.base import SubagentBase, SubagentConfig

__all__ = ["SubagentBase", "SubagentConfig"]
