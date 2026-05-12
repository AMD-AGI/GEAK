"""``CrossSessionMemoryAnalysisAgent`` — per-round subagent that synthesises KB insights.

Runs once per optimization round when
``GEAK_USE_CROSS_SESSION_MEMORY=1`` (the default).  Consumes:

  - ``target_code``: current kernel source the optimizer is working on
  - ``target_profile``: current baseline_metrics / profile.json for this kernel
  - ``retrieved``: top-k ExperienceRecords returned by
    ``cross_session.retrieve()`` (code-similarity ranked)

Emits ``cross_session_memory_insights.md`` (~5-25 KB markdown) to the
artefacts directory.  Content (per plan §0.5(b)):

  - Analysis Summary (applicability assessment, priority ranking)
  - Top Recommended Strategies (ranked, with reasoning + concrete patterns)
  - Avoid / Known Dead-Ends (with regression evidence from KB)
  - Reference: Full KB Entries (for EACH retrieved experience):
      * baseline_code           (full source of the KB kernel)
      * winning_diff            (unified patch that produced the speedup)
      * profiler_before / after (hotspots, roofline)
      * strategies_tried        (per-strategy speedup/regression)
      * dead_ends               (with reasons)
      * code_sim score + subagent-derived staleness signal
  - ``none_applicable: true/false``

The file contents are read back into the task body by
``compose_task_body()`` before each round.  **Single path — no
fast/slow branching.**

This implementation is a skeleton: class + config contract are in
place; the actual synthesis prompt + LLM call land in a follow-up
commit.  Until then, ``run_pipeline``'s per-round loop still calls
the deterministic ``assemble_memory_context`` retriever (which dumps
top-k experiences as raw markdown).  Swapping that call for this
agent is the final step of the memory-synthesis work.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from minisweagent.pipeline_workers.base import SubagentBase

logger = logging.getLogger(__name__)


class CrossSessionMemoryAnalysisAgent(SubagentBase):
    """Per-round memory-synthesis subagent.

    Subclass override: ``run()``.  One-shot per round (new agent
    instance per round).  Does not compose ``OptimizationAgent`` —
    direct model query is sufficient for the synthesis task.
    """

    def run(self, **inputs: Any) -> str | dict:
        """Synthesise retrieved KB entries into an insights markdown file.

        Expected inputs (when implemented):
          - ``target_code: str``      — current kernel source
          - ``target_profile: dict``  — baseline_metrics / profile.json
          - ``retrieved: list[ExperienceRecord]`` — top-k results
          - ``out_path: Path``        — where to write
                                          ``cross_session_memory_insights.md``

        Returns: ``Path`` (the insights file path).
        """
        raise NotImplementedError(
            "CrossSessionMemoryAnalysisAgent.run is not implemented yet. "
            "Until this lands, the per-round memory injection still uses "
            "``memory.integration.assemble_memory_context()`` which dumps "
            "raw top-k KB markdown.  The synthesis prompt + LLM call are "
            "scheduled for a subsequent commit."
        )


__all__ = ["CrossSessionMemoryAnalysisAgent"]
