"""Memory subagents — per-round runtime subagents for cross-session memory.

Current members:

  - ``CrossSessionMemoryAnalysisAgent`` — per-round synthesis of top-k
    retrieved KB entries into ``cross_session_memory_insights.md``.
    Gated by ``GEAK_USE_CROSS_SESSION_MEMORY=1`` (default).
"""

from minisweagent.pipeline_workers.memory.cross_session_memory_analysis import (
    CrossSessionMemoryAnalysisAgent,
)

__all__ = ["CrossSessionMemoryAnalysisAgent"]
