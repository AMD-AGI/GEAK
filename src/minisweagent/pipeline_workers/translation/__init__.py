"""Translation subagents — standalone verify-retry subagent for kernel language porting.

Translation runs as a **preprocess phase**, not a ``run_pipeline``
mode.  See ``translator.py`` for the full architectural note.

The agent is deliberately NOT derived from ``OptimizationAgent``: it is
a narrow ``SubagentBase`` subclass with a tight model-query +
``verify_fn`` loop.  Implementation lands with the preprocessing
refactor PR.
"""

from minisweagent.pipeline_workers.translation.translator import TranslationAgent

__all__ = ["TranslationAgent"]
