"""AVO — Agentic Variation Operators on top of GEAK.

This package implements a single-lineage AVO continuous-evolution run as an
*additive* layer over GEAK's existing preprocess, ``OptimizationAgent``,
``save_and_test``, ``strategy_manager``, RAG, and ``RunBudget``. It does not
modify any GEAK core module.

See ``docs/developer/avo_design.md`` for the full design.

Public surface:

- :class:`~minisweagent.run.avo.result.VariationResult` — one variation step's outcome.
- :class:`~minisweagent.run.avo.lineage_store.LineageStore` — the committed lineage ``P_t``.
- :class:`~minisweagent.run.avo.stagnation.StagnationDetector` — deterministic stall detector.
- :func:`~minisweagent.run.avo.controller.run_avo` — the outer evolution loop.
"""

from __future__ import annotations

from minisweagent.run.avo.result import AttemptRecord, VariationResult

__all__ = ["AttemptRecord", "VariationResult"]
