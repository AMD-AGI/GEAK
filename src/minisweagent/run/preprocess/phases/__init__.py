"""Preprocess phase scaffolding.

This package is being introduced incrementally. For now it exposes the shared
``Phase`` / ``PhaseContext`` contract plus small deterministic phases that do
not change the current ``run_preprocessor`` runtime path.
"""

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

__all__ = ["Phase", "PhaseContext"]
