"""Preprocess phases — named, explicit stages of the preprocessing pipeline.

The preprocessing pipeline is a sequence of explicit phases (per the
execution plan §0.5(b)):

    TranslationPhase (CONDITIONAL)  — only when target_language differs
    DiscoveryPhase                   — always
    HarnessPhase                     — always
    BaselinePhase                    — always
    ExplorePhase                     — always

Each phase owns a narrow concern and can be run / tested in isolation.
``PreprocessOrchestrator`` in ``preprocess/orchestrator.py`` drives them
in order.

Phases communicate by reading from and writing to a shared
``PhaseContext`` (a light dict-plus-accessors wrapper over the existing
``PreprocessContext`` structure).  A phase contract:

    phase.run(ctx) -> None     # mutates ctx in-place; raises on fatal errors
    phase.is_applicable(ctx)   # returns False to skip (e.g. Translation
                                # when target == source)

This is the additive scaffolding step.  Phase bodies in this commit
*delegate to the existing ``run_preprocessor`` monolith* so the new
architecture is in place without rewriting the existing logic.
Subsequent commits progressively move logic into each phase file.
"""

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

__all__ = ["Phase", "PhaseContext"]
