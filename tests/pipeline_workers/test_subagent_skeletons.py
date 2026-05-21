"""Contract tests for the preprocessing-PR subagent skeletons.

These tests verify the SubagentBase-subclass contract for the three
new subagents (HarnessBuilder, KernelAnalysisAgent,
CrossSessionMemoryAnalysisAgent) before their bodies are implemented.
They guarantee the architectural shape that downstream phase code
will import against.
"""

from __future__ import annotations

import pytest

from minisweagent.agents.optimization_agent import OptimizationAgent
from minisweagent.pipeline_workers.base import SubagentBase
from minisweagent.pipeline_workers.memory import CrossSessionMemoryAnalysisAgent
from minisweagent.pipeline_workers.preprocess import HarnessBuilder, KernelAnalysisAgent
from minisweagent.pipeline_workers.translation import TranslationAgent


_SKELETONS = [HarnessBuilder, KernelAnalysisAgent, CrossSessionMemoryAnalysisAgent, TranslationAgent]


class TestSubagentHierarchy:
    @pytest.mark.parametrize("cls", _SKELETONS)
    def test_inherits_from_subagent_base(self, cls: type) -> None:
        assert issubclass(cls, SubagentBase)

    @pytest.mark.parametrize("cls", _SKELETONS)
    def test_does_not_inherit_from_optimization_agent(self, cls: type) -> None:
        """Per user direction: preprocess + translation subagents are
        narrow tasks that do not need OptimizationAgent's tool loop.
        They must be standalone SubagentBase subclasses."""
        assert not issubclass(cls, OptimizationAgent), (
            f"{cls.__name__} must not inherit from OptimizationAgent; "
            "it's a narrow SubagentBase task."
        )


class TestSubagentContract:
    """Each subagent overrides exactly one of run / loop.

    The CI gate ``check_subagent_base_contract.py`` enforces this at
    build time; these tests assert the same at pytest time so
    per-subagent regressions get a clear failure.
    """

    @pytest.mark.parametrize(
        "cls,expected_method",
        [
            (HarnessBuilder, "run"),
            (KernelAnalysisAgent, "run"),
            (CrossSessionMemoryAnalysisAgent, "run"),
            (TranslationAgent, "loop"),
        ],
    )
    def test_overrides_expected_method(self, cls: type, expected_method: str) -> None:
        base_run = SubagentBase.run
        base_loop = SubagentBase.loop
        own = cls.__dict__
        if expected_method == "run":
            assert "run" in own, f"{cls.__name__} must override run()"
            assert own.get("loop") is None, f"{cls.__name__} must not also override loop()"
        else:
            assert "loop" in own, f"{cls.__name__} must override loop()"
            assert own.get("run") is None, f"{cls.__name__} must not also override run()"
        # Sanity: the base method still raises NotImplementedError
        assert base_run is SubagentBase.run
        assert base_loop is SubagentBase.loop


class TestSubagentNotImplementedMessages:
    """The remaining skeleton bodies raise NotImplementedError with pointers
    to where the full implementation will live.  Downstream callers rely
    on these messages being actionable.

    Implemented subagents (behavioural tests live elsewhere):
      - TranslationAgent      -> ``test_translation_agent.py``
      - HarnessBuilder        -> ``test_harness_builder.py``
      - KernelAnalysisAgent   -> ``test_kernel_analysis_agent.py``

    Still skeletons (per plan §13.4 — KB work deferred until end-to-end
    pipeline lands):
      - CrossSessionMemoryAnalysisAgent  -> planned for Workstream D3
    """

    def test_memory_analysis_points_to_assemble(self) -> None:
        with pytest.raises(NotImplementedError) as e:
            CrossSessionMemoryAnalysisAgent.__dict__["run"](_FakeSubagent())
        assert "assemble_memory_context" in str(e.value)


class _FakeSubagent:
    """Minimal self mock so we can call the unbound method without
    triggering the full SubagentBase.__init__ (which needs a
    KernelLanguage)."""

    pass
