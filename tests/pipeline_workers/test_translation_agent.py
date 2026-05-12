"""Tests for ``TranslationAgent`` — the standalone verify-retry subagent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from minisweagent.pipeline_workers.base import SubagentConfig
from minisweagent.pipeline_workers.translation import TranslationAgent


def _fake_language(name: str = "hip") -> MagicMock:
    """Minimal KernelLanguage stub."""
    lang = MagicMock()
    lang.name = name
    lang.system_prompt_path = None
    return lang


def _fake_config() -> SubagentConfig:
    return SubagentConfig(
        name="translation",
        model_name="dummy",
        system_template="",
        instance_template="",
        step_limit=1,
    )


def _agent_with_model(model_output: str, *, target: str = "hip") -> TranslationAgent:
    """Build a TranslationAgent whose model returns a fixed string."""
    agent = TranslationAgent(language=_fake_language(target), config=_fake_config())
    # Inject a mock model so _query_model doesn't try to load a real one.
    model = MagicMock()
    model.query.return_value = {"content": model_output}
    agent.model = model
    return agent


class TestLoopVerification:
    def test_returns_ok_on_first_success(self) -> None:
        agent = _agent_with_model("hip_code_here")
        result = agent.loop(
            max_attempts=3,
            verify_fn=lambda _c: True,
            source_code="def kernel(): ...",
            source_language="triton",
        )
        assert result.ok
        assert result.attempts_used == 1
        assert result.candidate_code == "hip_code_here"
        assert result.feedback_history == []

    def test_retries_on_verification_failure(self) -> None:
        agent = _agent_with_model("hip_draft")
        call_count = {"n": 0}

        def _verifier(_c: str) -> tuple[bool, str]:
            call_count["n"] += 1
            return call_count["n"] >= 3, f"mismatch at call {call_count['n']}"

        result = agent.loop(
            max_attempts=5,
            verify_fn=_verifier,
            source_code="src",
            source_language="triton",
        )
        assert result.ok
        assert result.attempts_used == 3
        # Two failed attempts accumulated feedback
        assert len(result.feedback_history) == 2
        assert "mismatch at call 1" in result.feedback_history[0]

    def test_exhausted_attempts_returns_not_ok(self) -> None:
        agent = _agent_with_model("draft")
        result = agent.loop(
            max_attempts=2,
            verify_fn=lambda _c: False,
            source_code="src",
            source_language="triton",
        )
        assert result.ok is False
        assert result.attempts_used == 2
        assert len(result.feedback_history) == 2
        assert "exhausted" in result.reason

    def test_tuple_verify_fn_extracts_failure_reason(self) -> None:
        agent = _agent_with_model("draft")
        result = agent.loop(
            max_attempts=1,
            verify_fn=lambda _c: (False, "tensor norm diff 3.14"),
            source_code="src",
            source_language="triton",
        )
        assert result.ok is False
        assert len(result.feedback_history) == 1
        assert "tensor norm diff 3.14" in result.feedback_history[0]


class TestLoopInputs:
    def test_missing_source_code_raises(self) -> None:
        agent = _agent_with_model("x")
        with pytest.raises(ValueError, match="source_code"):
            agent.loop(
                max_attempts=1,
                verify_fn=lambda _c: True,
                source_language="triton",
            )

    def test_missing_source_language_raises(self) -> None:
        agent = _agent_with_model("x")
        with pytest.raises(ValueError, match="source_code"):
            agent.loop(
                max_attempts=1,
                verify_fn=lambda _c: True,
                source_code="",
                source_language="triton",
            )

    def test_zero_max_attempts_raises(self) -> None:
        agent = _agent_with_model("x")
        with pytest.raises(ValueError, match="max_attempts"):
            agent.loop(
                max_attempts=0,
                verify_fn=lambda _c: True,
                source_code="src",
                source_language="triton",
            )


class TestPromptComposition:
    def test_prompt_includes_both_languages_and_source(self) -> None:
        agent = TranslationAgent(language=_fake_language("hip"), config=_fake_config())
        sys_p, inst_p = agent._compose_translation_prompt(
            source_language="triton",
            target_language="hip",
            source_code="def kernel(): ...",
        )
        assert "translator" in sys_p.lower()
        assert "triton" in inst_p.lower()
        assert "hip" in inst_p.lower()
        assert "def kernel(): ..." in inst_p

    def test_prompt_includes_hints_when_provided(self) -> None:
        agent = TranslationAgent(language=_fake_language("hip"), config=_fake_config())
        _sys, inst = agent._compose_translation_prompt(
            source_language="triton",
            target_language="hip",
            source_code="src",
            hints="Use __global__ attribute for entry points",
        )
        assert "Use __global__ attribute" in inst

    def test_prompt_feeds_back_previous_failure(self) -> None:
        agent = TranslationAgent(language=_fake_language("hip"), config=_fake_config())
        _sys, inst = agent._compose_translation_prompt(
            source_language="triton",
            target_language="hip",
            source_code="src",
            last_feedback="Attempt 1: shape mismatch on output[0]",
        )
        assert "PREVIOUS ATTEMPT FAILED" in inst
        assert "shape mismatch on output[0]" in inst
