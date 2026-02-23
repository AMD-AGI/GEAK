"""Tests for agent filtering via GEAK_ALLOWED_AGENTS / GEAK_EXCLUDED_AGENTS."""

from __future__ import annotations

import logging

from minisweagent.agents.agent_spec import (
    ALL_AGENT_TYPES,
    filter_agent_type,
    get_allowed_agent_types,
)

# ---------------------------------------------------------------------------
# get_allowed_agent_types()
# ---------------------------------------------------------------------------


class TestGetAllowedAgentTypes:
    def test_no_env_vars_returns_none(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        assert get_allowed_agent_types() is None

    def test_allowed_set(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "swe_agent,strategy_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        result = get_allowed_agent_types()
        assert result == {"swe_agent", "strategy_agent"}

    def test_excluded_set(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.setenv("GEAK_EXCLUDED_AGENTS", "openevolve")
        result = get_allowed_agent_types()
        assert result == ALL_AGENT_TYPES - {"openevolve"}
        assert result == {"swe_agent", "strategy_agent"}

    def test_both_set_allowed_takes_precedence(self, monkeypatch, caplog):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "swe_agent")
        monkeypatch.setenv("GEAK_EXCLUDED_AGENTS", "strategy_agent")
        with caplog.at_level(logging.WARNING):
            result = get_allowed_agent_types()
        assert result == {"swe_agent"}
        assert "takes precedence" in caplog.text

    def test_invalid_agent_name_filtered(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "swe_agent,bogus_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        result = get_allowed_agent_types()
        assert result == {"swe_agent"}

    def test_whitespace_handling(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", " swe_agent , openevolve ")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        result = get_allowed_agent_types()
        assert result == {"swe_agent", "openevolve"}


# ---------------------------------------------------------------------------
# filter_agent_type()
# ---------------------------------------------------------------------------


class TestFilterAgentType:
    def test_no_env_vars_passthrough(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        for agent in ("swe_agent", "strategy_agent", "openevolve"):
            assert filter_agent_type(agent) == agent

    def test_allowed_agents_pass_through(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "swe_agent,strategy_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_FALLBACK_AGENT", raising=False)
        assert filter_agent_type("swe_agent") == "swe_agent"
        assert filter_agent_type("strategy_agent") == "strategy_agent"

    def test_disallowed_agent_remapped_to_fallback(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "swe_agent,strategy_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_FALLBACK_AGENT", raising=False)
        assert filter_agent_type("openevolve") == "swe_agent"

    def test_excluded_agent_remapped(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.setenv("GEAK_EXCLUDED_AGENTS", "openevolve")
        monkeypatch.delenv("GEAK_FALLBACK_AGENT", raising=False)
        assert filter_agent_type("openevolve") == "swe_agent"
        assert filter_agent_type("swe_agent") == "swe_agent"
        assert filter_agent_type("strategy_agent") == "strategy_agent"

    def test_custom_fallback_agent(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        monkeypatch.setenv("GEAK_FALLBACK_AGENT", "strategy_agent")
        assert filter_agent_type("openevolve") == "strategy_agent"
        assert filter_agent_type("swe_agent") == "strategy_agent"

    def test_remap_logs_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("GEAK_EXCLUDED_AGENTS", "openevolve")
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_FALLBACK_AGENT", raising=False)
        with caplog.at_level(logging.WARNING):
            result = filter_agent_type("openevolve")
        assert result == "swe_agent"
        assert "not allowed" in caplog.text
        assert "openevolve" in caplog.text


# ---------------------------------------------------------------------------
# _parse_llm_response integration (safety net)
# ---------------------------------------------------------------------------


class TestParseResponseSafetyNet:
    """Verify that _parse_llm_response applies filter_agent_type."""

    TASK_JSON_WITH_OPENEVOLVE = (
        '[{"label": "oe-task", "priority": 0, "agent_type": "openevolve", '
        '"kernel_language": "python", "task_prompt": "Run OpenEvolve"}]'
    )

    def test_excluded_agent_remapped_in_parse(self, monkeypatch):
        monkeypatch.setenv("GEAK_EXCLUDED_AGENTS", "openevolve")
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_FALLBACK_AGENT", raising=False)

        from minisweagent.agents.swe_agent import SweAgent
        from minisweagent.run.task_generator import _parse_llm_response

        class _FakeDefault:
            pass

        tasks = _parse_llm_response(self.TASK_JSON_WITH_OPENEVOLVE, _FakeDefault)
        assert len(tasks) == 1
        assert tasks[0].agent_class is SweAgent

    def test_no_filtering_keeps_openevolve(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)

        from minisweagent.agents.openevolve_worker import OpenEvolveWorker
        from minisweagent.run.task_generator import _parse_llm_response

        class _FakeDefault:
            pass

        tasks = _parse_llm_response(self.TASK_JSON_WITH_OPENEVOLVE, _FakeDefault)
        assert len(tasks) == 1
        assert tasks[0].agent_class is OpenEvolveWorker


    def test_fallback_not_in_allowed_set(self, monkeypatch):
        """Test that fallback agent is validated against allowed set (Issue #20)."""
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "strategy_agent")
        monkeypatch.setenv("GEAK_FALLBACK_AGENT", "openevolve")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)
        
        result = filter_agent_type("swe_agent")
        # The fallback "openevolve" is not in allowed set, so it should
        # fall back to the first allowed type instead
        assert result != "openevolve"
        assert result in get_allowed_agent_types()
        assert result == "strategy_agent"


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------


class TestPromptInjection:
    def test_allowed_agents_adds_restriction(self, monkeypatch):
        monkeypatch.setenv("GEAK_ALLOWED_AGENTS", "swe_agent")
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)

        from minisweagent.run.task_generator import _build_agent_restriction_addendum

        addendum = _build_agent_restriction_addendum()
        assert "swe_agent" in addendum
        assert "MUST NOT" in addendum
        assert "only" in addendum.lower() or "Only" in addendum

    def test_excluded_agents_adds_restriction(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.setenv("GEAK_EXCLUDED_AGENTS", "openevolve")

        from minisweagent.run.task_generator import _build_agent_restriction_addendum

        addendum = _build_agent_restriction_addendum()
        assert "openevolve" in addendum
        assert "NOT available" in addendum

    def test_no_env_vars_returns_empty(self, monkeypatch):
        monkeypatch.delenv("GEAK_ALLOWED_AGENTS", raising=False)
        monkeypatch.delenv("GEAK_EXCLUDED_AGENTS", raising=False)

        from minisweagent.run.task_generator import _build_agent_restriction_addendum

        assert _build_agent_restriction_addendum() == ""
