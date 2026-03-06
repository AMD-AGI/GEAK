# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sub_agent tool."""

from unittest.mock import MagicMock, patch

from minisweagent.tools.sub_agent_tool import SubAgentTool


class TestSubAgentTool:
    def test_no_context_returns_error(self):
        tool = SubAgentTool()
        result = tool(task="do something")
        assert result["returncode"] == 1
        assert "not initialized" in result["output"]

    def test_set_context(self):
        tool = SubAgentTool()
        model = MagicMock()
        env = MagicMock()
        tool.set_context(model, env)
        assert tool._model is model
        assert tool._env is env

    @patch("minisweagent.agents.default.DefaultAgent")
    def test_successful_run(self, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = ("Submitted", "optimization complete")
        mock_agent_cls.return_value = mock_agent

        tool = SubAgentTool(model=MagicMock(), env=MagicMock())
        result = tool(task="optimize this kernel", step_limit=5)
        assert result["returncode"] == 0
        assert "Submitted" in result["output"]
        mock_agent_cls.assert_called_once()

    @patch("minisweagent.agents.default.DefaultAgent")
    def test_failed_run(self, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("LLM unavailable")
        mock_agent_cls.return_value = mock_agent

        tool = SubAgentTool(model=MagicMock(), env=MagicMock())
        result = tool(task="optimize this kernel")
        assert result["returncode"] == 1
        assert "error" in result["output"].lower()

    @patch("minisweagent.agents.default.DefaultAgent")
    def test_step_limit_passed(self, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = ("Submitted", "done")
        mock_agent_cls.return_value = mock_agent

        tool = SubAgentTool(model=MagicMock(), env=MagicMock())
        tool(task="test", step_limit=3, cost_limit=0.5)
        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["step_limit"] == 3
        assert call_kwargs["cost_limit"] == 0.5
