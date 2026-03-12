"""Tests for sub_agent tool."""

from dataclasses import replace
from unittest.mock import MagicMock, patch

from minisweagent.tools.save_and_test import SaveAndTestContext
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
    def test_low_budgets_are_promoted(self, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = ("Submitted", "done")
        mock_agent_cls.return_value = mock_agent

        tool = SubAgentTool(model=MagicMock(), env=MagicMock())
        tool(task="test", step_limit=3, cost_limit=0.5)
        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["step_limit"] == 150
        assert call_kwargs["cost_limit"] == 0.0

    @patch("minisweagent.agents.default.DefaultAgent")
    def test_higher_explicit_budgets_are_preserved(self, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = ("Submitted", "done")
        mock_agent_cls.return_value = mock_agent

        tool = SubAgentTool(model=MagicMock(), env=MagicMock())
        tool(task="test", step_limit=220, cost_limit=4.5)
        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["step_limit"] == 220
        assert call_kwargs["cost_limit"] == 4.5

    @patch("minisweagent.agents.default.DefaultAgent")
    def test_child_inherits_parent_save_and_test_context(self, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = ("Submitted", "done")
        mock_agent.toolruntime._tool_table = {"save_and_test": MagicMock()}
        mock_agent._save_and_test_context = SaveAndTestContext(
            cwd="/tmp/worktree",
            test_command=None,
            timeout=60,
            patch_output_dir=None,
            patch_counter=0,
        )
        mock_agent_cls.return_value = mock_agent

        tool = SubAgentTool(model=MagicMock(), env=MagicMock())
        parent_ctx = SaveAndTestContext(
            cwd="/tmp/worktree",
            test_command="python test_harness.py --correctness",
            timeout=3600,
            patch_output_dir="/tmp/patches",
            patch_counter=7,
        )
        tool.set_context(
            MagicMock(),
            MagicMock(),
            inherited_config={"test_command": parent_ctx.test_command, "patch_output_dir": parent_ctx.patch_output_dir},
            save_and_test_context=parent_ctx,
        )

        tool(task="test")

        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["test_command"] == parent_ctx.test_command
        assert call_kwargs["patch_output_dir"] == parent_ctx.patch_output_dir

        child_ctx = mock_agent.toolruntime._tool_table["save_and_test"].set_context.call_args[0][0]
        assert child_ctx.test_command == parent_ctx.test_command
        assert child_ctx.patch_output_dir == parent_ctx.patch_output_dir
        assert child_ctx.patch_counter == parent_ctx.patch_counter

    @patch("minisweagent.agents.default.DefaultAgent")
    def test_child_patch_counter_updates_parent(self, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.toolruntime._tool_table = {"save_and_test": MagicMock()}
        initial_child_ctx = SaveAndTestContext(
            cwd="/tmp/worktree",
            test_command="python test_harness.py --correctness",
            timeout=3600,
            patch_output_dir="/tmp/patches",
            patch_counter=2,
        )
        final_child_ctx = replace(initial_child_ctx, patch_counter=5)
        mock_agent._save_and_test_context = initial_child_ctx

        def _run(_task):
            mock_agent._save_and_test_context = final_child_ctx
            return ("Submitted", "done")

        mock_agent.run.side_effect = _run
        mock_agent_cls.return_value = mock_agent

        tool = SubAgentTool(model=MagicMock(), env=MagicMock())
        parent_ctx = SaveAndTestContext(
            cwd="/tmp/worktree",
            test_command="python test_harness.py --correctness",
            timeout=3600,
            patch_output_dir="/tmp/patches",
            patch_counter=2,
        )
        tool.set_context(MagicMock(), MagicMock(), save_and_test_context=parent_ctx)

        tool(task="test")

        assert parent_ctx.patch_counter == 5
