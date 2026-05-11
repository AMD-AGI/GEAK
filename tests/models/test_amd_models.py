"""Unit tests for AMD model backends: message formatting, tool conversion, response parsing."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.models.amd_base import AmdLlmModelConfig
from minisweagent.models.amd_claude import AmdClaudeModel, convert_openai_tools_to_claude
from minisweagent.models.amd_llm import AmdLlmModel
from minisweagent.models.amd_openai import AmdOpenAIModel

# ---------------------------------------------------------------------------
# convert_openai_tools_to_claude
# ---------------------------------------------------------------------------


class TestConvertOpenaiToolsToClaude:
    TOOLS = [
        {
            "function": {
                "name": "bash",
                "description": "Run a bash command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
            }
        },
        {
            "function": {
                "name": "submit",
                "description": "Submit answer",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        },
    ]

    def test_basic_conversion(self):
        result = convert_openai_tools_to_claude(self.TOOLS, cache_control=False)
        assert len(result) == 2
        assert result[0]["name"] == "bash"
        assert result[0]["input_schema"]["properties"]["cmd"]["type"] == "string"
        assert result[1]["name"] == "submit"

    def test_cache_control_on_last_tool(self):
        result = convert_openai_tools_to_claude(self.TOOLS, cache_control=True)
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_off(self):
        result = convert_openai_tools_to_claude(self.TOOLS, cache_control=False)
        for tool in result:
            assert "cache_control" not in tool

    def test_empty_tools(self):
        assert convert_openai_tools_to_claude([], cache_control=True) == []

    def test_flat_tool_without_function_wrapper(self):
        tools = [{"name": "bash", "description": "Run bash", "parameters": {"type": "object"}}]
        result = convert_openai_tools_to_claude(tools, cache_control=False)
        assert len(result) == 1
        assert result[0]["name"] == "bash"


# ---------------------------------------------------------------------------
# AmdClaudeModel.format_messages
# ---------------------------------------------------------------------------


def _make_claude_model():
    """Create an AmdClaudeModel with a mocked client (no real API key needed)."""
    config = AmdLlmModelConfig(model_name="claude-sonnet-4.5", api_key="test-key")
    with patch.object(AmdClaudeModel, "_init_client"):
        return AmdClaudeModel(config)


class TestAmdClaudeFormatMessages:
    def test_system_extracted(self):
        model = _make_claude_model()
        system, msgs = model.format_messages([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ])
        assert system == "You are helpful"
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_tool_result_becomes_user(self):
        model = _make_claude_model()
        _, msgs = model.format_messages([
            {"role": "tool", "content": "result text", "tool_call_id": "call_1"},
        ])
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["type"] == "tool_result"
        assert msgs[0]["content"][0]["tool_use_id"] == "call_1"

    def test_assistant_tool_call_single_dict(self):
        model = _make_claude_model()
        _, msgs = model.format_messages([
            {
                "role": "assistant",
                "content": "thinking...",
                "tool_calls": {
                    "id": "call_1",
                    "function": {"name": "bash", "arguments": {"cmd": "ls"}},
                },
            },
        ])
        blocks = msgs[0]["content"]
        assert blocks[0] == {"type": "text", "text": "thinking..."}
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["name"] == "bash"
        assert blocks[1]["input"] == {"cmd": "ls"}

    def test_assistant_tool_call_list(self):
        model = _make_claude_model()
        _, msgs = model.format_messages([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "bash", "arguments": {"cmd": "ls"}}},
                    {"id": "c2", "function": {"name": "submit", "arguments": {}}},
                ],
            },
        ])
        blocks = msgs[0]["content"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["name"] == "bash"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["name"] == "submit"

    def test_plain_assistant_message(self):
        model = _make_claude_model()
        _, msgs = model.format_messages([
            {"role": "assistant", "content": "just text"},
        ])
        assert msgs[0] == {"role": "assistant", "content": "just text"}


# ---------------------------------------------------------------------------
# AmdClaudeModel._parse_response
# ---------------------------------------------------------------------------


class TestAmdClaudeParseResponse:
    def test_text_only(self):
        model = _make_claude_model()
        response = MagicMock()
        response.content = [SimpleNamespace(type="text", text="Hello world")]
        result = model._parse_response(response)
        assert result["content"] == "Hello world"
        assert result["tools"] == ""

    def test_tool_use(self):
        model = _make_claude_model()
        text_block = SimpleNamespace(type="text", text="Let me run that")
        tool_block = SimpleNamespace(type="tool_use", id="call_1", name="bash", input={"cmd": "ls"})
        response = MagicMock()
        response.content = [text_block, tool_block]
        result = model._parse_response(response)
        assert result["content"] == "Let me run that"
        tools = result["tools"]
        assert isinstance(tools, dict)
        assert tools["id"] == "call_1"  # pylint: disable=invalid-sequence-index
        assert tools["function"]["name"] == "bash"  # pylint: disable=invalid-sequence-index
        assert tools["function"]["arguments"] == {"cmd": "ls"}  # pylint: disable=invalid-sequence-index

    def test_thinking_block_captured(self):
        model = _make_claude_model()
        thinking_block = SimpleNamespace(type="thinking", thinking="step 1: analyze\nstep 2: plan")
        text_block = SimpleNamespace(type="text", text="Here is the answer")
        response = MagicMock()
        response.content = [thinking_block, text_block]
        result = model._parse_response(response)
        assert result["content"] == "Here is the answer"
        assert result["thinking"] == "step 1: analyze\nstep 2: plan"

    def test_thinking_block_not_in_content(self):
        model = _make_claude_model()
        thinking_block = SimpleNamespace(type="thinking", thinking="internal reasoning")
        text_block = SimpleNamespace(type="text", text="visible output")
        response = MagicMock()
        response.content = [thinking_block, text_block]
        result = model._parse_response(response)
        assert "internal reasoning" not in result["content"]

    def test_no_thinking_key_when_absent(self):
        model = _make_claude_model()
        response = MagicMock()
        response.content = [SimpleNamespace(type="text", text="hello")]
        result = model._parse_response(response)
        assert "thinking" not in result

    def test_empty_content(self):
        model = _make_claude_model()
        response = MagicMock()
        response.content = []
        result = model._parse_response(response)
        assert result["content"] == ""
        assert result["tools"] == ""


# ---------------------------------------------------------------------------
# AmdOpenAIModel.format_messages
# ---------------------------------------------------------------------------


def _make_openai_model():
    """Create an AmdOpenAIModel with a mocked client."""
    config = AmdLlmModelConfig(model_name="gpt-5", api_key="test-key")
    with patch.object(AmdOpenAIModel, "_init_client"):
        return AmdOpenAIModel(config)


class TestAmdOpenAIFormatMessages:
    def test_basic_messages(self):
        model = _make_openai_model()
        result = model.format_messages([
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "hello"},
        ])
        assert result[0] == {"role": "system", "content": "sys prompt"}
        assert result[1] == {"role": "user", "content": "hello"}

    def test_tool_result_becomes_function_call_output(self):
        model = _make_openai_model()
        result = model.format_messages([
            {"role": "tool", "content": "output text", "tool_call_id": "call_1"},
        ])
        assert result[0]["type"] == "function_call_output"
        assert result[0]["call_id"] == "call_1"
        assert result[0]["output"] == "output text"

    def test_assistant_with_tool_calls(self):
        model = _make_openai_model()
        result = model.format_messages([
            {
                "role": "assistant",
                "content": "thinking...",
                "tool_calls": {
                    "id": "call_1",
                    "function": {"name": "bash", "arguments": {"cmd": "ls"}},
                },
            },
        ])
        assert result[0] == {"role": "assistant", "content": "thinking..."}
        assert result[1]["type"] == "function_call"
        assert result[1]["name"] == "bash"
        assert result[1]["arguments"] == json.dumps({"cmd": "ls"})


# ---------------------------------------------------------------------------
# AmdOpenAIModel._parse_response
# ---------------------------------------------------------------------------


class TestAmdOpenAIParseResponse:
    def test_text_content(self):
        model = _make_openai_model()
        item = SimpleNamespace(type="output_text", text="Hello world")
        out_msg = SimpleNamespace(content=[item])
        response = MagicMock()
        response.output = [out_msg]
        result = model._parse_response(response)
        assert result["content"] == "Hello world"
        assert result["tools"] == ""

    def test_function_call(self):
        model = _make_openai_model()
        fc = SimpleNamespace(type="function_call", call_id="c1", name="bash", arguments='{"cmd": "ls"}')
        response = MagicMock()
        response.output = [fc]
        result = model._parse_response(response)
        tools = result["tools"]
        assert isinstance(tools, dict)
        assert tools["id"] == "c1"  # pylint: disable=invalid-sequence-index
        assert tools["function"]["name"] == "bash"  # pylint: disable=invalid-sequence-index
        assert tools["function"]["arguments"] == {"cmd": "ls"}  # pylint: disable=invalid-sequence-index


# ---------------------------------------------------------------------------
# AmdLlmModel router
# ---------------------------------------------------------------------------


class TestAmdLlmModelRouter:
    def test_set_tools_forwarded(self):
        with patch.object(AmdClaudeModel, "_init_client"):
            model = AmdLlmModel(model_name="claude-sonnet-4.5", api_key="test-key")
        new_tools = [{"name": "submit", "description": "Submit", "parameters": {"type": "object"}}]
        model.set_tools(new_tools)
        assert model._impl.tools == new_tools

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            AmdLlmModel(model_name="llama-3", api_key="test-key")


# ---------------------------------------------------------------------------
# AmdLlmModelBase.set_tools
# ---------------------------------------------------------------------------


class TestAmdLlmModelBaseSetTools:
    def test_set_tools_stores_list_as_given(self):
        config = AmdLlmModelConfig(model_name="claude-test", api_key="key")
        with patch.object(AmdClaudeModel, "_init_client"):
            model = AmdClaudeModel(config)
        tools = [
            {"name": "bash", "description": "bash"},
            {"name": "profiling", "description": "profiling"},
            {"name": "submit", "description": "submit"},
        ]
        model.set_tools(tools)
        names = [t["name"] for t in model.tools]
        assert names == ["bash", "profiling", "submit"]
        assert model.tools is not tools
