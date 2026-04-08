"""Unit tests for LiteLLM model utilities: message normalisation, tool conversion, response parsing."""

import json
from types import SimpleNamespace

from minisweagent.models.litellm_model import (
    LITELLM_COMPLETION_PARAM_KEYS,
    _coerce_tool_arguments,
    _first_function_tool_call,
    _openai_tool_call_to_api_shape,
    convert_openai_tools_to_litellm,
    normalize_messages_for_litellm_api,
)

# ---------------------------------------------------------------------------
# convert_openai_tools_to_litellm
# ---------------------------------------------------------------------------


class TestConvertOpenaiToolsToLitellm:
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
        result = convert_openai_tools_to_litellm(self.TOOLS)
        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "bash"
        assert result[1]["function"]["name"] == "submit"

    def test_input_schema_used(self):
        result = convert_openai_tools_to_litellm(self.TOOLS)
        assert result[0]["function"]["input_schema"]["properties"]["cmd"]["type"] == "string"

    def test_skips_tools_without_name(self):
        tools = [{"function": {"description": "no name"}}]
        assert convert_openai_tools_to_litellm(tools) == []

    def test_cache_control_off_by_default(self):
        result = convert_openai_tools_to_litellm(self.TOOLS)
        assert "cache_control" not in result[-1]

    def test_cache_control_on(self):
        result = convert_openai_tools_to_litellm(self.TOOLS, tool_cache_control=True)
        assert result[-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[0]

    def test_empty_tools_with_cache_control(self):
        result = convert_openai_tools_to_litellm([], tool_cache_control=True)
        assert result == []

    def test_flat_tool_dict_without_function_wrapper(self):
        tools = [{"name": "bash", "description": "Run bash", "parameters": {"type": "object"}}]
        result = convert_openai_tools_to_litellm(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "bash"


# ---------------------------------------------------------------------------
# normalize_messages_for_litellm_api
# ---------------------------------------------------------------------------


class TestNormalizeMessagesForLitellmApi:
    def test_passes_through_non_assistant(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "system", "content": "sys"}]
        result = normalize_messages_for_litellm_api(msgs)
        assert result == msgs

    def test_wraps_single_dict_tool_calls_in_list(self):
        msgs = [
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": {
                    "id": "call_1",
                    "function": {"name": "bash", "arguments": {"cmd": "ls"}},
                },
            }
        ]
        result = normalize_messages_for_litellm_api(msgs)
        tc = result[0]["tool_calls"]
        assert isinstance(tc, list) and len(tc) == 1
        assert tc[0]["type"] == "function"
        assert tc[0]["function"]["arguments"] == json.dumps({"cmd": "ls"})

    def test_normalises_list_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "bash", "arguments": {"cmd": "echo"}}},
                ],
            }
        ]
        result = normalize_messages_for_litellm_api(msgs)
        tc = result[0]["tool_calls"]
        assert isinstance(tc, list) and len(tc) == 1
        assert tc[0]["type"] == "function"

    def test_assistant_without_tool_calls_unchanged(self):
        msgs = [{"role": "assistant", "content": "just text"}]
        result = normalize_messages_for_litellm_api(msgs)
        assert result[0] == msgs[0]

    def test_does_not_mutate_original(self):
        original_args = {"cmd": "ls"}
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": {"id": "c1", "function": {"name": "bash", "arguments": original_args}},
            }
        ]
        normalize_messages_for_litellm_api(msgs)
        assert msgs[0]["tool_calls"]["function"]["arguments"] == original_args


# ---------------------------------------------------------------------------
# _first_function_tool_call
# ---------------------------------------------------------------------------


class TestFirstFunctionToolCall:
    def test_extracts_first_tool_call(self):
        fn = SimpleNamespace(name="bash", arguments='{"cmd": "ls"}')
        call = SimpleNamespace(type="function", function=fn, id="call_1")
        message = SimpleNamespace(tool_calls=[call])
        result = _first_function_tool_call(message)
        assert result["id"] == "call_1"
        assert result["function"]["name"] == "bash"
        assert result["function"]["arguments"] == {"cmd": "ls"}

    def test_returns_empty_when_no_tool_calls(self):
        message = SimpleNamespace(tool_calls=None)
        assert _first_function_tool_call(message) == {}

    def test_returns_empty_when_tool_calls_empty(self):
        message = SimpleNamespace(tool_calls=[])
        assert _first_function_tool_call(message) == {}

    def test_skips_non_function_types(self):
        call = SimpleNamespace(type="code_interpreter", function=None, id="c1")
        fn = SimpleNamespace(name="bash", arguments="{}")
        func_call = SimpleNamespace(type="function", function=fn, id="c2")
        message = SimpleNamespace(tool_calls=[call, func_call])
        result = _first_function_tool_call(message)
        assert result["id"] == "c2"

    def test_handles_dict_arguments(self):
        fn = SimpleNamespace(name="bash", arguments={"cmd": "echo"})
        call = SimpleNamespace(type="function", function=fn, id="c1")
        message = SimpleNamespace(tool_calls=[call])
        result = _first_function_tool_call(message)
        assert result["function"]["arguments"] == {"cmd": "echo"}


# ---------------------------------------------------------------------------
# _openai_tool_call_to_api_shape
# ---------------------------------------------------------------------------


class TestOpenaiToolCallToApiShape:
    def test_adds_type_function(self):
        result = _openai_tool_call_to_api_shape({"id": "c1", "function": {"name": "bash", "arguments": "{}"}})
        assert result["type"] == "function"

    def test_serialises_dict_arguments(self):
        result = _openai_tool_call_to_api_shape(
            {"id": "c1", "function": {"name": "bash", "arguments": {"cmd": "ls"}}}
        )
        assert result["function"]["arguments"] == json.dumps({"cmd": "ls"})

    def test_none_arguments_become_empty_json(self):
        result = _openai_tool_call_to_api_shape({"id": "c1", "function": {"name": "bash", "arguments": None}})
        assert result["function"]["arguments"] == "{}"

    def test_missing_function_gets_default(self):
        result = _openai_tool_call_to_api_shape({"id": "c1"})
        assert result["function"] == {"name": "", "arguments": "{}"}


# ---------------------------------------------------------------------------
# _coerce_tool_arguments
# ---------------------------------------------------------------------------


class TestCoerceToolArguments:
    def test_none_returns_empty(self):
        assert _coerce_tool_arguments(None) == {}

    def test_dict_passes_through(self):
        d = {"a": 1}
        assert _coerce_tool_arguments(d) is d

    def test_json_string_parsed(self):
        assert _coerce_tool_arguments('{"a": 1}') == {"a": 1}

    def test_empty_string_returns_empty(self):
        assert _coerce_tool_arguments("") == {}


# ---------------------------------------------------------------------------
# LITELLM_COMPLETION_PARAM_KEYS includes expected keys
# ---------------------------------------------------------------------------


class TestCompletionParamKeys:
    def test_thinking_key_present(self):
        assert "thinking" in LITELLM_COMPLETION_PARAM_KEYS

    def test_reasoning_key_present(self):
        assert "reasoning" in LITELLM_COMPLETION_PARAM_KEYS

    def test_tools_key_present(self):
        assert "tools" in LITELLM_COMPLETION_PARAM_KEYS
