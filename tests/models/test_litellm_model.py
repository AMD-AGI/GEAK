"""Unit tests for LiteLLM model utilities: message normalisation, tool conversion, response parsing."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from minisweagent.models.litellm_model import (
    LITELLM_COMPLETION_PARAM_KEYS,
    LitellmModel,
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
        result = _openai_tool_call_to_api_shape({"id": "c1", "function": {"name": "bash", "arguments": {"cmd": "ls"}}})
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


# ---------------------------------------------------------------------------
# LitellmModel injects the AMD LLM gateway "user" header
# ---------------------------------------------------------------------------


def _fake_litellm_response() -> MagicMock:
    """Build a minimal response object covering the fields LitellmModel.query reads."""
    message = SimpleNamespace(content="hello", tool_calls=None)
    choice = SimpleNamespace(message=message)
    response = MagicMock()
    response.choices = [choice]
    response.model = "test-model"
    response.model_dump.return_value = {}
    return response


_AMD_GATEWAY_API_BASE = "https://llm-api.amd.com/Anthropic"
_NON_AMD_API_BASE = "https://api.openai.com/v1"


class TestLitellmModelUserHeader:
    """The ``"user"`` request header is only attached to AMD LLM gateway
    traffic — see :func:`_is_amd_llm_gateway_api_base`.  For non-gateway
    providers we must NOT leak the operator's local username over the
    wire."""

    def test_injects_resolved_user_when_no_headers(self, monkeypatch):
        monkeypatch.setenv("GEAK_USER", "alice")
        model = LitellmModel(
            model_name="anthropic/claude-opus-4.6",
            model_kwargs={"api_key": "k", "api_base": _AMD_GATEWAY_API_BASE},
        )
        with (
            patch(
                "minisweagent.models.litellm_model.litellm.completion", return_value=_fake_litellm_response()
            ) as comp,
            patch(
                "minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost",
                return_value=0.0,
            ),
        ):
            model.query([{"role": "user", "content": "hi"}])
        headers = comp.call_args.kwargs["extra_headers"]
        assert headers["user"] == "alice"

    def test_preserves_explicit_user_override(self, monkeypatch):
        """Caller-provided ``"user"`` wins for any provider (AMD gateway
        or not) — the override is what's intentional, not the default."""
        monkeypatch.setenv("GEAK_USER", "alice")
        model = LitellmModel(
            model_name="anthropic/claude-opus-4.6",
            model_kwargs={
                "api_key": "k",
                "api_base": _AMD_GATEWAY_API_BASE,
                "extra_headers": {"user": "explicit", "Ocp-Apim-Subscription-Key": "k"},
            },
        )
        with (
            patch(
                "minisweagent.models.litellm_model.litellm.completion", return_value=_fake_litellm_response()
            ) as comp,
            patch(
                "minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost",
                return_value=0.0,
            ),
        ):
            model.query([{"role": "user", "content": "hi"}])
        headers = comp.call_args.kwargs["extra_headers"]
        assert headers["user"] == "explicit"
        assert headers["Ocp-Apim-Subscription-Key"] == "k"

    def test_merges_with_other_extra_headers(self, monkeypatch):
        monkeypatch.setenv("GEAK_USER", "alice")
        model = LitellmModel(
            model_name="anthropic/claude-opus-4.6",
            model_kwargs={
                "api_key": "k",
                "api_base": _AMD_GATEWAY_API_BASE,
                "extra_headers": {"Ocp-Apim-Subscription-Key": "k"},
            },
        )
        with (
            patch(
                "minisweagent.models.litellm_model.litellm.completion", return_value=_fake_litellm_response()
            ) as comp,
            patch(
                "minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost",
                return_value=0.0,
            ),
        ):
            model.query([{"role": "user", "content": "hi"}])
        headers = comp.call_args.kwargs["extra_headers"]
        assert headers["user"] == "alice"
        assert headers["Ocp-Apim-Subscription-Key"] == "k"

    def test_does_not_inject_user_for_non_amd_gateway(self, monkeypatch):
        """Calling a non-AMD-gateway provider (e.g. openai.com direct)
        must not leak the local OS username as an HTTP header.
        Regression guard for PR #226 review note."""
        monkeypatch.setenv("GEAK_USER", "alice")
        model = LitellmModel(
            model_name="openai/gpt-4o",
            model_kwargs={"api_key": "k", "api_base": _NON_AMD_API_BASE},
        )
        with (
            patch(
                "minisweagent.models.litellm_model.litellm.completion", return_value=_fake_litellm_response()
            ) as comp,
            patch(
                "minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost",
                return_value=0.0,
            ),
        ):
            model.query([{"role": "user", "content": "hi"}])
        headers = comp.call_args.kwargs["extra_headers"]
        assert "user" not in headers, (
            "LitellmModel must NOT attach the local OS username as a "
            "request header for non-AMD-gateway providers; got headers="
            f"{headers!r}"
        )

    def test_non_amd_preserves_other_extra_headers(self, monkeypatch):
        """For non-AMD-gateway providers, other caller-supplied
        ``extra_headers`` must still pass through verbatim."""
        monkeypatch.setenv("GEAK_USER", "alice")
        model = LitellmModel(
            model_name="openai/gpt-4o",
            model_kwargs={
                "api_key": "k",
                "api_base": _NON_AMD_API_BASE,
                "extra_headers": {"X-Trace-Id": "abc123"},
            },
        )
        with (
            patch(
                "minisweagent.models.litellm_model.litellm.completion", return_value=_fake_litellm_response()
            ) as comp,
            patch(
                "minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost",
                return_value=0.0,
            ),
        ):
            model.query([{"role": "user", "content": "hi"}])
        headers = comp.call_args.kwargs["extra_headers"]
        assert headers.get("X-Trace-Id") == "abc123"
        assert "user" not in headers

    def test_dev_gateway_also_gets_user_header(self, monkeypatch):
        """The dev gateway hostname (used by ``scripts/run-docker.sh``
        default ``AMD_LLM_BASE_URL``) is also recognised as AMD-gateway
        traffic and gets the ``user`` header attribution."""
        monkeypatch.setenv("GEAK_USER", "alice")
        model = LitellmModel(
            model_name="anthropic/claude-opus-4.6",
            model_kwargs={
                "api_key": "k",
                "api_base": "https://llm-gateway-dev.apps.amdcloud.com/api/gateway/v1",
            },
        )
        with (
            patch(
                "minisweagent.models.litellm_model.litellm.completion", return_value=_fake_litellm_response()
            ) as comp,
            patch(
                "minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost",
                return_value=0.0,
            ),
        ):
            model.query([{"role": "user", "content": "hi"}])
        headers = comp.call_args.kwargs["extra_headers"]
        assert headers["user"] == "alice"
