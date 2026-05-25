"""LiteLLM-backed model

Swap usage (when ready)::

    from minisweagent.models.new_litellm_model import NewLitellmModel as LitellmModel
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.amd_base import AmdLlmModelConfig, get_amd_llm_user
from minisweagent.models.utils.cache_control import set_cache_control

# ---------------------------------------------------------------------------
# Logging — ``getLogger(...).setLevel()`` returns None; keep a real Logger.
# ---------------------------------------------------------------------------

logger = logging.getLogger("LiteLLM")
logger.setLevel(logging.WARNING)

CACHE_CONTROL_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}

# AMD LLM gateway hostname prefixes.  The ``"user"`` request header is only
# meaningful (and only consumed for request attribution) when the call is
# routed through one of these gateways; attaching the local OS username to
# arbitrary third-party providers (e.g. OpenAI direct, Anthropic direct) would
# leak the operator's identity over the wire for no benefit.
_AMD_LLM_GATEWAY_API_BASE_PREFIXES: tuple[str, ...] = (
    "https://llm-api.amd.com/",
    "https://llm-gateway-dev.apps.amdcloud.com/",
)


def _is_amd_llm_gateway_api_base(api_base: Any) -> bool:
    """Return True if *api_base* points at an AMD LLM gateway endpoint.

    Used by :class:`LitellmModel` to gate the ``"user"`` request header so
    only gateway-routed traffic carries the operator identity.
    """
    if not isinstance(api_base, str) or not api_base:
        return False
    return any(api_base.startswith(p) for p in _AMD_LLM_GATEWAY_API_BASE_PREFIXES)


# Parameters forwarded to ``litellm.completion`` (extend when providers add flags).
LITELLM_COMPLETION_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "stop_sequences",
        "stream",
        "metadata",
        "system",
        "tools",
        "api_key",
        "api_base",
        "extra_headers",
        "drop_params",
        # OpenAI / gateway “reasoning effort” style configs used in GEAK YAML.
        "reasoning_effort",
        "reasoning",
        # Anthropic extended thinking (e.g. {"type": "enabled", "budget_tokens": 10000}).
        "thinking",
        "text",
    }
)


def convert_openai_tools_to_litellm(
    tools: list[dict[str, Any]],
    *,
    tool_cache_control: bool = False,
) -> list[dict[str, Any]]:
    """Map OpenAI-style tool definitions to the shape LiteLLM expects.

    When *tool_cache_control* is True, the last tool receives Anthropic-style
    ``cache_control`` (parity with :func:`amd_claude.convert_openai_tools_to_claude`).
    """
    litellm_tools: list[dict[str, Any]] = []
    for raw in tools:
        func = raw.get("function", raw)
        name = func.get("name")
        if not name:
            continue
        function: dict[str, Any] = {
            "name": name,
            "description": func.get("description", ""),
            "parameters": func.get(
                "parameters",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        }
        entry: dict[str, Any] = {"type": "function", "function": function}
        litellm_tools.append(entry)

    if tool_cache_control and litellm_tools:
        # Same placement as ``claude_tools[-1]["cache_control"]`` in amd_claude.
        litellm_tools[-1]["cache_control"] = CACHE_CONTROL_EPHEMERAL

    return litellm_tools


def _ensure_nested_model_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Guarantee ``kwargs['model_kwargs']`` exists before in-place api_base writes."""
    mk = kwargs.get("model_kwargs")
    if mk is None:
        kwargs["model_kwargs"] = {}
    elif not isinstance(mk, dict):
        raise TypeError(f"model_kwargs must be a dict, got {type(mk).__name__}")
    return kwargs


def _normalize_api_base_from_kwargs(kwargs: dict[str, Any]) -> None:
    """Copy ``base_url`` from top-level or nested ``model_kwargs`` into ``api_base``."""
    model_kwargs = kwargs["model_kwargs"]
    base_url = model_kwargs.get("base_url")
    if base_url is not None:
        model_kwargs["api_base"] = base_url
    top_base = kwargs.get("base_url")
    if top_base is not None:
        model_kwargs["api_base"] = top_base


def _coerce_tool_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw) if raw else {}
    return json.loads(str(raw))


def _first_function_tool_call(message: Any) -> dict[str, Any]:
    """Build the legacy single-tool dict from the chat completion message."""
    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        if getattr(call, "type", "function") != "function":
            continue
        fn = getattr(call, "function", None)
        if fn is None:
            continue
        name = getattr(fn, "name", None) or ""
        args = _coerce_tool_arguments(getattr(fn, "arguments", None))
        return {
            "id": getattr(call, "id", "") or "",
            "function": {"arguments": args, "name": name},
        }
    return {}


def _openai_tool_call_to_api_shape(call: dict[str, Any]) -> dict[str, Any]:
    """Ensure one tool call matches OpenAI chat-completions shape for LiteLLM → Anthropic."""
    out = dict(call)
    if out.get("type") != "function":
        out["type"] = "function"
    fn = out.get("function")
    if not isinstance(fn, dict):
        fn = {"name": "", "arguments": "{}"}
        out["function"] = fn
    else:
        fn = dict(fn)
        args = fn.get("arguments")
        if isinstance(args, dict):
            fn["arguments"] = json.dumps(args)
        elif args is None:
            fn["arguments"] = "{}"
        elif not isinstance(args, str):
            fn["arguments"] = json.dumps(args)
        out["function"] = fn
    return out


def normalize_messages_for_litellm_api(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy *messages* and coerce assistant ``tool_calls`` to OpenAI list form.

    GEAK's :class:`~minisweagent.agents.default.DefaultAgent` stores a single
    ``response["tools"]`` dict under ``tool_calls``. LiteLLM's Anthropic / Vertex
    path expects ``tool_calls`` to be a **list** of ``{"type":"function",...}``
    objects; otherwise ``tool_use`` blocks are not emitted and the following
    ``role: tool`` turns into orphan ``tool_result`` blocks (400 invalid_request).
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        m = copy.deepcopy(msg)
        if m.get("role") != "assistant":
            out.append(m)
            continue
        tc = m.get("tool_calls")
        if tc is None:
            out.append(m)
            continue
        if isinstance(tc, dict):
            m["tool_calls"] = [_openai_tool_call_to_api_shape(tc)]
        elif isinstance(tc, list):
            m["tool_calls"] = [_openai_tool_call_to_api_shape(x) if isinstance(x, dict) else x for x in tc]
        out.append(m)
    return out


def _register_litellm_registry(path: Path | str | None) -> None:
    if not path:
        return
    p = Path(path)
    if p.is_file():
        litellm.utils.register_model(json.loads(p.read_text(encoding="utf-8")))


@dataclass
class LitellmModelConfig(AmdLlmModelConfig):
    """Configuration for :class:`NewLitellmModel`."""

    set_cache_control: Literal["default_end"] | None = None
    """Override parent default: LiteLLM should NOT apply cache control unless explicitly set
    (e.g. by ``get_model`` for Anthropic models). The parent ``AmdLlmModelConfig`` defaults to
    ``"default_end"`` which would incorrectly apply cache control to non-Anthropic models."""

    litellm_model_registry: Path | str | None = field(default_factory=lambda: os.getenv("LITELLM_MODEL_REGISTRY_PATH"))
    litellm_model_name_override: str = ""
    """If set, used only for ``litellm.cost_calculator.completion_cost`` (Portkey-style parity)."""
    tool_cache_control: bool = False
    """Attach ``cache_control`` to the last tool definition (Anthropic prompt caching)."""


def _merge_completion_kwargs(
    config: LitellmModelConfig,
    call_kwargs: dict[str, Any],
) -> dict[str, Any]:
    merged = config.model_kwargs | call_kwargs | asdict(config)
    filtered = {k: v for k, v in merged.items() if k in LITELLM_COMPLETION_PARAM_KEYS}
    # Omit api_key from the completion call when unset so LiteLLM can read provider keys
    # from the environment (e.g. OPENAI_API_KEY, AZURE_API_KEY); passing None/"" would override that.
    ak = filtered.get("api_key")
    if ak is None or (isinstance(ak, str) and not ak.strip()):
        filtered.pop("api_key", None)
    return filtered


class LitellmModel:
    """Query models through LiteLLM with GEAK-compatible tool and cost accounting."""

    def __init__(self, **kwargs: Any) -> None:
        self.cost = 0.0
        self.n_calls = 0

        # Subclass hook (e.g. AnthropicModel passing config_class) — not a dataclass field.
        kwargs.pop("config_class", None)

        _ensure_nested_model_kwargs(kwargs)
        model_kwargs: dict[str, Any] = kwargs["model_kwargs"]

        if model_kwargs.get("api_key") is not None and kwargs.get("api_key") is None:
            kwargs["api_key"] = model_kwargs["api_key"]

        _normalize_api_base_from_kwargs(kwargs)

        self.config = LitellmModelConfig(**kwargs)
        # Populated by :meth:`set_tools`; :class:`~minisweagent.agents.default.DefaultAgent`
        # passes ``toolruntime.get_tools_list()`` after constructing :class:`~minisweagent.tools.tools_runtime.ToolRuntime`.
        self.tools: list[dict[str, Any]] = []
        _register_litellm_registry(self.config.litellm_model_registry)

    def set_tools(self, tools: list[dict[str, Any]]) -> None:
        """Replace the active OpenAI-style tool definitions (no model-side filtering).

        Use the same list as :meth:`ToolRuntime.get_tools_list
        <minisweagent.tools.tools_runtime.ToolRuntime.get_tools_list>` from the
        agent's runtime so tool names match :meth:`ToolRuntime.dispatch`.
        """
        self.tools = list(tools)

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        filtered = _merge_completion_kwargs(self.config, kwargs)
        filtered["tools"] = convert_openai_tools_to_litellm(
            self.tools,
            tool_cache_control=self.config.tool_cache_control,
        )
        existing_headers = filtered.get("extra_headers") or {}
        if existing_headers.get("user"):
            # Caller-provided override wins regardless of provider — keep as-is.
            filtered["extra_headers"] = dict(existing_headers)
        elif _is_amd_llm_gateway_api_base(filtered.get("api_base")):
            filtered["extra_headers"] = {
                **existing_headers,
                "user": get_amd_llm_user(),
            }
        else:
            # Non-AMD-gateway provider: do NOT attach the local OS username
            # as a request header. Preserve any other caller-supplied
            # ``extra_headers`` verbatim.
            filtered["extra_headers"] = dict(existing_headers)
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                **filtered,
            )
        except litellm.exceptions.AuthenticationError as e:
            hint = " You can permanently set your API key with `config set KEY VALUE`."
            msg = getattr(e, "message", None)
            if isinstance(msg, str):
                e.message = msg + hint
            elif e.args:
                e.args = (str(e.args[0]) + hint,) + e.args[1:]
            raise

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        messages = normalize_messages_for_litellm_api(messages)

        response = self._query(messages, **kwargs)

        response_for_cost = response
        if self.config.litellm_model_name_override:
            try:
                response_for_cost = response.model_copy(deep=False)
            except (AttributeError, TypeError):
                response_for_cost = response
            if getattr(response_for_cost, "model", None) is not None:
                response_for_cost.model = self.config.litellm_model_name_override

        try:
            cost = litellm.cost_calculator.completion_cost(
                response_for_cost,
                model=self.config.litellm_model_name_override or None,
            )
        except Exception as e:
            logger.warning(
                "Error calculating cost for model %s: %s. "
                "See 'Updating the model registry' at https://klieret.short.gy/litellm-model-registry",
                self.config.model_name,
                e,
            )
            cost = 0.0

        self.n_calls += 1
        assert cost >= 0.0, f"Cost is negative: {cost}"
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        message = response.choices[0].message
        tool_dict = _first_function_tool_call(message)

        return {
            "content": message.content or "",
            "tools": tool_dict,
            "extra": {"response": response.model_dump()},
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}


if __name__ == "__main__":
    # Quick smoke test
    from minisweagent.tools.tools_runtime import ToolRuntime

    model_configs = [
        {
            "model_name": "openai/gpt-5",
            "model_kwargs": {
                "extra_headers": {"Ocp-Apim-Subscription-Key": ""},
                "temperature": 1.0,
                "max_tokens": 16000,
                "api_key": "",
                "api_base": "https://llm-api.amd.com/azure/engines/gpt-5",
            },
        },
        {
            "model_name": "anthropic/claude-opus-4.6",
            "model_kwargs": {
                "extra_headers": {"Ocp-Apim-Subscription-Key": ""},
                "temperature": 1.0,
                "max_tokens": 16000,
                "api_key": "",
                "api_base": "https://llm-api.amd.com/Anthropic",
            },
        },
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"},
        {
            "role": "user",
            "content": "Use tools I provided to you to create a silu kernel",
        },
    ]
    for model_config in model_configs:
        model = LitellmModel(**model_config)
        model.set_tools(ToolRuntime.fetch_tools_list())
        response = model.query(messages)
        print(response)
