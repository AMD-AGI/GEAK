# Modifications Copyright(C)[2026] Advanced Micro Devices, Inc. All rights reserved.

import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
import logging
import anthropic
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.key_per_thread import get_key_per_thread
from minisweagent.tools.tools_runtime import get_tools_list

logger = logging.getLogger("anthropic_model")
CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}


@dataclass
class AnthropicModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    api_key: str | None = None
    base_url: str | None = None
    api_version: str = "2023-06-01"
    set_cache_control: Literal["default_end"] | None = "default_end"
    cost_per_1k_input_tokens: float = 0.003
    cost_per_1k_output_tokens: float = 0.003
    bash_tool: bool = True
    profiling: bool = False
    use_strategy_manager: bool = False


def convert_openai_tools_to_claude(tools: list[dict], *, cache_control: bool = True) -> list[dict]:
    """Convert OpenAI-style tool definitions to Anthropic format."""
    claude_tools = []
    for tool in tools:
        func = tool.get("function", tool)
        claude_tool = {
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get(
                "parameters",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        }
        claude_tools.append(claude_tool)
    if cache_control and claude_tools:
        claude_tools[-1]["cache_control"] = CACHE_CONTROL_EPHEMERAL
    return claude_tools


class AnthropicModel:
    """Direct Anthropic Claude API model.

    This class is not auto-selected by `get_model` unless explicitly requested
    via `model_class=anthropic_model`.
    """

    def __init__(self, **kwargs):
        self.config = AnthropicModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self.tools = get_tools_list(use_strategy_manager=self.config.use_strategy_manager)
        if not self.config.profiling:
            self.tools = [tool for tool in self.tools if tool["name"] != "profiling"]
        if not self.config.bash_tool:
            self.tools = [tool for tool in self.tools if tool["name"] != "bash"]
        self._init_client()

    def _resolve_api_key(self) -> str:
        if self.config.api_key:
            return self.config.api_key
        if key := os.getenv("ANTHROPIC_API_KEY"):
            return key
        # Legacy only: rotating key list
        if rotating_keys := os.getenv("ANTHROPIC_API_KEYS"):
            warnings.warn(
                "ANTHROPIC_API_KEYS is deprecated and will be removed in the future. "
                "Simply use the ANTHROPIC_API_KEY environment variable instead. "
                "Key rotation is no longer required."
            )
            return get_key_per_thread(rotating_keys.split("::"))
        raise ValueError(
            "Anthropic API key not provided. Set it via model config `api_key`, "
            "or environment variable ANTHROPIC_API_KEY."
        )

    def _init_client(self):
        api_key = self._resolve_api_key()
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=self.config.base_url,
            default_headers={"anthropic-version": self.config.api_version},
        )

    def format_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert standard messages to Anthropic message format."""
        system_message: str | None = None
        anthropic_messages: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_message = content
                continue

            if role == "tool":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": content,
                            }
                        ],
                    }
                )
                continue

            if role == "assistant" and (msg.get("tool_calls") or msg.get("tools")):
                content_blocks: list[dict] = []
                if content:
                    content_blocks.append({"type": "text", "text": content})

                tool_info = msg.get("tool_calls") or msg.get("tools")
                if isinstance(tool_info, list):
                    tool_info = tool_info[0] if tool_info else {}
                func = tool_info.get("function", {}) if isinstance(tool_info, dict) else {}
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}

                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_info.get("id", "") if isinstance(tool_info, dict) else "",
                        "name": tool_name,
                        "input": tool_args,
                    }
                )
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
                continue

            anthropic_role = "assistant" if role == "assistant" else "user"
            anthropic_messages.append({"role": anthropic_role, "content": content})

        return system_message, anthropic_messages

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=4, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt, anthropic.AuthenticationError, anthropic.NotFoundError)),
    )
    def _query_api(self, messages: list[dict], **kwargs):
        supported_params = {
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "stop_sequences",
            "stream",
            "metadata",
            "system",
            "tools",
            "tool_choice",
        }
        all_kwargs = self.config.model_kwargs | kwargs
        filtered_kwargs = {k: v for k, v in all_kwargs.items() if k in supported_params}

        if "max_tokens" not in filtered_kwargs:
            filtered_kwargs["max_tokens"] = 4096

        filtered_kwargs["tools"] = convert_openai_tools_to_claude(self.tools, cache_control=True)

        system_message, anthropic_messages = self.format_messages(messages)
        if self.config.set_cache_control:
            if system_message and "system" not in filtered_kwargs:
                filtered_kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_message,
                        "cache_control": CACHE_CONTROL_EPHEMERAL,
                    }
                ]

            for msg in reversed(anthropic_messages):
                if msg.get("role") != "user":
                    continue
                msg_content = msg.get("content")
                if isinstance(msg_content, list) and msg_content:
                    msg_content[-1] = {**msg_content[-1], "cache_control": CACHE_CONTROL_EPHEMERAL}
                elif isinstance(msg_content, str) and msg_content:
                    msg["content"] = [
                        {
                            "type": "text",
                            "text": msg_content,
                            "cache_control": CACHE_CONTROL_EPHEMERAL,
                        }
                    ]
                break
        elif system_message and "system" not in filtered_kwargs:
            filtered_kwargs["system"] = system_message

        return self.client.messages.create(
            model=self.config.model_name,
            messages=anthropic_messages,
            **filtered_kwargs,
        )

    def _parse_response(self, response) -> dict:
        output_dict: dict = {"content": "", "tools": ""}
        try:
            if response.content:
                text_parts: list[str] = []
                for block in response.content:
                    block_text = getattr(block, "text", None)
                    if isinstance(block_text, str) and block_text:
                        text_parts.append(block_text)
                output_dict["content"] = "".join(text_parts)

                for block in response.content:
                    if getattr(block, "type", "") == "tool_use":
                        output_dict["tools"] = {
                            "id": getattr(block, "id", ""),
                            "function": {
                                "arguments": getattr(block, "input", {}),
                                "name": getattr(block, "name", ""),
                            },
                        }
                        break
        except Exception as e:
            logger.warning(f"Failed to parse anthropic response content: {e}")
        return output_dict

    def query(self, messages: list[dict], **kwargs) -> dict:
        response = self._query_api(messages, **kwargs)
        content = self._parse_response(response)

        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            cost = (
                (input_tokens / 1000) * self.config.cost_per_1k_input_tokens
                + (output_tokens / 1000) * self.config.cost_per_1k_output_tokens
            )
        else:
            cost = 0.0

        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        try:
            if hasattr(response, "model_dump"):
                response_dump = response.model_dump()
            elif hasattr(response, "to_dict"):
                response_dump = response.to_dict()
            elif hasattr(response, "dict"):
                response_dump = response.dict()
            else:
                response_dump = str(response)
        except Exception:
            response_dump = str(response)

        content["extra"] = {"response": response_dump}
        return content

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
