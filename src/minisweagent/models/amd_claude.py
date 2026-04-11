"""Anthropic (Claude) model implementation for AMD LLM gateway."""

import logging
import os

import anthropic
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models.amd_base import AmdLlmModelBase, logger

CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}


def convert_openai_tools_to_claude(tools: list[dict], *, cache_control: bool = True) -> list[dict]:
    """Convert OpenAI-style tool (function calling) definitions into
    Claude tool-use compatible format.

    If *cache_control* is ``True`` the last tool gets a
    ``cache_control`` marker so that the entire tool-definition prefix
    is cached by Anthropic.
    """
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


class AmdClaudeModel(AmdLlmModelBase):
    """Anthropic (Claude) model via AMD LLM gateway."""

    def _init_client(self):
        api_key = self._get_api_key()
        user = self._get_user()
        base_url = self.config.base_url or "https://llm-api.amd.com/Anthropic"
        self.client = anthropic.Anthropic(
            api_key="dummy",
            base_url=base_url,
            default_headers={
                "Ocp-Apim-Subscription-Key": api_key,
                "user": user,
                "anthropic-version": self.config.api_version,
            },
        )

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def format_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert standard messages to Anthropic format.

        Returns:
            ``(system_message, anthropic_messages)``

        Handles:
        - system messages → extracted as the ``system`` parameter
        - assistant messages with ``tool_calls`` → content blocks with ``tool_use``
        - tool result messages → user message with ``tool_result`` content block
        """
        system_message: str | None = None
        anthropic_messages: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_message = content
            elif role == "tool":
                # Tool result → user message with tool_result content block
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
            elif role == "assistant" and msg.get("tool_calls"):
                # Assistant message with tool call → structured content blocks.
                # Accept both a single dict and a list (for future multi-tool support).
                content_blocks: list[dict] = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                raw_tool_calls = msg["tool_calls"]
                tool_list = raw_tool_calls if isinstance(raw_tool_calls, list) else [raw_tool_calls]
                for tool_info in tool_list:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_info.get("id", ""),
                            "name": tool_info["function"]["name"],
                            "input": tool_info["function"]["arguments"],
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            else:
                anthropic_role = "assistant" if role == "assistant" else "user"
                anthropic_messages.append({"role": anthropic_role, "content": content})

        return system_message, anthropic_messages

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=4, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt, anthropic.AuthenticationError, anthropic.NotFoundError)),
    )
    def _query_api(self, messages: list[dict], **kwargs):
        # Anthropic API supported parameters
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
            "thinking",
        }

        all_kwargs = self.config.model_kwargs | kwargs
        filtered_kwargs = {k: v for k, v in all_kwargs.items() if k in supported_params}
        filtered_kwargs["tools"] = convert_openai_tools_to_claude(
            self.tools,
            cache_control=False,
        )

        system_message, anthropic_messages = self.format_messages(messages)

        if "max_tokens" not in filtered_kwargs:
            filtered_kwargs["max_tokens"] = 4096

        if self.config.set_cache_control:
            if system_message and "system" not in filtered_kwargs:
                filtered_kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_message,
                    }
                ]
        else:
            if system_message and "system" not in filtered_kwargs:
                filtered_kwargs["system"] = system_message

        return self.client.messages.create(
            model=self.config.model_name,
            messages=anthropic_messages,
            **filtered_kwargs,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response) -> dict:
        output_dict: dict = {"content": "", "tools": ""}
        try:
            if response.content:
                content_parts = []
                thinking_parts = []
                for block in response.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "thinking":
                        thinking_text = getattr(block, "thinking", None)
                        if isinstance(thinking_text, str) and thinking_text:
                            thinking_parts.append(thinking_text)
                        continue
                    block_text = getattr(block, "text", None)
                    if isinstance(block_text, str) and block_text:
                        content_parts.append(block_text)
                output_dict["content"] = "".join(content_parts)
                if thinking_parts:
                    output_dict["thinking"] = "".join(thinking_parts)

                for block in response.content:
                    if block.type == "tool_use":
                        output_dict["tools"] = {
                            "id": block.id,
                            "function": {
                                "arguments": block.input,
                                "name": block.name,
                            },
                        }
                        break
        except Exception:
            logger.warning("Failed to parse anthropic response content")
        return output_dict
