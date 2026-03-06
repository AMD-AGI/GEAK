# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI (GPT) model implementation for AMD LLM gateway."""

import json
import logging
import os

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models.amd_base import AmdLlmModelBase, logger


class AmdOpenAIModel(AmdLlmModelBase):
    """OpenAI (GPT) model via AMD LLM gateway using the Responses API."""

    def _init_client(self):
        api_key = self._get_api_key()
        base_url = self.config.base_url or f"https://llm-api.amd.com/openai/{self.config.model_name}"
        self.client = openai.AzureOpenAI(
            api_key="dummy",
            api_version=self.config.api_version,
            base_url=base_url,
            default_headers={
                "Ocp-Apim-Subscription-Key": api_key,
            },
        )

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def format_messages(self, messages: list[dict]) -> list[dict]:
        """Convert standard messages to OpenAI Responses API input format.

        Handles:
        - assistant messages with ``tool_calls`` → ``function_call`` items
        - tool result messages → ``function_call_output`` items
        """
        formatted: list[dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "tool":
                # Tool result → function_call_output
                formatted.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": content,
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                # Assistant with tool call
                if content:
                    formatted.append({"role": "assistant", "content": content})
                tool_info = msg["tool_calls"]
                args = tool_info["function"]["arguments"]
                formatted.append(
                    {
                        "type": "function_call",
                        "call_id": tool_info.get("id", ""),
                        "name": tool_info["function"]["name"],
                        "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                    }
                )
            else:
                formatted.append({"role": role, "content": content})

        return formatted

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=4, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt, openai.AuthenticationError, openai.NotFoundError)),
    )
    def _query_api(self, messages: list[dict], **kwargs):
        # AMD Responses API supported parameters
        supported_params = {
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "stream",
            "n",
            "seed",
            "response_format",
            "tools",
            "tool_choice",
            "reasoning",
            "text",
        }

        all_kwargs = self.config.model_kwargs | kwargs
        filtered_kwargs = {k: v for k, v in all_kwargs.items() if k in supported_params}
        filtered_kwargs["tools"] = self.tools
        filtered_kwargs["tool_choice"] = "auto"

        # NOTE: OpenAI handles prompt caching automatically server-side;
        # explicit cache_control markers are not supported/needed.

        formatted_messages = self.format_messages(messages)

        logger.debug("OpenAI formatted messages: %s", formatted_messages)
        return self.client.responses.create(
            model=self.config.model_name,
            input=formatted_messages,
            **filtered_kwargs,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response) -> dict:
        output_dict: dict = {"content": "", "tools": ""}
        try:
            outs = response.output
            # Collect *all* output_text items across all output messages.
            # Some Responses API implementations return multiple content items
            # where output_text may not be at index 0 (e.g., tool calls / annotations first).
            content_parts: list[str] = []
            for out in outs or []:
                if not out or not hasattr(out, "content") or out.content is None:
                    continue
                for item in out.content or []:
                    try:
                        item_type = getattr(item, "type", None)
                        item_text = getattr(item, "text", None)
                        if item_type == "output_text" and isinstance(item_text, str) and item_text:
                            content_parts.append(item_text)
                    except Exception:
                        continue
            if content_parts:
                output_dict["content"] = "".join(content_parts)
            for out in outs:
                if out and hasattr(out, "type") and out.type == "function_call":
                    output_dict["tools"] = {
                        "id": getattr(out, "call_id", ""),
                        "function": {
                            "arguments": json.loads(out.arguments),
                            "name": out.name,
                        },
                    }
                    break
        except Exception as e:
            logger.warning(f"Failed to parse openai response content: {e}")
        return output_dict
