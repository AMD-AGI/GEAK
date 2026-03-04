# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""OpenAI GPT-5.2 model via AMD LLM gateway (Azure Chat Completions API).

GPT-5.2 uses the Azure OpenAI *Chat Completions* API with a deployment-based
URL, unlike GPT-5 which uses the Responses API.  This module handles:

- ``/openai/deployments/{deployment_id}`` URL routing
- ``chat.completions.create`` instead of ``responses.create``
- Standard OpenAI tool-calling format
"""

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

_GPT52_DEPLOYMENT = os.getenv("GEAK_GPT52_DEPLOYMENT", "dvue-aoai-001-gpt-5.2")
_GPT52_API_VERSION = os.getenv("GEAK_GPT52_API_VERSION", "2025-04-01-preview")


class AmdOpenAIChatModel(AmdLlmModelBase):
    """GPT-5.2 via AMD gateway using Azure Chat Completions API."""

    def _init_client(self):
        api_key = self._get_api_key()
        base_url = f"https://llm-api.amd.com/openai/deployments/{_GPT52_DEPLOYMENT}"
        self.client = openai.AzureOpenAI(
            api_key="dummy",
            api_version=_GPT52_API_VERSION,
            base_url=base_url,
            default_headers={
                "Ocp-Apim-Subscription-Key": api_key,
            },
        )

    def format_messages(self, messages: list[dict]) -> list[dict]:
        """Convert standard messages to OpenAI Chat Completions format."""
        formatted: list[dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "tool":
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": content,
                })
            elif role == "assistant" and msg.get("tool_calls"):
                tool_info = msg["tool_calls"]
                args = tool_info["function"]["arguments"]
                formatted.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [{
                        "id": tool_info.get("id", "call_0"),
                        "type": "function",
                        "function": {
                            "name": tool_info["function"]["name"],
                            "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                        },
                    }],
                })
            else:
                formatted.append({"role": role, "content": content})

        return formatted

    def _convert_tools(self) -> list[dict]:
        """Convert internal tool format to OpenAI function-calling format."""
        oai_tools = []
        for tool in self.tools:
            func = tool.get("function", tool)
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }),
                },
            })
        return oai_tools

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=4, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt, openai.AuthenticationError, openai.NotFoundError)),
    )
    def _query_api(self, messages: list[dict], **kwargs):
        formatted_messages = self.format_messages(messages)
        oai_tools = self._convert_tools()

        call_kwargs = {}
        if oai_tools:
            call_kwargs["tools"] = oai_tools
            call_kwargs["tool_choice"] = "auto"

        return self.client.chat.completions.create(
            model=_GPT52_DEPLOYMENT,
            messages=formatted_messages,
            **call_kwargs,
        )

    def _parse_response(self, response) -> dict:
        output_dict: dict = {"content": "", "tools": ""}
        try:
            choice = response.choices[0]
            msg = choice.message

            if msg.content:
                output_dict["content"] = msg.content

            if msg.tool_calls:
                tc = msg.tool_calls[0]
                output_dict["tools"] = {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    },
                }
        except Exception as e:
            logger.warning("Failed to parse GPT-5.2 response: %s", e)
        return output_dict
