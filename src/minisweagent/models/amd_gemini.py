# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Google (Gemini) model implementation for AMD LLM gateway."""

import base64
import logging
import os
import uuid

from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models.amd_base import AmdLlmModelBase, logger


def convert_openai_tools_to_gemini(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-style tool (function calling) definitions
    into Gemini-compatible ``function_declarations`` format.
    """
    gemini_tools = []
    for tool in tools:
        func = tool.get("function", tool)
        gemini_func = {
            "name": func["name"],
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
        gemini_tools.append(gemini_func)
    return gemini_tools


class AmdGeminiModel(AmdLlmModelBase):
    """Google (Gemini) model via AMD LLM gateway."""

    def _init_client(self):
        api_key = self._get_api_key()
        user = self._get_user()
        base_url = self.config.base_url or "https://llm-api.amd.com/VertexGen"
        self.client = genai.Client(
            vertexai=True,
            api_key="dummy",
            http_options=HttpOptions(
                base_url=base_url,
                api_version="v1",
                headers={
                    "Ocp-Apim-Subscription-Key": api_key,
                    "user": user,
                },
            ),
        )

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def format_messages(self, messages: list[dict]) -> tuple[str | None, list]:
        """Convert standard messages to Gemini ``types.Content`` objects.

        Returns:
            ``(system_message, contents)``

        Building real SDK types (instead of raw dicts) avoids the
        dict → Content Pydantic validation inside the SDK.
        ``thought_signature`` for Gemini 3+ thinking models is set on
        ``Part`` (its native location in the SDK), not on ``FunctionCall``.
        """
        system_message: str | None = None
        contents: list = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_message = content
            elif role == "tool":
                # Tool result → user Content with FunctionResponse part
                fr = types.FunctionResponse(
                    name=msg.get("name", ""),
                    response={"result": content},
                )
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(function_response=fr)],
                    )
                )
            elif role == "assistant" and (msg.get("tool_calls") or msg.get("tools")):
                # Model Content with FunctionCall part
                # Accept both "tool_calls" and "tools" keys for robustness.
                tool_info = msg.get("tool_calls") or msg.get("tools")
                # OpenAI-style tool_calls is a list; take the first entry.
                if isinstance(tool_info, list):
                    tool_info = tool_info[0]

                parts: list[types.Part] = []
                if content:
                    parts.append(types.Part(text=content))
                fc_kwargs: dict = {
                    "name": tool_info["function"]["name"],
                    "args": tool_info["function"]["arguments"],
                }
                # Gemini 3+ requires thought_signature on function-call parts.
                # NOTE: thought_signature is a field on Part, NOT on FunctionCall.
                thought_sig = tool_info.get("thought_signature")
                if thought_sig is None:
                    thought_sig = tool_info.get("function", {}).get("thought_signature")
                # The Part.thought_signature field is typed as bytes.
                # Convert from base64 string (stored in messages) to bytes.
                if isinstance(thought_sig, str):
                    thought_sig = base64.b64decode(thought_sig)
                part_kwargs: dict = {
                    "function_call": types.FunctionCall(**fc_kwargs),
                }
                if thought_sig is not None:
                    part_kwargs["thought_signature"] = thought_sig
                    logger.debug(
                        "Attaching thought_signature (len=%d) to Part for FunctionCall '%s'",
                        len(thought_sig),
                        fc_kwargs["name"],
                    )
                parts.append(types.Part(**part_kwargs))
                contents.append(types.Content(role="model", parts=parts))
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part(text=content)],
                    )
                )

        return system_message, contents

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=4, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(KeyboardInterrupt),
    )
    def _query_api(self, messages: list[dict], **kwargs):
        # Google genai API supported parameters
        supported_params = {
            "temperature",
            "max_output_tokens",
            "top_p",
            "top_k",
            "stop_sequences",
            "candidate_count",
            "safety_settings",
            "config",
        }

        all_kwargs = self.config.model_kwargs | kwargs
        filtered_kwargs = {k: v for k, v in all_kwargs.items() if k in supported_params}

        gemini_tools = convert_openai_tools_to_gemini(self.tools)
        tools = [types.Tool(function_declarations=gemini_tools)]

        system_message, contents = self.format_messages(messages)

        # Build GenerateContentConfig with generation params
        config_params: dict = {}
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "stop_sequences",
            "candidate_count",
            "safety_settings",
        ):
            if key in filtered_kwargs:
                config_params[key] = filtered_kwargs.pop(key)

        filtered_kwargs["config"] = types.GenerateContentConfig(
            tools=tools,
            system_instruction=system_message or None,
            **config_params,
        )

        return self.client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            **filtered_kwargs,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response) -> dict:
        output_dict: dict = {"content": "", "tools": ""}
        content = ""
        try:
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                parts = (
                    candidate.content.parts
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts
                    else []
                )

                if parts:
                    # Collect all text parts; be robust to SDK/object shape differences.
                    content_parts: list[str] = []
                    for part in parts:
                        part_text = None
                        if hasattr(part, "text"):
                            part_text = getattr(part, "text", None)
                        elif isinstance(part, dict):
                            part_text = part.get("text")
                        if isinstance(part_text, str) and part_text:
                            content_parts.append(part_text)
                    content = "".join(content_parts)

                for part in parts:
                    fc = None
                    if hasattr(part, "functionCall") and part.functionCall is not None:
                        fc = part.functionCall
                    elif hasattr(part, "function_call") and part.function_call is not None:
                        fc = part.function_call
                    if fc is None:
                        continue
                    # Preserve thought_signature for multi-turn tool use (Gemini 3+ requirement).
                    thought_signature = None
                    for attr in ("thought_signature", "thoughtSignature"):
                        if hasattr(fc, attr):
                            thought_signature = getattr(fc, attr)
                            if thought_signature is not None:
                                break
                    if thought_signature is None and hasattr(part, "thought_signature"):
                        thought_signature = getattr(part, "thought_signature", None)
                    output_dict["tools"] = {
                        "id": str(uuid.uuid4()),  # Gemini doesn't provide call IDs
                        "function": {
                            "arguments": getattr(fc, "args", None) or {},
                            "name": getattr(fc, "name", None) or "",
                        },
                    }
                    if thought_signature is not None:
                        # Vertex REST API expects thought_signature as a
                        # base64 *string*, not raw bytes.  The SDK may return
                        # bytes from the proto layer – convert here so the
                        # value survives JSON / Pydantic round-trips later.
                        if isinstance(thought_signature, (bytes, bytearray)):
                            thought_signature = base64.b64encode(thought_signature).decode("ascii")
                        output_dict["tools"]["thought_signature"] = thought_signature
                    break
            else:
                text = getattr(response, "text", "")
                if isinstance(text, str) and text:
                    content = text
        except Exception as e:
            logger.warning(f"Failed to parse gemini response content: {e}")

        output_dict["content"] = content

        return output_dict
