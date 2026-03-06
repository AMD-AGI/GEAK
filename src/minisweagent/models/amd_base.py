# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Base class and shared configuration for AMD LLM models."""

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.tools.tools_runtime import get_tools_list

logger = logging.getLogger("amd_llm")


@dataclass
class AmdLlmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    api_key: str | None = None
    base_url: str | None = None
    api_version: str = "2023-10-16"
    cost_per_1k_input_tokens: float = 0.01
    cost_per_1k_output_tokens: float = 0.01
    set_cache_control: Literal["default_end"] | None = "default_end"
    reasoning: dict[str, Any] = field(default_factory=dict)
    bash_tool: bool = True
    profiling: bool = False
    use_strategy_manager: bool = False


class AmdLlmModelBase:
    """Base class for AMD LLM model implementations.

    Subclasses must override:
        - ``_init_client``  – set up the vendor SDK client
        - ``_query_api``    – send the request and return the raw response
        - ``_parse_response`` – convert the raw response to a standard dict
        - ``format_messages`` – convert standard messages to vendor format

    The standard **response dict** returned by ``query`` has the shape::

        {
            "content": str,        # text content (may be empty)
            "tools": {             # present when the model requests a tool call
                "id": str,         # unique call id
                "function": {
                    "name": str,
                    "arguments": dict,
                },
            } | "",                # empty string when no tool call
        }

    The standard **message format** stored in ``agent.messages``::

        {"role": "system",    "content": "..."}
        {"role": "user",      "content": "..."}
        {"role": "assistant", "content": "...", "tool_calls": {...}}   # optional tool_calls
        {"role": "tool",      "content": "...", "tool_call_id": "...", "name": "..."}
    """

    def __init__(self, config: AmdLlmModelConfig):
        self.config = config
        self.cost = 0.0
        self.n_calls = 0
        # Load tools list
        self.tools = get_tools_list(use_strategy_manager=self.config.use_strategy_manager)
        if not self.config.bash_tool:
            self.tools = [tool for tool in self.tools if tool["name"] != "bash"]
        self._init_client()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_api_key(self) -> str:
        api_key = self.config.api_key or os.getenv("AMD_LLM_API_KEY") or os.getenv("LLM_GATEWAY_KEY")
        if not api_key:
            raise ValueError(
                "API key not provided. Please set it via:\n"
                "  1. VSCode settings (mini-swe-agent.apiKey), or\n"
                "  2. Environment variable AMD_LLM_API_KEY, or\n"
                "  3. Environment variable LLM_GATEWAY_KEY"
            )
        return api_key

    def _get_user(self) -> str:
        try:
            return os.getlogin()
        except OSError:
            return os.getenv("USER", "unknown")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _init_client(self):
        """Initialize the vendor API client. Override in subclasses."""
        raise NotImplementedError

    def _query_api(self, messages: list[dict], **kwargs):
        """Send the query and return the *raw* vendor response object."""
        raise NotImplementedError

    def _parse_response(self, response) -> dict:
        """Parse the raw vendor response into the standard response dict."""
        raise NotImplementedError

    def format_messages(self, messages: list[dict]) -> Any:
        """Convert standard messages to vendor-specific format."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_tools(self, tools: list[dict]) -> None:
        """Replace the tool schemas visible to the LLM."""
        self.tools = tools

    def query(self, messages: list[dict], **kwargs) -> dict:
        """Query the model and return a standardised response dict."""
        response = self._query_api(messages, **kwargs)
        content = self._parse_response(response)

        # Calculate cost from usage metadata
        usage = getattr(response, "usage", None)
        if usage:
            try:
                cost = (usage.input_tokens / 1000) * self.config.cost_per_1k_input_tokens + (
                    usage.output_tokens / 1000
                ) * self.config.cost_per_1k_output_tokens
            except (AttributeError, TypeError):
                logger.debug("Usage information available but format unexpected")
                cost = 0.0
        else:
            cost = 0.0

        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        # Attach raw response for trajectory logging.
        try:
            response_dump = None
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

        if isinstance(content, dict):
            content["extra"] = {"response": response_dump}

        return content

    def get_template_vars(self):
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
