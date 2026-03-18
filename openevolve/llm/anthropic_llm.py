# Copyright(C)[2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Anthropic API interface for LLMs -- native Claude support without OpenAI proxy.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import anthropic

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class AnthropicLLM(LLMInterface):
    """LLM interface using the Anthropic Messages API directly."""

    def __init__(self, model_cfg: Optional[object] = None):
        api_key = model_cfg.api_key or os.environ.get("ANTHROPIC_API_KEY")
        assert api_key, (
            "API key must be provided either in config.yaml or as "
            "environment variable 'ANTHROPIC_API_KEY'"
        )

        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_key = api_key

        self.client = anthropic.Anthropic(api_key=self.api_key)

        logger.info(f"Initialized Anthropic LLM with model: {self.model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context."""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)

        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or 4096,
            "messages": messages,
        }
        if system_message:
            params["system"] = system_message
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p

        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._call_api(params), timeout=timeout
                )
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"All {retries + 1} attempts failed with error: {e}"
                    )
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual Anthropic API call in a thread pool executor."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.messages.create(**params)
        )
        text = response.content[0].text
        logger.debug(f"API parameters: model={params['model']}")
        logger.debug(f"API response: {text[:200]}...")
        return text
