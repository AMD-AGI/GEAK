import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import openai
import anthropic
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

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


class AmdLlmModel:
    def __init__(self, **kwargs):
        # Extract api_key from model_kwargs if present
        model_kwargs = kwargs.get("model_kwargs", {})
        if "api_key" in model_kwargs and model_kwargs["api_key"] is not None and "api_key" not in kwargs:
            kwargs["api_key"] = model_kwargs["api_key"]

        self.config = AmdLlmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

        api_key = self.config.api_key or os.getenv("AMD_LLM_API_KEY") or os.getenv("LLM_GATEWAY_KEY") or os.getenv("ANTHROPIC_API_KEY")

        # Get user name safely
        try:
            user = os.getlogin()
        except OSError:
            user = os.getenv("USER", "unknown")

        # 初始化 client
        if "gpt" in self.config.model_name:
            base_url = self.config.base_url or f"https://llm-api.amd.com/openai/{self.config.model_name}"
            self.client = openai.AzureOpenAI(
                api_key="dummy",
                api_version=self.config.api_version,
                base_url=base_url,
                default_headers={
                    "Ocp-Apim-Subscription-Key": api_key,
                    # "user": os.getlogin(),
                },
            )
        elif "claude" in self.config.model_name:
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


    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=4, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt, openai.AuthenticationError, openai.NotFoundError)),
    )
    def _query_anthropic(self, messages: list[dict[str, str]], **kwargs):
        # Anthropic API supported parameters
        supported_params = {
            "temperature", "max_tokens", "top_p", "top_k",
            "stop_sequences", "stream", "metadata", "system"
        }

        # Filter parameters
        all_kwargs = self.config.model_kwargs | kwargs
        filtered_kwargs = {
            k: v for k, v in all_kwargs.items()
            if k in supported_params
        }

        # Convert messages format for Anthropic API
        # Anthropic expects messages with role and content
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_message = content
            else:
                # Map OpenAI roles to Anthropic roles
                anthropic_role = "assistant" if role == "assistant" else "user"
                anthropic_messages.append({
                    "role": anthropic_role,
                    "content": content
                })

        # Anthropic API requires max_tokens
        if "max_tokens" not in filtered_kwargs:
            filtered_kwargs["max_tokens"] = 4096

        # Use system parameter if available
        if system_message:
            filtered_kwargs["system"] = system_message

        # Call Anthropic API
        response = self.client.messages.create(
            model=self.config.model_name,
            messages=anthropic_messages,
            **filtered_kwargs
        )

        return response
    
    def _query_openai(self, messages: list[dict[str, str]], **kwargs):
        # AMD Responses API 允许的参数
        supported_params = {
            "top_p", "frequency_penalty",
            "presence_penalty", "stop", "stream", "n",
            "seed", "response_format", "tools", "tool_choice",
            "reasoning", "text"
        }

        # 参数过滤 (also filter out api_key since it's already in headers)
        all_kwargs = self.config.model_kwargs | kwargs
        filtered_kwargs = {
            k: v for k, v in all_kwargs.items()
            if k in supported_params
        }

        # 拼 prompt
        prompt = "\n".join([msg["content"] for msg in messages])

        # 调用 AMD Responses API
        response = self.client.responses.create(
            model=self.config.model_name,
            input=prompt,
            **filtered_kwargs,
        )

        return response
    
    def _parse_response_openai(self, response):
        content = ""
        try:
            out = response.output
            if out and hasattr(out[0], "content"):
                if out[0].content and out[0].content[0].type == "output_text":
                    content = out[0].content[0].text
            # resp from gpt-5-codex
            elif out and hasattr(out[-1], "content"):
                if out[-1].content and out[-1].content[0].type == "output_text":
                    content = out[-1].content[0].text
        except Exception:
            logger.warning("Failed to parse response content")

        return content
    
    def _parse_response_anthropic(self, response):
        content = ""
        try:
            if response.content:
                # Anthropic returns content as a list of content blocks
                content_parts = []
                for block in response.content:
                    if block.type == "text":
                        content_parts.append(block.text)
                content = "".join(content_parts)
        except Exception:
            logger.warning("Failed to parse response content")


        return content
    
    def query(self, messages: list[dict[str, str]], **kwargs):
        if "gpt" in self.config.model_name:
            response = self._query_openai(messages, **kwargs)
            content = self._parse_response_openai(response)
        elif "claude" in self.config.model_name:
            response = self._query_anthropic(messages, **kwargs)
            content = self._parse_response_anthropic(response)
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")

        usage = response.usage
        if usage:
            cost = (
                (usage.input_tokens / 1000) * self.config.cost_per_1k_input_tokens +
                (usage.output_tokens / 1000) * self.config.cost_per_1k_output_tokens
            )
        else:
            cost = 0.0

        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        return {
            "content": content,
            "extra": {"response": response.model_dump()},
        }

    def get_template_vars(self):
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}


if __name__ == "__main__":
    # API key is read from AMD_LLM_API_KEY environment variable
    # Example: export AMD_LLM_API_KEY=your_key_here
    model = AmdLlmModel(
        model_name="claude-opus-4.5",
        # api_key will be read from environment variable
    )
    
    
    query_result_extracted_path = "/home/ethany/three_projects_clean/query_result_extracted_2.txt"
    with open(query_result_extracted_path, "r") as f:
        query_result_extracted = f.read()

    response = model.query([
    {
        "role": "system",
        "content": """You are a filtering and summarization subagent in a code generation/optimization system. Your task is to process retrieved chunks from the RAG database based on the user's query.

Input: 
- User query: [QUERY]
- Retrieved chunks: [CHUNKS] (list of text snippets)

Steps:
1. Evaluate each chunk for relevance to the query (score 0-10; discard if <5).
2. Remove duplicates or highly similar chunks.
3. Summarize the remaining chunks into concise, key points or a coherent paragraph, focusing on code-related insights, optimizations, example codes or generation techniques.
4. Output only the filtered summary; no explanations or additional content.

Output format:
- Relevant summary
- If no relevant chunks: "No relevant information found." """ \
},
    {"role": "user", "content": f"{query_result_extracted}"}
])

    print(response["content"])

    # response = model.query([{"role": "system", "content": "You are a sub-agent that ."}])
    # print(response["content"])
    
    # response = model.query([{"role": "user", "content": "Explain briefly what 'neural text degeneration' means in LLMs."}])
    # print(response["content"])

    # response = model.query([{"role": "user", "content": "What was my last question?"}])
    # print(response["content"])