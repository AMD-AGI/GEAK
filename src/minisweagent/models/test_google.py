import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("amd_llm")

try:
    from google import genai
    from google.genai.types import HttpOptions
except ImportError:
    genai = None
    HttpOptions = None


@dataclass
class AmdLlmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    api_key: str | None = None
    base_url: str = "https://llm-api.amd.com/VertexGen"
    api_version: str = "v1"
    cost_per_1k_input_tokens: float = 0.01
    cost_per_1k_output_tokens: float = 0.01
    set_cache_control: Literal["default_end"] | None = "default_end"


class AmdLlmModel:
    def __init__(self, **kwargs):
        if genai is None or HttpOptions is None:
            raise ImportError(
                "The google-genai package is required to use AmdLlmModel. "
                "Please install it with: pip install google-genai"
            )
        
        # Extract api_key from model_kwargs if present
        model_kwargs = kwargs.get("model_kwargs", {})
        if "api_key" in model_kwargs and model_kwargs["api_key"] is not None and "api_key" not in kwargs:
            kwargs["api_key"] = model_kwargs["api_key"]
        
        self.config = AmdLlmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

        api_key = self.config.api_key or os.getenv("AMD_LLM_API_KEY") or os.getenv("LLM_GATEWAY_KEY")
        if not api_key:
            api_key = ""

        # Get user name safely
        try:
            user = os.getlogin()
        except OSError:
            user = os.getenv("USER", "unknown")

        # Initialize Google genai client with AMD endpoint
        self.client = genai.Client(
            vertexai=True,
            api_key="dummy",
            http_options=HttpOptions(
                base_url=self.config.base_url,
                api_version=self.config.api_version,
                headers={
                    "Ocp-Apim-Subscription-Key": api_key,
                    "user": user,
                }
            )
        )

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=4, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(KeyboardInterrupt),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        # Google genai API supported parameters
        supported_params = {
            "temperature", "max_output_tokens", "top_p", "top_k",
            "stop_sequences", "candidate_count", "safety_settings"
        }

        # Filter parameters
        all_kwargs = self.config.model_kwargs | kwargs
        filtered_kwargs = {
            k: v for k, v in all_kwargs.items()
            if k in supported_params
        }

        # Convert messages format for Google genai API
        # Google genai expects contents as a list of Content objects with role and parts
        # System messages should be passed as system_instruction parameter
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            
            if role == "system":
                genai_role = "system"
            elif role == "assistant":
                genai_role = "model" 
            else:
                genai_role = "user"

            contents.append({
                "role": genai_role,
                "parts": [
                    {"text": text}
                ]
            })

        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            **filtered_kwargs
        )


        return response

    def query(self, messages: list[dict[str, str]], **kwargs):
        response = self._query(messages, **kwargs)

        content = ""
        try:
            # Google genai returns response.text for text content
            if hasattr(response, "text") and response.text:
                content = response.text
            elif hasattr(response, "candidates") and response.candidates:
                # Fallback: try to extract from candidates
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    text_parts = [
                        part.text for part in candidate.content.parts
                        if hasattr(part, "text")
                    ]
                    content = "".join(text_parts)
        except Exception as e:
            logger.warning(f"Failed to parse response content: {e}")

        # Calculate cost
        # Google genai may provide usage information in different formats
        usage = None
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
        elif hasattr(response, "usage"):
            usage = response.usage

        if usage:
            input_tokens = getattr(usage, "prompt_token_count", 0) or getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or getattr(usage, "output_tokens", 0) or 0
            cost = (
                (input_tokens / 1000) * self.config.cost_per_1k_input_tokens +
                (output_tokens / 1000) * self.config.cost_per_1k_output_tokens
            )
        else:
            cost = 0.0

        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        # Prepare response dict
        response_dict = {}
        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        elif hasattr(response, "dict"):
            response_dict = response.dict()
        elif hasattr(response, "__dict__"):
            response_dict = response.__dict__

        return {
            "content": content,
            "extra": {"response": response_dict},
        }

    def get_template_vars(self):
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}


if __name__ == "__main__":
    model = AmdLlmModel(
        model_name="gemini-3-pro-preview",
        api_key="",
    )
    response = model.query([{"role": "user", "content": "How does AI work?"}])
    print(response["content"])

