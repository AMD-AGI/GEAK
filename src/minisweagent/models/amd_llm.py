"""AMD LLM Model Router.

Routes to the appropriate model implementation (OpenAI / Claude / Gemini)
based on ``model_name``.  All heavy lifting lives in the per-vendor modules:

- :mod:`minisweagent.models.amd_openai`
- :mod:`minisweagent.models.amd_claude`
- :mod:`minisweagent.models.amd_gemini`
"""

import logging

from minisweagent.models.amd_base import AmdLlmModelConfig

logger = logging.getLogger("amd_llm")


class AmdLlmModel:
    """Thin router that delegates to the correct vendor model implementation.

    Instantiation inspects ``model_name`` and creates the matching backend:

    - ``"gpt*"``    → :class:`~minisweagent.models.amd_openai.AmdOpenAIModel`
    - ``"claude*"`` → :class:`~minisweagent.models.amd_claude.AmdClaudeModel`
    - ``"gemini*"`` → :class:`~minisweagent.models.amd_gemini.AmdGeminiModel`

    Every public attribute (``cost``, ``n_calls``, ``config``, …) is
    transparently forwarded to the underlying implementation so that calling
    code does not need to know which vendor is active.
    """

    def __init__(self, **kwargs):
        # Extract api_key from model_kwargs if present (backward compat)
        model_kwargs = kwargs.get("model_kwargs", {})
        if "api_key" in model_kwargs and model_kwargs["api_key"] is not None and "api_key" not in kwargs:
            kwargs["api_key"] = model_kwargs["api_key"]

        config = AmdLlmModelConfig(**kwargs)

        if "gpt" in config.model_name:
            from minisweagent.models.amd_openai import AmdOpenAIModel
            self._impl = AmdOpenAIModel(config)
        elif "claude" in config.model_name:
            from minisweagent.models.amd_claude import AmdClaudeModel
            self._impl = AmdClaudeModel(config)
        elif "gemini" in config.model_name:
            from minisweagent.models.amd_gemini import AmdGeminiModel
            self._impl = AmdGeminiModel(config)
        else:
            raise ValueError(f"Unsupported model: {config.model_name}")

    # ------------------------------------------------------------------
    # Forwarded properties
    # ------------------------------------------------------------------

    @property
    def cost(self):
        return self._impl.cost

    @cost.setter
    def cost(self, value):
        self._impl.cost = value

    @property
    def n_calls(self):
        return self._impl.n_calls

    @n_calls.setter
    def n_calls(self, value):
        self._impl.n_calls = value

    @property
    def config(self):
        return self._impl.config

    # ------------------------------------------------------------------
    # Forwarded methods
    # ------------------------------------------------------------------

    def query(self, messages: list[dict], **kwargs) -> dict:
        return self._impl.query(messages, **kwargs)

    def get_template_vars(self) -> dict:
        return self._impl.get_template_vars()


if __name__ == "__main__":
    # Quick smoke test
    model_list = [
        "gpt-5",
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "gemini-3-pro-preview",
    ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"},
        {"role": "user", "content": "Use tool named as str_replace_editor to view file '/home/chaox/kernel_agent/read_mini.py' and output your thinking"},
    ]
    for model_name in model_list:
        print(f"Testing {model_name}...")
        model = AmdLlmModel(
            model_name=model_name,
            api_key="",
        )
        response = model.query(messages)
        print(response)
