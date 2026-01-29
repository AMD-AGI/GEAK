import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
import json
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
from minisweagent.tools.tools_runtime import get_tools_list
try:
    from google import genai
    from google.genai import types
    from google.genai.types import HttpOptions
except ImportError:
    raise ImportError("You should install google-genai to use Gemini models. pip install google-genai")

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

def convert_openai_tools_to_claude(tools: list[dict]) -> list[dict]:
    """
    Convert OpenAI-style tool (function calling) definitions into
    Claude tool-use compatible format.
    """
    claude_tools = []
    for tool in tools:
        func = tool.get("function", tool)
        claude_tool = {
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {
                "type": "object",
                "properties": {},
                "required": []
            })
        }
        claude_tools.append(claude_tool)
    return claude_tools

def convert_openai_tools_to_gemini(tools: list[dict]) -> list[dict]:
    """
    Convert OpenAI-style tool (function calling) definitions
    into Gemini-compatible function_declarations format.
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
                    "required": []
                }
            )
        }
        gemini_tools.append(gemini_func)
    return gemini_tools


class AmdLlmModel:
    def __init__(self, **kwargs):
        # Extract api_key from model_kwargs if present
        model_kwargs = kwargs.get("model_kwargs", {})
        if "api_key" in model_kwargs and model_kwargs["api_key"] is not None and "api_key" not in kwargs:
            kwargs["api_key"] = model_kwargs["api_key"]

        self.config = AmdLlmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        # Get tools list based on strategy manager setting
        self.tools = get_tools_list(use_strategy_manager=self.config.use_strategy_manager)
        if not self.config.profiling:
            self.tools = [tool for tool in self.tools if tool["name"] != "profiling"]
        if not self.config.bash_tool:
            self.tools = [tool for tool in self.tools if tool["name"] != "bash"]

        api_key = self.config.api_key or os.getenv("AMD_LLM_API_KEY") or os.getenv("LLM_GATEWAY_KEY")
        
        # Validate API key
        if not api_key:
            raise ValueError(
                "API key not provided. Please set it via:\n"
                "  1. VSCode settings (mini-swe-agent.apiKey), or\n"
                "  2. Environment variable AMD_LLM_API_KEY, or\n"
                "  3. Environment variable LLM_GATEWAY_KEY"
            )

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
        elif "gemini" in self.config.model_name:
            # Initialize Google genai client with AMD endpoint
            # Ensure base_url and api_version are set correctly for Gemini
            base_url = self.config.base_url or "https://llm-api.amd.com/VertexGen"
            api_version = "v1"  # Gemini always uses "v1"
            self.client = genai.Client(
                vertexai=True,
                api_key="dummy",
                http_options=HttpOptions(
                    base_url=base_url,
                    api_version=api_version,
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
        retry=retry_if_not_exception_type((KeyboardInterrupt, openai.AuthenticationError, openai.NotFoundError)),
    )
    def _query_anthropic(self, messages: list[dict[str, str]], **kwargs):
        # Anthropic API supported parameters
        supported_params = {
            "temperature", "max_tokens", "top_p", "top_k",
            "stop_sequences", "stream", "metadata", "system", "tools"
        }

        # Filter parameters
        all_kwargs = self.config.model_kwargs | kwargs
        filtered_kwargs = {
            k: v for k, v in all_kwargs.items()
            if k in supported_params
        }

        filtered_kwargs["tools"] = convert_openai_tools_to_claude(self.tools)

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
                prompt = "\n".join([msg["content"] for msg in messages])
        anthropic_messages.append({
            "role": "user",
            "content": prompt
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

        filtered_kwargs["tools"] = self.tools
        filtered_kwargs["tool_choice"] = "auto"

        cleaned_messages = []
        prompt = "\n".join([msg["content"] for msg in messages])
        cleaned_messages.append({"role": "user", "content":prompt})

        response = self.client.responses.create(
            model=self.config.model_name,
            input=cleaned_messages,
            **filtered_kwargs
        )

        return response
    
    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=4, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(KeyboardInterrupt),
    )
    def _query_gemini(self, messages: list[dict[str, str]], **kwargs):
        # Google genai API supported parameters
        supported_params = {
            "temperature", "max_output_tokens", "top_p", "top_k",
            "stop_sequences", "candidate_count", "safety_settings", "config"
        }

        # Filter parameters
        all_kwargs = self.config.model_kwargs | kwargs
        filtered_kwargs = {
            k: v for k, v in all_kwargs.items()
            if k in supported_params
        }
        
        test_tools = convert_openai_tools_to_gemini(self.tools)
        tools = [types.Tool(function_declarations=test_tools)]
        filtered_kwargs["config"] = types.GenerateContentConfig(tools=tools)

        # Convert messages format for Google genai API
        # Google genai expects contents as a list of Content objects with role and parts
        contents = []
        prompt = "\n".join([msg["content"] for msg in messages])
        contents.append({"role": "user", "content":prompt})
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            **filtered_kwargs
        )

        return response

    def _parse_response_openai(self, response):
        output_dict ={
            "content": "",
            "tools": ""
        }
        content = ""
        try:
            outs = response.output
            for out in outs:
                if out and hasattr(out, "content") and out.content is not None and out.content[0].type == "output_text":
                    content = out.content[0].text
                    output_dict["content"] = content
                    break
            for out in outs:
                if out and hasattr(out, "type") and out.type == "function_call":
                    tool_call = {
                        "id": out.call_id,
                        "function": {
                            "arguments": json.loads(out.arguments),
                            "name": out.name,
                        },
                    }
                    output_dict["tools"] = tool_call
                    break
        except Exception as e:
            logger.warning(f"Failed to parse openai response content {e}")

        return output_dict
    
    def _parse_response_anthropic(self, response):
        output_dict ={
            "content": "",
            "tools": ""
        }
        content = ""
        try:
            if response.content:
                # Anthropic returns content as a list of content blocks
                content_parts = []
                for block in response.content:
                    if block.type == "text":
                        content_parts.append(block.text)
                content = "".join(content_parts)
                output_dict["content"] = content
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_call = {
                            "id": block.id,
                            "function": {
                                "arguments": block.input,
                                "name": block.name,
                            },
                        }
                        output_dict["tools"] = tool_call
                        break
        except Exception:
            logger.warning("Failed to parse response content")

    
        return output_dict
    
    def _parse_response_gemini(self, response):
        output_dict ={
            "content": "",
            "tools": ""
        }
        content = ""
        try:
            # Google genai returns response.text for text content
            if hasattr(response, "text") and response.text:
                content = response.text
            elif hasattr(response, "candidates") and response.candidates:
                # Fallback: try to extract from candidates
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
                    text_parts = [
                        part.text for part in candidate.content.parts
                        if hasattr(part, "text")
                    ]
                    content = "".join(text_parts)
                    output_dict["content"] = content
            
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_message"):
                    tools_call = candidate.finish_message
                    m = re.search(r"call:(\w+)\s*\{([\s\S]*?)\}\s*$", tools_call.strip())
                    if m:
                        name = m.group(1)
                        raw_args = m.group(2)
                        json_like = re.sub(
                            r'(\w+)\s*:',
                            r'"\1":',
                            raw_args
                        )
                        # import pdb; pdb.set_trace()
                        json_text = "{" + json_like + "}"
                        arguments = json.loads(json_text)
                        tool_call = {
                            "id": None,
                            "function": {
                                "arguments": arguments,
                                "name": name,
                            },
                        }
                        output_dict["tools"] = tool_call            
        except Exception as e:
            logger.warning(f"Failed to parse response content: {e}")

        return output_dict

    def query(self, messages: list[dict[str, str]], **kwargs):
        if "gpt" in self.config.model_name:
            response = self._query_openai(messages, **kwargs)
            content = self._parse_response_openai(response)
        elif "claude" in self.config.model_name:
            response = self._query_anthropic(messages, **kwargs)
            content = self._parse_response_anthropic(response)
        elif "gemini" in self.config.model_name:
            response = self._query_gemini(messages, **kwargs)
            content = self._parse_response_gemini(response)
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
    
        try:
            usage = response.usage
        except Exception:
            logger.warning("Failed to get usage information")
            usage = None

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

        return content

    def get_template_vars(self):
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}


if __name__ == "__main__":
    # gpt-5, gpt-5-codex, gpt-5.1, claude-opus-4.5, claude-sonnet-4.5，gemini-3-pro-preview
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
        {"role": "user", "content": "Use tool named as str_replace_editor to veiw file '/home/chaox/kernel_agent/read_mini.py' and output your thinking"},
    ]
    for model_name in model_list:
        print(f"Testing {model_name}...")
        model = AmdLlmModel(
            model_name=model_name,
            api_key="",
        )
        response = model.query(messages)
        print(response)