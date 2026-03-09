# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

from typing import List
import os
import anthropic
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
from geak_agent.models.Base import BaseModel


class StandardClaudeModel(BaseModel):
    """Standard Claude API (api.anthropic.com)"""
    def __init__(self, 
                 model_id="claude-sonnet-4-20250514", 
                 api_key=None):
        assert api_key is not None, "No API key provided."
        self.model_id = model_id
        client_kwargs = {"api_key": api_key}
        client_kwargs["base_url"] = "https://api.anthropic.com"
        self.client = anthropic.Anthropic(**client_kwargs)
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature=0, 
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=16000) -> str:
        max_tokens = min(max_tokens, 16000)

        api_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = self.client.messages.create(**api_kwargs)
        except anthropic.APIStatusError as e:
            raise ValueError(f"Anthropic API error: {e.status_code} - {e.message}")
        except anthropic.APIConnectionError as e:
            raise ValueError(f"API connection error: {str(e)}")
        except anthropic.AuthenticationError as e:
            raise ValueError(f"Authentication error - check your API key: {str(e)}")
        except Exception as e:
            raise ValueError(f"API call failed: {type(e).__name__}: {str(e)}")
        
        if not response or not hasattr(response, 'content') or len(response.content) == 0:
            raise ValueError("No response content returned from the API.")
        
        return response.content[0].text


class ClaudeModel(BaseModel):
    """Cambricon LLM API Gateway - Claude client"""
    def __init__(self, 
                 model_id="claude-sonnet-4", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.SERVER = "https://llm-api.Cambricon.com/claude3"
        self.headers = {
            'Ocp-Apim-Subscription-Key': api_key
        }
    
    @retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List,
                 temperature=1.0,
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=16000,
                 max_completion_tokens=16000
                 ) -> str:
        # Cap max_tokens
        max_tokens = min(max_tokens, 16000)
        
        body = {
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "max_completion_tokens": max_tokens,
            "max_tokens": max_tokens,
            "presence_Penalty": 0,
            "frequency_Penalty": 0,
        }
        try:
            response = requests.post(
                        url=f"{self.SERVER}/{self.model_id}/chat/completions",
                        json=body,
                        headers=self.headers,
                        timeout=600
                    )
        except Exception as e:
            raise ValueError(f"No response from the API: {str(e)}")
        
        if response.status_code != 200:
            raise ValueError(f"API returned status {response.status_code}: {response.text}")
        
        result = response.json()
        if 'content' in result and len(result['content']) > 0:
            return result['content'][0]['text']
        elif 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise ValueError(f"Unexpected response format: {result}")
