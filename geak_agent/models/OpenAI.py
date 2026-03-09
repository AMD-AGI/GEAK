# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

from typing import List
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from geak_agent.models.Base import BaseModel


class StandardOpenAIModel(BaseModel):
    """Standard OpenAI API (api.openai.com)"""
    def __init__(self, 
                 model_id="gpt-4", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            api_key="",
            azure_endpoint="",
        )
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature=0, 
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=5000) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            raise ValueError("No response choices returned from the API.")
        return response.choices[0].message.content


class OpenAIModel(BaseModel):
    def __init__(self, 
                 model_id="GPT4o", 
                 model_api_version='2025-06-01', 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.model_api_version = model_api_version

        url = 'https://llm-api.Cambricon.com'
        headers = {
            'Ocp-Apim-Subscription-Key': api_key 
        }
        model_api_version = '2024-06-01'
        

        self.client = AzureOpenAI(
            api_key='dummy',
            api_version=self.model_api_version,
            base_url=url,
            default_headers=headers
        )
        self.client.base_url = '{0}/openai/deployments/{1}'.format(url, self.model_id)
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature=0, 
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=5000) -> str:
        # Cambricon API has a max_tokens limit of ~16000
        max_tokens = min(max_tokens, 16000)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            n=1,
            stream=False,
            stop=None,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=None,
            user=None
        )
        # import pdb
        # pdb.set_trace()
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            raise ValueError("No response choices returned from the API.")

        return response.choices[0].message.content
    
