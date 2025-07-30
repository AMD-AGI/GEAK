import os
from typing import List
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from models.Base import BaseModel


class OpenAIModel(BaseModel):
    def __init__(self, 
                 model_id="GPT4o", 
                 model_api_version='2024-06-01', 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.model_api_version = model_api_version

        assert 'MODEL_API_URL' in os.environ, "MODEL_API_URL environment variable is not set."
        MODEL_API_URL = os.environ['MODEL_API_URL']

        url = MODEL_API_URL
        headers = {
            'Ocp-Apim-Subscription-Key': api_key 
        }
        model_api_version = '2024-06-01'
        

        self.client = openai.AzureOpenAI(
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
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            raise ValueError("No response choices returned from the API.")

        return response.choices[0].message.content
    