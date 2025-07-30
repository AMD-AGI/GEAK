from typing import List
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
from Base import BaseModel


class ClaudeModel(BaseModel):
    def __init__(self, 
                 model_id="claude-3.7", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.model_api_version = model_api_version

        assert 'MODEL_API_URL' in os.environ, "MODEL_API_URL environment variable is not set."
        MODEL_API_URL = os.environ['MODEL_API_URL']

        url = MODEL_API_URL

        self.SERVER = url
        self.headers = {
            'Ocp-Apim-Subscription-Key': api_key
        }
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List,
                 temperature=0,
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=50000,
                 max_completion_tokens=50000
                 ) -> str:
        body = {
            "messages": messages,
            "temperature": 0,
            "stream": False,
            "max_completion_tokens": 50000,
            "max_tokens": 50000,
            "presence_Penalty": 0,
            "frequency_Penalty": 0,
        }
        response = requests.post(
                    url=f"{self.SERVER}/{self.model_id}/chat/completions",
                    json=body,
                    headers=self.headers
                )
        return response.json()['content'][0]['text']
    
