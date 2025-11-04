from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential
import requests
from models.Base import BaseModel
import time
 
class GeminiModel(BaseModel):
    def __init__(self, 
                 model_id="gemini-2.5-pro", 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.SERVER = "https://llm-api.amd.com/vertex/gemini"
        self.HEADERS = {"Ocp-Apim-Subscription-Key": api_key}
 
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature=1.0, 
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=30000) -> str:
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature":temperature,
            "top_P": 0.95,
            "presence_Penalty": presence_penalty,
            "frequency_Penalty": frequency_penalty,
        }
        response_gemine = requests.post(url=f"{self.SERVER}/{self.model_id}/chat", 
                            json=body,
                            headers=self.HEADERS)
        assert response_gemine.status_code == 200
        time.sleep(10)
        code_chat_completion_result = response_gemine.json()
        return code_chat_completion_result['candidates'][0]['content']['parts'][0]['text']