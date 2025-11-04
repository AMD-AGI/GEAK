from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    def generate(self, messages: List, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
