# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    def generate(self, messages: List, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
