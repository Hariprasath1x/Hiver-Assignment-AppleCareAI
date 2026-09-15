from abc import ABC, abstractmethod
from typing import Any, Type
from pydantic import BaseModel

class LLMProvider(ABC):
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Type[BaseModel], temperature: float = 0.0) -> BaseModel:
        """Generates a structured Pydantic object from the LLM."""
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        """Generates a raw string from the LLM."""
        pass
