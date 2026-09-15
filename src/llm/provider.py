import os
import json
from pydantic import BaseModel
from typing import Type
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "gpt-4o")
        self.client = OpenAI(api_key=api_key) if api_key else None
        
    def generate_structured(self, prompt: str, schema: Type[BaseModel], temperature: float = 0.0) -> BaseModel:
        if not self.client:
            raise ValueError("OPENAI_API_KEY is not set.")
            
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
            temperature=temperature
        )
        return completion.choices[0].message.parsed
        
    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        if not self.client:
            raise ValueError("OPENAI_API_KEY is not set.")
            
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return completion.choices[0].message.content
