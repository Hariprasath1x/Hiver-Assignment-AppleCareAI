import os
from typing import Type, Optional
from pydantic import BaseModel

from google import genai
from google.genai import types

from src.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No Gemini API key found. Set GEMINI_API_KEY_1 or GEMINI_API_KEY.")
        self.client = genai.Client(api_key=api_key)
        # Default to the configured working model; never guess
        self.model = model or os.getenv("LLM_MODEL_GEMINI", os.getenv("LLM_MODEL", "gemini-3.8-flash"))

    def generate_structured(self, prompt: str, schema_class: Type[BaseModel], temperature: float = 0.0) -> BaseModel:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_class,
            temperature=temperature
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )

        return schema_class.model_validate_json(response.text)

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        config = types.GenerateContentConfig(
            temperature=temperature
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )

        return response.text
