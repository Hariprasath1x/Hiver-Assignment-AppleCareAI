import os
import json
from pydantic import BaseModel
from typing import Type, Optional
from groq import Groq
from src.llm.base import LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        api_key = api_key or os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("No Groq API key found. Set GROQ_API_KEY_1 or GROQ_API_KEY.")
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        self.client = Groq(api_key=api_key)

    def generate_structured(self, prompt: str, schema_class: Type[BaseModel], temperature: float = 0.0) -> BaseModel:
        schema_json = schema_class.model_json_schema()
        system_prompt = (
            "You are a helpful assistant. "
            f"You MUST output valid JSON that strictly matches this schema: {json.dumps(schema_json)}"
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=temperature
        )

        raw_content = completion.choices[0].message.content
        return schema_class.model_validate_json(raw_content)

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return completion.choices[0].message.content
