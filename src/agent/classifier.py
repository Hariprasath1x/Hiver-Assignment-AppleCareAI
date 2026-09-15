from src.llm.base import LLMProvider
from src.agent.schemas import IntentResult
from src.agent.prompts import INTENT_CLASSIFICATION_PROMPT

class IntentClassifier:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        
    def classify(self, text: str) -> IntentResult:
        prompt = INTENT_CLASSIFICATION_PROMPT.format(customer_message=text)
        # Use low temperature for deterministic classification
        result = self.provider.generate_structured(prompt, IntentResult, temperature=0.0)
        return result
