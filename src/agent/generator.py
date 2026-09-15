from typing import List
from src.llm.base import LLMProvider
from src.agent.schemas import Evidence
from src.agent.prompts import GROUNDED_GENERATION_PROMPT

class GroundedGenerator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        
    def generate(self, text: str, intent: str, evidence: List[Evidence]) -> str:
        if not evidence:
            return "I want to make sure you get the right help with this. I'll escalate this to a support specialist who can look into it further."
            
        evidence_text = ""
        for i, ev in enumerate(evidence):
            evidence_text += f"[{i+1}] Similar Customer: {ev.customer_message}\n"
            evidence_text += f"    Apple Support Reply: {ev.historical_response}\n\n"
            
        prompt = GROUNDED_GENERATION_PROMPT.format(
            customer_message=text,
            intent=intent,
            evidence_text=evidence_text
        )
        
        reply = self.provider.generate_text(prompt, temperature=0.0)
        return reply.strip()
