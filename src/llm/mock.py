from pydantic import BaseModel
from typing import Type
from .base import LLMProvider


class MockProvider(LLMProvider):
    """Mock LLM provider for testing. Returns deterministic structured results based on keywords."""

    def generate_structured(self, prompt: str, schema: Type[BaseModel], temperature: float = 0.0) -> BaseModel:
        if schema.__name__ == 'IntentResult':
            prompt_lower = prompt.lower()
            msg_start = prompt_lower.find('customer message: "')
            if msg_start != -1:
                msg = prompt_lower[msg_start:]
            else:
                msg = prompt_lower

            if 'battery' in msg:
                return schema(
                    intent="battery_power_issue", confidence=0.9, reasoning="Mentions battery.",
                    severity="LOW", severity_confidence=0.9, severity_reason="Routine battery issue."
                )
            if 'screen' in msg or 'damage' in msg or 'crack' in msg or 'broken' in msg:
                return schema(
                    intent="hardware_damage_repair", confidence=0.95, reasoning="Mentions physical damage.",
                    severity="HIGH", severity_confidence=0.95, severity_reason="Physical damage requires service."
                )
            if 'password' in msg or 'hack' in msg or 'unauthorized' in msg or 'stolen' in msg:
                return schema(
                    intent="account_security_activation", confidence=0.95, reasoning="Security issue.",
                    severity="HIGH", severity_confidence=0.95, severity_reason="Security incident."
                )
            if 'yes' in msg and len(msg.split()) < 10:
                return schema(
                    intent="other_unclear_context_dependent", confidence=0.9, reasoning="Too short.",
                    severity="LOW", severity_confidence=0.85, severity_reason="Vague message."
                )
            if 'order' in msg or 'delivery' in msg or 'ship' in msg:
                return schema(
                    intent="order_delivery_inquiry", confidence=0.85, reasoning="Delivery.",
                    severity="MEDIUM", severity_confidence=0.8, severity_reason="Order inquiry needs backend check."
                )
            if 'wifi' in msg or 'bluetooth' in msg or 'connection' in msg or 'network' in msg:
                return schema(
                    intent="network_connectivity", confidence=0.85, reasoning="Network issue.",
                    severity="LOW", severity_confidence=0.85, severity_reason="Routine connectivity troubleshooting."
                )

            return schema(
                intent="software_system_issue", confidence=0.8, reasoning="Default mock intent.",
                severity="LOW", severity_confidence=0.75, severity_reason="General software issue."
            )

        raise ValueError(f"MockProvider doesn't support structured schema: {schema.__name__}")

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        prompt_lower = prompt.lower()
        if "escalate" in prompt_lower and "force" in prompt_lower:
            return "I will escalate this."
        return "Mock response based on evidence: Please try restarting your device."
