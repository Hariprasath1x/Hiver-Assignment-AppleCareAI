from typing import Optional
from src.llm.base import LLMProvider
from src.agent.schemas import AgentResponse
from src.agent.classifier import IntentClassifier
from src.agent.escalation import EscalationPolicy, check_deterministic_safety
from src.agent.generator import GroundedGenerator
from src.retrieval.retriever import SemanticRetriever

ESCALATE_REPLY = "I want to make sure you get the right help with this. I'll escalate this to a support specialist who can look into it further."

# Patterns that indicate a DM-only or context-dependent historical response
# (we should not use these verbatim)
_DM_PHRASES = ["DM us", "send us a DM", "direct message", "reach out via DM", "please follow"]
_SHORT_RESPONSE_MIN_LEN = 30  # historical response must have enough content to be useful


class SupportLensAgent:
    def __init__(
        self,
        provider: LLMProvider,
        retriever: SemanticRetriever,
        escalation_policy: Optional[EscalationPolicy] = None,
    ):
        self.provider = provider
        self.retriever = retriever
        self.classifier = IntentClassifier(provider)
        self.generator = GroundedGenerator(provider)
        self.escalation_policy = escalation_policy or EscalationPolicy()

    def process_message(self, text: str, exclude_conversation_id: str = None) -> AgentResponse:
        # 1. Intent + Severity Classification (LLM)
        intent_result = self.classifier.classify(text)

        # 2. Dense Semantic Retrieval (MiniLM + FAISS — no LLM)
        evidence = self.retriever.retrieve(
            text, top_k=5, exclude_conversation_id=exclude_conversation_id
        )

        # 3. Escalation Decision (deterministic policy)
        action_decision = self.escalation_policy.decide(
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            evidence=evidence,
            severity=intent_result.severity,
            text=text,
        )

        # 4. Grounded Response Generation
        if action_decision.action == "ESCALATE":
            draft_reply = ESCALATE_REPLY
        else:
            draft_reply = self.generator.generate(
                text=text,
                intent=intent_result.intent,
                evidence=evidence,
            )

        return AgentResponse(
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            severity=intent_result.severity,
            severity_confidence=intent_result.severity_confidence,
            severity_reason=intent_result.severity_reason,
            action=action_decision.action,
            action_reason=action_decision.reason,
            draft_reply=draft_reply,
            evidence=evidence,
        )

    def process_local_fallback(
        self, text: str, exclude_conversation_id: str = None
    ) -> AgentResponse:
        """
        LLM-free fallback. Uses only MiniLM + FAISS retrieval and deterministic rules.
        Does NOT call any LLM provider.
        """
        # Run deterministic safety check
        safety_trigger = check_deterministic_safety(text)

        # Retrieve evidence
        evidence = self.retriever.retrieve(
            text, top_k=5, exclude_conversation_id=exclude_conversation_id
        )

        top_sim = evidence[0].similarity if evidence else 0.0
        top_response = evidence[0].historical_response if evidence else ""

        # Conditions for safe local AUTO_HANDLE:
        # - no safety trigger
        # - top similarity > 0.85
        # - historical response is non-empty and substantial
        # - response does not appear DM-only or context-dependent
        can_auto_handle = (
            safety_trigger is None
            and top_sim > 0.85
            and len(top_response.strip()) >= _SHORT_RESPONSE_MIN_LEN
            and not any(phrase.lower() in top_response.lower() for phrase in _DM_PHRASES)
        )

        if can_auto_handle:
            action = "AUTO_HANDLE"
            action_reason = (
                f"Local MiniLM fallback: strong evidence (sim={top_sim:.3f}), "
                f"safe to use historical response verbatim."
            )
            draft_reply = top_response.strip()
        else:
            action = "ESCALATE"
            reason_parts = []
            if safety_trigger:
                reason_parts.append(safety_trigger)
            if top_sim <= 0.85:
                reason_parts.append(f"Insufficient similarity ({top_sim:.3f} ≤ 0.85).")
            if not evidence:
                reason_parts.append("No retrieval evidence found.")
            action_reason = " | ".join(reason_parts) if reason_parts else "Local fallback: escalating for safety."
            draft_reply = ESCALATE_REPLY

        return AgentResponse(
            intent="UNKNOWN",
            intent_confidence=0.0,
            severity="UNKNOWN",
            severity_confidence=0.0,
            severity_reason="LLM unavailable — severity not assessed.",
            action=action,
            action_reason=action_reason,
            draft_reply=draft_reply,
            evidence=evidence,
        )
