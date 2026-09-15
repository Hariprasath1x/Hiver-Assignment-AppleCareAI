import re
from typing import List, Optional
from src.agent.schemas import ActionDecision, Evidence

# Deterministic high-risk keyword patterns — conservative, case-insensitive
# These trigger ESCALATE regardless of LLM output
SAFETY_PATTERNS = [
    r'\bhack(ed|ing|er)?\b',
    r'\bcompromis(ed|e)\b',
    r'\bunauthori[sz]ed\b',
    r'\bstolen\b',
    r'\bfraud\b',
    r'\bsuspicious\b',
    r'\bsecurity incident\b',
    r'\bidentity theft\b',
    r'\bscam\b',
    r'\bphishing\b',
    r'\bacccount tak[eo]ver\b',
    r'\bsomeone else.{0,20}(account|phone|device)\b',
    r'\b(account|phone|device).{0,20}stolen\b',
    r'\bcan.?t (get in|access|log in|sign in)\b',
    r'\blocked out\b',
    r'\bactivation lock\b',
    r'\btalk to (a |an )?(human|person|agent|rep|representative|specialist|someone)\b',
    r'\bspeak (to|with) (a |an )?(human|person|agent|rep|representative|specialist|someone)\b',
    r'\b(need|want) (a |an )?(human|person|agent|rep|representative|specialist)\b',
    r'\bphysical damage\b',
    r'\bcracked screen\b',
    r'\bbroken screen\b',
    r'\bscreen (is |)shatter(ed)?\b',
    r'\bscreen (is |)crack(ed)?\b',
    r'\bwater damage\b',
    r'\bdrop(ped)? (my |the )?(iphone|phone|device|mac)\b',
    r'\brepair (needed|required|center|service)\b',
    r'\bapple (store|care|repair)\b',
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SAFETY_PATTERNS]


def check_deterministic_safety(text: str) -> Optional[str]:
    """
    Returns a trigger description if a deterministic safety rule fires, else None.
    This check runs independently of LLM availability.
    """
    for pattern in _COMPILED_PATTERNS:
        m = pattern.search(text)
        if m:
            return f"Deterministic safety trigger: matched pattern '{pattern.pattern}' in message."
    return None


class EscalationPolicy:
    def __init__(
        self,
        min_intent_confidence: float = 0.70,
        min_retrieval_sim: float = 0.60,
        min_retrieval_sim_medium: float = 0.75,
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_sim = min_retrieval_sim
        self.min_retrieval_sim_medium = min_retrieval_sim_medium

    def decide(
        self,
        intent: str,
        confidence: float,
        evidence: List[Evidence],
        severity: str = "HIGH",
        text: str = "",
    ) -> ActionDecision:
        """
        Deterministic escalation policy.

        Order of checks:
        1. Deterministic safety keywords (LLM-independent)
        2. HIGH severity
        3. Low intent confidence
        4. No retrieval evidence
        5. Top similarity below minimum
        6. MEDIUM severity — requires stronger evidence
        7. Otherwise AUTO_HANDLE (only for LOW severity with sufficient signals)
        """

        # 1. Deterministic safety trigger (runs even when LLM unavailable)
        if text:
            safety_trigger = check_deterministic_safety(text)
            if safety_trigger:
                return ActionDecision(action="ESCALATE", reason=safety_trigger)

        # 2. HIGH severity → always escalate
        if severity == "HIGH":
            return ActionDecision(
                action="ESCALATE",
                reason=f"Severity is HIGH — automatic escalation required."
            )

        # 3. Intent confidence too low
        if confidence < self.min_intent_confidence:
            return ActionDecision(
                action="ESCALATE",
                reason=f"Intent confidence ({confidence:.2f}) below threshold ({self.min_intent_confidence})."
            )

        # 4. No retrieval evidence
        if not evidence:
            return ActionDecision(
                action="ESCALATE",
                reason="No retrieval evidence found."
            )

        # 5. Top similarity below minimum
        top_sim = evidence[0].similarity
        if top_sim < self.min_retrieval_sim:
            return ActionDecision(
                action="ESCALATE",
                reason=f"Top evidence similarity ({top_sim:.2f}) below threshold ({self.min_retrieval_sim})."
            )

        # 6. MEDIUM severity — requires stronger evidence
        if severity == "MEDIUM":
            if top_sim < self.min_retrieval_sim_medium:
                return ActionDecision(
                    action="ESCALATE",
                    reason=(
                        f"MEDIUM severity requires top similarity >= {self.min_retrieval_sim_medium}, "
                        f"but got {top_sim:.2f}."
                    )
                )

        # 7. All checks passed → AUTO_HANDLE
        return ActionDecision(
            action="AUTO_HANDLE",
            reason="Confidence and evidence quality are sufficient for safe auto-handling."
        )
