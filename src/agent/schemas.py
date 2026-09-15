from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class IntentResult(BaseModel):
    intent: str = Field(description="The predicted operational intent.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    reasoning: str = Field(description="Short rationale for the chosen intent.")
    severity: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="Operational severity/risk level: LOW (routine), MEDIUM (needs care), HIGH (must escalate)."
    )
    severity_confidence: float = Field(
        description="Confidence in the severity assessment, between 0.0 and 1.0."
    )
    severity_reason: str = Field(
        description="Short reason for the severity rating."
    )

class Evidence(BaseModel):
    source_id: str
    customer_message: str
    historical_response: str
    similarity: float

class ActionDecision(BaseModel):
    action: str = Field(description="Must be 'AUTO_HANDLE' or 'ESCALATE'")
    reason: str = Field(description="Reasoning for why this action was chosen.")

class AgentResponse(BaseModel):
    intent: str
    intent_confidence: float
    severity: str
    severity_confidence: float
    severity_reason: str
    action: str
    action_reason: str
    draft_reply: str
    evidence: List[Evidence]
