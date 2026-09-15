import pytest
from src.agent.schemas import ActionDecision, Evidence
from src.agent.escalation import EscalationPolicy

def test_escalation_high_risk_severity():
    """HIGH severity forces ESCALATE regardless of confidence and retrieval."""
    policy = EscalationPolicy()

    ev = [Evidence(source_id="1", customer_message="x", historical_response="y", similarity=0.99)]
    decision = policy.decide("account_security_activation", 0.99, ev, severity="HIGH", text="")

    assert decision.action == "ESCALATE"
    assert "HIGH" in decision.reason

def test_escalation_low_confidence():
    policy = EscalationPolicy(min_intent_confidence=0.70)
    ev = [Evidence(source_id="1", customer_message="x", historical_response="y", similarity=0.99)]

    decision = policy.decide("battery_power_issue", 0.65, ev, severity="LOW", text="my battery")

    assert decision.action == "ESCALATE"
    assert "confidence" in decision.reason.lower()

def test_escalation_low_retrieval():
    policy = EscalationPolicy(min_retrieval_sim=0.60)
    ev = [Evidence(source_id="1", customer_message="x", historical_response="y", similarity=0.55)]

    decision = policy.decide("battery_power_issue", 0.95, ev, severity="LOW", text="my battery")

    assert decision.action == "ESCALATE"
    assert "similarity" in decision.reason.lower()

def test_escalation_no_evidence():
    policy = EscalationPolicy()
    decision = policy.decide("battery_power_issue", 0.95, [], severity="LOW", text="my battery")

    assert decision.action == "ESCALATE"
    assert "No retrieval evidence" in decision.reason

def test_auto_handle_success():
    policy = EscalationPolicy(min_intent_confidence=0.70, min_retrieval_sim=0.60)
    ev = [Evidence(source_id="1", customer_message="x", historical_response="y", similarity=0.85)]

    decision = policy.decide("battery_power_issue", 0.90, ev, severity="LOW", text="my battery drains")

    assert decision.action == "AUTO_HANDLE"
