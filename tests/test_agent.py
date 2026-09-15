import pytest
from src.llm.mock import MockProvider
from src.agent.agent import SupportLensAgent
from src.agent.schemas import Evidence

class MockRetriever:
    def retrieve(self, text: str, top_k: int = 5, exclude_conversation_id: str = None):
        return [
            Evidence(source_id="1", customer_message="Mock query", historical_response="Try restarting.", similarity=0.85)
        ]

class MockRetrieverLowSim:
    def retrieve(self, text: str, top_k: int = 5, exclude_conversation_id: str = None):
        return [
            Evidence(source_id="1", customer_message="Mock query", historical_response="Try restarting.", similarity=0.20)
        ]

def test_agent_auto_handle_success():
    provider = MockProvider()
    retriever = MockRetriever()
    agent = SupportLensAgent(provider, retriever)
    
    # "battery" triggers the mock provider to return high confidence "battery_power_issue"
    response = agent.process_message("My battery is draining.")
    
    assert response.action == "AUTO_HANDLE"
    assert response.intent == "battery_power_issue"
    assert response.intent_confidence == 0.9
    assert len(response.evidence) == 1
    assert "Mock response" in response.draft_reply

def test_agent_hard_fallback_low_sim():
    provider = MockProvider()
    retriever = MockRetrieverLowSim()
    agent = SupportLensAgent(provider, retriever)
    
    # "battery" gives high intent confidence, but retriever returns low similarity
    response = agent.process_message("My battery is draining.")
    
    assert response.action == "ESCALATE"
    assert "similarity" in response.action_reason
    assert "escalate this to a support specialist" in response.draft_reply

def test_agent_hard_fallback_high_risk_intent():
    provider = MockProvider()
    retriever = MockRetriever()
    agent = SupportLensAgent(provider, retriever)
    
    # "password" triggers high risk intent
    response = agent.process_message("I forgot my password.")
    
    assert response.action == "ESCALATE"
    assert "account_security_activation" in response.intent
    assert "escalate this to a support specialist" in response.draft_reply
