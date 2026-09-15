import faiss
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.mock import MockProvider
from src.llm.provider import OpenAIProvider
from src.llm.gemini import GeminiProvider
from src.llm.groq import GroqProvider
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import SemanticRetriever
from src.agent.agent import SupportLensAgent
from dotenv import load_dotenv

def demo():
    print("Loading Agent...")
    load_dotenv()
    llm_provider = os.getenv("LLM_PROVIDER", "mock").lower()
    
    if llm_provider == "gemini":
        provider = GeminiProvider()
    elif llm_provider == "groq":
        provider = GroqProvider()
    elif llm_provider == "openai":
        provider = OpenAIProvider()
    else:
        provider = MockProvider()
        
    print(f"Using Provider: {provider.__class__.__name__}")
    embedder = Embedder()
    
    # Needs the index built first
    index_path = 'data/processed/retrieval_index.faiss'
    corpus_path = 'data/processed/retrieval_corpus.parquet'
    
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}. Run scripts/build_faiss_index.py first.")
        return
        
    retriever = SemanticRetriever(embedder, index_path, corpus_path)
    agent = SupportLensAgent(provider, retriever)
    
    examples = [
        "My iPhone 7 battery is draining super fast after the update.",
        "Yes I did", # Context-dependent, should escalate
        "I need a refund for an unauthorized charge from iTunes." # High-risk, should escalate
    ]
    
    for text in examples:
        print("\n" + "="*50)
        print(f"CUSTOMER: {text}")
        print("="*50)
        
        response = agent.process_message(text)
        
        print(f"Intent: {response.intent}")
        print(f"Confidence: {response.intent_confidence}")
        print(f"Action: {response.action}")
        print(f"Reason: {response.action_reason}\n")
        print(f"Draft Reply:\n{response.draft_reply}\n")
        print("Historical Evidence:")
        for i, ev in enumerate(response.evidence):
            print(f" {i+1}. [Sim: {ev.similarity:.2f}] {ev.customer_message}")
            print(f"    -> {ev.historical_response}\n")

if __name__ == "__main__":
    demo()
