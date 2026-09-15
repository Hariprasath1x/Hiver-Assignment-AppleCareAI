import os
import sys
import time
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.provider import OpenAIProvider
from src.llm.gemini import GeminiProvider
from src.llm.groq import GroqProvider
from src.agent.schemas import IntentResult
from src.agent.prompts import INTENT_CLASSIFICATION_PROMPT

def run_smoke_test():
    load_dotenv()
    
    print("API Key loaded:", "Yes" if os.getenv("OPENAI_API_KEY") else "No")
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    print("LLM_PROVIDER:", llm_provider)
    print("LLM_MODEL:", os.getenv("LLM_MODEL", "gpt-4o"))
    
    if llm_provider == "gemini":
        provider = GeminiProvider()
    elif llm_provider == "groq":
        provider = GroqProvider()
    else:
        provider = OpenAIProvider()
    
    # Test text
    test_msg = "My battery keeps draining really fast after the latest update."
    prompt = INTENT_CLASSIFICATION_PROMPT.format(customer_message=test_msg)
    
    print("\nExecuting Structured Output Call...")
    start_time = time.time()
    
    try:
        result = provider.generate_structured(prompt, IntentResult, temperature=0.0)
        end_time = time.time()
        
        print("\n--- SMOKE TEST SUCCESS ---")
        print(f"Latency: {end_time - start_time:.2f} seconds")
        print(f"Parsed Pydantic Instance: {type(result)}")
        print(f"Intent: {result.intent}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")
        
    except Exception as e:
        print("\n--- SMOKE TEST FAILED ---")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_smoke_test()
