# Phase 7: AI Agent Implementation & RAG

## Architecture Overview
The `SupportLens` agent implements an explainable, pipeline-driven Support AI.
The flow is:
1. **Input Normalization & Intent Classification** via LLM using our operational 10-intent taxonomy.
2. **Dense Semantic Retrieval** using `sentence-transformers` (`all-MiniLM-L6-v2`) and `FAISS`.
3. **Escalation Policy Evaluation** based on multiple safety signals.
4. **Grounded Generation** synthesizing the retrieved historical responses.

## Retrieval Strategy
- **Embedding Model**: `all-MiniLM-L6-v2` (Chosen for extremely fast CPU-based inference and decent semantic clustering).
- **Index**: `FAISS` Inner Product (Cosine Similarity).
- **Corpus**: ~51,500 substantive historical interactions. We strictly filtered out DM-only deflections from the AppleSupport dataset so that the RAG model pulls actionable troubleshooting steps rather than "DM us your serial number."
- **Leakage Prevention**: During retrieval, if the incoming customer message happens to be from a known historical conversation ID, that specific conversation ID is strictly excluded from the FAISS lookup results. Furthermore, the 200 Golden Set conversations were entirely dropped from the FAISS corpus during indexing.

## Evidence Filtering & Escalation
Escalation is treated as an independent logical layer rather than directly trusting the LLM. 
The agent will **force an escalation** if:
1. **HIGH Severity**: The LLM classifies the issue as HIGH severity (security risk, physical damage, account compromise).
2. **Low Confidence**: The LLM's intent confidence is `< 0.70`.
3. **Low Evidence Quality**: The top retrieved historical interaction has a cosine similarity `< 0.60`.
4. **No Evidence**: If the retriever returns 0 results.
5. **Deterministic Safety Keywords**: Hardcoded regex patterns for fraud, hacking, cracked screens, etc. — fires even without an LLM.

## Grounded Generation constraints
When generating a reply, the LLM is given strict systemic rules:
- ONLY use the provided evidence.
- DO NOT invent Apple policies, URLs, or prices.
- If the evidence doesn't answer the question, do not hallucinate an answer. Output exactly the fallback escalation text.

*(Note: AppleSupport's historical responses are not definitive proof that the customer's underlying issue was resolved. The agent synthesizes these responses to provide the most historically accurate support path, but assumes no inherent success.)*

## Provider Abstraction & Reproducibility
The agent abstracts the LLM via `LLMProvider`. The final benchmark used the following chain:
- **Primary**: `GroqProvider` → `openai/gpt-oss-120b` (190/200 golden examples)
- **Fallback 1**: `GeminiProvider` → `gemini-3.8-flash` via Key 1
- **Fallback 2**: `GeminiProvider` → `gemini-3.8-flash` via Key 2
- **Local Fallback**: MiniLM + FAISS only, no LLM (used for URL-only messages)
- **MockProvider**: Matches strings deterministically to return mocked Pydantic JSON schemas. Ensures the entire CI/CD and testing pipeline can execute without any API keys.

To run the deterministic demo (no API key required):
`python scripts/demo_agent.py`
