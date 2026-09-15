# Phase 6: Baseline Evaluation

This report presents the baseline evaluation metrics on the Golden Evaluation Set (v1.0.0). These baselines establish a credible performance floor before deploying the final AI Agent.

## 1. Baseline Architectures
- **Baseline 1 (Majority Classifier)**: Trivial baseline that always predicts the majority intent (`other_unclear_context_dependent`), always returns a generic reply, and always chooses `ESCALATE`.
- **Baseline 2 (TF-IDF Simple ML)**: TF-IDF vectorization with Logistic Regression for intent classification, and TF-IDF cosine similarity for retrieval. Escalate if intent confidence < 0.35 or retrieval similarity < 0.40. Both were trained/tuned exclusively on a 20k Weakly Labeled Dev Set.

## 2. Leakage Validation
The evaluation suite explicitly checks for data contamination. Tests confirm:
- No golden conversation IDs exist in the 20k Dev Set.
- No golden conversation IDs exist in the 47k Retrieval Corpus.
- Exact duplicate customer text from the golden set is rigorously excluded from retrieval pool.

## 3. Comparison Table

| Metric | Majority (B1) | TF-IDF (B2) | Later AI Agent |
| :--- | :--- | :--- | :--- |
| Intent Accuracy | 26.0% | 84.0% | - |
| Intent Macro F1 | 4.1% | 83.3% | - |
| Intent Weighted F1 | 10.7% | 83.5% | - |
| Auto-handle Precision | N/A (0.0%) | 92.0% | - |
| Escalation Recall | 100% | 94.1% | - |
| Escalation Rate | 100% | 62.5% | - |
| Avg Top-1 Similarity | N/A | 0.63 | - |

*(Note: Semantic relevance of the replies was deferred to the Phase 7 LLM Judge, but the retrieval similarity indicates TF-IDF often fails to find a high-confidence exact match).*

## 4. Confusion Matrix Analysis (TF-IDF)
The TF-IDF model performed surprisingly well on standard technical intents (almost 100% on network, hardware, and app store issues), largely due to the distinct vocabulary of those problems (e.g. "wifi", "screen", "music").

**Difficult Intents**:
- `feature_inquiry_how_to` is heavily confused with `other_unclear_context_dependent`. TF-IDF correctly classified only 5/16 examples, predicting `other` for 9 examples. Reason: Feature inquiries are often short ("how do I do this?"), lacking the strong technical keywords of bug reports, making them look like vague context-dependent messages to a bag-of-words model.

## 5. Threshold Behavior & Auto-Handle Safety
Baseline 2 achieved a remarkable **92% Auto-handle Precision** while recalling **94% of Escalations**. It did this by heavily leveraging the threshold fallback: if it wasn't extremely confident, it escalated. As a result, its overall **Escalation Rate is 62.5%** (compared to the true dataset distribution of 51.5%). 

*Note on Golden Set Limitations*: The golden set intentionally oversamples rare and escalation-sensitive examples for safety testing. Therefore, this 62.5% escalation rate is highly adversarial and not representative of the real-world baseline escalation rate on standard traffic.

## 6. What the Baselines Teach Us
1. **Intent Classification is Easy for Technical Bugs**: The eventual AI Agent does not need to over-engineer standard intent classification. Simple ML solves it.
2. **Context and Nuance are the Real Bottlenecks**: TF-IDF completely fails to distinguish between a vague complaint ("this is trash") and a genuine question ("how do I use this?"). The LLM's true value will be semantically parsing short, context-dependent feature requests.
3. **Retrieval needs Semantic Search**: An average TF-IDF top-1 similarity of 0.63 suggests lexical retrieval misses synonymous issues. We need semantic embeddings in the final agent.
