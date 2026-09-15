# Phase 8: Final Agent Evaluation

## Overview
This document summarizes the end-to-end evaluation of the SupportLens AI Agent against the frozen Golden Set (v1.0.0).

**Configuration Frozen:**
- `LLM_PROVIDER`: groq
- `LLM_MODEL`: openai/gpt-oss-120b
- `temperature`: 0
- `top_k`: 5
- `retrieval_similarity_threshold`: 0.60
- `intent_confidence_threshold`: 0.70

## 1. Comparison vs. Baselines

| Model | Intent Acc | Intent F1 (Weighted) | Escalation Precision | Escalation Recall |
|-------|------------|----------------------|----------------------|-------------------|
| B1 - Majority | 0.260 | 0.041 | 0.515 | 1.000 |
| B2 - TF-IDF | 0.840 | 0.833 | 0.776 | 0.942 |
| B3 - SupportLens (Groq) | 0.695 | 0.711 | 0.912 | 0.806 |

## 2. Intent Classification

- **Accuracy**: 0.695
- **Macro F1**: 0.675
- **Weighted F1**: 0.711

### Per-Intent Performance

| Intent | Precision | Recall | F1 |
|--------|-----------|--------|----|
| account_security_activation | 0.800 | 1.000 | 0.889 |
| app_store_media_services | 0.333 | 0.438 | 0.378 |
| battery_power_issue | 0.667 | 0.625 | 0.645 |
| feature_inquiry_how_to | 0.471 | 0.500 | 0.485 |
| hardware_damage_repair | 0.750 | 0.562 | 0.643 |
| keyboard_text_input_issue | 0.917 | 0.688 | 0.786 |
| network_connectivity | 0.857 | 0.750 | 0.800 |
| order_delivery_inquiry | 0.944 | 0.548 | 0.694 |
| other_unclear_context_dependent | 0.960 | 0.923 | 0.941 |
| software_system_issue | 0.361 | 0.765 | 0.491 |
| FAILED | 0.000 | 0.000 | 0.000 |

## 3. Escalation & Action Routing

- **Auto-Handle**: Precision 0.817 | Recall 0.918 | F1 0.864
- **Escalate**: Precision 0.912 | Recall 0.806 | F1 0.856
- **Escalation Rate**: 0.455

### Confusion Matrix (Action)
- TP (Escalate correctly): 83
- TN (Auto correctly): 89
- FP (Over-escalate): 8
- FN (Missed escalate): 20

## 4. Retrieval & Evidence

- **Mean Top-1 Similarity**: 0.777
- **Mean Top-3 Similarity**: 0.761
- **Mean Top-5 Similarity**: 0.752
- **Evidence Availability Rate**: 0.945
- **Weak-Evidence Rate (<0.60)**: 0.065
- **Retrieval-Triggered Escalation Rate**: 0.000

## 5. Grounding Flags

The following automated rules caught potential grounding violations:
- Total flagged responses: 16

*(See `evaluation/final_results/grounding_flags.json` for details)*

## 6. Execution Stats
- **Golden Set Size**: 200
- **Total LLM Calls**: 200 (completed successfully)
- **Leakage Checks**: Exact `customer_text` and `conversation_id` matches strictly removed. FAISS index untouched.

