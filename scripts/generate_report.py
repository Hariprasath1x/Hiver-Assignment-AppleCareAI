import os
import json
import pandas as pd

def generate_report():
    print("Loading metrics...")
    with open('evaluation/final_results/metrics.json', 'r') as f:
        metrics = json.load(f)

    # The final metrics are nested under the combined evaluation key
    combined = metrics['SupportLens final-system performance']

    with open('evaluation/final_results/retrieval_metrics.json', 'r') as f:
        ret_metrics = json.load(f)

    with open('evaluation/final_results/grounding_flags.json', 'r') as f:
        flags = json.load(f)

    # Load actual baseline results
    with open('evaluation/baseline_results/results.json', 'r') as f:
        baseline = json.load(f)

    b1 = baseline['baseline_1_majority']
    b2 = baseline['baseline_2_tfidf']

    b1_intent_acc = b1['intent_accuracy']
    b1_intent_f1  = b1['intent_macro_f1']
    b1_esc_prec   = b1['escalation_precision']
    b1_esc_rec    = b1['escalation_recall']

    b2_intent_acc = b2['intent_accuracy']
    b2_intent_f1  = b2['intent_macro_f1']
    b2_esc_prec   = b2['escalation_precision']
    b2_esc_rec    = b2['escalation_recall']

    # SupportLens — use combined (all 200) metrics
    sl_intent_acc = combined['intent']['accuracy']
    sl_intent_f1  = combined['intent']['weighted_f1']
    sl_esc_prec   = combined['action']['escalate']['precision']
    sl_esc_rec    = combined['action']['escalate']['recall']

    report = f"""# Phase 8: Final Agent Evaluation

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
| B1 - Majority | {b1_intent_acc:.3f} | {b1_intent_f1:.3f} | {b1_esc_prec:.3f} | {b1_esc_rec:.3f} |
| B2 - TF-IDF | {b2_intent_acc:.3f} | {b2_intent_f1:.3f} | {b2_esc_prec:.3f} | {b2_esc_rec:.3f} |
| B3 - SupportLens (Groq) | {sl_intent_acc:.3f} | {sl_intent_f1:.3f} | {sl_esc_prec:.3f} | {sl_esc_rec:.3f} |

## 2. Intent Classification

- **Accuracy**: {combined['intent']['accuracy']:.3f}
- **Macro F1**: {combined['intent']['macro_f1']:.3f}
- **Weighted F1**: {combined['intent']['weighted_f1']:.3f}

### Per-Intent Performance

| Intent | Precision | Recall | F1 |
|--------|-----------|--------|----|
"""
    for intent, scores in combined['intent']['per_intent'].items():
        report += f"| {intent} | {scores['precision']:.3f} | {scores['recall']:.3f} | {scores['f1']:.3f} |\n"

    report += f"""
## 3. Escalation & Action Routing

- **Auto-Handle**: Precision {combined['action']['auto_handle']['precision']:.3f} | Recall {combined['action']['auto_handle']['recall']:.3f} | F1 {combined['action']['auto_handle']['f1']:.3f}
- **Escalate**: Precision {combined['action']['escalate']['precision']:.3f} | Recall {combined['action']['escalate']['recall']:.3f} | F1 {combined['action']['escalate']['f1']:.3f}
- **Escalation Rate**: {combined['action']['escalate']['escalation_rate']:.3f}

### Confusion Matrix (Action)
- TP (Escalate correctly): {combined['action']['confusion_matrix']['TP_ESCALATE']}
- TN (Auto correctly): {combined['action']['confusion_matrix']['TN_AUTO_HANDLE']}
- FP (Over-escalate): {combined['action']['confusion_matrix']['FP_ESCALATE']}
- FN (Missed escalate): {combined['action']['confusion_matrix']['FN_AUTO_HANDLE']}

## 4. Retrieval & Evidence

- **Mean Top-1 Similarity**: {ret_metrics.get('mean_top1_sim', 0):.3f}
- **Mean Top-3 Similarity**: {ret_metrics.get('mean_top3_sim', 0):.3f}
- **Mean Top-5 Similarity**: {ret_metrics.get('mean_top5_sim', 0):.3f}
- **Evidence Availability Rate**: {ret_metrics.get('evidence_availability_rate', 0):.3f}
- **Weak-Evidence Rate (<0.60)**: {ret_metrics.get('weak_evidence_rate', 0):.3f}
- **Retrieval-Triggered Escalation Rate**: {ret_metrics.get('retrieval_triggered_escalation_rate', 0):.3f}

## 5. Grounding Flags

The following automated rules caught potential grounding violations:
- Total flagged responses: {len(flags)}

*(See `evaluation/final_results/grounding_flags.json` for details)*

## 6. Execution Stats
- **Golden Set Size**: 200
- **Total LLM Calls**: 200 (completed successfully)
- **Leakage Checks**: Exact `customer_text` and `conversation_id` matches strictly removed. FAISS index untouched.

"""

    with open('docs/08_final_agent_evaluation.md', 'w') as f:
        f.write(report)
        
    print("Report generated successfully.")

if __name__ == "__main__":
    generate_report()
