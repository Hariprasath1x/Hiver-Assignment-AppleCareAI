"""
SupportLens Final Evaluation Runner — Phase 8 Lockdown
======================================================

Provider fallback order (exactly one Groq key):
  1. Groq Key 1     (GROQ_API_KEY_1 / openai/gpt-oss-120b)
  2. Gemini Key 1   (GEMINI_API_KEY_1 / gemini-3.8-flash)
  3. Gemini Key 2   (GEMINI_API_KEY_2 / gemini-3.8-flash)
  4. Local MiniLM + FAISS (no LLM)
  5. ESCALATE

Usage:
  python evaluation/run_final_agent.py               # run all 200 (fresh)
  python evaluation/run_final_agent.py --retry-failed  # retry only FAILED/missing

IMPORTANT: Do NOT modify the golden set, taxonomy, thresholds, or any successful predictions.
"""

import faiss
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import json
import time
import re
import argparse
import pandas as pd
import numpy as np
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from src.llm.groq import GroqProvider
from src.llm.gemini import GeminiProvider
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import SemanticRetriever
from src.agent.agent import SupportLensAgent
from src.agent.escalation import check_deterministic_safety

AGENT_VERSION = "supportlens-final-v1"

# Frozen thresholds — do NOT change
INTENT_CONFIDENCE_THRESHOLD = 0.70
SIMILARITY_THRESHOLD = 0.60
TOP_K = 5

PREDICTIONS_FILE = 'evaluation/final_results/predictions.jsonl'
GOLDEN_SET_FILE  = 'evaluation/golden_set.csv'

ESCALATE_REPLY = "I want to make sure you get the right help with this. I'll escalate this to a support specialist who can look into it further."


# ---------------------------------------------------------------------------
# Grounding check (unchanged from original)
# ---------------------------------------------------------------------------
def check_grounding(reply: str, evidence: list) -> list:
    flags = []
    evidence_text = " ".join([ev.historical_response for ev in evidence]).lower()
    reply_lower = reply.lower()

    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', reply_lower)
    for url in urls:
        if url not in evidence_text:
            flags.append(f"URL not in evidence: {url}")

    phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', reply_lower)
    for phone in phones:
        if phone not in evidence_text:
            flags.append(f"Phone number not in evidence: {phone}")

    prices = re.findall(r'\$\d+(?:\.\d{2})?|\£\d+(?:\.\d{2})?', reply_lower)
    if prices:
        flags.append("Price mentioned in reply")

    return flags


# ---------------------------------------------------------------------------
# Build the provider fallback list
# ---------------------------------------------------------------------------
def build_provider_list():
    """
    Constructs a list of (provider_label, model_name, provider_instance)
    from available env keys, in strict fallback order.
    No GROQ_API_KEY_2 — only one Groq key.
    """
    providers = []

    groq_key1 = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY")
    if groq_key1:
        try:
            p = GroqProvider(api_key=groq_key1, model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))
            providers.append({
                "label": "groq_key1",
                "provider_used": "groq",
                "model_used": p.model,
                "instance": p,
            })
        except Exception as e:
            print(f"  [WARN] Groq Key 1 init failed: {e}")

    gemini_model = os.getenv("LLM_MODEL_GEMINI", "gemini-3.8-flash")

    gemini_key1 = os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY")
    if gemini_key1:
        try:
            p = GeminiProvider(api_key=gemini_key1, model=gemini_model)
            providers.append({
                "label": "gemini_key1",
                "provider_used": "gemini",
                "model_used": p.model,
                "instance": p,
            })
        except Exception as e:
            print(f"  [WARN] Gemini Key 1 init failed: {e}")

    gemini_key2 = os.getenv("GEMINI_API_KEY_2")
    if gemini_key2:
        try:
            p = GeminiProvider(api_key=gemini_key2, model=gemini_model)
            providers.append({
                "label": "gemini_key2",
                "provider_used": "gemini",
                "model_used": p.model,
                "instance": p,
            })
        except Exception as e:
            print(f"  [WARN] Gemini Key 2 init failed: {e}")

    return providers


# ---------------------------------------------------------------------------
# Classify error type
# ---------------------------------------------------------------------------
def is_rate_limit(err_msg: str) -> bool:
    return '429' in err_msg or 'ratelimit' in err_msg.replace('_', '').replace('-', '') or 'rate limit' in err_msg

def is_transient(err_msg: str) -> bool:
    return any(x in err_msg for x in ['timeout', 'connection', 'network', 'unavailable', '502', '503', '504', '500'])


# ---------------------------------------------------------------------------
# Attempt one example across the provider chain
# ---------------------------------------------------------------------------
def run_with_fallback(q_id, text, conv_id, providers, retriever):
    """
    Attempt prediction for one example through the full provider chain.
    Returns (response, provider_meta) or (None, provider_meta).
    provider_meta contains: provider_used, model_used, fallback_used, fallback_reason, system_status
    """
    provider_meta = {
        "provider_used": None,
        "model_used": None,
        "fallback_used": False,
        "fallback_reason": None,
        "system_status": "LLM_UNAVAILABLE",
    }

    for i, prov in enumerate(providers):
        label = prov["label"]
        print(f"    [{label}] Attempting...")

        max_transient = 2
        for attempt in range(max_transient + 1):
            try:
                # Always instantiate a fresh SupportLensAgent to guarantee
                # all internal components (classifier, generator) use this provider
                agent = SupportLensAgent(prov["instance"], retriever)
                response = agent.process_message(text, exclude_conversation_id=conv_id)

                provider_meta["provider_used"] = prov["provider_used"]
                provider_meta["model_used"] = prov["model_used"]
                provider_meta["fallback_used"] = (i > 0)
                if i > 0:
                    provider_meta["fallback_reason"] = f"prior_provider_failed"
                provider_meta["system_status"] = "NORMAL" if i == 0 else "DEGRADED"

                print(f"    [{label}] SUCCESS (attempt {attempt+1})")
                return response, provider_meta

            except Exception as e:
                err_msg = str(e)
                # 429 / rate-limit → immediately move on, no retry of same key
                if is_rate_limit(err_msg.lower()):
                    print(f"    [{label}] 429 rate limit — moving to next provider immediately.")
                    break
                # Transient → bounded retry
                elif is_transient(err_msg.lower()) and attempt < max_transient:
                    wait = 3 * (attempt + 1)
                    print(f"    [{label}] Transient error (attempt {attempt+1}): {err_msg[:80]}. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    # Fatal or exhausted transient retries
                    print(f"    [{label}] Failed: {err_msg[:120]}")
                    break

    # All LLM providers exhausted — try local fallback
    print(f"    [local_fallback] All LLM providers failed. Attempting local MiniLM + FAISS only...")
    provider_meta["fallback_used"] = True
    provider_meta["system_status"] = "LLM_UNAVAILABLE"
    provider_meta["fallback_reason"] = "all_llm_providers_failed"
    provider_meta["provider_used"] = "local"
    provider_meta["model_used"] = "local-minilm-faiss"

    try:
        # Use first available provider's retriever (they all share the same retriever)
        local_agent = SupportLensAgent.__new__(SupportLensAgent)
        local_agent.retriever = retriever
        from src.agent.escalation import EscalationPolicy
        local_agent.escalation_policy = EscalationPolicy()
        response = local_agent.process_local_fallback(text, exclude_conversation_id=conv_id)
        print(f"    [local_fallback] Result: {response.action}")
        return response, provider_meta
    except Exception as e:
        print(f"    [local_fallback] Failed: {e}")
        return None, provider_meta


# ---------------------------------------------------------------------------
# Build a single prediction record
# ---------------------------------------------------------------------------
def build_prediction(q_id, row, response, provider_meta, is_failed=False):
    if is_failed or response is None:
        return {
            'id': q_id,
            'customer_message': row['customer_text'],
            'expected_intent': row['expected_intent'],
            'predicted_intent': 'FAILED',
            'intent_correct': 0,
            'expected_action': row['expected_action'],
            'predicted_action': 'FAILED',
            'action_correct': 0,
            'intent_confidence': 0.0,
            'severity': None,
            'severity_confidence': None,
            'severity_reason': None,
            'top_similarity': 0.0,
            'top3_similarity': 0.0,
            'top5_similarity': 0.0,
            'evidence_count': 0,
            'evidence_ids': [],
            'draft_reply': 'FAILED',
            'action_reason': 'All providers failed.',
            'grounding_flags': [],
            **provider_meta,
            'agent_version': AGENT_VERSION,
        }

    sims = [ev.similarity for ev in response.evidence]
    flags = check_grounding(response.draft_reply, response.evidence)

    return {
        'id': q_id,
        'customer_message': row['customer_text'],
        'expected_intent': row['expected_intent'],
        'predicted_intent': response.intent,
        'intent_correct': int(response.intent == row['expected_intent']),
        'expected_action': row['expected_action'],
        'predicted_action': response.action,
        'action_correct': int(response.action == row['expected_action']),
        'intent_confidence': response.intent_confidence,
        'severity': response.severity,
        'severity_confidence': response.severity_confidence,
        'severity_reason': response.severity_reason,
        'top_similarity': sims[0] if sims else 0.0,
        'top3_similarity': sum(sims[:3]) / len(sims[:3]) if sims[:3] else 0.0,
        'top5_similarity': sum(sims[:5]) / len(sims[:5]) if sims[:5] else 0.0,
        'evidence_count': len(sims),
        'evidence_ids': [ev.source_id for ev in response.evidence],
        'draft_reply': response.draft_reply,
        'action_reason': response.action_reason,
        'grounding_flags': flags,
        **provider_meta,
        'agent_version': AGENT_VERSION,
    }


# ---------------------------------------------------------------------------
# Compute metrics for a given subset of records
# ---------------------------------------------------------------------------
def compute_metrics(records, golden_df, label):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, precision_recall_fscore_support
    )

    if not records:
        return None

    df = pd.DataFrame(records)
    all_intents = sorted(golden_df['expected_intent'].unique())
    intent_classes = all_intents + ['FAILED']
    action_labels = ["AUTO_HANDLE", "ESCALATE"]

    y_true_intent = df['expected_intent']
    y_pred_intent = df['predicted_intent']
    y_true_action = df['expected_action']
    y_pred_action = df['predicted_action']

    p, r, f1, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, labels=intent_classes, zero_division=0
    )
    per_intent = {cls: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f1[i])}
                  for i, cls in enumerate(intent_classes)}

    intent_metrics = {
        'accuracy': float(accuracy_score(y_true_intent, y_pred_intent)),
        'macro_f1': float(f1_score(y_true_intent, y_pred_intent, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true_intent, y_pred_intent, average='weighted', zero_division=0)),
        'per_intent': per_intent,
    }

    ap = precision_score(y_true_action, y_pred_action, labels=action_labels, average=None, zero_division=0)
    ar = recall_score(y_true_action, y_pred_action, labels=action_labels, average=None, zero_division=0)
    af = f1_score(y_true_action, y_pred_action, labels=action_labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true_action, y_pred_action, labels=action_labels)
    TN, FP, FN, TP = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    action_metrics = {
        'auto_handle': {'precision': float(ap[0]), 'recall': float(ar[0]), 'f1': float(af[0])},
        'escalate': {
            'precision': float(ap[1]), 'recall': float(ar[1]), 'f1': float(af[1]),
            'escalation_rate': float((df['predicted_action'] == 'ESCALATE').mean()),
        },
        'confusion_matrix': {
            'TP_ESCALATE': int(TP), 'TN_AUTO_HANDLE': int(TN),
            'FP_ESCALATE': int(FP), 'FN_AUTO_HANDLE': int(FN),
        },
    }

    retrieval_metrics = {
        'mean_top1_sim': float(df['top_similarity'].mean()),
        'median_top1_sim': float(df['top_similarity'].median()),
        'mean_top3_sim': float(df['top3_similarity'].mean()) if 'top3_similarity' in df else 0,
        'evidence_availability_rate': float((df['evidence_count'] > 0).mean()) if 'evidence_count' in df else 1.0,
        'weak_evidence_rate': float((df['top_similarity'] < SIMILARITY_THRESHOLD).mean()),
    }

    return {
        'label': label,
        'sample_size': len(records),
        'intent': intent_metrics,
        'action': action_metrics,
        'retrieval': retrieval_metrics,
    }


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry only FAILED/missing examples.')
    args = parser.parse_args()

    print("=" * 60)
    print("SupportLens Final Evaluation")
    print("=" * 60)

    # Validate at least one LLM provider is configured
    has_groq = bool(os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY"))
    if not has_groq and not has_gemini:
        print("WARNING: No LLM provider configured. Will use local fallback only.")

    os.makedirs('evaluation/final_results/confusion_matrices', exist_ok=True)

    print("\nInitializing components...")
    embedder = Embedder()
    retriever = SemanticRetriever(
        embedder,
        'data/processed/retrieval_index.faiss',
        'data/processed/retrieval_corpus.parquet'
    )

    providers = build_provider_list()
    if providers:
        print(f"Provider chain: {' → '.join(p['label'] for p in providers)} → local_fallback")
    else:
        print("Provider chain: local_fallback only")

    print("\nLoading Golden Set...")
    golden_df = pd.read_csv(GOLDEN_SET_FILE)
    golden_ids = set(golden_df['example_id'].astype(str))
    total_examples = len(golden_df)

    # -----------------------------------------------------------------------
    # Load existing predictions — do NOT touch successful ones
    # -----------------------------------------------------------------------
    existing_records = {}     # q_id -> record (all existing)
    failed_ids = set()

    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                qid = record['id']
                if record.get('predicted_intent') == 'FAILED':
                    failed_ids.add(qid)
                    existing_records[qid] = record  # keep track but will retry
                else:
                    existing_records[qid] = record  # successful — preserve

    successful_ids = {qid for qid, r in existing_records.items()
                      if r.get('predicted_intent') != 'FAILED'}
    missing_ids = golden_ids - set(existing_records.keys())
    incomplete_ids = failed_ids | missing_ids   # the 4 to retry

    if args.retry_failed:
        print(f"\nRetry mode: preserving {len(successful_ids)} successes, retrying {len(incomplete_ids)} incomplete.")
        print(f"  Failed IDs: {sorted(failed_ids)}")
        print(f"  Missing IDs: {sorted(missing_ids)}")
        target_ids = incomplete_ids
    else:
        target_ids = golden_ids  # full run

    # -----------------------------------------------------------------------
    # Write initial snapshot: all existing successful records to file
    # -----------------------------------------------------------------------
    # This gives us a clean state. Successful records are preserved verbatim.
    if args.retry_failed:
        with open(PREDICTIONS_FILE, 'w') as f:
            for qid, record in existing_records.items():
                if qid not in incomplete_ids:
                    f.write(json.dumps(record) + '\n')
                    f.flush()

    # -----------------------------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------------------------
    retry_count = 0
    provider_tally = Counter()
    new_records = []

    for idx, row in golden_df.iterrows():
        q_id = str(row['example_id'])

        if args.retry_failed and q_id not in target_ids:
            continue
        if not args.retry_failed and q_id in successful_ids:
            continue

        retry_count += 1
        total_targets = len(target_ids)

        if args.retry_failed:
            print(f"\n[RETRY {retry_count}/{total_targets}] {q_id}")
        else:
            print(f"\n[{retry_count}/{total_examples}] {q_id}")

        text = row['customer_text']
        conv_id = str(row['conversation_id'])

        response, provider_meta = run_with_fallback(q_id, text, conv_id, providers, retriever)

        if response is not None:
            record = build_prediction(q_id, row, response, provider_meta)
            provider_tally[provider_meta['provider_used']] += 1
            print(f"  → {record['predicted_intent']} | {record['predicted_action']} | {provider_meta['system_status']}")
        else:
            record = build_prediction(q_id, row, None, provider_meta, is_failed=True)
            provider_tally['failed'] += 1
            print(f"  → FAILED after all providers exhausted.")

        new_records.append(record)
        with open(PREDICTIONS_FILE, 'a') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()

        time.sleep(0.5)

    # -----------------------------------------------------------------------
    # Post-run validation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Post-run validation...")
    all_records = []
    with open(PREDICTIONS_FILE, 'r') as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    all_ids = [r['id'] for r in all_records]
    unique_ids_set = set(all_ids)
    duplicates = [k for k, v in Counter(all_ids).items() if v > 1]
    missing_final = golden_ids - unique_ids_set
    unexpected = unique_ids_set - golden_ids

    print(f"Total records: {len(all_records)}")
    print(f"Unique IDs: {len(unique_ids_set)}")
    print(f"Duplicates: {duplicates}")
    print(f"Missing IDs: {missing_final}")
    print(f"Unexpected IDs: {unexpected}")

    successful_all = [r for r in all_records if r.get('predicted_intent') != 'FAILED']
    failed_all = [r for r in all_records if r.get('predicted_intent') == 'FAILED']
    print(f"Successful: {len(successful_all)}")
    print(f"Failed: {len(failed_all)}")

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    print("\nCalculating metrics...")

    coverage = len(successful_all) / total_examples

    provider_dist = Counter(r.get('provider_used') for r in all_records)
    system_dist = Counter(r.get('system_status') for r in all_records)
    severity_dist = Counter(r.get('severity') for r in all_records)
    fallback_count = sum(1 for r in all_records if r.get('fallback_used') is True)

    groq_records = [r for r in successful_all if r.get('provider_used') == 'groq']
    combined_records = successful_all  # all genuine predictions regardless of provider

    df_all = pd.DataFrame(all_records)
    df_all.to_csv('evaluation/final_results/error_analysis.csv', index=False)

    metrics = {
        "coverage": {
            "expected": total_examples,
            "successful": len(successful_all),
            "failed": len(failed_all),
            "coverage_pct": round(coverage * 100, 2),
        },
        "provider_distribution": dict(provider_dist),
        "system_status_distribution": dict(system_dist),
        "severity_distribution": dict(severity_dist),
        "reliability": {
            "fallback_attempts": fallback_count,
            "remaining_failures": len(failed_all),
            "final_coverage_pct": round(coverage * 100, 2),
        },
    }

    groq_metrics = compute_metrics(groq_records, golden_df, "Primary Groq evaluation")
    combined_metrics = compute_metrics(combined_records, golden_df, "SupportLens final-system performance")

    if groq_metrics:
        metrics["Primary Groq evaluation"] = groq_metrics
    if combined_metrics:
        metrics["SupportLens final-system performance"] = combined_metrics

    # Save
    with open('evaluation/final_results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Grounding flags
    grounding_flags = [
        r for r in all_records if r.get('grounding_flags')
    ]
    with open('evaluation/final_results/grounding_flags.json', 'w') as f:
        json.dump(grounding_flags, f, indent=2)

    # Confusion matrices
    from sklearn.metrics import confusion_matrix
    if combined_records:
        df_c = pd.DataFrame(combined_records)
        all_intents = sorted(golden_df['expected_intent'].unique())
        intent_classes = all_intents + ['FAILED']
        cm_intent = confusion_matrix(
            df_c['expected_intent'], df_c['predicted_intent'], labels=intent_classes
        )
        pd.DataFrame(cm_intent, index=intent_classes, columns=intent_classes).to_csv(
            'evaluation/final_results/confusion_matrices/intent_cm.csv'
        )
        cm_action = confusion_matrix(
            df_c['expected_action'], df_c['predicted_action'],
            labels=["AUTO_HANDLE", "ESCALATE"]
        )
        pd.DataFrame(
            cm_action, index=["AUTO_HANDLE", "ESCALATE"], columns=["AUTO_HANDLE", "ESCALATE"]
        ).to_csv('evaluation/final_results/confusion_matrices/action_cm.csv')

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Coverage: {coverage:.2%} ({len(successful_all)}/{total_examples})")
    print(f"Provider distribution: {dict(provider_dist)}")
    if groq_metrics:
        print(f"Groq intent accuracy: {groq_metrics['intent']['accuracy']:.4f}")
        print(f"Groq macro F1: {groq_metrics['intent']['macro_f1']:.4f}")
    if combined_metrics:
        print(f"Combined intent accuracy: {combined_metrics['intent']['accuracy']:.4f}")
        print(f"Combined macro F1: {combined_metrics['intent']['macro_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
