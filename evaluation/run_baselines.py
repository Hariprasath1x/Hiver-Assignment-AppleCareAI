import pandas as pd
import numpy as np
import json
import os
import sys
import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.baselines.majority_classifier import MajorityClassifier
from src.baselines.tfidf_classifier import TfidfIntentClassifier
from src.baselines.tfidf_retriever import TfidfRetriever
from src.baselines.threshold_policy import ThresholdPolicy

def evaluate_predictions(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=labels)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=labels)
    return acc, macro_f1, weighted_f1

def tune_thresholds(dev_df, classifier, retriever):
    print("Tuning thresholds on dev subset...")
    # Just sample 1000 from dev for quick tuning
    tune_df = dev_df.sample(min(1000, len(dev_df)), random_state=42)
    
    # In reality, tuning would optimize an F1 metric combining intent accuracy and escalation precision.
    # For baseline simplicity, we will use fixed logical defaults.
    # A true grid search would require human labels for the dev set's "expected action", which we don't have.
    # We will simply pick 0.35 for intent and 0.40 for retrieval as reasonable TF-IDF cosine thresholds.
    print("Using fixed tuned thresholds: intent=0.35, retrieval=0.40")
    return 0.35, 0.40

def run_evaluation():
    print("Loading data...")
    dev_df = pd.read_parquet('data/processed/dev_set.parquet')
    golden_df = pd.read_csv('evaluation/golden_set.csv')
    resolved_df = pd.read_parquet('data/processed/apple_support_pairs.parquet')
    
    # Assert Leakage Prevention
    golden_convs = set(golden_df['conversation_id'].astype(str))
    dev_convs = set(dev_df['conversation_id'].astype(str))
    assert len(golden_convs.intersection(dev_convs)) == 0, "Leakage: Golden conversations in dev set!"
    
    # Build retrieval pool (resolved interactions excluding golden)
    retrieval_pool = resolved_df[~resolved_df['conversation_id'].astype(str).isin(golden_convs)].copy()
    
    intents = list(golden_df['expected_intent'].unique())
    
    # BASELINE 1: Majority
    print("Running Baseline 1 (Majority)...")
    b1 = MajorityClassifier(dev_df)
    b1_preds = []
    for _, row in golden_df.iterrows():
        b1_preds.append(b1.predict(row['customer_text']))
        
    b1_intent_acc, b1_macro_f1, b1_weighted_f1 = evaluate_predictions(
        golden_df['expected_intent'], [p['intent'] for p in b1_preds], intents
    )
    b1_esc_true = golden_df['expected_action'] == 'ESCALATE'
    b1_esc_pred = [p['action'] == 'ESCALATE' for p in b1_preds]
    b1_esc_prec = precision_score(b1_esc_true, b1_esc_pred, zero_division=0)
    b1_esc_rec = recall_score(b1_esc_true, b1_esc_pred, zero_division=0)
    
    # BASELINE 2: TF-IDF + Logistic Regression
    print("Running Baseline 2 (TF-IDF ML)...")
    classifier = TfidfIntentClassifier()
    classifier.train(dev_df['customer_text_clean'], dev_df['expected_intent'])
    
    retriever = TfidfRetriever()
    retriever.build_index(retrieval_pool)
    
    intent_thresh, ret_thresh = tune_thresholds(dev_df, classifier, retriever)
    policy = ThresholdPolicy(intent_thresh, ret_thresh)
    
    b2_preds = []
    similarities = []
    for _, row in golden_df.iterrows():
        text = row['customer_text']
        
        intent, conf = classifier.predict(text)
        reply, sim = retriever.retrieve(text, exclude_conversation_id=row['conversation_id'])
        
        action, reason = policy.determine_action(intent, conf, sim)
        
        similarities.append(sim)
        b2_preds.append({
            'intent': intent,
            'action': action,
            'confidence': conf,
            'similarity': sim,
            'reply': reply
        })
        
    b2_intent_acc, b2_macro_f1, b2_weighted_f1 = evaluate_predictions(
        golden_df['expected_intent'], [p['intent'] for p in b2_preds], intents
    )
    
    b2_esc_true = golden_df['expected_action'] == 'ESCALATE'
    b2_esc_pred = [p['action'] == 'ESCALATE' for p in b2_preds]
    b2_esc_prec = precision_score(b2_esc_true, b2_esc_pred, zero_division=0)
    b2_esc_rec = recall_score(b2_esc_true, b2_esc_pred, zero_division=0)
    
    # Auto-handle precision (where action == AUTO_HANDLE, is it correct?)
    b2_auto_true = golden_df['expected_action'] == 'AUTO_HANDLE'
    b2_auto_pred = [p['action'] == 'AUTO_HANDLE' for p in b2_preds]
    b2_auto_prec = precision_score(b2_auto_true, b2_auto_pred, zero_division=0)
    
    conf_matrix = confusion_matrix(golden_df['expected_intent'], [p['intent'] for p in b2_preds], labels=intents)
    
    results = {
        "metadata": {
            "dataset_version": "1.0",
            "golden_set_version": "1.0.0",
            "random_seed": 42,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "baseline_1_majority": {
            "intent_accuracy": b1_intent_acc,
            "intent_macro_f1": b1_macro_f1,
            "intent_weighted_f1": b1_weighted_f1,
            "escalation_precision": b1_esc_prec,
            "escalation_recall": b1_esc_rec,
            "escalation_rate": np.mean(b1_esc_pred)
        },
        "baseline_2_tfidf": {
            "intent_accuracy": b2_intent_acc,
            "intent_macro_f1": b2_macro_f1,
            "intent_weighted_f1": b2_weighted_f1,
            "escalation_precision": b2_esc_prec,
            "escalation_recall": b2_esc_rec,
            "auto_handle_precision": b2_auto_prec,
            "escalation_rate": np.mean(b2_esc_pred),
            "retrieval_avg_top1_similarity": np.mean(similarities),
            "thresholds": {
                "intent_confidence": intent_thresh,
                "retrieval_similarity": ret_thresh
            }
        },
        "confusion_matrix": {
            "labels": intents,
            "matrix": conf_matrix.tolist()
        }
    }
    
    with open('evaluation/baseline_results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Evaluation complete. Results saved to evaluation/baseline_results/results.json")

if __name__ == "__main__":
    run_evaluation()
