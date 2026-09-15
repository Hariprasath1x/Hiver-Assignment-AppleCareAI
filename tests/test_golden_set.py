import pandas as pd
import pytest
import os
import json

@pytest.fixture(scope="module")
def data():
    golden_path = 'evaluation/golden_set.csv'
    resolved_path = 'data/processed/apple_support_pairs.parquet'
    unresolved_path = 'data/processed/apple_support_unresolved.parquet'
    
    assert os.path.exists(golden_path), "Golden set not generated"
    
    golden = pd.read_csv(golden_path)
    res = pd.read_parquet(resolved_path)
    unres = pd.read_parquet(unresolved_path)
    
    return {'golden': golden, 'res': res, 'unres': unres}

def test_golden_size(data):
    golden = data['golden']
    assert 190 <= len(golden) <= 210, f"Expected ~200 examples, got {len(golden)}"

def test_valid_fields(data):
    golden = data['golden']
    required = ['example_id', 'conversation_id', 'customer_tweet_id', 'customer_text', 
                'available_context', 'expected_intent', 'expected_action', 
                'expected_reason', 'difficulty', 'source_type', 'annotator_notes']
    
    for req in required:
        assert req in golden.columns, f"Missing required column: {req}"
        assert golden[req].notna().all(), f"Column {req} contains NaNs"

def test_valid_intents(data):
    golden = data['golden']
    with open('data/processed/intent_taxonomy.json', 'r') as f:
        taxonomy = json.load(f)
        
    valid_intents = [t['intent_id'] for t in taxonomy]
    
    for intent in golden['expected_intent'].unique():
        assert intent in valid_intents, f"Invalid intent found in golden set: {intent}"

def test_valid_actions(data):
    golden = data['golden']
    valid_actions = ['AUTO_HANDLE', 'ESCALATE']
    for act in golden['expected_action'].unique():
        assert act in valid_actions, f"Invalid action: {act}"

def test_no_duplicates(data):
    golden = data['golden']
    assert golden['example_id'].is_unique, "Duplicate example_ids"
    assert golden['customer_tweet_id'].is_unique, "Duplicate customer_tweet_ids"
    assert golden['conversation_id'].is_unique, "Duplicate conversation_ids, preventing independent evaluation"

def test_no_leakage(data):
    # This test verifies that the golden set can be completely isolated.
    # In a real evaluation pipeline, the retrieval corpus will exclude any conversation_id present in the golden set.
    # Here we just check that no exact customer text in the golden set is duplicated outside its own conversation_id.
    
    golden = data['golden']
    res = data['res']
    
    # Check if a golden customer text exists in any OTHER conversation in the retrieval corpus
    for _, row in golden.iterrows():
        text = row['customer_text']
        conv_id = str(row['conversation_id'])
        
        matches = res[(res['customer_text_clean'] == text) & (res['conversation_id'] != conv_id)]
        assert len(matches) == 0, f"Leakage Risk: Exact text '{text}' exists in multiple conversations."
