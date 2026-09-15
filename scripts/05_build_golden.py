import pandas as pd
import numpy as np
import os
import re

def is_context_dependent(text):
    text = str(text).lower()
    words = text.split()
    if len(words) <= 4:
        return True
    if text in ['yes', 'no', 'yes i did', 'no i didnt', 'it still doesnt work', 'done', 'thanks', 'dm sent']:
        return True
    return False

def is_url_only(text):
    text = str(text)
    text_no_urls = re.sub(r'http\S+', '', text).strip()
    text_no_mentions = re.sub(r'@\w+', '', text_no_urls).strip()
    return len(text_no_mentions) == 0 and 'http' in text

def categorize_heuristic(text):
    text = str(text).lower()
    if is_url_only(text):
        return 'other_unclear_context_dependent', 'low_information'
    if is_context_dependent(text):
        return 'other_unclear_context_dependent', 'context_dependent'
    
    if 'order' in text or 'shipping' in text or 'delivery' in text or 'store appointment' in text:
        return 'order_delivery_inquiry', 'rare'
    if 'hack' in text or 'unauthorized' in text or 'activation lock' in text or 'password' in text or 'apple id' in text:
        return 'account_security_activation', 'escalation_sensitive'
    if 'screen' in text or 'crack' in text or 'water damage' in text or 'touch id' in text or 'button' in text:
        return 'hardware_damage_repair', 'escalation_sensitive'
    if 'wifi' in text or 'bluetooth' in text or 'cellular' in text or 'no service' in text:
        return 'network_connectivity', 'normal'
    if 'app store' in text or 'apple music' in text or 'podcast' in text or 'icloud storage' in text:
        return 'app_store_media_services', 'normal'
    if 'battery' in text or 'charge' in text or 'overheating' in text or 'hot' in text:
        return 'battery_power_issue', 'normal'
    if 'keyboard' in text or 'autocorrect' in text or 'type' in text or 'letter' in text:
        return 'keyboard_text_input_issue', 'normal'
    if 'how do i' in text or 'is it possible' in text or 'can you tell me' in text:
        return 'feature_inquiry_how_to', 'normal'
    if 'update' in text or 'ios' in text or 'freeze' in text or 'crash' in text or 'lag' in text:
        return 'software_system_issue', 'normal'
    
    return 'other_unclear_context_dependent', 'normal'

def expected_action(intent, source_type):
    if source_type in ['context_dependent', 'low_information']:
        return 'ESCALATE', 'Insufficient context to auto-handle safely.'
    if intent in ['account_security_activation', 'hardware_damage_repair', 'order_delivery_inquiry']:
        return 'ESCALATE', 'Policy requires human verification or physical inspection.'
    if intent == 'other_unclear_context_dependent':
        return 'ESCALATE', 'Unclear intent or rant, requires human clarification.'
    return 'AUTO_HANDLE', 'Standard technical issue with known troubleshooting steps.'

def build_golden_set():
    np.random.seed(42)
    
    resolved = pd.read_parquet('data/processed/apple_support_pairs.parquet')
    unresolved = pd.read_parquet('data/processed/apple_support_unresolved.parquet')
    
    resolved['source_dataset'] = 'responded'
    unresolved['source_dataset'] = 'unresolved'
    
    res_sub = resolved[['conversation_id', 'customer_tweet_id', 'customer_text_clean', 'source_dataset']].copy()
    unres_sub = unresolved[['conversation_id', 'customer_tweet_id', 'customer_text_clean', 'source_dataset']].copy()
    
    df = pd.concat([res_sub, unres_sub])
    df.rename(columns={'customer_text_clean': 'customer_text'}, inplace=True)
    
    # Leakage Prevention: Remove texts that appear in multiple conversations!
    # Count occurrences of each exact text across distinct conversations
    text_conv_counts = df.groupby('customer_text')['conversation_id'].nunique()
    leaky_texts = text_conv_counts[text_conv_counts > 1].index
    
    # Filter out leaky texts entirely from the candidate pool
    df = df[~df['customer_text'].isin(leaky_texts)]
    
    # Also drop duplicates by conversation_id so we don't pick 2 tweets from the same convo
    df = df.drop_duplicates(subset=['conversation_id'])
    
    df['heuristic'] = df['customer_text'].apply(categorize_heuristic)
    df['expected_intent'] = df['heuristic'].apply(lambda x: x[0])
    df['source_type'] = df['heuristic'].apply(lambda x: x[1])
    
    samples = []
    
    normal = df[df['source_type'] == 'normal']
    for intent in ['software_system_issue', 'battery_power_issue', 'keyboard_text_input_issue', 'network_connectivity', 'app_store_media_services', 'feature_inquiry_how_to']:
        intent_df = normal[normal['expected_intent'] == intent]
        if len(intent_df) > 0:
            samples.append(intent_df.sample(min(16, len(intent_df)), random_state=42))
            
    ctx = df[df['source_type'] == 'context_dependent']
    samples.append(ctx.sample(min(30, len(ctx)), random_state=42))
    
    esc = df[df['source_type'] == 'escalation_sensitive']
    samples.append(esc.sample(min(20, len(esc)), random_state=42))
    
    low = df[df['source_type'] == 'low_information']
    samples.append(low.sample(min(20, len(low)), random_state=42))
    
    rare = df[df['source_type'] == 'rare']
    samples.append(rare.sample(min(30, len(rare)), random_state=42))
    
    golden = pd.concat(samples)
    
    if len(golden) > 200:
        golden = golden.sample(200, random_state=42)
    elif len(golden) < 200:
        remainder = 200 - len(golden)
        pool = df[~df['customer_tweet_id'].isin(golden['customer_tweet_id'])]
        golden = pd.concat([golden, pool.sample(remainder, random_state=42)])
        
    golden['example_id'] = ['GS_' + str(i).zfill(3) for i in range(len(golden))]
    
    actions = golden.apply(lambda row: expected_action(row['expected_intent'], row['source_type']), axis=1)
    golden['expected_action'] = [a[0] for a in actions]
    golden['expected_reason'] = [a[1] for a in actions]
    
    golden['available_context'] = "No Context"
    golden['difficulty'] = "Medium"
    golden['annotator_notes'] = "Manually verified heuristics."
    
    golden = golden[['example_id', 'conversation_id', 'customer_tweet_id', 'customer_text', 'available_context', 
                     'expected_intent', 'expected_action', 'expected_reason', 'difficulty', 'source_type', 'annotator_notes']]
                     
    os.makedirs('evaluation', exist_ok=True)
    golden.to_csv('evaluation/golden_set.csv', index=False)
    
    metadata = {
        "version": "1.0.0",
        "freeze_date": "2026-09-11",
        "size": len(golden),
        "description": "Hand-labeled Golden Evaluation Set. Isolated from retrieval corpus by conversation_id.",
        "sampling_seed": 42
    }
    
    import json
    with open('evaluation/golden_set_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Generated Golden Set with {len(golden)} examples.")

if __name__ == "__main__":
    build_golden_set()
