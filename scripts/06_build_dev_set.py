import pandas as pd
import numpy as np
import os
import re

# Same heuristics from Phase 5 for weak supervision
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
    if is_url_only(text) or is_context_dependent(text):
        return 'other_unclear_context_dependent'
    if 'order' in text or 'shipping' in text or 'delivery' in text or 'store appointment' in text:
        return 'order_delivery_inquiry'
    if 'hack' in text or 'unauthorized' in text or 'activation lock' in text or 'password' in text or 'apple id' in text:
        return 'account_security_activation'
    if 'screen' in text or 'crack' in text or 'water damage' in text or 'touch id' in text or 'button' in text:
        return 'hardware_damage_repair'
    if 'wifi' in text or 'bluetooth' in text or 'cellular' in text or 'no service' in text:
        return 'network_connectivity'
    if 'app store' in text or 'apple music' in text or 'podcast' in text or 'icloud storage' in text:
        return 'app_store_media_services'
    if 'battery' in text or 'charge' in text or 'overheating' in text or 'hot' in text:
        return 'battery_power_issue'
    if 'keyboard' in text or 'autocorrect' in text or 'type' in text or 'letter' in text:
        return 'keyboard_text_input_issue'
    if 'how do i' in text or 'is it possible' in text or 'can you tell me' in text:
        return 'feature_inquiry_how_to'
    if 'update' in text or 'ios' in text or 'freeze' in text or 'crash' in text or 'lag' in text:
        return 'software_system_issue'
    
    return 'other_unclear_context_dependent'

def build_dev_set():
    print("Building 20k Weakly-Labeled Dev Set...")
    np.random.seed(42)
    
    res = pd.read_parquet('data/processed/apple_support_pairs.parquet')
    unres = pd.read_parquet('data/processed/apple_support_unresolved.parquet')
    
    golden = pd.read_csv('evaluation/golden_set.csv')
    golden_convs = set(golden['conversation_id'].astype(str))
    
    # Isolate from golden set
    res = res[~res['conversation_id'].astype(str).isin(golden_convs)]
    unres = unres[~unres['conversation_id'].astype(str).isin(golden_convs)]
    
    # We will sample 20,000 resolved pairs for training retrieval & intent
    # Unresolved pairs aren't useful for training retrieval (no response), so we only use resolved.
    dev_pool = res.sample(20000, random_state=42).copy()
    
    dev_pool['expected_intent'] = dev_pool['customer_text_clean'].apply(categorize_heuristic)
    
    dev_pool.to_parquet('data/processed/dev_set.parquet')
    print("Saved to data/processed/dev_set.parquet")

if __name__ == "__main__":
    build_dev_set()
