import pandas as pd
import json
import re
import os

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Normalize usernames (keep the @ to indicate it was a mention, but anonymize numbers if desired, 
    # though the dataset already replaces user handles with numbers e.g. @115854)
    # Let's just remove duplicate spaces and strip.
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_data():
    print("Loading interim datasets...")
    resolved = pd.read_parquet('data/interim/apple_interactions_resolved.parquet')
    unresolved = pd.read_parquet('data/interim/apple_interactions_unresolved.parquet')
    
    print("Reconstructing and aggregating interactions...")
    # Some customer tweets received multiple replies from AppleSupport.
    # We should aggregate them so each customer_tweet_id is exactly 1 row.
    
    # Sort by support_created_at to maintain chronological order of replies
    resolved['support_created_at_dt'] = pd.to_datetime(resolved['support_created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    resolved = resolved.sort_values(['customer_tweet_id', 'support_created_at_dt'])
    
    grouped = resolved.groupby('customer_tweet_id').agg({
        'conversation_id': 'first',
        'customer_text': 'first',
        'customer_created_at': 'first',
        'support_tweet_id': lambda x: list(x),
        'support_text': lambda x: list(x),
        'support_created_at': lambda x: list(x)
    }).reset_index()
    
    print("Cleaning text...")
    grouped['customer_text_clean'] = grouped['customer_text'].apply(clean_text)
    
    # Join multiple support responses into a single combined response for easier LLM use
    grouped['support_text_combined'] = grouped['support_text'].apply(lambda texts: " ".join(texts))
    grouped['support_text_clean'] = grouped['support_text_combined'].apply(clean_text)
    
    unresolved['customer_text_clean'] = unresolved['customer_text'].apply(clean_text)
    
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed pairs
    grouped.to_parquet('data/processed/apple_support_pairs.parquet', index=False)
    unresolved.to_parquet('data/processed/apple_support_unresolved.parquet', index=False)
    
    stats = {
        'total_resolved_unique_customer_tweets': len(grouped),
        'total_unresolved_customer_tweets': len(unresolved),
        'total_customer_inbound': len(grouped) + len(unresolved)
    }
    
    with open('data/processed/cleaning_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"Data cleaning complete. Processed pairs saved to data/processed/apple_support_pairs.parquet")
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    clean_data()
