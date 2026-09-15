import pandas as pd
import numpy as np
import json
import os

def build_conversation_roots(df):
    """
    Builds a mapping from tweet_id to its root conversation_id
    by following in_response_to_tweet_id up the chain.
    """
    parent_map = dict(zip(df['tweet_id'], df['in_response_to_tweet_id']))
    
    roots = {}
    for tweet_id in df['tweet_id']:
        curr = tweet_id
        visited = set()
        while pd.notna(curr) and curr in parent_map and pd.notna(parent_map[curr]):
            if curr in visited:
                break # cycle detection just in case
            visited.add(curr)
            
            # Move to parent
            parent = parent_map[curr]
            if parent not in parent_map:
                break
            curr = parent
        roots[tweet_id] = curr
    return roots

def extract_data(input_csv, output_dir):
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv, dtype={
        'tweet_id': str, 
        'author_id': str, 
        'in_response_to_tweet_id': str, 
        'response_tweet_id': str
    })
    
    total_source_rows = len(df)
    
    print("Identifying AppleSupport interactions...")
    apple_outbound = df[df['author_id'] == 'AppleSupport']
    
    # Customer tweets that mention AppleSupport
    customer_mentions = df[(df['inbound'] == True) & (df['text'].str.contains('@AppleSupport', case=False, na=False))]
    
    # Customer tweets that AppleSupport replied to
    apple_in_response_to = apple_outbound['in_response_to_tweet_id'].dropna().unique()
    customer_parents = df[df['tweet_id'].isin(apple_in_response_to)]
    
    # Customer tweets that replied to AppleSupport
    apple_tweet_ids = apple_outbound['tweet_id'].dropna().unique()
    customer_replies = df[(df['inbound'] == True) & (df['in_response_to_tweet_id'].isin(apple_tweet_ids))]
    
    all_customer_tweets = pd.concat([customer_mentions, customer_parents, customer_replies]).drop_duplicates(subset=['tweet_id'])
    
    apple_subset = pd.concat([apple_outbound, all_customer_tweets]).drop_duplicates(subset=['tweet_id'])
    
    print("Calculating conversation IDs...")
    roots = build_conversation_roots(apple_subset)
    apple_subset['conversation_id'] = apple_subset['tweet_id'].map(roots)
    all_customer_tweets['conversation_id'] = all_customer_tweets['tweet_id'].map(roots)
    
    os.makedirs(output_dir, exist_ok=True)
    raw_output_path = os.path.join(output_dir, 'apple_support_raw.parquet')
    apple_subset.to_parquet(raw_output_path, index=False)
    
    # Now build interaction pairs
    print("Building interaction pairs...")
    
    # Map from customer tweet to its apple support replies
    # Since an AppleSupport tweet can reply to a customer tweet, we join on customer_tweet.tweet_id == apple_tweet.in_response_to_tweet_id
    
    interactions = []
    unresolved = []
    
    apple_reply_dict = {}
    for _, row in apple_outbound.iterrows():
        parent_id = row['in_response_to_tweet_id']
        if pd.notna(parent_id):
            if parent_id not in apple_reply_dict:
                apple_reply_dict[parent_id] = []
            apple_reply_dict[parent_id].append(row)

    for _, cust_row in all_customer_tweets.iterrows():
        cust_id = cust_row['tweet_id']
        if cust_id in apple_reply_dict:
            for support_row in apple_reply_dict[cust_id]:
                interactions.append({
                    'conversation_id': cust_row['conversation_id'],
                    'customer_tweet_id': cust_row['tweet_id'],
                    'customer_text': cust_row['text'],
                    'customer_created_at': cust_row['created_at'],
                    'support_tweet_id': support_row['tweet_id'],
                    'support_text': support_row['text'],
                    'support_created_at': support_row['created_at']
                })
        else:
            unresolved.append({
                'conversation_id': cust_row['conversation_id'],
                'customer_tweet_id': cust_row['tweet_id'],
                'customer_text': cust_row['text'],
                'customer_created_at': cust_row['created_at']
            })
            
    interactions_df = pd.DataFrame(interactions)
    unresolved_df = pd.DataFrame(unresolved)
    
    interactions_df.to_parquet(os.path.join(output_dir, 'apple_interactions_resolved.parquet'), index=False)
    unresolved_df.to_parquet(os.path.join(output_dir, 'apple_interactions_unresolved.parquet'), index=False)
    
    # Deduplication metrics
    dup_cust = interactions_df.duplicated(subset=['customer_text']).sum()
    
    unique_resolved_customers = interactions_df['customer_tweet_id'].nunique()
    total_inbound = len(all_customer_tweets)
    total_unresolved = len(unresolved_df)
    
    # Mathematical reconciliation
    assert unique_resolved_customers + total_unresolved == total_inbound, "Reconciliation failed: resolved + unresolved != total inbound"
    
    extra_replies = len(interactions_df) - unique_resolved_customers
    
    stats = {
        'total_source_rows': total_source_rows,
        'apple_subset_rows': len(apple_subset),
        'inbound_customer_rows': total_inbound,
        'outbound_apple_rows': len(apple_outbound),
        'unique_resolved_customer_tweets': unique_resolved_customers,
        'total_resolved_interaction_pairs': len(interactions_df),
        'multiple_replies_to_same_tweet': extra_replies,
        'unresolved_interactions': total_unresolved,
        'duplicate_customer_texts_in_resolved': int(dup_cust),
        'note': 'Resolved means the customer message has an explicit AppleSupport response linked, NOT that the underlying problem was actually solved.'
    }
    
    with open(os.path.join(output_dir, 'extraction_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
        
    print("Extraction complete. Stats saved.")
    return stats

if __name__ == '__main__':
    extract_data('datasets/twcs.csv', 'data/interim')
