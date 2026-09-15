import pandas as pd
import pytest
import os

@pytest.fixture(scope="module")
def data():
    raw_path = 'data/interim/apple_support_raw.parquet'
    resolved_path = 'data/interim/apple_interactions_resolved.parquet'
    
    assert os.path.exists(raw_path), "Raw data not extracted"
    assert os.path.exists(resolved_path), "Resolved interactions not extracted"
    
    df_raw = pd.read_parquet(raw_path)
    df_res = pd.read_parquet(resolved_path)
    
    return {'raw': df_raw, 'resolved': df_res}

def test_applesupport_only_filtering(data):
    df = data['raw']
    
    # Assert that all outbound tweets are strictly from AppleSupport
    outbound = df[df['inbound'] == False]
    assert (outbound['author_id'] == 'AppleSupport').all(), "Found non-AppleSupport outbound tweets"
    
    # Assert that all inbound tweets are not from AppleSupport
    inbound = df[df['inbound'] == True]
    assert (inbound['author_id'] != 'AppleSupport').all(), "AppleSupport is marked as inbound"

def test_customer_support_role_assignment(data):
    df_res = data['resolved']
    
    # customer_text should exist and not be empty
    assert df_res['customer_text'].notna().all()
    assert df_res['support_text'].notna().all()
    
    # In raw data, ensure that support_tweet_id actually belongs to AppleSupport
    support_tweet_ids = df_res['support_tweet_id'].unique()
    raw = data['raw']
    support_raw = raw[raw['tweet_id'].isin(support_tweet_ids)]
    assert (support_raw['author_id'] == 'AppleSupport').all()

def test_duplicate_ids(data):
    df = data['raw']
    # tweet_id must be unique in the raw extraction
    assert df['tweet_id'].is_unique, "Duplicate tweet_ids found in raw extraction"

def test_chronological_ordering(data):
    df_res = data['resolved'].copy()
    
    # Parse dates — utc=True avoids an intermittent native segfault in
    # pandas 3.x / Python 3.14 where _ensure_nanosecond_dtype crashes.
    # All timestamps are UTC (+0000) so this is semantically identical.
    df_res['cust_dt'] = pd.to_datetime(df_res['customer_created_at'], format='%a %b %d %H:%M:%S +0000 %Y', utc=True)
    df_res['supp_dt'] = pd.to_datetime(df_res['support_created_at'], format='%a %b %d %H:%M:%S +0000 %Y', utc=True)
    
    # Support reply must happen after or at the exact same second as customer tweet
    # We allow equality just in case of weird timestamp rounding, but support_dt >= cust_dt
    assert (df_res['supp_dt'] >= df_res['cust_dt']).all(), "Found support replies that occurred before the customer message"

def test_conversation_reconstruction_consistency(data):
    df_res = data['resolved']
    df_raw = data['raw']
    
    # In resolved interactions, support_tweet_id's in_response_to_tweet_id MUST match customer_tweet_id
    raw_support = df_raw[df_raw['author_id'] == 'AppleSupport'][['tweet_id', 'in_response_to_tweet_id']]
    merged = df_res.merge(raw_support, left_on='support_tweet_id', right_on='tweet_id', how='left')
    
    assert (merged['in_response_to_tweet_id'] == merged['customer_tweet_id']).all(), "Interaction parent mismatch"

def test_broken_parent_references(data):
    df = data['raw']
    
    # If an AppleSupport tweet has a parent, does it exist in the raw set?
    apple_outbound = df[df['author_id'] == 'AppleSupport']
    parents = apple_outbound['in_response_to_tweet_id'].dropna()
    
    # They should all exist in the raw set because we specifically extracted customer_parents, 
    # EXCEPT for tweets that were deleted/missing from the original Kaggle twcs.csv dataset.
    missing_parents = parents[~parents.isin(df['tweet_id'])]
    
    # We found empirically that 71 parent tweets are completely missing from the 3M dataset.
    assert len(missing_parents) <= 100, f"Found {len(missing_parents)} broken parent references, expected around 71."
