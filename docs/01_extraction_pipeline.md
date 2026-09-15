# Data Extraction Pipeline

This document explains the extraction pipeline for Phase 1 of the AppleCareAI project.

## Overview
The goal of this phase is to extract all interactions related to the `AppleSupport` brand from the raw Kaggle `twcs.csv` dataset, and to reconstruct these interactions into a clean, interaction-level dataset suitable for training and evaluation.

## 1. AppleSupport Row Extraction
To find all relevant tweets, we perform the following steps on `twcs.csv`:
1. **Outbound Tweets**: We filter all rows where `author_id == 'AppleSupport'`. This yields the complete set of responses sent by the brand.
2. **Inbound Customers (Parents)**: We collect the `in_response_to_tweet_id` from all AppleSupport outbound tweets. We then extract these parent tweets from the dataset. These are the inbound customer issues.
3. **Inbound Customers (Mentions)**: We also extract all inbound tweets (`inbound == True`) containing `@AppleSupport` in their text, to capture customers reaching out who may not have received a response.
4. **Inbound Customers (Follow-ups)**: We collect any inbound tweets that replied *to* an AppleSupport tweet.

All of these are combined and deduplicated by `tweet_id` into a subset dataframe, saved as `data/interim/apple_support_raw.parquet`.

## 2. Conversation Reconstruction
A simple interaction is defined as a Customer Tweet and the corresponding AppleSupport response.
1. We group the AppleSupport outbound tweets by their `in_response_to_tweet_id`.
2. We iterate over every extracted customer tweet.
3. If the customer tweet ID matches an AppleSupport `in_response_to_tweet_id`, we create a **Resolved Interaction** pair.
4. If it does not match (the customer didn't receive a direct response), we append it to an **Unresolved Interaction** list.

## 3. Conversation IDs
The `twcs.csv` dataset does not have native thread IDs. We compute a `conversation_id` for each tweet by recursively following the `in_response_to_tweet_id` reference up the chain until the root tweet is found. All tweets in the same thread share this root `conversation_id`.

## 4. Artifacts Produced
- `data/interim/apple_support_raw.parquet`: The raw subset of all AppleSupport-related tweets.
- `data/interim/apple_interactions_resolved.parquet`: Interaction-level pairs (Customer text -> Support text) with timestamps and IDs.
- `data/interim/apple_interactions_unresolved.parquet`: Customer messages that did not receive a direct reply.
- `data/interim/extraction_stats.json`: Pipeline metrics and deduplication stats.

## 5. Validation
An automated test suite (`tests/test_extraction.py`) verifies:
- Filtering accuracy (no non-Apple support accounts mixed in).
- Valid parent references (all referenced customer tweets exist in our subset).
- Chronological validity (support replies occur on or after the customer message).
- Duplicate ID checks and relational consistency.
