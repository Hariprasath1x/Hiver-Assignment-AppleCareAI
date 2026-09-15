import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.embedder import Embedder
from src.retrieval.index import FAISSIndex

def build_index():
    print("Loading data...")
    res = pd.read_parquet('data/processed/apple_support_pairs.parquet')
    golden = pd.read_csv('evaluation/golden_set.csv')
    
    # Isolate golden set conversations
    golden_convs = set(golden['conversation_id'].astype(str))
    
    # Final retrieval corpus
    # Exclude by conversation_id
    corpus = res[~res['conversation_id'].astype(str).isin(golden_convs)].copy()
    
    # Exclude by exact customer text to prevent twin leakage
    golden_texts = set(golden['customer_text'])
    corpus = corpus[~corpus['customer_text'].isin(golden_texts)]
    
    # Filter out DM deflections (as specified in prompt: candidate substantive corpus ~47,634)
    corpus = corpus[~corpus['support_text_clean'].str.contains(r'\bdm\b', case=False, na=False)]
    
    corpus = corpus.reset_index(drop=True)
    
    print(f"Building index for {len(corpus)} items...")
    embedder = Embedder()
    
    # Batch embeddings to avoid memory spikes
    batch_size = 1000
    all_embeddings = []
    
    for i in range(0, len(corpus), batch_size):
        batch_texts = corpus['customer_text_clean'].iloc[i:i+batch_size].tolist()
        batch_emb = embedder.embed(batch_texts)
        all_embeddings.append(batch_emb)
        print(f"Embedded {i+len(batch_texts)} / {len(corpus)}")
        
    embeddings = np.vstack(all_embeddings)
    
    # Save index
    print("Saving FAISS index...")
    faiss_index = FAISSIndex(dimension=embeddings.shape[1])
    faiss_index.build(embeddings)
    faiss_index.save('data/processed/retrieval_index.faiss')
    
    # Save corresponding corpus for retriever lookups
    corpus.to_parquet('data/processed/retrieval_corpus.parquet')
    
    print("Done!")

if __name__ == "__main__":
    build_index()
