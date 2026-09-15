from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class TfidfRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.is_trained = False
        self.corpus_X = None
        self.corpus_df = None
        
    def build_index(self, corpus_df):
        self.corpus_df = corpus_df.reset_index(drop=True)
        self.corpus_X = self.vectorizer.fit_transform(self.corpus_df['customer_text_clean'].astype(str))
        self.is_trained = True
        
    def retrieve(self, text, exclude_conversation_id=None, top_k=1):
        if not self.is_trained:
            raise ValueError("Index not built.")
            
        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.corpus_X)[0]
        
        # Sort indices by similarity descending
        sorted_indices = np.argsort(sims)[::-1]
        
        results = []
        for idx in sorted_indices:
            row = self.corpus_df.iloc[idx]
            
            # Leakage prevention: do not retrieve from the same conversation
            if exclude_conversation_id and str(row['conversation_id']) == str(exclude_conversation_id):
                continue
                
            results.append({
                'similarity': float(sims[idx]),
                'historical_customer_text': row['customer_text_clean'],
                'historical_support_response': row['support_text_clean']
            })
            
            if len(results) == top_k:
                break
                
        if not results:
            return None, 0.0
            
        return results[0]['historical_support_response'], results[0]['similarity']
