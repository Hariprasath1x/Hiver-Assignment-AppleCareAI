import pandas as pd
from typing import List
from src.retrieval.embedder import Embedder
from src.retrieval.index import FAISSIndex
from src.agent.schemas import Evidence

class SemanticRetriever:
    def __init__(self, embedder: Embedder, index_path: str, corpus_path: str):
        self.embedder = embedder
        self.faiss_index = FAISSIndex()
        self.faiss_index.load(index_path)
        
        # Load corpus for lookup
        self.corpus_df = pd.read_parquet(corpus_path)
        
    def retrieve(self, query: str, top_k: int = 5, exclude_conversation_id: str = None) -> List[Evidence]:
        q_vec = self.embedder.embed([query])
        
        # Request a few extra in case we need to filter out the excluded conv_id
        similarities, indices = self.faiss_index.search(q_vec, top_k + 2)
        
        results = []
        for sim, idx in zip(similarities, indices):
            row = self.corpus_df.iloc[idx]
            
            if exclude_conversation_id and str(row['conversation_id']) == str(exclude_conversation_id):
                continue
                
            results.append(Evidence(
                source_id=str(row['conversation_id']),
                customer_message=row['customer_text_clean'],
                historical_response=row['support_text_clean'],
                similarity=float(sim)
            ))
            
            if len(results) == top_k:
                break
                
        return results
