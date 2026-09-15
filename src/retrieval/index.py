import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized embeddings (Cosine Similarity)
        
    def build(self, embeddings: np.ndarray):
        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = query_embedding.astype(np.float32)
        similarities, indices = self.index.search(query_embedding, top_k)
        return similarities[0], indices[0]
        
    def save(self, filepath: str):
        faiss.write_index(self.index, filepath)
        
    def load(self, filepath: str):
        self.index = faiss.read_index(filepath)
