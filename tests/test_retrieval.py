import pytest
import pandas as pd
from src.retrieval.index import FAISSIndex
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.embedder import Embedder

class MockEmbedder:
    def embed(self, texts):
        import numpy as np
        # Return dummy embeddings of size 384
        return np.random.rand(len(texts), 384)

@pytest.fixture
def mock_corpus(tmp_path):
    df = pd.DataFrame({
        'conversation_id': ['100', '101', '102'],
        'customer_text_clean': ['My screen broke', 'WiFi not working', 'Battery drain'],
        'support_text_clean': ['Go to Apple Store', 'Reset network settings', 'Check battery health']
    })
    path = tmp_path / "mock_corpus.parquet"
    df.to_parquet(path)
    return str(path)

@pytest.fixture
def mock_index(tmp_path):
    import numpy as np
    index = FAISSIndex(dimension=384)
    # 3 random vectors
    vectors = np.random.rand(3, 384).astype(np.float32)
    index.build(vectors)
    path = tmp_path / "mock_index.faiss"
    index.save(str(path))
    return str(path)

def test_retrieval_excludes_conversation(mock_corpus, mock_index):
    retriever = SemanticRetriever(MockEmbedder(), mock_index, mock_corpus)
    
    # Exclude conv 100
    results = retriever.retrieve("screen broke", top_k=2, exclude_conversation_id="100")
    
    for ev in results:
        assert ev.source_id != "100", "Retriever failed to exclude the specified conversation_id"
