import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import pickle
import os

class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity
        self.documents = []
        
    def add_documents(self, documents: List[str]):
        """Add documents to the vector store."""
        self.documents.extend(documents)
        embeddings = self.model.encode(documents)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents."""
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, filepath: str):
        """Save the vector store to disk."""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save documents
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
    
    def load(self, filepath: str):
        """Load the vector store from disk."""
        if os.path.exists(f"{filepath}.faiss") and os.path.exists(f"{filepath}.pkl"):
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load documents
            with open(f"{filepath}.pkl", 'rb') as f:
                self.documents = pickle.load(f)
            
            return True
        return False
