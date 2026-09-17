import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any

class FAISSVectorStore:
    def __init__(self, dim: int = 512, data_dir: str = "../data"):
        self.dim = dim
        self.index_file = os.path.join(data_dir, "index.faiss")
        self.meta_file = os.path.join(data_dir, "metadata.json")
        self.metadata: Dict[int, Any] = {}
        
        # IndexFlatIP calculates inner product. 
        # With L2-normalized vectors, Inner Product == Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dim) 
        
        os.makedirs(data_dir, exist_ok=True)
        self._load()

    def _load(self):
        """Loads the FAISS index and metadata from disk if they exist."""
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            print(f"Loaded FAISS index with {self.index.ntotal} vectors.")
            
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r") as f:
                # JSON dictionary keys are stored as strings, cast them back to integers
                self.metadata = {int(k): v for k, v in json.load(f).items()}

    def save(self):
        """Saves the current FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def add_image(self, embedding: List[float], meta: dict):
        """Adds a single image embedding and its metadata to the store."""
        vector = np.array([embedding], dtype=np.float32)
        current_id = self.index.ntotal
        
        self.index.add(vector)
        self.metadata[current_id] = meta
        self.save()

    def search(self, query_embedding: List[float], top_k: int = 20) -> List[dict]:
        """Searches the index for the top_k most similar images."""
        if self.index.ntotal == 0:
            return []
            
        vector = np.array([query_embedding], dtype=np.float32)
        
        # faiss returns similarity scores and index IDs
        scores, indices = self.index.search(vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 means FAISS didn't find enough results
                item = self.metadata[idx].copy()
                item["similarity_score"] = float(score) 
                results.append(item)
                
        return results

    def get_stats(self) -> dict:
        return {"total_indexed": self.index.ntotal}
