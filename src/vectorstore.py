import os
import faiss
import numpy as np
import faiss
import pickle
from typing import List, Any
import torch
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                 chunk_size: int = 800, 
                 chunk_overlap: int = 150):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("[INFO] FAISS index moved to GPU")
        self.metadata = []
        self.embedding_model = embedding_model
        # Choose device dynamically and fall back to CPU on failure
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = SentenceTransformer(embedding_model, device=device)
            print(f"[INFO] Loaded embedding model: {embedding_model} on device {device}")
        except Exception as e:
            if device != "cpu":
                print(f"[WARN] Failed to load model on device '{device}': {e}. Falling back to 'cpu'.")
            self.model = SentenceTransformer(embedding_model, device="cpu")
            print(f"[INFO] Loaded embedding model: {embedding_model} on device cpu")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        
        metadatas = []
        for chunk in chunks:
            metadata = {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown") if hasattr(chunk, "metadata") else "unknown"
            }
            metadatas.append(metadata)
        
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built with {len(chunks)} chunks and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        # Defensive handling: ensure numpy array, handle empty and 1-D inputs
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.size == 0:
            print("[WARN] No embeddings to add (empty array). Skipping.")
            return

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.ndim != 2:
            raise ValueError(f"Unexpected embeddings array shape: {embeddings.shape}")

        # Normalize embeddings safely (avoid divide-by-zero)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings_normalized = embeddings / norms

        dim = embeddings_normalized.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings_normalized)
        if metadatas:
            if isinstance(metadatas, dict):
                self.metadata.append(metadatas)
            else:
                self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings_normalized.shape[0]} normalized vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index with {self.index.ntotal} vectors from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        D, I = self.index.search(query_normalized, min(top_k, self.index.ntotal))
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({
                    "index": int(idx), 
                    "score": float(score),
                    "metadata": meta
                })
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text[:100]}...'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)
