import os
import numpy as np
import pickle
from typing import List, Any
from uuid import uuid4
import chromadb
from chromadb.config import Settings
import torch
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class ChromaVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                 chunk_size: int = 800, 
                 chunk_overlap: int = 150):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        # Initialize Chroma client (try local duckdb+parquet persistence)
        try:
            self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_dir))
        except Exception:
            self.client = chromadb.Client()

        # Use a default collection name for backward compatibility
        self.collection = self.client.get_or_create_collection(name="default")
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

    def add_embeddings(self, embeddings, metadatas: List[Any] = None, documents: List[str] = None):
        # Accept numpy arrays or lists
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.size == 0:
            print("[WARN] No embeddings to add (empty array). Skipping.")
            return

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.ndim != 2:
            raise ValueError(f"Unexpected embeddings array shape: {embeddings.shape}")

        # Convert embeddings to nested lists for Chroma
        embeddings_list = embeddings.tolist()

        n = len(embeddings_list)
        ids = [uuid4().hex for _ in range(n)]

        # Prepare documents list if not provided
        if documents is None and metadatas is not None:
            documents = [m.get("text", "") for m in metadatas]

        # Chroma expects lists for ids/documents/metadatas/embeddings
        add_kwargs = {
            "ids": ids,
            "embeddings": embeddings_list,
        }
        if documents is not None:
            add_kwargs["documents"] = documents
        if metadatas is not None:
            add_kwargs["metadatas"] = metadatas

        self.collection.add(**add_kwargs)
        # Persist to disk
        try:
            self.client.persist()
        except Exception:
            # Some chroma clients persist automatically; ignore failures
            pass

        # Maintain a local metadata list for compatibility with previous API
        if metadatas:
            if isinstance(metadatas, dict):
                self.metadata.append(metadatas)
            else:
                self.metadata.extend(metadatas)

        print(f"[INFO] Added {n} vectors to Chroma collection 'default'.")

    def save(self):
        # Persist chroma client/collection
        try:
            self.client.persist()
            # also save metadata for compatibility
            meta_path = os.path.join(self.persist_dir, "metadata.pkl")
            with open(meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
            print(f"[INFO] Persisted Chroma collection and metadata to {self.persist_dir}")
        except Exception as e:
            print(f"[WARN] Chroma persist failed: {e}")

    def load(self):
        # Re-initialize client and collection from the persist directory
        try:
            self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_dir))
            self.collection = self.client.get_or_create_collection(name="default")
            meta_path = os.path.join(self.persist_dir, "metadata.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
            print(f"[INFO] Loaded Chroma collection from {self.persist_dir}")
        except Exception as e:
            print(f"[WARN] Failed to load Chroma collection from {self.persist_dir}: {e}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        # Accept numpy array or list
        emb = np.asarray(query_embedding, dtype="float32")
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        emb_list = emb.tolist()
        try:
            # Avoid requesting 'ids' in include (not accepted by this chroma version)
            res = self.collection.query(query_embeddings=emb_list, n_results=top_k, include=["metadatas", "distances", "documents"])
        except Exception as e:
            print(f"[ERROR] Chroma query failed: {e}")
            return []
        results = []
        ids = res.get("ids", [[]])[0] if "ids" in res else []
        dists = res.get("distances", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        docs = res.get("documents", [[]])[0]

        for _id, dist, meta, doc in zip(ids, dists, metadatas, docs):
            results.append({"id": _id, "distance": float(dist), "metadata": meta, "document": doc})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying Chroma collection for: '{query_text[:100]}...'")
        query_emb = self.model.encode([query_text])
        return self.search(query_emb, top_k=top_k)
