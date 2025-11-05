import os
from typing import List, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.embedding import EmbeddingPipeline


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = "chroma_store",
        embedding_model: str = "BAAI/bge-m3",
        chunk_size: int = 600,
        chunk_overlap: int = 80,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.db: Chroma | None = None
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Using Chroma vector store with embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building Chroma store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        # Let Chroma + LangChain handle embeddings internally via HuggingFaceEmbeddings
        self.db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        # Ensure persistence to disk
        self.db.persist()
        print(f"[INFO] Chroma store built and persisted to {self.persist_dir}")

    def load(self):
        # Lazy init of persistent Chroma store
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )
        print(f"[INFO] Loaded Chroma store from {self.persist_dir}")

    def query(self, query_text: str, top_k: int = 5):
        if self.db is None:
            # Auto-load if not already loaded
            self.load()
        print(f"[INFO] Querying Chroma for: '{query_text}'")
        results = self.db.similarity_search_with_score(query_text, k=top_k)
        # Normalize to previous FAISS-like schema
        norm = []
        for i, (doc, score) in enumerate(results):
            meta = dict(doc.metadata or {})
            meta["text"] = doc.page_content
            norm.append({"index": i, "distance": score, "metadata": meta})
        return norm


# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents

    docs = load_all_documents("data")
    store = ChromaVectorStore("chroma_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
