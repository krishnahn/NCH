from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# If your project keeps this path, leave it as-is.
# Otherwise, adjust to your loader location.
from src.data_loader import load_all_documents


class EmbeddingPipeline:

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        normalize: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.normalize = normalize

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        print(f"[INFO] Loaded embedding model: {model_name} on {device}")

        # BGE recommended instructions
        self._doc_prompt = "Represent this document for retrieval: "
        self._qry_prompt = "Represent this sentence for searching relevant passages: "

    def _maybe_normalize(self, vecs: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return vecs
        # L2-normalize row-wise for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into overlapping chunks. Uses heading-aware separators first,
        then falls back to paragraph/sentence boundaries.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n### ", "\n## ", "\n# ",
                "\n\n", "\n",
                ". ", "! ", "? ",
                " ", ""
            ],
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Embed a list of chunk Documents using the BGE document instruction.
        Returns a (N, D) numpy array (optionally L2-normalized).
        """
        texts = [self._doc_prompt + chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        # show_progress_bar uses tqdm under the hood
        emb = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize ourselves for clarity
        )
        emb = self._maybe_normalize(emb)
        print(f"[INFO] Embeddings shape: {emb.shape}")
        return emb

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string using the BGE query instruction.
        Returns a (D,) numpy array (optionally L2-normalized).
        """
        q = self._qry_prompt + query
        vec = self.model.encode(
            q,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize ourselves
        )
        vec = self._maybe_normalize(vec.reshape(1, -1))[0]
        return vec


# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline(
        model_name="BAAI/bge-m3",
        chunk_size=600,
        chunk_overlap=80,
        normalize=True,
    )
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)

    # Example query embedding
    qvec = emb_pipe.embed_query("What is the B.Sc. Nursing eligibility in Tamil Nadu?")
    print("[INFO] Query embedding dim:", qvec.shape[0])
