import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        llm_model: str = "gemini-1.5-flash"
    ):
        # --- Load FAISS VectorStore ---
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        # --- Build or Load FAISS index ---
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # --- Configure Gemini ---
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set in environment variables.")

        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=0.3,
            top_p=0.9,
            max_output_tokens=1024,
        )
        print(f"[INFO] Gemini LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant chunks, build a context prompt, and summarize using Gemini."""
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
        context = "\n\n".join(texts)

        if not context.strip():
            return "No relevant documents found."

        prompt = f"""
You are a professional assistant specialized in nursing and healthcare education.
Answer the following question based strictly on the provided context.

Question:
{query}

Context:
{context}

Instructions:
- Provide a concise, factual summary.
- If answer not found, say 'I don’t know based on the provided context.'
- Cite relevant sections or documents if possible.

Answer:
"""
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"[ERROR] Gemini API call failed: {e}"


# --- Example Usage ---
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What are the eligibility criteria for B.Sc. Nursing?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("\nSummary:\n", summary)
