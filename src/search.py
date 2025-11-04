import os
import torch
from dotenv import load_dotenv
from src.vectorstore import ChromaVectorStore
from transformers import AutoTokenizer, AutoModelForCausalLM
# Try to use langdetect if installed, otherwise provide a lightweight fallback
try:
    from langdetect import detect  # type: ignore
except Exception:
    def detect(text: str) -> str:
        """
        Lightweight fallback language detector based on Unicode script ranges.
        Returns 'ta' for Tamil, 'hi' for Devanagari (Hindi), 'ml' for Malayalam, else 'en'.
        This is intentionally simple and only intended as a robust fallback.
        """
        if not text:
            return 'en'
        for ch in text:
            code = ord(ch)
            # Tamil: U+0B80–U+0BFF
            if 0x0B80 <= code <= 0x0BFF:
                return 'ta'
            # Devanagari (used by Hindi): U+0900–U+097F
            if 0x0900 <= code <= 0x097F:
                return 'hi'
            # Malayalam: U+0D00–U+0D7F
            if 0x0D00 <= code <= 0x0D7F:
                return 'ml'
        return 'en'
from typing import List, Dict
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
load_dotenv()


class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store",
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 local_model_path: str = r"C:\Users\hamsa\OneDrive\Desktop\my proj\Nursing chatbot\rag1\rag1\sarvam-1"):

        self.vectorstore = ChromaVectorStore(persist_dir, embedding_model)
        self.last_context = ""
        self.local_model_path = local_model_path
        self.supported_languages = {
            'en': 'English',
            'ta': 'Tamil',
            'hi': 'Hindi',
            'ml': 'Malayalam'
        }

        # Load persisted collection (Chroma)
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(persist_dir) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        print("🔹 Loading Sarvam model from local path...")

        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local model path not found: {local_model_path}")

        try:
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            # Model (optimized for inference)
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            self.model.eval()

            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                print(f" Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(" Running on CPU (inference will be slower)")

            print(" Sarvam model loaded successfully!")

        except Exception as e:
            print(f" Error loading local model: {e}")
            raise

        print("🌐 Supported languages: Tamil, English, Hindi, Malayalam")

    # ---------------- LANGUAGE & RELEVANCE CHECKS ---------------- #

    def detect_language(self, text: str) -> str:
        """Detect language; fallback to English."""
        try:
            lang = detect(text)
            return lang if lang in self.supported_languages else 'en'
        except:
            return 'en'

    def check_context_relevance(self, results: List[Dict], threshold: float = 0.15) -> bool:
        """Check if the retrieved context is relevant."""
        if not results:
            return False
        top_score = results[0].get("score", 0)
        print(f"[DEBUG] Top retrieval score: {top_score:.4f}")
        return top_score > threshold

    # ---------------- LANGUAGE MESSAGES ---------------- #

    def get_language_instruction(self, lang_code: str) -> str:
        instructions = {
            'en': "Respond in English.",
            'ta': "தமிழில் பதிலளிக்கவும்.",
            'hi': "हिंदी में उत्तर दें।",
            'ml': "മലയാളത്തിൽ മറുപടി നൽകുക."
        }
        return instructions.get(lang_code, instructions['en'])

    def get_fallback_message(self, lang_code: str, message_type: str) -> str:
        messages = {
            'en': {
                'no_info': "I couldn't find specific information about this in my knowledge base.",
                'insufficient': "I don't have enough information to answer this question properly."
            },
            'ta': {
                'no_info': "இது பற்றிய குறிப்பிட்ட தகவல் கிடைக்கவில்லை. தயவுசெய்து கேள்வியை வேறுவிதமாகக் கேளுங்கள்.",
                'insufficient': "இந்த கேள்விக்கு பதிலளிக்க போதுமான தகவல் இல்லை."
            },
            'hi': {
                'no_info': "मुझे इसके बारे में विशिष्ट जानकारी नहीं मिली।",
                'insufficient': "इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।"
            },
            'ml': {
                'no_info': "ഇതിനെക്കുറിച്ച് വിവരങ്ങൾ കണ്ടെത്താനായില്ല.",
                'insufficient': "ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ മതിയായ വിവരങ്ങളില്ല."
            }
        }
        return messages.get(lang_code, messages['en']).get(message_type, messages['en'][message_type])

    # ---------------- PROMPT CREATION ---------------- #

    def create_rag_prompt(self, query, context, query_lang):
        """Create strong, factual prompt to reduce hallucination."""
        prompt = f"""
You are a helpful and factual assistant.
You must answer ONLY using the information provided in the Context below.

If the Context does not contain the answer, say exactly:
"I don’t have this information."

Do NOT make up facts, numbers, or names.
Do NOT use any outside knowledge.

Context:
<<<
{context}
>>>

Question:
{query}

Answer in the same language as the question.
"""
        return prompt.strip()

    # ---------------- RESPONSE GENERATION ---------------- #

    def generate_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate low-hallucination response."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,            
                temperature=0.3,             
                top_p=0.8,                  
                repetition_penalty=1.2,      
                no_repeat_ngram_size=3,      
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True              
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the answer part
        if "Answer:" in decoded:
            response = decoded.split("Answer:")[-1].strip()
        else:
            response = decoded[len(prompt):].strip()

        # Filter uncertain starts
        if any(x in response[:80].lower() for x in ["i don’t", "i don't", "i cannot"]):
            return response.split('.')[0] + '.'

        return response.strip()

    # ---------------- GUARDRAILS ---------------- #

    def apply_guardrails(self, response: str, lang_code: str) -> str:
        """Post-process output for clarity and safety."""
        if len(response.strip()) < 15:
            return self.get_fallback_message(lang_code, 'insufficient')

        if len(response) > 1000:
            response = '. '.join(response.split('.')[:4]) + '.'

        if any(marker in response for marker in ['[CONTEXT', 'QUESTION', '###']):
            return self.get_fallback_message(lang_code, 'insufficient')

        return response.strip()

    # ---------------- MAIN PIPELINE ---------------- #

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Main search + generation pipeline."""
        query_lang = self.detect_language(query)
        print(f"[INFO] Language detected: {self.supported_languages.get(query_lang, 'English')}")

        # Retrieve relevant documents
        results = self.vectorstore.query(query, top_k=top_k)

        # Check if retrieved context is meaningful
        if not self.check_context_relevance(results, threshold=0.15):
            has_content = any(
                r.get("metadata") and r["metadata"].get("text") and len(r["metadata"]["text"]) > 100
                for r in results
            )
            if not has_content:
                return self.get_fallback_message(query_lang, 'no_info')

        # Build concise context
        contexts = []
        for r in results[:4]:
            if r.get("metadata") and r["metadata"].get("text"):
                text = r["metadata"]["text"].strip()
                if len(text) > 30:
                    contexts.append(text)
        if not contexts:
            return self.get_fallback_message(query_lang, 'insufficient')

        context = "\n\n".join(contexts)
        if len(context) > 1800:
            context = context[:1800] + "..."

        self.last_context = context

        # Generate final response
        prompt = self.create_rag_prompt(query, context, query_lang)
        response = self.generate_response(prompt, max_new_tokens=200)
        response = self.apply_guardrails(response, query_lang)
        return response
