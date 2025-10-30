import os
import torch
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from transformers import AutoTokenizer, AutoModelForCausalLM
from langdetect import detect, LangDetectException
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
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.last_context = ""
        self.local_model_path = local_model_path
        self.supported_languages = {
            'en': 'English',
            'ta': 'Tamil',
            'hi': 'Hindi',
            'ml': 'Malayalam'
        }
        
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        print(f"🔹 Loading Sarvam model from local path...")
        
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local model path not found: {local_model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            # SPEED OPTIMIZATION 1: Load model in 8-bit for faster inference
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # SPEED OPTIMIZATION 2: Set model to eval mode and compile if possible
            self.model.eval()
            
            # SPEED OPTIMIZATION 3: Enable inference optimizations
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ Running on CPU - inference will be slower")
            
            print("✅ Sarvam model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading local model: {e}")
            raise
        
        print(f"✅ Supporting languages: Tamil, English, Hindi, Malayalam")

    def detect_language(self, text: str) -> str:
        """Fast language detection with caching"""
        try:
            lang = detect(text)
            if lang in self.supported_languages:
                return lang
            return 'en'
        except:
            return 'en'

    def is_nursing_related(self, query: str) -> bool:
        """IMPROVED: More lenient keyword matching"""
        nursing_keywords = {
            'en': ['nursing', 'nurse', 'gnm', 'bsc', 'msc', 'anm', 'college', 
                   'admission', 'course', 'eligibility', 'syllabus', 'fee', 
                   'hospital', 'clinical', 'midwife', 'student', 'degree', 
                   'diploma', 'program', 'training', 'education', 'medical',
                   'health', 'care', 'colleges', 'university', 'institute',
                   'b.sc', 'm.sc', 'pg', 'ug'],  # Added more variations
            'ta': ['நர்சிங்', 'கல்லூரி', 'பாடநெறி', 'மருத்துவ', 'செவிலியர்', 
                   'படிப்பு', 'சேர்க்கை', 'தகுதி', 'பயிற்சி'],
            'hi': ['नर्सिंग', 'कॉलेज', 'पाठ्यक्रम', 'प्रवेश', 'योग्यता', 
                   'अस्पताल', 'शिक्षा', 'प्रशिक्षण', 'नर्स'],
            'ml': ['നഴ്‌സിംഗ്', 'കോളേജ്', 'കോഴ്‌സ്', 'പ്രവേശനം', 'യോഗ്യത', 
                   'ആശുപത്രി', 'വിദ്യാഭ്യാസം', 'പരിശീലനം']
        }
        
        query_lower = query.lower()
        
        # Check all language keywords
        for lang_keywords in nursing_keywords.values():
            if any(keyword in query_lower for keyword in lang_keywords):
                return True
        
        # FALLBACK: If query contains question words + general education terms, allow it
        question_indicators = ['how many', 'what', 'which', 'where', 'when', 'list', 'tell']
        education_terms = ['college', 'course', 'admission', 'fee', 'eligibility']
        
        has_question = any(indicator in query_lower for indicator in question_indicators)
        has_education = any(term in query_lower for term in education_terms)
        
        if has_question and has_education:
            return True
        
        return False

    def check_context_relevance(self, results: List[Dict], threshold: float = 0.15) -> bool:
        """IMPROVED: Lower threshold to prevent false rejections"""
        if not results or len(results) == 0:
            return False
        top_score = results[0].get("score", 0)
        print(f"[DEBUG] Top retrieval score: {top_score:.4f}")
        return top_score > threshold  # Lowered from 0.25 to 0.15

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
                'out_of_scope': "I can only answer questions about nursing education in Tamil Nadu. Please ask about nursing colleges, courses, admissions, or eligibility.",
                'no_info': "I couldn't find specific information about this in my knowledge base. Please try rephrasing your question or ask about specific nursing colleges in Tamil Nadu.",
                'insufficient': "I don't have enough information to answer this question properly."
            },
            'ta': {
                'out_of_scope': "என்னால் தமிழ்நாட்டில் உள்ள நர்சிங் கல்வி பற்றிய கேள்விகளுக்கு மட்டுமே பதிலளிக்க முடியும்.",
                'no_info': "இது பற்றிய குறிப்பிட்ட தகவல் கிடைக்கவில்லை. தயவுசெய்து உங்கள் கேள்வியை வேறுவிதமாகக் கேளுங்கள்.",
                'insufficient': "இந்த கேள்விக்கு பதிலளிக்க போதுமான தகவல் இல்லை."
            },
            'hi': {
                'out_of_scope': "मैं केवल तमिलनाडु में नर्सिंग शिक्षा के बारे में प्रश्नों का उत्तर दे सकता हूं।",
                'no_info': "मुझे इसके बारे में विशिष्ट जानकारी नहीं मिली। कृपया अपना प्रश्न फिर से लिखें।",
                'insufficient': "इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।"
            },
            'ml': {
                'out_of_scope': "തമിഴ്‌നാട്ടിലെ നഴ്‌സിംഗ് വിദ്യാഭ്യാസത്തെക്കുറിച്ചുള്ള ചോദ്യങ്ങൾക്ക് മാത്രമേ എനിക്ക് ഉത്തരം നൽകാൻ കഴിയൂ.",
                'no_info': "ഇതിനെക്കുറിച്ച് വിവരങ്ങൾ കണ്ടെത്താനായില്ല. ചോദ്യം മാറ്റിയെഴുതുക.",
                'insufficient': "ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ മതിയായ വിവരങ്ങളില്ല."
            }
        }
        return messages.get(lang_code, messages['en']).get(message_type, messages['en'][message_type])

    def create_rag_prompt(self, query: str, context: str, lang_code: str) -> str:
        """SPEED OPTIMIZATION 4: Shorter, more efficient prompt"""
        language_instruction = self.get_language_instruction(lang_code)
        
        prompt = f"""You are a nursing education assistant for Tamil Nadu.

Rules:
1. Answer using ONLY the Context below
2. If no relevant info in Context, say "I don't have this information"
3. Be concise and direct
4. {language_instruction}

Context:
{context}

Question: {query}

Answer:"""
        return prompt

    def generate_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """SPEED OPTIMIZATION 5: Reduced tokens and faster sampling"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1536  # Reduced from 2048
        ).to(self.model.device)
        
        with torch.no_grad():
            # SPEED OPTIMIZATION 6: Faster generation settings
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,  # Reduced from 300
                do_sample=True,
                top_p=0.9,  # Slightly increased for speed
                temperature=0.5,  # Balanced
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,  # Reduced from 3
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Answer:" in full_response:
            response = full_response.split("Answer:")[-1].strip()
        else:
            response = full_response[len(prompt):].strip()
        
        # Quick uncertainty check
        if any(phrase in response[:80].lower() for phrase in ["i don't", "i do not", "i cannot"]):
            return response.split('.')[0] + '.' if '.' in response else response
        
        return response

    def apply_guardrails(self, response: str, lang_code: str) -> str:
        """Lightweight guardrails"""
        if len(response.strip()) < 15:
            return self.get_fallback_message(lang_code, 'insufficient')
        
        if len(response) > 1000:
            sentences = response.split('.')
            response = '. '.join(sentences[:4]) + '.'
        
        if any(marker in response for marker in ['[CONTEXT', 'QUESTION', '###']):
            return self.get_fallback_message(lang_code, 'insufficient')
        
        return response

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Main search method with optimizations"""
        query_lang = self.detect_language(query)
        print(f"[INFO] Language: {self.supported_languages.get(query_lang, 'English')}")
        
        # Relaxed domain check
        if not self.is_nursing_related(query):
            return self.get_fallback_message(query_lang, 'out_of_scope')
        
        # Retrieve documents
        results = self.vectorstore.query(query, top_k=top_k)
        
        # More lenient relevance check
        if not self.check_context_relevance(results, threshold=0.15):
            # Before giving up, check if ANY result has decent content
            has_content = any(
                r.get("metadata") and r["metadata"].get("text") and len(r["metadata"]["text"]) > 100
                for r in results
            )
            if not has_content:
                return self.get_fallback_message(query_lang, 'no_info')
        
        # Build context from top results
        contexts = []
        for i, r in enumerate(results[:4], 1):  # Increased from 3 to 4
            if r.get("metadata") and r["metadata"].get("text"):
                text = r["metadata"]["text"].strip()
                if text and len(text) > 30:  # Lowered from 50
                    contexts.append(text)  # Removed source labeling for shorter context
        
        if not contexts:
            return self.get_fallback_message(query_lang, 'insufficient')
        
        # SPEED OPTIMIZATION 7: Shorter context window
        context = "\n\n".join(contexts)
        if len(context) > 1800:  # Reduced from 2500
            context = context[:1800] + "..."
        
        self.last_context = context
        
        # Generate response
        prompt = self.create_rag_prompt(query, context, query_lang)
        response = self.generate_response(prompt, max_new_tokens=200)
        response = self.apply_guardrails(response, query_lang)
        
        return response
