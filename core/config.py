import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN        = os.getenv("HF_TOKEN", "")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index.bin")
WHISPER_MODEL   = os.getenv("WHISPER_MODEL_SIZE", "base")
RAG_TOP_K       = int(os.getenv("RAG_TOP_K", "5"))
MAX_HISTORY     = int(os.getenv("CONVERSATION_HISTORY_SIZE", "10"))
SAFETY_LEVEL    = os.getenv("SAFETY_FILTER_LEVEL", "strict")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# Validation au démarrage
if not GROQ_API_KEY:
    import logging
    logging.warning("[Config] GROQ_API_KEY manquante — mode fallback offline")
