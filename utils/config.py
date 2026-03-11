# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Centralized Configuration
  All paths, model IDs, and settings in one place.
═══════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Root ──
PROJECT_ROOT = Path(__file__).parent.parent

# ── API Keys ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Model Paths ──
MODELS_DIR = PROJECT_ROOT / "models"
FAISS_INDEX_PATH = str(MODELS_DIR / "faiss_index.bin")
RETRIEVER_DATA_PATH = str(MODELS_DIR / "retriever_data.pkl")
CLASSIFIER_MODEL_PATH = str(MODELS_DIR / "intent_classifier.pkl")
CANCER_MODEL_PATH = str(MODELS_DIR / "breast_cancer_model_v2.keras")
MONAI_DIR = str(MODELS_DIR / "monai_breast_density")

# ── Hugging Face Model IDs ──
HF_MONAI_MODEL = "MONAI/breast_density_classification"
HF_BRAIN_TUMOR_MODEL = "Devarshi/Brain_Tumor_Classification"
HF_XRAY_MODEL = "codewithdark/vit-chest-xray"
HF_DERM_MODEL = "avanishd/efficient-net-v2-m-finetuned-skin-lesion-classification"
HF_WHISPER_MODEL = "openai/whisper-small"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
USE_WHISPER_STT = os.getenv("USE_WHISPER", "").strip().lower() in ("1", "true", "yes")

# ── LLM Settings ──
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 900
LLM_HISTORY_LIMIT = 6  # max past messages sent to LLM

# ── Retriever Settings ──
FAISS_TOP_K = 5
FAISS_THRESHOLD = 0.45

# ── App Settings ──
HISTORY_FILE = str(PROJECT_ROOT / "consultation_history.json")
LOGO_FILENAME = "Stylized_Heart_and_Cross_Logo_for_SHIFA_AI__1_-removebg-preview.png"
LOGO_PATH = str(PROJECT_ROOT / LOGO_FILENAME)

# ── Safety ──
MAX_INPUT_LENGTH = 2000  # max chars for user input
