"""
SHIFA-Mental — Arabic Psychotherapy LLM Fine-Tuning Guide
=========================================================
Stratégie de fine-tuning d'un LLM sur la psychothérapie arabe
"""

# ════════════════════════════════════════════════════════════════
# ARCHITECTURE DE FINE-TUNING
# ════════════════════════════════════════════════════════════════
"""
Modèles de base recommandés (par ordre de priorité) :

1. Jais-13B (أبوظبي — G42/MBZUAI)
   - Meilleur LLM arabe open-source
   - 13B paramètres, pré-entraîné sur 116B tokens arabes
   - HuggingFace: inceptionai/jais-13b-chat

2. AceGPT-13B (CUHK)
   - Fine-tuné sur instructions arabes
   - Bon pour le dialogue clinique

3. ALLaM-7B (SDAIA — Arabie Saoudite)
   - Optimisé pour l'arabe dialectal et classique

4. Mistral-7B + Arabic LoRA
   - Alternative légère avec LoRA adapters arabes
"""

# ════════════════════════════════════════════════════════════════
# DATASETS DE FINE-TUNING
# ════════════════════════════════════════════════════════════════
"""
Sources de données pour la psychothérapie arabe :

1. ArabicMentalHealth Dataset
   - Twitter/X arabe annoté en détresse psychologique
   - Classes : dépression, anxiété, PTSD, normal

2. MultiArabic Mental Health (MAMS)
   - Multi-dialectal Arabic mental health corpus
   - 5 dialectes arabes + MSA

3. Synthetic Data Generation (à créer) :
   - Générer 10K conversations CBT simulées via GPT-4
   - Prompt: "Generate a CBT therapy conversation in Arabic about [topic]"
   - Topics: anxiety, depression, grief, relationship issues, work stress

4. Dataset Darija (Marocain) :
   - DarijaBERT corpus (Université Mohammed V)
   - Adapter pour le contexte thérapeutique marocain
"""

SYNTHETIC_DATA_GENERATOR_PROMPT = """
Tu génères des données d'entraînement pour un LLM de soutien psychologique en arabe.

Crée une conversation thérapeutique réaliste (CBT) en arabe standard entre :
- Patient : souffrant de {condition}
- Thérapeute : appliquant les techniques CBT (restructuration cognitive, activation comportementale)

Format JSON strict :
{
  "condition": "{condition}",
  "dialect": "MSA|Moroccan|Egyptian|Levantine",
  "severity": 1|2|3,
  "conversation": [
    {"role": "patient", "content": "..."},
    {"role": "therapist", "content": "..."},
    ...
  ],
  "techniques_used": ["CBT", "mindfulness", "behavioral_activation"],
  "crisis_flag": false
}

Conditions cibles : anxiety, depression, grief, insomnia, relationship_issues, 
                    work_burnout, trauma, social_phobia, OCD_mild
"""

# ════════════════════════════════════════════════════════════════
# FINE-TUNING PIPELINE (QLoRA)
# ════════════════════════════════════════════════════════════════

FINETUNING_CONFIG = {
    "base_model": "inceptionai/jais-13b-chat",
    "method": "QLoRA",
    "quantization": "4-bit NF4",
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    "training_args": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "fp16": True,
        "logging_steps": 25,
        "save_steps": 200,
        "evaluation_strategy": "steps",
        "eval_steps": 200
    },
    "data_format": "chatml",  # <|im_start|>system\n...\n<|im_end|>
    "estimated_vram": "16GB (A100/V100)",
    "estimated_time": "4-6 hours on A100"
}

TRAINING_SCRIPT = '''
#!/usr/bin/env python
"""
SHIFA-Mental Fine-Tuning Script
Requires: transformers, peft, trl, bitsandbytes, datasets
"""

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# ── 1. Load base model (Jais-13B) ──────────────────────────────
MODEL_ID = "inceptionai/jais-13b-chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ── 2. LoRA Config ─────────────────────────────────────────────
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 3. Dataset ─────────────────────────────────────────────────
SYSTEM_PROMPT = """أنت معالج نفسي متخصص يستخدم أساليب العلاج المعرفي السلوكي.
تتحدث بالعربية مع تعاطف واحترام، وتساعد المرضى على فهم مشاعرهم وتطوير
استراتيجيات للتكيف الإيجابي. أولويتك القصوى هي سلامة المريض."""

def format_conversation(example):
    """Format dataset entry into ChatML format."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(example["conversation"])
    
    formatted = ""
    for msg in messages:
        formatted += f"<|im_start|>{msg[\'role\']}\\n{msg[\'content\']}<|im_end|>\\n"
    return {"text": formatted}

# Load your custom Arabic psychotherapy dataset
# dataset = load_dataset("json", data_files="arabic_psychotherapy_dataset.jsonl")
# dataset = dataset.map(format_conversation)

# ── 4. Training ────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./shifa-mental-jais-13b",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=25,
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="shifa-mental-v1"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
)

trainer.train()
trainer.save_model("./shifa-mental-jais-13b-final")

# ── 5. Merge & Push ────────────────────────────────────────────
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
merged_model = PeftModel.from_pretrained(base_model, "./shifa-mental-jais-13b-final")
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("./shifa-mental-merged")
# merged_model.push_to_hub("your-org/shifa-mental-v1")
'''

# ════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ════════════════════════════════════════════════════════════════

EVALUATION_FRAMEWORK = {
    "automatic_metrics": {
        "BLEU-4": "Qualité de génération vs references humaines",
        "BERTScore (AraBERT)": "Similarité sémantique en arabe",
        "Perplexity": "Fluidité du modèle sur corpus test"
    },
    "clinical_metrics": {
        "Empathy Score": "Présence de formulations empathiques (lexicon-based)",
        "Safety Score": "Détection correcte des crises (F1-score)",
        "Therapeutic Alliance": "Évaluation humaine par experts cliniques",
        "Crisis Detection F1": "Rappel des cas de crise (priorité recall > precision)"
    },
    "safety_tests": [
        "Test 1: Faux positifs — textes neutres ne déclenchant pas d'alarme",
        "Test 2: Vrais positifs — phrases de crise détectées (recall ≥ 0.98)",
        "Test 3: Refus de prescription médicamenteuse",
        "Test 4: Redirection appropriée vers professionnels",
        "Test 5: Cohérence culturelle arabe/marocaine"
    ]
}

# ════════════════════════════════════════════════════════════════
# DEPLOYMENT (Streamlit Cloud + Hugging Face Inference)
# ════════════════════════════════════════════════════════════════

DEPLOYMENT_OPTIONS = {
    "option_1_hf_inference": {
        "description": "Hugging Face Inference API (gratuit tier)",
        "model_repo": "your-org/shifa-mental-jais-13b",
        "latency": "2-5s",
        "cost": "Gratuit jusqu'à 30k tokens/mois",
        "code": """
import requests

HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/your-org/shifa-mental-jais-13b"

def call_hf_model(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}
    r = requests.post(API_URL, headers=headers, json=payload)
    return r.json()[0]["generated_text"]
"""
    },
    "option_2_ollama": {
        "description": "Ollama local (développement)",
        "command": "ollama run shifa-mental",
        "latency": "<1s local",
        "cost": "Gratuit",
        "note": "Créer Modelfile avec GGUF quantized version"
    },
    "option_3_openrouter_fallback": {
        "description": "OpenRouter API (solution actuelle dans mental_module.py)",
        "model": "anthropic/claude-3-haiku",
        "latency": "1-3s",
        "cost": "~$0.25/1M input tokens via OpenRouter",
        "note": "Compatible OpenAI SDK — supporte Jais, Claude, Llama via un seul endpoint",
        "code": """
import requests

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_openrouter(messages: list, model: str = "anthropic/claude-3-haiku") -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://shifa-ai.ma",
        "X-Title": "SHIFA-Mental"
    }
    r = requests.post(API_URL, headers=headers, json={
        "model": model,
        "messages": messages,
        "max_tokens": 512
    })
    return r.json()["choices"][0]["message"]["content"]
"""
    }
}
