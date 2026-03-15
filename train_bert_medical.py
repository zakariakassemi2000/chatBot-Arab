# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — AraBERT Medical Fine-Tuning Script
  
  Fine-tunes AraBERT (aubmindlab/bert-base-arabertv2) on Arabic
  medical text data for intent classification and domain adaptation.
  
  Data Sources:
    1. Existing RAG data (retriever_data.pkl)
    2. HuggingFace datasets (arbml/CQA_MD_ar, MedQA)
    3. Custom medical Q&A pairs
  
  USAGE:
    python train_bert_medical.py
    python train_bert_medical.py --epochs 10 --batch_size 32
    python train_bert_medical.py --model CAMeL-Lab/bert-base-arabic-camelbert-mix
═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch

# ─── Configuration ────────────────────────────────────────────────
CONFIG = {
    "model_name": "aubmindlab/bert-base-arabertv2",
    "max_samples": 8000,
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "max_length": 256,
    "val_split": 0.15,
    "output_dir": "models/bert_medical",
    "seed": 42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
#  Step 0: Parse Arguments
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AraBERT for medical classification")
    parser.add_argument("--model", default=CONFIG["model_name"], help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--max_samples", type=int, default=CONFIG["max_samples"])
    parser.add_argument("--skip_download", action="store_true", help="Use only local data")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  Step 1: Load & Prepare Data
# ═══════════════════════════════════════════════════════════════════

def load_existing_data():
    """Load existing RAG data from retriever_data.pkl."""
    pkl_path = "models/retriever_data.pkl"
    if not os.path.exists(pkl_path):
        print("  ⚠️  retriever_data.pkl not found, skipping local data")
        return [], []

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    texts = []
    labels = []

    if isinstance(data, dict):
        if "texts" in data and "intents" in data:
            texts = data["texts"]
            labels = data["intents"]
        elif "df" in data:
            df = data["df"]
            texts = df["text"].tolist() if "text" in df.columns else []
            labels = df["intent"].tolist() if "intent" in df.columns else []
    elif isinstance(data, pd.DataFrame):
        texts = data["text"].tolist() if "text" in data.columns else \
                data["question"].tolist() if "question" in data.columns else []
        labels = data["intent"].tolist() if "intent" in data.columns else \
                 data["category"].tolist() if "category" in data.columns else []

    print(f"  ✅ Loaded {len(texts)} samples from local RAG data")
    return texts, labels


def load_huggingface_data(max_samples=8000):
    """Download and prepare Arabic medical datasets from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ⚠️  'datasets' library not installed. Run: pip install datasets")
        return [], []

    texts = []
    labels = []

    # ── Dataset 1: Arabic Medical Q&A ──
    try:
        print("  📥 Downloading arbml/CQA_MD_ar...")
        ds = load_dataset("arbml/CQA_MD_ar", split="train")
        for row in ds:
            q = row.get("question", "")
            a = row.get("answer", "")
            if q and len(q) > 10:
                texts.append(q)
                # Heuristic intent assignment based on content
                labels.append(_classify_intent(q))
        print(f"     ✅ Loaded {len(texts)} from CQA_MD_ar")
    except Exception as e:
        print(f"     ⚠️  CQA_MD_ar failed: {e}")

    # ── Dataset 2: Medical Q&A (multilingual) ──
    try:
        print("  📥 Downloading medical_questions_pairs...")
        ds = load_dataset("medical_questions_pairs", split="train")
        count = 0
        for row in ds:
            q = row.get("question_1", "") or row.get("question", "")
            if q and len(q) > 10:
                texts.append(q)
                labels.append(_classify_intent(q))
                count += 1
                if count >= max_samples // 4:
                    break
        print(f"     ✅ Loaded {count} from medical_questions_pairs")
    except Exception as e:
        print(f"     ⚠️  medical_questions_pairs failed: {e}")

    # Limit total samples
    if len(texts) > max_samples:
        indices = np.random.RandomState(42).choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]

    print(f"  📊 Total HuggingFace samples: {len(texts)}")
    return texts, labels


def _classify_intent(text: str) -> str:
    """Heuristic intent classification for training data bootstrap."""
    text_lower = text.lower() if text else ""

    emergency_kw = ["طوارئ", "إسعاف", "نزيف", "إغماء", "سكتة", "نوبة قلبية",
                    "ضيق تنفس شديد", "فقدان وعي", "emergency", "urgent"]
    treatment_kw = ["علاج", "دواء", "عقار", "حبوب", "مضاد", "جرعة",
                    "treatment", "medication", "drug", "dose"]
    symptom_kw = ["أعاني", "ألم", "صداع", "حمى", "سعال", "غثيان", "دوخة",
                  "أشعر", "pain", "headache", "fever", "cough"]
    guidance_kw = ["طبيب", "مستشفى", "أخصائي", "تحويل", "فحص",
                   "doctor", "hospital", "specialist", "referral"]

    for kw in emergency_kw:
        if kw in text_lower:
            return "استشارة_طارئة"
    for kw in treatment_kw:
        if kw in text_lower:
            return "طلب_علاج"
    for kw in symptom_kw:
        if kw in text_lower:
            return "وصف_أعراض"
    for kw in guidance_kw:
        if kw in text_lower:
            return "طلب_توجيه"
    return "طلب_معلومات"


def generate_synthetic_data():
    """Generate synthetic Arabic medical Q&A for augmentation."""
    synthetic = [
        # وصف_أعراض
        ("أعاني من صداع شديد في الجانب الأيمن من الرأس منذ ثلاثة أيام", "وصف_أعراض"),
        ("لدي ألم في الصدر عند التنفس العميق", "وصف_أعراض"),
        ("أشعر بدوخة مستمرة وغثيان خاصة في الصباح", "وصف_أعراض"),
        ("ظهرت لدي حمى وسعال جاف منذ أسبوع", "وصف_أعراض"),
        ("أعاني من ألم في أسفل الظهر يمتد إلى الساق اليسرى", "وصف_أعراض"),
        ("لاحظت تورم في القدمين مع ضيق في التنفس", "وصف_أعراض"),

        # طلب_معلومات
        ("ما هي أسباب ارتفاع ضغط الدم؟", "طلب_معلومات"),
        ("ما الفرق بين السكري من النوع الأول والثاني؟", "طلب_معلومات"),
        ("ما هي فوائد فيتامين D للجسم؟", "طلب_معلومات"),
        ("كيف يعمل الجهاز المناعي في مكافحة العدوى؟", "طلب_معلومات"),
        ("ما هي أعراض نقص الحديد في الدم؟", "طلب_معلومات"),
        ("ما هي مضاعفات السكري إذا لم يُعالج؟", "طلب_معلومات"),

        # طلب_علاج
        ("ما هو أفضل دواء لعلاج الصداع النصفي؟", "طلب_علاج"),
        ("هل يوجد علاج طبيعي لارتفاع الكوليسترول؟", "طلب_علاج"),
        ("ما جرعة الباراسيتامول المناسبة للأطفال؟", "طلب_علاج"),
        ("ما هو العلاج المناسب لحساسية الأنف الموسمية؟", "طلب_علاج"),
        ("هل المضادات الحيوية فعالة ضد نزلات البرد؟", "طلب_علاج"),
        ("ما هو بديل الأسبرين لمرضى المعدة؟", "طلب_علاج"),

        # استشارة_طارئة
        ("أشعر بألم شديد في الصدر وضيق تنفس مفاجئ", "استشارة_طارئة"),
        ("طفلي سقط على رأسه وفقد الوعي لثوانٍ", "استشارة_طارئة"),
        ("هناك شخص يعاني من نزيف حاد لا يتوقف", "استشارة_طارئة"),
        ("أشعر بخدر مفاجئ في الذراع الأيسر وصعوبة في الكلام", "استشارة_طارئة"),
        ("تناول ابني كمية كبيرة من الدواء بالخطأ", "استشارة_طارئة"),
        ("حرق شديد في الجلد مع ظهور فقاعات كبيرة", "استشارة_طارئة"),

        # طلب_توجيه
        ("أي طبيب متخصص يجب أن أزوره لآلام المفاصل؟", "طلب_توجيه"),
        ("هل أحتاج فحص دم شامل أم فحص كوليسترول فقط؟", "طلب_توجيه"),
        ("متى يجب عمل أشعة رنين مغناطيسي؟", "طلب_توجيه"),
        ("هل حالتي تستدعي زيارة طبيب أم يمكنني الانتظار؟", "طلب_توجيه"),
        ("ما هي الفحوصات الدورية المناسبة لعمر 40 سنة؟", "طلب_توجيه"),
        ("هل أحتاج جراحة أم العلاج الدوائي كافٍ؟", "طلب_توجيه"),
    ]

    texts = [t for t, _ in synthetic]
    labels = [l for _, l in synthetic]
    print(f"  ✅ Generated {len(texts)} synthetic samples")
    return texts, labels


# ═══════════════════════════════════════════════════════════════════
#  Step 2: Train
# ═══════════════════════════════════════════════════════════════════

def train_model(args):
    """Main training pipeline."""
    from engine.bert_medical import MedicalBERT

    start_time = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   SHIFA AI — AraBERT Medical Fine-Tuning                   ║")
    print(f"║   Model: {args.model:<49}║")
    print(f"║   Device: {str(DEVICE):<48}║")
    print(f"║   GPU: {str(torch.cuda.is_available()):<51}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Load Data ──
    print("━" * 60)
    print("  📦 Step 1: Loading Training Data")
    print("━" * 60)

    all_texts, all_labels = [], []

    # Local RAG data
    t, l = load_existing_data()
    all_texts.extend(t)
    all_labels.extend(l)

    # HuggingFace data
    if not args.skip_download:
        t, l = load_huggingface_data(args.max_samples)
        all_texts.extend(t)
        all_labels.extend(l)

    # Synthetic augmentation
    t, l = generate_synthetic_data()
    all_texts.extend(t)
    all_labels.extend(l)

    # Validate data
    if len(all_texts) < 50:
        print("  ❌ Not enough training data (minimum 50 samples)")
        print("  💡 Run without --skip_download to fetch HuggingFace data")
        sys.exit(1)

    # Filter valid labels
    from engine.bert_medical import MEDICAL_INTENTS
    valid_indices = [i for i, l in enumerate(all_labels) if l in MEDICAL_INTENTS]
    all_texts = [all_texts[i] for i in valid_indices]
    all_labels = [all_labels[i] for i in valid_indices]

    print(f"\n  📊 Total training samples: {len(all_texts)}")
    print("  📊 Label distribution:")
    from collections import Counter
    for label, count in Counter(all_labels).most_common():
        print(f"     {label}: {count}")

    # ── Fine-tune ──
    print(f"\n{'━' * 60}")
    print("  🧠 Step 2: Fine-tuning AraBERT")
    print("━" * 60)

    engine = MedicalBERT(model_name=args.model, num_classes=len(MEDICAL_INTENTS))

    history = engine.fine_tune(
        texts=all_texts,
        labels=all_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        verbose=True,
    )

    # ── Save ──
    print(f"\n{'━' * 60}")
    print("  💾 Step 3: Saving Model")
    print("━" * 60)

    engine.save()

    # Save training history
    history_path = os.path.join(CONFIG["output_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  ✅ History saved: {history_path}")

    # ── Evaluate ──
    print(f"\n{'━' * 60}")
    print("  📊 Step 4: Final Evaluation")
    print("━" * 60)

    # Quick eval on synthetic data to verify
    eval_texts, eval_labels = generate_synthetic_data()
    results = engine.evaluate_detailed(eval_texts, eval_labels)

    print(f"\n  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    print(f"\n{results['classification_report']}")

    total_time = time.time() - start_time
    print("═" * 60)
    print(f"  🎉 Training Complete! Total time: {total_time/60:.1f} minutes")
    print(f"  📁 Model saved: {CONFIG['output_dir']}/")
    print("═" * 60)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
