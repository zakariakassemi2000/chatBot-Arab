"""
Setup Script - Arabic Health Assistant
Run this ONCE to download data, build FAISS index, train classifier.
"""

import sys
import os
import io
import time

# Fix Windows console encoding for Arabic/Unicode
if sys.platform == 'win32' and not os.environ.get('_UTF8_FIX_APPLIED'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        os.environ['_UTF8_FIX_APPLIED'] = '1'
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.knowledge_base import load_and_prepare_datasets
from engine.retriever import FAISSRetriever
from engine.classifier import IntentClassifier


def setup(max_samples: int = 8000):
    """Complete setup: download data, build index, train classifier."""
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   🏥 إعداد المساعد الصحي الذكي بالعربية                  ║")
    print("║   Arabic Health Assistant — Full Setup                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    start = time.time()

    # ── Step 1: Load and prepare knowledge base ──
    print("━" * 55)
    print("  📚 الخطوة 1: تحميل وتجهيز قاعدة المعرفة")
    print("━" * 55)
    df = load_and_prepare_datasets(max_samples=max_samples)

    # ── Step 2: Build FAISS index ──
    print("\n" + "━" * 55)
    print("  🔍 الخطوة 2: بناء فهرس FAISS للبحث السريع")
    print("━" * 55)
    retriever = FAISSRetriever()
    embeddings = retriever.build_index(df, verbose=True)

    # ── Step 3: Train intent classifier ──
    print("\n" + "━" * 55)
    print("  🧠 الخطوة 3: تدريب مصنّف النوايا")
    print("━" * 55)
    classifier = IntentClassifier()
    intents = df['intent'].tolist()
    classifier.train(embeddings, intents, verbose=True)

    # ── Step 4: Save everything ──
    print("\n" + "━" * 55)
    print("  💾 الخطوة 4: حفظ النماذج والفهارس")
    print("━" * 55)
    retriever.save()
    classifier.save()

    elapsed = time.time() - start
    print(f"\n{'═' * 55}")
    print(f"  ✅ الإعداد اكتمل بنجاح! ⏱️ الوقت: {elapsed:.1f} ثانية")
    print(f"  📊 قاعدة المعرفة: {len(df)} سؤال/جواب")
    print(f"  🎯 النوايا: {df['intent'].nunique()} نوع")
    print(f"  📁 التصنيفات: {df['category'].nunique()} تصنيف")
    print(f"{'═' * 55}")
    print()
    print("  🚀 لتشغيل المساعد:")
    print("     streamlit run app.py")
    print()


if __name__ == "__main__":
    setup()
