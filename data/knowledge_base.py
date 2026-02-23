"""
Knowledge Base (The Brain) - Arabic Medical Q&A Data
Loads, normalises and categorises Arabic medical Q&A data.
"""

import sys
import os
import re
import pandas as pd
from datasets import load_dataset


# ─── Medical categories mapping (Arabic) ───────────────────────────
CATEGORY_KEYWORDS = {
    "أمراض القلب والأوعية الدموية": [
        "قلب", "ضغط", "كوليسترول", "شريان", "أوعية", "دموية",
        "خفقان", "صمام", "جلطة", "تصلب", "ذبحة", "نبض",
    ],
    "الجهاز الهضمي": [
        "معدة", "بطن", "إسهال", "إمساك", "قولون", "كبد",
        "مرارة", "هضم", "غثيان", "تقيؤ", "قرحة", "انتفاخ",
        "بواسير", "مريء", "ارتجاع",
    ],
    "الجهاز التنفسي": [
        "رئة", "تنفس", "سعال", "كحة", "ربو", "حساسية",
        "أنف", "جيوب", "التهاب رئوي", "حلق", "لوز",
        "بلغم", "ضيق تنفس", "صدر",
    ],
    "العظام والمفاصل": [
        "عظم", "مفصل", "ظهر", "ركبة", "فقرات", "غضروف",
        "عمود فقري", "كسر", "هشاشة", "روماتيزم", "التهاب مفاصل",
        "عضلات", "أوتار", "رقبة",
    ],
    "الأمراض الجلدية": [
        "جلد", "بشرة", "حب شباب", "أكزيما", "صدفية",
        "طفح", "حكة", "ثآليل", "فطريات", "حروق",
        "شعر", "تساقط", "أظافر",
    ],
    "المسالك البولية والكلى": [
        "كلى", "بول", "مثانة", "حصوات", "بروستات",
        "التهاب بولي", "غسيل كلوي", "حالب",
    ],
    "الغدد الصماء والسكري": [
        "سكري", "غدة درقية", "هرمون", "أنسولين", "سكر",
        "غدة صماء", "كورتيزول", "تيروكسين",
    ],
    "الحمل والنساء": [
        "حمل", "رحم", "مبيض", "دورة شهرية", "ولادة",
        "جنين", "رضاعة", "تبويض", "إجهاض", "حامل",
        "نفاس", "قيصرية", "تكيس",
    ],
    "صحة الأطفال": [
        "طفل", "رضيع", "أطفال", "تطعيم", "لقاح",
        "نمو", "تسنين", "حليب أم", "حفاضات",
    ],
    "الأعصاب والصحة النفسية": [
        "عصب", "صداع", "شقيقة", "اكتئاب", "قلق",
        "أرق", "نوم", "صرع", "تشنج", "وسواس",
        "توتر", "ذاكرة", "دماغ", "مخ",
    ],
    "أمراض العيون": [
        "عين", "بصر", "نظر", "نظارة", "قرنية",
        "شبكية", "ماء أبيض", "ماء أزرق", "جفاف العين",
    ],
    "الأنف والأذن والحنجرة": [
        "أذن", "سمع", "حنجرة", "صوت", "لوزتين",
        "أنف", "جيوب أنفية", "طنين", "دوخة",
    ],
    "الأدوية والعلاج": [
        "دواء", "جرعة", "مضاد حيوي", "مسكن", "مكمل",
        "فيتامين", "علاج", "وصفة", "حبوب", "إبرة",
        "تأثيرات جانبية", "تفاعل دوائي",
    ],
    "التغذية والصحة العامة": [
        "غذاء", "وزن", "سمنة", "تخسيس", "حمية",
        "رياضة", "تمارين", "فيتامينات", "معادن",
        "ماء", "نظام غذائي", "سعرات",
    ],
}

# ─── Intent labels for the classifier ──────────────────────────────
INTENT_MAP = {
    "وصف_أعراض": [
        "أشعر", "أعاني", "عندي", "يظهر لي", "أحس",
        "ألم", "وجع", "حكة", "انتفاخ", "حرارة",
        "صداع", "دوخة", "غثيان", "تعب", "إرهاق",
    ],
    "طلب_معلومات": [
        "ما هو", "ما هي", "ماهو", "ماهي", "كيف",
        "لماذا", "هل", "أريد معرفة", "ما سبب",
        "ما معنى", "شرح", "ما الفرق",
    ],
    "طلب_علاج": [
        "علاج", "دواء", "كيف أعالج", "ما العلاج",
        "وصفة", "أدوية", "جرعة", "مضاد",
    ],
    "استشارة_طارئة": [
        "نزيف", "لا أستطيع التنفس", "ألم شديد",
        "فقدت الوعي", "تسمم", "إغماء", "حادث",
        "كسر", "حروق شديدة", "ألم في الصدر المفاجئ",
    ],
    "طلب_توجيه": [
        "أي طبيب", "أي تخصص", "إلى من أذهب",
        "أحتاج استشارة", "موعد", "تحاليل", "أشعة",
        "فحص", "تشخيص",
    ],
}


def detect_category(text: str) -> str:
    """Assign a medical category based on keyword matching."""
    text_lower = text.strip()
    best_cat = "عام"
    best_count = 0
    for cat, keywords in CATEGORY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_cat = cat
    return best_cat


def detect_intent(text: str) -> str:
    """Detect the user's intent from their message."""
    text_lower = text.strip()
    best_intent = "وصف_أعراض"
    best_count = 0
    for intent, keywords in INTENT_MAP.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_intent = intent
    return best_intent


def clean_text(text: str) -> str:
    """Clean and normalise Arabic medical text."""
    if not isinstance(text, str):
        return ""
    # Remove doctor names and signatures
    text = re.sub(r'(أ\.?د\.?|ا\s*د\.?)\s+[\u0600-\u06FF\s]{3,30}', '', text)
    text = re.sub(r'والله\s+الشافي', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Normalise Arabic characters
    text = re.sub(r'[إأآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    # Remove excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def load_and_prepare_datasets(max_samples: int = 8000) -> pd.DataFrame:
    """
    Download Arabic medical Q&A datasets from HuggingFace,
    normalize, categorise, and return a structured DataFrame.
    """
    print("╔══════════════════════════════════════════════════╗")
    print("║  📚 تحميل وتجهيز قاعدة المعرفة الطبية العربية     ║")
    print("╚══════════════════════════════════════════════════╝")

    datasets_config = [
        {
            "name": "MustafaIbrahim/medical-arabic-qa",
            "label": "Medical Arabic QA",
        },
        {
            "name": "tarekys5/Arabic_Healthcare_QA",
            "label": "Arabic Healthcare QA",
        },
    ]

    all_dfs = []

    for cfg in datasets_config:
        try:
            print(f"\n  ▸ جاري تحميل {cfg['label']}...")
            ds = load_dataset(cfg["name"], split="train")
            df = ds.to_pandas()
            df.columns = [c.lower().strip() for c in df.columns]

            # Normalise column names
            rename_map = {}
            for col in df.columns:
                if any(k in col for k in ['question', 'instruction', 'input']):
                    rename_map[col] = 'question'
                elif any(k in col for k in ['answer', 'response', 'output']):
                    rename_map[col] = 'answer'
                elif 'category' in col or 'topic' in col:
                    rename_map[col] = 'category'
            df.rename(columns=rename_map, inplace=True)

            if 'question' in df.columns and 'answer' in df.columns:
                # Filter out very short / empty answers
                df = df[df['answer'].astype(str).str.len() > 30]
                df = df[df['question'].astype(str).str.len() > 5]
                keep_cols = [c for c in ['question', 'answer', 'category'] if c in df.columns]
                df = df[keep_cols].copy()
                print(f"    ✓ تم تحميل {len(df)} زوج سؤال/جواب")
                all_dfs.append(df)
            else:
                print(f"    ✗ لم يتم التعرف على أعمدة السؤال/الجواب")
        except Exception as e:
            print(f"    ✗ خطأ: {e}")

    if not all_dfs:
        raise RuntimeError("لم يتم تحميل أي مجموعة بيانات!")

    # Combine all datasets
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  📊 إجمالي البيانات المجمعة: {len(full_df)} زوج")

    # Remove duplicates
    full_df.drop_duplicates(subset=['question'], inplace=True)
    print(f"  📊 بعد إزالة المكررات: {len(full_df)} زوج")

    # Sample for performance
    if len(full_df) > max_samples:
        print(f"  📊 أخذ عينة من {max_samples} زوج للأداء...")
        full_df = full_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    # Clean texts
    print("  🧹 تنظيف النصوص...")
    full_df['question'] = full_df['question'].apply(clean_text)
    full_df['answer'] = full_df['answer'].apply(lambda x: clean_text(str(x)))

    # Auto-detect categories where missing
    if 'category' not in full_df.columns:
        full_df['category'] = ''
    mask = full_df['category'].isna() | (full_df['category'].astype(str).str.strip() == '') | (full_df['category'] == 'nan')
    full_df.loc[mask, 'category'] = full_df.loc[mask, 'question'].apply(detect_category)
    full_df.loc[~mask, 'category'] = full_df.loc[~mask, 'category'].apply(
        lambda x: x if isinstance(x, str) and x.strip() else 'عام'
    )

    # Detect intents
    print("  🎯 تصنيف النوايا...")
    full_df['intent'] = full_df['question'].apply(detect_intent)

    # Final clean
    full_df = full_df[full_df['answer'].str.len() > 30].reset_index(drop=True)

    print(f"\n  ✅ قاعدة المعرفة جاهزة: {len(full_df)} زوج سؤال/جواب")
    print(f"  📁 التصنيفات: {full_df['category'].nunique()} تصنيف")
    print(f"  🎯 النوايا: {full_df['intent'].nunique()} نية")

    return full_df
