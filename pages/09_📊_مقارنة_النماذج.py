# -*- coding: utf-8 -*-
"""
SHIFA AI — Model Comparison Results Page
مقارنة النماذج
"""

import os
import json
import streamlit as st
import numpy as np

st.set_page_config(page_title="SHIFA AI | مقارنة النماذج", page_icon="📊", layout="wide")

# ── CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Cairo', sans-serif !important;
    }
    .main-title {
        background: linear-gradient(135deg, #E53935, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 32px;
        font-weight: 900;
        text-align: center;
    }
    .model-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #f0e0e0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        margin-bottom: 16px;
        transition: transform 0.2s;
    }
    .model-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.1); }
    .winner-badge {
        display: inline-block;
        background: linear-gradient(135deg, #FFD700, #FFA000);
        color: #333;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

RESULTS_FILE = "models/benchmark_results/benchmark_results.json"
PLOTS_DIR = "models/benchmark_results"

st.markdown('<h1 class="main-title">📊 مقارنة نماذج تشخيص سرطان الثدي</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:#5A6072; margin-bottom:24px;">
    مقارنة 6 خوارزميات ذكاء اصطناعي على مجموعة بيانات Wisconsin لسرطان الثدي
</p>
""", unsafe_allow_html=True)

# ── Load Results ──
if not os.path.exists(RESULTS_FILE):
    st.warning("لم يتم العثور على نتائج المقارنة. يرجى تشغيل:")
    st.code("python benchmark_models.py")
    st.info("سيقوم البرنامج بتدريب 6 نماذج وحفظ النتائج تلقائياً.")

    if st.button("▶️ تشغيل المقارنة الآن", type="primary"):
        with st.spinner("جاري تشغيل المقارنة (قد يستغرق بضع دقائق)..."):
            import subprocess
            result = subprocess.run(
                ["python", "benchmark_models.py"],
                capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
            )
            if result.returncode == 0:
                st.success("تم الانتهاء بنجاح!")
                st.rerun()
            else:
                st.error(f"خطأ: {result.stderr[:500]}")
    st.stop()

with open(RESULTS_FILE, "r") as f:
    results = json.load(f)

# ── Find Best Model ──
best_model = max(results.keys(), key=lambda m: results[m].get("f1_score", 0))

# ── Tab Layout ──
tab1, tab2, tab3, tab4 = st.tabs(["📊 ملخص النتائج", "📈 الرسوم البيانية", "📋 تفاصيل النماذج", "🔬 تقرير التصنيف"])

with tab1:
    # Summary metrics table
    st.markdown("### 📊 جدول المقارنة")

    import pandas as pd
    table_data = []
    for name, m in results.items():
        is_best = name == best_model
        table_data.append({
            "النموذج": f"🏆 {name}" if is_best else name,
            "الدقة (Accuracy)": f"{m['accuracy']:.4f}",
            "الضبط (Precision)": f"{m['precision']:.4f}",
            "الاستدعاء (Recall)": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "AUC-ROC": f"{m.get('auc_roc', 0):.4f}",
            "وقت التدريب": f"{m.get('train_time', 0):.3f}s",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Best model card
    best_m = results[best_model]
    st.markdown(f"""
    <div class="model-card" style="border-left: 5px solid #FFD700;">
        <span class="winner-badge">🏆 أفضل نموذج</span>
        <h3 style="margin: 10px 0 5px;">{best_model}</h3>
        <p>الدقة: <b>{best_m['accuracy']:.4f}</b> | F1: <b>{best_m['f1_score']:.4f}</b> |
           AUC: <b>{best_m.get('auc_roc', 0):.4f}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Quick comparison cards
    st.markdown("### نظرة سريعة على كل نموذج")
    cols = st.columns(3)
    colors = ["#E53935", "#1976D2", "#388E3C", "#FF9800", "#9C27B0", "#00ACC1"]

    for i, (name, m) in enumerate(results.items()):
        with cols[i % 3]:
            color = colors[i % len(colors)]
            badge = '<span class="winner-badge">🏆</span>' if name == best_model else ""
            st.markdown(f"""
            <div class="model-card" style="border-top: 4px solid {color};">
                {badge}
                <h4 style="color:{color}; margin:8px 0;">{name}</h4>
                <p style="font-size:28px; font-weight:900; color:{color}; margin:5px 0;">
                    {m['f1_score']*100:.1f}%
                </p>
                <p style="font-size:12px; color:#888;">F1-Score</p>
                <hr style="border-color:#f0f0f0;">
                <p style="font-size:13px;">
                    Acc: {m['accuracy']:.3f} | AUC: {m.get('auc_roc', 0):.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📈 الرسوم البيانية")

    plots = [
        ("metrics_comparison.png", "مقارنة المقاييس"),
        ("roc_curves.png", "منحنيات ROC"),
        ("confusion_matrices.png", "مصفوفات الالتباس"),
        ("training_time.png", "مقارنة وقت التدريب"),
    ]

    for filename, title in plots:
        path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(path):
            st.markdown(f"#### {title}")
            st.image(path, use_container_width=True)
        else:
            st.info(f"الرسم البياني '{title}' غير متاح.")

with tab3:
    st.markdown("### 📋 تفاصيل كل نموذج")

    for name, m in results.items():
        with st.expander(f"{'🏆 ' if name == best_model else ''}{name}"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{m['accuracy']:.4f}")
            col2.metric("Precision", f"{m['precision']:.4f}")
            col3.metric("Recall", f"{m['recall']:.4f}")
            col4.metric("F1-Score", f"{m['f1_score']:.4f}")

            if "confusion_matrix" in m:
                cm = np.array(m["confusion_matrix"])
                st.markdown("**مصفوفة الالتباس:**")
                cm_df = pd.DataFrame(cm,
                    index=["True Malignant", "True Benign"],
                    columns=["Pred Malignant", "Pred Benign"]
                )
                st.dataframe(cm_df)

            if "cv_mean" in m:
                st.markdown(f"**Cross-Validation:** {m['cv_mean']:.4f} ± {m['cv_std']:.4f}")

            st.markdown(f"**وقت التدريب:** {m.get('train_time', 0):.3f}s")
            st.markdown(f"**وقت الاستنتاج:** {m.get('inference_time', 0):.5f}s")

with tab4:
    st.markdown("### 🔬 تقارير التصنيف المفصلة")

    for name, m in results.items():
        if "report" in m and m["report"]:
            with st.expander(f"{name}"):
                st.code(m["report"])
