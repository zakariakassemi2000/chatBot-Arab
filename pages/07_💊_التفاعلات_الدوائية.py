# -*- coding: utf-8 -*-
"""
SHIFA AI — Drug-Drug Interaction Checker Page
التفاعلات الدوائية
"""

import streamlit as st
from engine.ddi_detector import DrugInteractionDetector, DDI_TYPES, KNOWN_INTERACTIONS

st.set_page_config(page_title="SHIFA AI | التفاعلات الدوائية", page_icon="💊", layout="centered")

# ── CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Cairo', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }
    .block-container { max-width: 860px !important; }
    .main-title {
        background: linear-gradient(135deg, #E53935, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 32px;
        font-weight: 900;
        text-align: center;
    }
    .interaction-card {
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        border: 1px solid #f0e0e0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }
    .severity-high { background: linear-gradient(135deg, #fff5f5, #ffe0e0); border-left: 5px solid #DC3545; }
    .severity-moderate { background: linear-gradient(135deg, #fffbf0, #fff3cd); border-left: 5px solid #FFC107; }
    .severity-safe { background: linear-gradient(135deg, #f0fff4, #d4edda); border-left: 5px solid #28A745; }
    .severity-unknown { background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-left: 5px solid #6C757D; }
</style>
""", unsafe_allow_html=True)

# ── Init ──
@st.cache_resource
def load_ddi():
    return DrugInteractionDetector()

ddi = load_ddi()

# ── Header ──
st.markdown('<h1 class="main-title">💊 فاحص التفاعلات الدوائية</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:#5A6072; margin-bottom:24px;">
    تحقق من التفاعلات المحتملة بين الأدوية التي تتناولها
</p>
""", unsafe_allow_html=True)

# ── Disclaimer ──
st.warning("⚠️ هذه الأداة للأغراض التعليمية فقط. استشر الصيدلي أو الطبيب دائماً قبل تغيير أدويتك.")

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["🔍 فحص دوائين", "📋 فحص قائمة أدوية", "📖 قاعدة البيانات"])

with tab1:
    st.markdown("### فحص التفاعل بين دوائين")
    col1, col2 = st.columns(2)
    with col1:
        drug1 = st.text_input("💊 الدواء الأول", placeholder="مثال: وارفارين", key="drug1")
    with col2:
        drug2 = st.text_input("💊 الدواء الثاني", placeholder="مثال: أسبرين", key="drug2")

    context = st.text_input("📝 سياق إضافي (اختياري)", placeholder="مثال: مريض يتناول مميعات الدم")

    if st.button("🔍 فحص التفاعل", use_container_width=True, type="primary"):
        if not drug1 or not drug2:
            st.warning("يرجى إدخال اسم الدوائين.")
        else:
            with st.spinner("جاري تحليل التفاعل الدوائي..."):
                result = ddi.predict_interaction(drug1.strip(), drug2.strip(), context.strip())

            severity = result.get("severity", "unknown")
            css_class = f"severity-{severity}"

            severity_icons = {"high": "🔴", "moderate": "🟡", "safe": "🟢", "unknown": "⚪"}
            severity_labels = {"high": "خطورة عالية", "moderate": "خطورة متوسطة",
                             "safe": "آمن", "unknown": "غير محدد"}

            icon = severity_icons.get(severity, "⚪")
            label = severity_labels.get(severity, "غير محدد")

            st.markdown(f"""
            <div class="interaction-card {css_class}">
                <h3>{icon} {result.get('type_ar', 'غير معروف')}</h3>
                <p><b>الدواء الأول:</b> {result['drug1']}</p>
                <p><b>الدواء الثاني:</b> {result['drug2']}</p>
                <p><b>مستوى الخطورة:</b> {label}</p>
                <hr style="border-color: rgba(0,0,0,0.1);">
                <p><b>الوصف:</b> {result.get('description', '')}</p>
                <p><b>التوصية:</b> {result.get('recommendation', '')}</p>
                <p style="font-size:12px; color:#888;">المصدر: {result.get('source', 'غير محدد')}</p>
            </div>
            """, unsafe_allow_html=True)

            # Show confidence if from BERT
            if "confidence" in result:
                st.markdown("#### 📊 توزيع الاحتمالات")
                for label_ar, prob in result.get("all_probabilities", {}).items():
                    st.progress(prob, text=f"{label_ar}: {prob*100:.1f}%")

with tab2:
    st.markdown("### فحص قائمة أدوية متعددة")
    st.markdown("أدخل أسماء الأدوية (دواء واحد في كل سطر):")

    drugs_text = st.text_area(
        "قائمة الأدوية",
        placeholder="وارفارين\nأسبرين\nأوميبرازول\nكلوبيدوغريل",
        height=150,
        label_visibility="collapsed",
    )

    if st.button("🔍 فحص جميع التفاعلات", use_container_width=True, type="primary", key="check_all"):
        drugs = [d.strip() for d in drugs_text.strip().split("\n") if d.strip()]

        if len(drugs) < 2:
            st.warning("يرجى إدخال دوائين على الأقل.")
        else:
            with st.spinner(f"جاري فحص {len(drugs)} أدوية ({len(drugs)*(len(drugs)-1)//2} تفاعل محتمل)..."):
                interactions = ddi.check_multiple_drugs(drugs)

            if not interactions:
                st.success("✅ لم يُكتشف أي تفاعل دوائي بين الأدوية المُدخلة.")
            else:
                st.error(f"⚠️ تم اكتشاف {len(interactions)} تفاعل(ات) دوائية!")

                for inter in interactions:
                    severity = inter.get("severity", "unknown")
                    css_class = f"severity-{severity}"
                    severity_icons = {"high": "🔴", "moderate": "🟡", "safe": "🟢", "unknown": "⚪"}
                    icon = severity_icons.get(severity, "⚪")

                    st.markdown(f"""
                    <div class="interaction-card {css_class}">
                        <h4>{icon} {inter['drug1']} ↔ {inter['drug2']}</h4>
                        <p><b>النوع:</b> {inter.get('type_ar', '')}</p>
                        <p>{inter.get('description', '')}</p>
                        <p><b>التوصية:</b> {inter.get('recommendation', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 📖 قاعدة بيانات التفاعلات المعروفة")
    st.markdown(f"تحتوي قاعدة البيانات على **{len(KNOWN_INTERACTIONS)}** تفاعل دوائي موثق.")

    for (d1, d2), info in KNOWN_INTERACTIONS.items():
        severity = info["severity"]
        severity_icons = {"high": "🔴", "moderate": "🟡", "safe": "🟢"}
        icon = severity_icons.get(severity, "⚪")

        with st.expander(f"{icon} {d1} ↔ {d2}"):
            st.markdown(f"**النوع:** {info.get('type', '')}")
            st.markdown(f"**الوصف:** {info.get('description_ar', '')}")
            st.markdown(f"**التوصية:** {info.get('recommendation_ar', '')}")
