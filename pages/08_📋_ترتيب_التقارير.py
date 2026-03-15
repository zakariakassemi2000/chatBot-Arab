# -*- coding: utf-8 -*-
"""
SHIFA AI — Radiology Report Prioritization Page
ترتيب التقارير الطبية
"""

import streamlit as st
from engine.report_prioritizer import ReportPrioritizer, PRIORITY_LEVELS

st.set_page_config(page_title="SHIFA AI | ترتيب التقارير", page_icon="📋", layout="centered")

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
    .priority-card {
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        transition: transform 0.2s;
    }
    .priority-card:hover { transform: translateY(-2px); }
    .priority-urgent { background: linear-gradient(135deg, #fff5f5, #ffe0e0); border-left: 5px solid #DC3545; }
    .priority-semi { background: linear-gradient(135deg, #fff8e1, #fff3cd); border-left: 5px solid #FF9800; }
    .priority-routine { background: linear-gradient(135deg, #f0fff4, #d4edda); border-left: 5px solid #28A745; }
    .queue-number {
        display: inline-block;
        width: 36px;
        height: 36px;
        line-height: 36px;
        text-align: center;
        border-radius: 50%;
        font-weight: 900;
        font-size: 16px;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Init ──
@st.cache_resource
def load_prioritizer():
    p = ReportPrioritizer()
    p.load()  # Try to load trained model; falls back to rule-based
    return p

prioritizer = load_prioritizer()

# ── Header ──
st.markdown('<h1 class="main-title">📋 نظام ترتيب التقارير الطبية</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:#5A6072; margin-bottom:24px;">
    ترتيب تقارير الأشعة حسب الأولوية باستخدام BERT + Attention + GRU
</p>
""", unsafe_allow_html=True)

# ── Model Status ──
if prioritizer._initialized:
    st.success("🤖 النموذج المتقدم (BERT+Attention+GRU) جاهز")
else:
    st.info("📌 يعمل بالنظام القائم على القواعد. لتدريب النموذج المتقدم:")
    st.code("python train_report_prioritizer.py")

# ── Priority Legend ──
st.markdown("""
<div style="background:#f8f9fa; border-radius:14px; padding:16px; margin-bottom:20px; border:1px solid #e0e0e0;">
    <h4 style="margin:0 0 10px; color:#1E2028;">مستويات الأولوية</h4>
    <div style="display:flex; gap:12px; flex-wrap:wrap;">
        <div style="flex:1; min-width:150px; background:#d4edda; padding:10px; border-radius:10px; text-align:center;">
            <b>🟢 روتيني</b><br><small>خلال 30 يوم</small>
        </div>
        <div style="flex:1; min-width:150px; background:#fff3cd; padding:10px; border-radius:10px; text-align:center;">
            <b>🟠 شبه طارئ</b><br><small>خلال 7 أيام</small>
        </div>
        <div style="flex:1; min-width:150px; background:#f8d7da; padding:10px; border-radius:10px; text-align:center;">
            <b>🔴 طارئ</b><br><small>خلال 24 ساعة</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer ──
st.warning("⚠️ هذه الأداة تجريبية للأغراض التعليمية فقط. القرار النهائي يعود للطبيب المختص.")

# ── Tabs ──
tab1, tab2 = st.tabs(["📝 تقرير واحد", "📋 تقارير متعددة"])

with tab1:
    st.markdown("### تحليل تقرير أشعة واحد")

    report_text = st.text_area(
        "أدخل نص التقرير الطبي",
        placeholder="مثال: أشعة الصدر تُظهر عقدة رئوية مشبوهة في الفص العلوي الأيمن بحجم 12 ملم...",
        height=200,
    )

    # Example reports
    st.markdown("##### أمثلة سريعة:")
    examples = [
        ("🟢 تقرير روتيني", "الأشعة السينية للصدر: لا توجد علامات مرضية واضحة. القلب والرئتان طبيعيان. لا يوجد انصباب جنبي."),
        ("🟠 تقرير شبه طارئ", "تصوير الثدي: كتلة ذات حدود غير واضحة في الثدي الأيسر. BI-RADS 4. يُنصح بأخذ خزعة."),
        ("🔴 تقرير طارئ", "التصوير المقطعي للدماغ: نزيف داخل الجمجمة حاد. ورم دموي تحت الجافية. تحول في خط المنتصف. جراحة طارئة."),
    ]

    cols_ex = st.columns(3)
    for i, (label, text) in enumerate(examples):
        with cols_ex[i]:
            if st.button(label, key=f"example_{i}", use_container_width=True):
                st.session_state["report_example"] = text
                st.rerun()

    if "report_example" in st.session_state:
        report_text = st.session_state.pop("report_example")

    if st.button("🔍 تحليل الأولوية", use_container_width=True, type="primary"):
        if not report_text or len(report_text.strip()) < 10:
            st.warning("يرجى إدخال نص التقرير (10 أحرف على الأقل).")
        else:
            with st.spinner("جاري تحليل التقرير..."):
                result = prioritizer.predict(report_text.strip())

            level = PRIORITY_LEVELS[result["priority"]]
            css_class = {0: "priority-routine", 1: "priority-semi", 2: "priority-urgent"}[result["priority"]]

            st.markdown(f"""
            <div class="priority-card {css_class}">
                <h2 style="margin:0;">{result['icon']} {result['label_ar']}</h2>
                <p style="font-size:18px; margin:8px 0;">
                    <b>الثقة:</b> {result['confidence']*100:.1f}%
                </p>
                <p style="margin:8px 0;">{result['description']}</p>
                <p><b>الإطار الزمني المقترح:</b> خلال {result['max_days']} يوم</p>
                <p style="font-size:12px; color:#888;">المصدر: {result.get('source', 'غير محدد')}</p>
            </div>
            """, unsafe_allow_html=True)

            # Show probabilities if from BERT
            if "all_probabilities" in result:
                st.markdown("#### 📊 توزيع الاحتمالات")
                for label_ar, prob in result["all_probabilities"].items():
                    st.progress(prob, text=f"{label_ar}: {prob*100:.1f}%")

            # Show keywords if rule-based
            if "keywords_found" in result:
                kw = result["keywords_found"]
                if kw.get("urgent"):
                    st.markdown(f"**كلمات طارئة:** {', '.join(kw['urgent'])}")
                if kw.get("semi_urgent"):
                    st.markdown(f"**كلمات شبه طارئة:** {', '.join(kw['semi_urgent'])}")

with tab2:
    st.markdown("### ترتيب تقارير متعددة حسب الأولوية")
    st.markdown("أدخل كل تقرير في سطر منفصل (افصل بـ `---`):")

    multi_reports = st.text_area(
        "التقارير",
        placeholder="التقرير الأول...\n---\nالتقرير الثاني...\n---\nالتقرير الثالث...",
        height=300,
        label_visibility="collapsed",
    )

    if st.button("📋 ترتيب حسب الأولوية", use_container_width=True, type="primary", key="sort_all"):
        reports = [r.strip() for r in multi_reports.split("---") if r.strip() and len(r.strip()) > 10]

        if len(reports) < 2:
            st.warning("يرجى إدخال تقريرين على الأقل (مفصولين بـ ---).")
        else:
            with st.spinner(f"جاري تحليل {len(reports)} تقارير..."):
                results = prioritizer.predict_batch(reports)

            st.success(f"✅ تم ترتيب {len(results)} تقارير حسب الأولوية")

            # Summary
            urgent_count = sum(1 for r in results if r["priority"] == 2)
            semi_count = sum(1 for r in results if r["priority"] == 1)
            routine_count = sum(1 for r in results if r["priority"] == 0)

            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 طارئ", urgent_count)
            col2.metric("🟠 شبه طارئ", semi_count)
            col3.metric("🟢 روتيني", routine_count)

            st.markdown("---")
            st.markdown("### 📋 الترتيب (الأكثر طوارئ أولاً)")

            for i, result in enumerate(results):
                css_class = {0: "priority-routine", 1: "priority-semi", 2: "priority-urgent"}[result["priority"]]
                queue_color = PRIORITY_LEVELS[result["priority"]]["color"]

                st.markdown(f"""
                <div class="priority-card {css_class}">
                    <span class="queue-number" style="background:{queue_color}; color:white;">
                        {i+1}
                    </span>
                    <b>{result['icon']} {result['label_ar']}</b>
                    (ثقة: {result['confidence']*100:.0f}%)
                    <span style="float:left; font-size:12px; color:#888;">خلال {result['max_days']} يوم</span>
                </div>
                """, unsafe_allow_html=True)
