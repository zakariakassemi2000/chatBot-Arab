# -*- coding: utf-8 -*-
import streamlit as st
import sys
import os
import base64
import re
from PIL import Image as PILImage

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.llm import GroqVision

st.set_page_config(page_title="تحليل الصور الطبية", page_icon="🔬", layout="wide")


# ── Parser helper to structure the medical vision report ──
def parse_vision_response(text: str) -> dict:
    res = {
        "type": "غير محدد",
        "observations": "",
        "diagnosis": "",
        "severity": "غير محدد",
        "recommendation": "",
        "when_to_see_doctor": ""
    }
    
    # Remove markdown bold markers for simpler matching
    text_clean = text.replace("**", "")
    
    # Regex parsing for structured headers
    type_match = re.search(r"1\.\s*نوع\s*الصورة\s*:\s*(.*?)(?=\d\.)", text_clean, re.DOTALL)
    if not type_match:
        type_match = re.search(r"نوع\s*الصورة\s*:\s*(.*?)(?=\n)", text_clean)
        
    obs_match = re.search(r"2\.\s*الملاحظات\s*الرئيسية\s*:\s*(.*?)(?=\d\.)", text_clean, re.DOTALL)
    if not obs_match:
        obs_match = re.search(r"الملاحظات\s*الرئيسية\s*:\s*(.*?)(?=\n\n)", text_clean, re.DOTALL)
        
    diag_match = re.search(r"3\.\s*التشخيص\s*المحتمل\s*:\s*(.*?)(?=\d\.)", text_clean, re.DOTALL)
    if not diag_match:
        diag_match = re.search(r"التشخيص\s*المحتمل\s*:\s*(.*?)(?=\n\n)", text_clean, re.DOTALL)
        
    sev_match = re.search(r"4\.\s*مستوى\s*الخطورة\s*:\s*(.*?)(?=\d\.)", text_clean, re.DOTALL)
    if not sev_match:
        sev_match = re.search(r"مستوى\s*الخطورة\s*:\s*(.*?)(?=\n\n)", text_clean, re.DOTALL)
        
    rec_match = re.search(r"5\.\s*التوصية\s*الطبية\s*:\s*(.*)", text_clean, re.DOTALL)
    
    if type_match: res["type"] = type_match.group(1).strip()
    if obs_match: res["observations"] = obs_match.group(1).strip()
    if diag_match: res["diagnosis"] = diag_match.group(1).strip()
    if sev_match: res["severity"] = sev_match.group(1).strip()
    if rec_match: res["recommendation"] = rec_match.group(1).strip()

    # Fallback if text format differs
    if not any([obs_match, diag_match, rec_match]):
        res["diagnosis"] = text
        res["observations"] = "تم تضمين الملاحظات التفصيلية في ملخص التحليل أعلاه."
        res["recommendation"] = "يرجى مراجعة ملخص الفحص للحصول على التوجيهات الطبية."

    # Determine "When to consult a doctor" based on severity
    sev_lower = res["severity"].lower()
    if any(k in sev_lower for k in ["حرج", "critical", "حرج جدا", "حرج جداً"]):
        res["when_to_see_doctor"] = "🚨 يُوصى بالتوجه فوراً لأقرب قسم طوارئ أو استدعاء الطبيب المختص نظراً لارتفاع مستوى الخطورة المكتشف."
    elif any(k in sev_lower for k in ["مرتفع", "high", "عالية", "عالي"]):
        res["when_to_see_doctor"] = "⚠️ يُنصح بجدولة موعد طبي عاجل مع الطبيب المختص لمراجعة نتائج الفحص السريرية."
    elif any(k in sev_lower for k in ["متوسط", "medium", "معتدل"]):
        res["when_to_see_doctor"] = "📅 يُفضل استشارة طبيبك المعالج في غضون أيام قليلة للتحقق من الملاحظات الظاهرة."
    else:
        res["when_to_see_doctor"] = "🛡️ يُنصح بعرض النتائج على طبيب الأسرة في الزيارة الدورية القادمة أو في حال ظهور أي أعراض سريرية مقلقة."

    return res


# ── Premium Light Theme CSS Injection ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal;
        font-style: normal;
        font-size: 20px;
        display: inline-block;
        line-height: 1;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: normal;
        white-space: nowrap;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stSidebarNav"] {display: none;}

    /* ═══ Base Healthcare Light Mode ═══ */
    .stApp {
        background-color: #f0fdfa;
        background-image: 
            radial-gradient(circle at 5% 15%, rgba(8, 145, 178, 0.03), transparent 40%),
            radial-gradient(circle at 95% 85%, rgba(13, 148, 136, 0.03), transparent 40%);
        color: #134E4A;
    }
    
    @keyframes ordFadeIn {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ═══ Header Styles ═══ */
    .lab-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 2.2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .lab-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e293b;
        margin: 0 0 10px 0;
        font-family: 'Cairo', sans-serif;
    }
    .lab-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin: 0 0 1.5rem 0;
        font-family: 'Cairo', sans-serif;
    }
    
    /* ═══ Trust Indicators ═══ */
    .trust-container {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    .trust-badge {
        background: rgba(8, 145, 178, 0.05);
        border: 1px solid rgba(8, 145, 178, 0.15);
        color: #0891B2;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.88rem;
        font-weight: 600;
        font-family: 'Cairo', sans-serif;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* ═══ Glassmorphic Components ═══ */
    .upload-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .result-section {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        animation: ordFadeIn 0.5s ease both;
    }
    .result-section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.8rem;
        font-family: 'Cairo', sans-serif;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .result-content {
        font-size: 0.95rem;
        color: #334155;
        line-height: 1.8;
        font-family: 'Cairo', sans-serif;
        text-align: right;
        direction: rtl;
    }

    /* ═══ Sidebar custom styling ═══ */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-left: 1px solid #e2e8f0 !important;
        backdrop-filter: none;
    }

    /* ═══ Buttons ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #0891B2, #0e7490) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        font-family: 'Cairo', sans-serif !important;
        min-height: 50px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(8, 145, 178, 0.25) !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0e7490, #155E75) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(8, 145, 178, 0.4) !important;
    }
    
    /* ═══ Chat Message Glassmorphism ═══ */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-bottom: 0.8rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }
</style>
<script>
const _shifaInitLocal = () => {
    const doc = window.parent.document || document;
    doc.querySelectorAll('button').forEach(btn => {
        if (btn.textContent.trim() === 'Browse files') btn.textContent = 'تصفح الملفات';
    });
    doc.querySelectorAll('p, span, div, small, label').forEach(el => {
        if (el.childElementCount === 0) {
            let t = el.textContent.trim();
            if (t === 'Drag and drop file here') el.textContent = 'اسحب وأفلت الملف هنا';
            if (t.startsWith('Limit') || (t.includes('Limit') && t.includes('per file'))) {
                el.textContent = 'الحد الأقصى 10 ميغا • JPG, JPEG, PNG, WEBP';
            }
        }
    });
};
const _obsLocal = new MutationObserver(_shifaInitLocal);
_obsLocal.observe(window.parent.document.body || document.body, {childList: true, subtree: true});
setTimeout(_shifaInitLocal, 500);
setTimeout(_shifaInitLocal, 2000);
</script>
""", unsafe_allow_html=True)

# ── Clean Sidebar Setup ──
with st.sidebar:
    st.page_link("app.py", label="الرئيسية", icon="🏠")
    st.page_link("pages/06_🦩_المساعد_متعدد_الوسائط.py", label="تحليل الصور الطبية", icon="🖼️", disabled=True)
    st.markdown("---")
    
    # New chat button replacing technical clean button
    if st.button("🗑️ محادثة جديدة", width="stretch", key="new_chat"):
        st.session_state.vision_history = []
        st.rerun()
        
    st.markdown("---")
    
    # Move account settings to a user menu
    _current_user = st.session_state.get("_user", {})
    _uname = _current_user.get("full_name", "مستخدم")
    _role = _current_user.get("role", "guest")
    
    with st.expander(f"👤 {_uname}", expanded=False):
        role_label = "حساب زائر" if _role == "guest" else "حساب طبيب" if _role == "doctor" else "مستخدم مسجل"
        st.markdown(f"<div style='font-size:0.85rem; color:#475569; font-family:\"Cairo\"; margin-bottom:8px;'>النوع: {role_label}</div>", unsafe_allow_html=True)
        if st.button("🚪 تسجيل الخروج", width="stretch", key="sidebar_logout"):
            for k in ["_authenticated", "_user", "_is_guest", "messages", "local_history", "vision_history"]:
                st.session_state.pop(k, None)
            st.switch_page("app.py")


# ── Header Section ──
st.markdown("""
<div class="lab-header">
    <p class="lab-title">🔬 تحليل الصور الطبية</p>
    <p class="lab-subtitle">ارفع صورة طبية للحصول على تحليل أولي مدعوم بالذكاء الاصطناعي</p>
    <div class="trust-container">
        <div class="trust-badge"><span class="material-symbols-rounded" style="font-size:16px;">privacy_tip</span>حماية الخصوصية</div>
        <div class="trust-badge"><span class="material-symbols-rounded" style="font-size:16px;">verified_user</span>تحليل آمن</div>
        <div class="trust-badge"><span class="material-symbols-rounded" style="font-size:16px;">format_list_bulleted</span>نتائج منظمة</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Vision Analyzer Initialization (Zero technical log displayed to user) ──
if "vision_analyzer" not in st.session_state:
    st.session_state.vision_analyzer = GroqVision()

vision_analyzer = st.session_state.vision_analyzer

if not vision_analyzer.client:
    st.error("❌ خدمة التحليل الطبي غير متاحة حالياً. يرجى مراجعة المشرف على النظام.")
    st.stop()

if "vision_history" not in st.session_state:
    st.session_state.vision_history = []


# ── Layout: Upload & Configuration ──
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'Cairo\'; font-weight:700; color:#1e293b; margin:0 0 10px 0; display:flex; align-items:center; gap:6px;"><span class="material-symbols-rounded" style="color:#0891B2;">cloud_upload</span> 📷 ارفع صورة طبية</p>', unsafe_allow_html=True)

    # Large styled dropzone uploader wrapper
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    uploaded_img = st.file_uploader(
        "ارفع صورة طبية",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="vision_upload",
        help="اسحب الصورة هنا أو اضغط للاختيار · الصيغ المدعومة: JPG • PNG • WEBP (الحد الأقصى: 10 ميغابايت)"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_img:
        st.image(uploaded_img, width="stretch", caption="الصورة المرفوعة للفحص")

        # Type selector with clean Arabic medical labels
        st.markdown('<p style="font-family:\'Cairo\'; font-weight:700; color:#1e293b; margin:16px 0 10px 0; display:flex; align-items:center; gap:6px;"><span class="material-symbols-rounded" style="color:#0891B2;">category</span> حدد نوع الفحص الطبي</p>', unsafe_allow_html=True)
        image_type = st.selectbox(
            "نوع الصورة",
            options=["auto", "xray", "mri", "dermato", "eye", "dental", "general"],
            format_func=lambda x: {
                "auto":    "🤖 تحديد تلقائي لنوع الفحص",
                "xray":    "🩻 تصوير بالأشعة السينية (X-Ray)",
                "mri":     "🧠 الرنين المغناطيسي (MRI)",
                "dermato": "🩹 فحص طب الأمراض الجلدية",
                "eye":     "👁️ فحص شبكية وقاع العين",
                "dental":  "🦷 فحص أشعة الأسنان",
                "general": "📷 صورة طبية عامة",
            }[x],
            label_visibility="collapsed"
        )

        if st.button("🔬 بدء الفحص والتحليل الطبي الذكي", key="auto_analyze"):
            with st.status("🔬 يجري الآن فحص وتحليل الصورة طبياً...", expanded=True) as status_box:
                try:
                    st.write("⏳ جاري تهيئة الصورة وقراءتها رقمياً...")
                    bytes_data = uploaded_img.getvalue()
                    b64 = base64.b64encode(bytes_data).decode("utf-8")

                    detected_type = image_type
                    if image_type == "auto":
                        st.write("🔍 جاري التعرف التلقائي على نوع الصورة...")
                        detected_type = vision_analyzer.detect_image_type(b64)

                    st.write("🧠 جاري مطابقة الأنماط الحيوية وتوليد التقييم الطبي الأول...")
                    prompt = ""
                    answer = vision_analyzer.analyze_image(
                        base64_image=b64,
                        prompt=prompt,
                        image_type=detected_type,
                    )

                    label_map = {
                        "xray": "أشعة سينية", "mri": "رنين مغناطيسي",
                        "dermato": "جلدية", "eye": "عيون",
                        "dental": "أسنان", "general": "طبية عامة",
                        "non_medical": "غير طبية", "auto": "تلقائي"
                    }
                    user_msg = f"تحليل الصورة الطبية — [{label_map.get(detected_type, detected_type)}]"
                    st.session_state.vision_history.append({"role": "user", "content": user_msg})
                    st.session_state.vision_history.append({"role": "assistant", "content": answer})
                    
                    status_box.update(label="✅ اكتمل فحص الصورة الطبية!", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status_box.update(label="❌ فشل تحليل الصورة الطبية", state="error", expanded=True)
                    st.error(f"مشكلة برمجية أثناء التحليل: {e}")
                    logger.error(f"Vision error: {e}", exc_info=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Layout: Structured Report & Conversation ──
with col_right:
    # ── Empty State / Visual Illustration ──
    if not st.session_state.vision_history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1.5rem; background: #ffffff; border-radius: 20px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); animation: ordFadeIn 0.5s ease both;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#0891B2" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="width: 80px; height: 80px; margin-bottom: 1.5rem; opacity: 0.85;">
                <path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9" />
                <path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5" />
                <circle cx="12" cy="12" r="2" fill="#0891B2" />
                <path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5" />
                <path d="M19.1 4.9C23 8.8 23 15.2 19.1 19.1" />
                <path d="M12 2v2M12 20v2M2 12h2M20 12h2" />
            </svg>
            <p style="font-size: 1.25rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem; font-family: 'Cairo', sans-serif;">يرجى رفع صورة طبية لبدء الفحص</p>
            <p style="font-size: 0.9rem; color: #64748b; max-width: 400px; margin: 0 auto; line-height: 1.6; font-family: 'Cairo', sans-serif;">ادمج صور الأشعة السينية (X-Ray)، الرنين المغناطيسي (MRI)، أو الفحوصات الجلدية للحصول على قراءة استرشادية أولية منظمة ومحمية بالكامل.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Separate the main initial report from follow-up chat
        analysis_message = None
        chat_messages = []
        for msg in st.session_state.vision_history:
            if msg["role"] == "assistant" and analysis_message is None:
                analysis_message = msg["content"]
            else:
                chat_messages.append(msg)
        
        # Display the parsed structured report panels
        if analysis_message:
            report = parse_vision_response(analysis_message)
            
            # Determine severity badge style
            sev_clean = report["severity"].strip()
            sev_badge = f'<span style="background:rgba(16, 185, 129, 0.15); color:#059669; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:700;">{sev_clean}</span>'
            if any(k in sev_clean.lower() for k in ["حرج", "critical", "حرج جدا", "حرج جداً"]):
                sev_badge = f'<span style="background:rgba(239, 68, 68, 0.15); color:#dc2626; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:700;">🚨 حرج جداً</span>'
            elif any(k in sev_clean.lower() for k in ["مرتفع", "high", "عالية", "عالي"]):
                sev_badge = f'<span style="background:rgba(249, 115, 22, 0.15); color:#d97706; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:700;">⚠️ خطورة عالية</span>'
            elif any(k in sev_clean.lower() for k in ["متوسط", "medium", "معتدل"]):
                sev_badge = f'<span style="background:rgba(234, 179, 8, 0.15); color:#b45309; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:700;">خطورة متوسطة</span>'

            # Panel 1: Analysis Summary
            st.markdown(f"""
            <div class="result-section">
                <div class="result-section-title">
                    <span class="material-symbols-rounded" style="color:#0891B2;">radiology</span>
                    🩻 ملخص التحليل
                </div>
                <div class="result-content">
                    <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
                        <span><b>نوع الفحص المكتشف:</b> {report["type"]}</span>
                        <span><b>مستوى الخطورة:</b> {sev_badge}</span>
                    </div>
                    <div style="margin-top: 10px;">
                        <b>التشخيص الأولي المحتمل:</b><br/>
                        {report["diagnosis"]}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Panel 2: Potential Observations
            st.markdown(f"""
            <div class="result-section">
                <div class="result-section-title">
                    <span class="material-symbols-rounded" style="color:#0891B2;">visibility</span>
                    🔍 الملاحظات المحتملة
                </div>
                <div class="result-content">
                    {report["observations"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Panel 3: Recommendations
            st.markdown(f"""
            <div class="result-section">
                <div class="result-section-title">
                    <span class="material-symbols-rounded" style="color:#0891B2;">assignment</span>
                    📋 التوصيات
                </div>
                <div class="result-content">
                    {report["recommendation"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Panel 4: Emergency Guidelines
            st.markdown(f"""
            <div class="result-section" style="border: 1px solid #fee2e2; border-right: 4px solid #ef4444; background: #fef2f2;">
                <div class="result-section-title" style="color:#dc2626;">
                    <span class="material-symbols-rounded" style="color:#ef4444;">error_outline</span>
                    ⚠️ متى يجب مراجعة الطبيب
                </div>
                <div class="result-content" style="color:#991b1b;">
                    {report["when_to_see_doctor"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Panel 5: Conversation Header & Thread
            st.markdown("<h4 style='font-family:\"Cairo\"; color:#1e293b; margin: 2rem 0 1rem; display:flex; align-items:center; gap:8px;'><span class='material-symbols-rounded' style='color:#0891B2;'>question_answer</span> ❓ أسئلة إضافية</h4>", unsafe_allow_html=True)
            
            # Print conversation threads
            for msg in chat_messages:
                avatar = "👤" if msg["role"] == "user" else "🩺"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])
                    
    # Chat input for follow-up questions
    if prompt := st.chat_input("اسأل سؤالاً إضافياً حول الصورة..."):
        if not uploaded_img:
            st.warning("⚠️ يرجى تحميل صورة أولاً.")
        else:
            st.session_state.vision_history.append({"role": "user", "content": prompt})
            with st.spinner("جاري التقييم الطبي..."):
                bytes_data = uploaded_img.getvalue()
                b64 = base64.b64encode(bytes_data).decode("utf-8")
                answer = vision_analyzer.analyze_image(
                    base64_image=b64,
                    prompt=prompt,
                    image_type="auto",
                    history=st.session_state.vision_history[:-1],
                )
            st.session_state.vision_history.append({"role": "assistant", "content": answer})
            st.rerun()

# ── Footer Medical Disclaimer ──
st.markdown("""
<hr style="opacity:0.08; margin-top:3rem;"/>
<div style="text-align:center; padding: 1.5rem; color:#475569; font-size:0.8rem; font-family:'Cairo',sans-serif; direction:rtl;">
    <b>⚠️ تنبيه قانوني هام:</b> جميع تقارير الاستشارات والصور الطبية الصادرة عن النظام هي تحليلات آلية تهدف للتعليم والتثقيف الصحي فقط. لا تعتبر هذه النتائج تشخيصاً طبياً نهائياً، ولا يجب استعمالها لوصف أو تعديل العلاجات دون مراجعة طبية مباشرة مع المهنيين الصحيين المؤهلين.
</div>
""", unsafe_allow_html=True)
