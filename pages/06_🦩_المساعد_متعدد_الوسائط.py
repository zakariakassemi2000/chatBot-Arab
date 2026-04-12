import streamlit as st
import sys
import os
import base64
from PIL import Image as PILImage

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.llm import GroqVision

st.set_page_config(page_title="معمل تحليل الصور الطبية", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebarNav"] {display: none;}

    /* ── Dark theme ── */
    .stApp { background-color: #0f1117; }

    .lab-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid rgba(220, 38, 38, 0.3);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        text-align: center;
    }
    .lab-title {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 6px 0;
        font-family: 'Segoe UI', sans-serif;
    }
    .lab-subtitle {
        font-size: 14px;
        color: #9ca3af;
        margin: 0;
    }
    .model-badge {
        display: inline-block;
        background: rgba(220, 38, 38, 0.15);
        border: 1px solid rgba(220, 38, 38, 0.4);
        color: #f87171;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 10px;
    }
    .upload-card {
        background: #1a1f2e;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .result-card {
        background: #1a1f2e;
        border: 1px solid rgba(220, 38, 38, 0.25);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 16px;
    }
    .result-card h4 {
        color: #f87171;
        font-size: 14px;
        font-weight: 700;
        margin: 0 0 12px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-label {
        color: #6b7280;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .type-chip {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        margin: 4px 2px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #b91c1c, #991b1b) !important;
        transform: translateY(-1px);
    }
    /* Chat */
    [data-testid="stChatMessage"] {
        background: #1a1f2e !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.page_link("app.py", label="الرجوع للرئيسية", icon="🏠")
    st.markdown("---")
    if st.button("🗑️ مسح المحادثة"):
        st.session_state.vision_history = []
        st.rerun()

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="lab-header">
    <p class="lab-title">🔬 معمل تحليل الصور الطبية</p>
    <p class="lab-subtitle">تحليل ذكي للصور الطبية بالذكاء الاصطناعي — أشعة X، رنين مغناطيسي، صور جلدية وأكثر</p>
    <span class="model-badge">⚡ Llama 4 Scout · Groq Vision</span>
</div>
""", unsafe_allow_html=True)

# ── Init analyzer — NO cache, direct instantiation ───────────────────
if "vision_analyzer" not in st.session_state:
    st.session_state.vision_analyzer = GroqVision()

vision_analyzer = st.session_state.vision_analyzer

if not vision_analyzer.client:
    st.error("❌ مفتاح `GROQ_API_KEY` مفقود أو غير صالح. يرجى إضافته في ملف `.env`.")
    st.stop()

# ── Layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📤 رفع الصورة</p>', unsafe_allow_html=True)

    uploaded_img = st.file_uploader(
        "اختر صورة طبية",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_img:
        st.image(uploaded_img, width='stretch', caption="الصورة المرفوعة")

        # Type selector
        st.markdown('<p class="section-label" style="margin-top:14px;">🏷️ نوع الصورة</p>', unsafe_allow_html=True)
        image_type = st.selectbox(
            "نوع الصورة",
            options=["auto", "xray", "mri", "dermato", "eye", "dental", "general"],
            format_func=lambda x: {
                "auto":    "🤖 تلقائي (الذكاء الاصطناعي يحدد)",
                "xray":    "🦴 أشعة X",
                "mri":     "🧠 رنين مغناطيسي (MRI)",
                "dermato": "🩹 صورة جلدية",
                "eye":     "👁️ فحص عيون",
                "dental":  "🦷 أسنان",
                "general": "📷 صورة طبية عامة",
            }[x],
            label_visibility="collapsed"
        )

        if st.button("🔬 تحليل الصورة تلقائياً", key="auto_analyze"):
            with st.spinner("جاري تحليل الصورة بالذكاء الاصطناعي..."):
                bytes_data = uploaded_img.getvalue()
                b64 = base64.b64encode(bytes_data).decode("utf-8")

                # Auto-detect type if needed
                detected_type = image_type
                if image_type == "auto":
                    with st.spinner("جاري تحديد نوع الصورة..."):
                        detected_type = vision_analyzer.detect_image_type(b64)

                prompt = ""  # Empty = use default structured medical prompt
                answer = vision_analyzer.analyze_image(
                    base64_image=b64,
                    prompt=prompt,
                    image_type=detected_type,
                )

                if "vision_history" not in st.session_state:
                    st.session_state.vision_history = []

                label_map = {
                    "xray": "أشعة X", "mri": "رنين مغناطيسي",
                    "dermato": "جلدية", "eye": "عيون",
                    "dental": "أسنان", "general": "طبية عامة",
                    "non_medical": "غير طبية", "auto": "تلقائي"
                }
                user_msg = f"🔬 تحليل تلقائي — [{label_map.get(detected_type, detected_type)}]"
                st.session_state.vision_history.append({"role": "user", "content": user_msg})
                st.session_state.vision_history.append({"role": "assistant", "content": answer})
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<h4>💬 نتائج التحليل والمحادثة</h4>', unsafe_allow_html=True)

    if "vision_history" not in st.session_state:
        st.session_state.vision_history = []

    if not st.session_state.vision_history:
        st.markdown("""
        <div style="text-align:center; padding: 40px 20px; color: #4b5563;">
            <div style="font-size:48px; margin-bottom:12px;">🩻</div>
            <p style="font-size:15px; color:#6b7280;">ارفع صورة طبية واضغط "تحليل الصورة"<br>أو اكتب سؤالك مباشرة بعد رفع الصورة</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.vision_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input for follow-up questions
    if prompt := st.chat_input("اسأل سؤالاً إضافياً عن الصورة..."):
        if not uploaded_img:
            st.warning("⚠️ يرجى تحميل صورة أولاً.")
        else:
            st.session_state.vision_history.append({"role": "user", "content": prompt})
            with st.spinner("جاري التحليل..."):
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
