# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Stable Production Version
═══════════════════════════════════════════════════════════════════════
"""

import os, sys, io, base64, time
import streamlit as st
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# 1️⃣ GLOBAL SAFETY
# ─────────────────────────────────────────────────────────────
st.set_option('client.showErrorDetails', False)
load_dotenv()

# Windows UTF8 Fix
if sys.platform == 'win32' and not os.environ.get('_UTF8_FIX_APPLIED'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        os.environ['_UTF8_FIX_APPLIED'] = '1'
    except:
        pass

# ─────────────────────────────────────────────────────────────
# 2️⃣ IMPORT ENGINES
# ─────────────────────────────────────────────────────────────
from engine.retriever import FAISSRetriever
from engine.classifier import IntentClassifier, format_response
from engine.safety import SafetyGuard
from engine.llm import GroqGenerator
from engine.audio import speech_to_text_arabic, convert_audio_to_wav

# ─────────────────────────────────────────────────────────────
# 3️⃣ PAGE CONFIG
# ─────────────────────────────────────────────────────────────
LOGO_PATH = r"c:\Users\zakar\chatBot Arab\Stylized_Heart_and_Cross_Logo_for_SHIFA_AI__1_-removebg-preview.png"

def get_b64(path):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return None

LOGO_B64 = get_b64(LOGO_PATH)
LOGO_SRC = f"data:image/png;base64,{LOGO_B64}" if LOGO_B64 else ""

def get_logo():
    if os.path.exists(LOGO_PATH):
        return LOGO_PATH
    return "🩺"

st.set_page_config(
    page_title="SHIFA AI | مساعدك الطبي",
    page_icon=get_logo(),
    layout="centered",
)

# ── ADVANCED UI CSS (RED & WHITE THEME) ──
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&display=swap');
    
    /* Global Styles */
    html, body, [class*="st-"] {{
        font-family: 'Cairo', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }}
    
    .stApp {{
        background-color: #FFFFFF !important;
    }}
    
    #MainMenu, footer, header {{
        display: none !important;
    }}
    
    [data-testid="stSidebar"] {{
        display: none !important;
    }}
    
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 850px !important;
    }}

    /* Title & Logo Styling */
    .header-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
        padding: 1rem;
        animation: fadeInDown 0.8s ease-out;
    }}
    
    .logo-img {{
        width: 120px;
        height: auto;
        margin-bottom: 1rem;
        filter: drop-shadow(0 4px 8px rgba(229, 57, 53, 0.2));
    }}
    
    .main-title {{
        color: #E53935;
        font-size: 46px;
        font-weight: 900;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: -1px;
    }}
    
    .sub-title {{
        color: #5A6072;
        font-size: 16px;
        font-weight: 600;
        margin-top: 0.2rem;
    }}

    /* Chat Message Styling */
    [data-testid="stChatMessage"] {{
        background: transparent !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        animation: fadeInUp 0.4s ease-out;
    }}
    
    [data-testid="stChatMessageAvatarContainer"] img, 
    [data-testid="stChatMessageAvatarContainer"] div {{
        border: 2px solid #E53935 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
    }}

    /* Assistant Bubble (White/Red theme) */
    [data-testid="stChatMessage"]:not(:has([data-testid="chatAvatarIcon-user"])) div[data-testid="stChatMessageContent"] {{
        background: #FFFFFF !important;
        border: 1.5px solid #F5F7F9 !important;
        border-right: 4px solid #E53935 !important;
        border-radius: 4px 20px 20px 20px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        padding: 1.2rem !important;
        color: #1E2028 !important;
    }}

    /* User Bubble (Soft Gray/Red tint) */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"] {{
        background: #FEEEEE !important;
        border-right: 4px solid #C62828 !important;
        border-radius: 20px 4px 20px 20px !important;
        padding: 1rem !important;
        color: #1A202C !important;
    }}

    /* Welcome Card */
    .welcome-card {{
        background: white;
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        border: 1px solid #F1F3F5;
        box-shadow: 0 10px 40px rgba(0,0,0,0.03);
        margin-top: 2rem;
        margin-bottom: 2rem;
        direction: rtl;
    }}

    /* Input Area */
    .stChatInputContainer {{
        border: 2px solid #F1F3F5 !important;
        border-radius: 20px !important;
        background: white !important;
        padding: 4px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04) !important;
    }}
    
    .stChatInputContainer:focus-within {{
        border-color: #E53935 !important;
        box-shadow: 0 4px 20px rgba(229, 57, 53, 0.1) !important;
    }}

    /* Mic Button Styling */
    .mic-container {{
        position: fixed;
        bottom: 32px;
        left: 30px;
        z-index: 1000;
        cursor: pointer;
        background: white;
        padding: 12px;
        border-radius: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #F1F3F5;
        transition: all 0.2s ease;
    }}
    
    .mic-container:hover {{
        transform: scale(1.05);
        border-color: #E53935;
    }}

    /* Animations */
    @keyframes fadeInDown {{
        from {{ opacity: 0; transform: translateY(-20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 4️⃣ LOAD SYSTEM
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    try:
        r = FAISSRetriever()
        c = IntentClassifier()
        g = SafetyGuard()

        db_ok = r.load()
        if db_ok:
            c.load()

        return r, c, g, db_ok
    except:
        return None, None, None, False

retriever, classifier, guard, DB_STATUS = load_system()

# ─────────────────────────────────────────────────────────────
# 5️⃣ SESSION STATE
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

# AI Engine relies on environment variable GROQ_API_KEY
ai_engine = GroqGenerator()
AI_STATUS = bool(ai_engine.api_key)

# ─────────────────────────────────────────────────────────────
# 7️⃣ HEADER & LOGO
# ─────────────────────────────────────────────────────────────
logo_tag = f'<img src="{LOGO_SRC}" class="logo-img">' if LOGO_SRC else ''
st.markdown(f"""
    <div class="header-container">
        {logo_tag}
        <h1 class="main-title">SHIFA IA MAROC</h1>
        <p class="sub-title">مساعدك الطبي الذكي - رفيقك الصحي في كل مكان 🩺🇲🇦</p>
    </div>
""", unsafe_allow_html=True)

if not DB_STATUS:
    st.error("⚠️ قاعدة البيانات غير متاحة حالياً.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 8️⃣ WELCOME CARD (If no messages)
# ─────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
        <div class="welcome-card">
            <h2 style="color:#1E2028; font-weight:800; margin-bottom:12px;">مرحباً بك</h2>
            <p style="color:#5A6072; font-size:16px;">أنا هنا للإجابة على جميع استفساراتك الصحية باللغة العربية. كيف يمكنني مساعدتك اليوم؟</p>
            <div style="margin-top:20px; display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
                <span style="background:#FEEEEE; color:#C62828; padding:8px 16px; border-radius:12px; font-size:14px; font-weight:600;">👨‍⚕️ مستشار طبي</span>
                <span style="background:#FEEEEE; color:#C62828; padding:8px 16px; border-radius:12px; font-size:14px; font-weight:600;">🚨 إرشادات طوارئ</span>
                <span style="background:#FEEEEE; color:#C62828; padding:8px 16px; border-radius:12px; font-size:14px; font-weight:600;">🕌 خدمة مغربية</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 9️⃣ DISPLAY HISTORY
# ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=get_logo() if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────────────────────
# 9️⃣ INPUTS
# ─────────────────────────────────────────────────────────────
user_q = None

# Voice (Safe)
try:
    from audio_recorder_streamlit import audio_recorder
    st.markdown('<div class="mic-container">', unsafe_allow_html=True)
    raw_audio = audio_recorder(
        text="",
        recording_color="#E53935",
        neutral_color="#94A3B8",
        icon_size="1.2rem",
        key="mic"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if raw_audio and len(raw_audio) > 1000:
        with st.spinner("⏳ جاري تحويل الصوت..."):
            wav = convert_audio_to_wav(raw_audio, src_format="wav")
            text, err = speech_to_text_arabic(wav)
            if text:
                user_q = text
except:
    pass

# Text
typed = st.chat_input("اكتب استفسارك هنا...")
if typed:
    user_q = typed

# ─────────────────────────────────────────────────────────────
# 🔟 ADD USER MESSAGE
# ─────────────────────────────────────────────────────────────
if user_q and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.messages.append({
        "role": "user",
        "content": user_q.strip()
    })
    st.rerun()

# ─────────────────────────────────────────────────────────────
# 1️⃣1️⃣ GENERATE RESPONSE
# ─────────────────────────────────────────────────────────────
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and st.session_state.processing
):

    q = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant", avatar=get_logo()):
        placeholder = st.empty()
        placeholder.markdown("⏳ جاري التحليل...")

    try:
        # Safety check
        safe = guard.check(q)

        if safe["level"] in ("emergency", "boundary"):
            answer = safe["override_response"]

        else:
            enc = retriever.encode_query(q)

            try:
                intent, _ = classifier.predict(enc)
            except:
                intent = "general"

            res = retriever.get_best_answer(q)

            if res:
                base, _, _, _ = res

                if AI_STATUS:
                    try:
                        llm = ai_engine.generate_answer(q, base, intent)
                        answer = llm if llm else format_response(base, intent)
                    except:
                        answer = format_response(base, intent)
                else:
                    answer = format_response(base, intent)

                if safe["level"] == "caution":
                    answer = guard.format_caution_response(answer)

                answer = guard.add_disclaimer(answer)

            else:
                answer = "عذراً، لا أملك إجابة دقيقة حالياً. يرجى استشارة طبيب مختص."

    except Exception:
        answer = "⚠️ حدث خطأ غير متوقع. حاول مرة أخرى."

    placeholder.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.processing = False