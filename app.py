# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Enhanced Medical Platform
═══════════════════════════════════════════════════════════════════════
"""

import os, sys, io, base64, time, json
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import PIL.Image as PILImage
from utils.logger import get_logger

logger = get_logger("shifa.app")

# ─────────────────────────────────────────────────────────────
# GLOBAL SAFETY & CONFIG
# ─────────────────────────────────────────────────────────────
st.set_option('client.showErrorDetails', False)
load_dotenv()

# Windows UTF8 Fix
if sys.platform == 'win32' and not os.environ.get('_UTF8_FIX_APPLIED'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        os.environ['_UTF8_FIX_APPLIED'] = '1'
    except Exception as e:
        logger.error(f"[AppError] {e}")
        st.error("حدث خطأ غير متوقع — يرجى إعادة المحاولة")

# ─────────────────────────────────────────────────────────────
# IMPORT ENGINES
# ─────────────────────────────────────────────────────────────
from engine.retriever import FAISSRetriever
from engine.classifier import IntentClassifier, format_response
from engine.safety import SafetyGuard
from engine.llm import GroqGenerator
from engine.audio import speech_to_text_arabic, convert_audio_to_wav
from engine.vision_router import VisionRouter
from utils.gradcam import generate_gradcam_heatmap, is_available as gradcam_available
from utils.image_validator import validate_medical_image



# ─────────────────────────────────────────────────────────────
# CONSTANTS & UTILS
# ─────────────────────────────────────────────────────────────
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Stylized_Heart_and_Cross_Logo_for_SHIFA_AI__1_-removebg-preview.png")
HISTORY_FILE = "consultation_history.json"

def get_b64(path):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception as e:
        logger.error(f"[AppError] {e}")
        st.error("حدث خطأ غير متوقع — يرجى إعادة المحاولة")
    return None

LOGO_B64 = get_b64(LOGO_PATH)
LOGO_SRC = f"data:image/png;base64,{LOGO_B64}" if LOGO_B64 else ""

def get_logo():
    if os.path.exists(LOGO_PATH):
        return LOGO_PATH
    return None

def save_history(messages):
    if not messages: return
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            logger.error(f"[AppError] {e}")
            st.error("حدث خطأ غير متوقع — يرجى إعادة المحاولة")
    
    # Check if last session is already saved
    session_id = str(st.session_state.get("session_id", time.time()))
    existing = next((idx for idx, s in enumerate(history) if s.get("id") == session_id), None)
    
    entry = {
        "id": session_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": messages[0]["content"][:50] + "...",
        "messages": messages
    }
    
    if existing is not None:
        history[existing] = entry
    else:
        history.insert(0, entry)
        
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[:50], f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHIFA AI | Medical Intelligence Platform",
    page_icon=get_logo(),
    layout="centered",
)

# ── ADVANCED UI CSS ──
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Alexandria:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {{
        font-family: 'Alexandria', 'Outfit', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }}

    /* Global Background */
    .stApp {{
        background: linear-gradient(120deg, #0B0E14 0%, #151A28 100%) !important;
        background-attachment: fixed !important;
        color: #E2E8F0 !important;
    }}
    
    .stApp::before {{
        content: "";
        position: fixed;
        top: -200px;
        right: -200px;
        width: 600px;
        height: 600px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(229, 57, 53, 0.15) 0%, rgba(0,0,0,0) 70%);
        z-index: 0;
        pointer-events: none;
    }}

    .stApp::after {{
        content: "";
        position: fixed;
        bottom: -200px;
        left: -100px;
        width: 500px;
        height: 500px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.1) 0%, rgba(0,0,0,0) 70%);
        z-index: 0;
        pointer-events: none;
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: rgba(16, 23, 42, 0.6) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: -10px 0 30px rgba(0,0,0,0.5) !important;
    }}

    [data-testid="stSidebarNav"] {{
        display: none !important;
    }}

    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton > button {{
        color: #94A3B8 !important;
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        font-weight: 500 !important;
        padding: 12px !important;
        justify-content: flex-start !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(229,57,53,0.15) !important;
        border-color: rgba(229,57,53,0.4) !important;
        color: #FFF !important;
        transform: scale(1.02) translateX(-4px) !important;
        box-shadow: 0 4px 15px rgba(229,57,53,0.2) !important;
    }}
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #E53935 0%, #ef4444 100%) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
        box-shadow: 0 8px 25px rgba(229,57,53,0.3) !important;
        font-weight: 600 !important;
    }}
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {{
        color: #e2e8f0 !important;
    }}
    
    /* ── Typography & Cards ── */
    .block-container {{
        padding-top: 2.5rem !important;
        padding-bottom: 6rem !important;
        max-width: 900px !important;
        z-index: 1;
        position: relative;
    }}
    
    .main-title {{
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        text-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    
    .welcome-card, .data-card {{
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 30px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 24px;
        transition: transform 0.4s ease, box-shadow 0.4s ease;
        color: #E2E8F0 !important;
    }}
    .welcome-card:hover, .data-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.12);
    }}

    /* ── Suggestions ── */
    .suggestion-btn {{
        display: inline-block;
        background: rgba(15, 23, 42, 0.6);
        color: #94A3B8;
        padding: 10px 20px;
        border-radius: 20px;
        margin: 5px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.3s ease;
        backdrop-filter: blur(8px);
    }}
    .suggestion-btn:hover {{
        background: linear-gradient(135deg, rgba(229,57,53,0.8), rgba(198,40,40,0.8));
        color: white;
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 20px rgba(229,57,53,0.3);
        border-color: rgba(255,255,255,0.2);
    }}

    /* ── Stats badge ── */
    .stat-badge {{
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 14px 18px;
        margin: 8px 0;
        text-align: center;
        transition: all 0.3s ease;
    }}
    .stat-badge:hover {{
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(255,255,255,0.1);
    }}
    .stat-badge .stat-num {{
        font-size: 26px;
        font-weight: 800;
        color: #E53935;
        display: block;
        text-shadow: 0 2px 10px rgba(229,57,53,0.3);
    }}
    .stat-badge .stat-label {{
        font-size: 12px;
        color: #94A3B8;
        font-weight: 500;
    }}

    /* ── Chat ── */
    [data-testid="stChatMessage"] {{
        animation: slideUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        background: transparent !important;
        padding: 1rem !important;
    }}
    
    [data-testid="stChatMessageAvatarContainer"] img,
    [data-testid="stChatMessageAvatarContainer"] div {{
        border: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        background: #1E293B !important;
        color: #F8FAFC !important;
    }}
    
    [data-testid="stChatMessage"]:not(:has([data-testid="chatAvatarIcon-user"])) div[data-testid="stChatMessageContent"] {{
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        border-right: 3px solid #E53935 !important;
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
        border-left: 1px solid rgba(255,255,255,0.05) !important;
        border-top: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 8px 24px 24px 24px !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.2) !important;
        color: #E2E8F0 !important;
        padding: 1.2rem !important;
    }}
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"] {{
        background: linear-gradient(135deg, rgba(229,57,53,0.15) 0%, rgba(229,57,53,0.05) 100%) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(229,57,53,0.2) !important;
        border-radius: 24px 8px 24px 24px !important;
        color: #F8FAFC !important;
        padding: 1.2rem !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15) !important;
    }}
    
    [data-testid="stChatInput"] {{
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 24px !important;
        padding-bottom: 5px !important;
        box-shadow: 0 -10px 40px rgba(0,0,0,0.3) !important;
    }}
    
    .stChatInput textarea {{
        background: transparent !important;
        color: #F8FAFC !important;
        border: none !important;
        font-family: 'Alexandria', sans-serif !important;
    }}
    .stChatInput textarea:focus {{
        box-shadow: none !important;
    }}

    /* ── Animations ── */
    @keyframes slideUpFade {{
        from {{ opacity: 0; transform: translateY(20px); filter: blur(5px); }}
        to   {{ opacity: 1; transform: translateY(0); filter: blur(0); }}
    }}
    @keyframes pulseGlow {{
        0%, 100% {{ opacity: 1; box-shadow: 0 0 15px rgba(229,57,53,0.3); }}
        50%      {{ opacity: 0.6; box-shadow: 0 0 5px rgba(229,57,53,0.1); }}
    }}
    .typing-indicator {{ 
        display: inline-block;
        padding: 5px 15px;
        background: rgba(229,57,53,0.1);
        border-radius: 20px;
        color: #ff8a80;
        font-weight: 500;
        animation: pulseGlow 1.5s infinite; 
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent !important;
        gap: 8px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px 12px 0 0 !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(229,57,53,0.1) !important;
        color: #E53935 !important;
        border-top: 2px solid #E53935 !important;
        border-bottom: none !important;
    }}
    
    /* Texts and Elements within dark mode */
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: inherit;
    }}
    
    .stMarkdown p {{
        color: #CBD5E1 !important;
        line-height: 1.7 !important;
        font-size: 1.05rem !important;
    }}
    
    [data-testid="stAlert"] {{
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        color: #E2E8F0 !important;
    }}
    [data-testid="stAlert"] div[role="alert"] {{
        color: #E2E8F0 !important;
    }}
    
    /* Inputs */
    .stNumberInput input, .stTextInput input, .stTextArea textarea, .stSelectbox select {{
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
    }}
    .stNumberInput input:focus, .stTextInput input:focus, .stTextArea textarea:focus {{
        border-color: #E53935 !important;
        box-shadow: 0 0 0 2px rgba(229,57,53,0.2) !important;
    }}
    
    /* Expander */
    [data-testid="stExpander"] {{
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 16px !important;
    }}
    [data-testid="stExpander"] details {{
        border-color: transparent !important;
    }}
    [data-testid="stExpander"] summary {{
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }}
    [data-testid="stExpander"] p {{
        color: #94A3B8 !important;
    }}

    /* Landing Page Specific */
    .shifa-title {{
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #E53935, #ff6b6b, #E53935);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        margin-bottom: 0.5rem;
    }}
    @keyframes shine {{
        to {{ background-position: 200% center; }}
    }}
    .tagline {{
        text-align: center;
        color: #94A3B8;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }}
    .landing-card {{
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }}
    .landing-card:hover {{
        transform: translateY(-8px);
        background: rgba(229, 57, 53, 0.1);
        border-color: rgba(229, 57, 53, 0.3);
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
    }}
    .card-icon {{
        font-size: 2.5rem;
        margin-bottom: 10px;
    }}
    .card-title {{
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.1rem;
    }}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD SYSTEM
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_medical_system():
    try:
        r = FAISSRetriever()
        c = IntentClassifier()
        g = SafetyGuard()
        ai = GroqGenerator()
        v = VisionRouter()
        db_ok = r.load()
        if db_ok: c.load()
        return r, c, g, ai, v, db_ok
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None, None, None, None, None, False

retriever, classifier, guard, ai_engine, vision_router, DB_STATUS = load_medical_system()

AI_STATUS = bool(ai_engine and ai_engine.api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "home"
if "session_id" not in st.session_state:
    st.session_state.session_id = str(time.time())
if "quick_question" not in st.session_state:
    st.session_state.quick_question = None
if "tts_last" not in st.session_state:
    st.session_state.tts_last = None

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    logo_img = f'<img src="{LOGO_SRC}" style="width:75px; filter:drop-shadow(0 4px 12px rgba(229,57,53,0.5));">' if LOGO_SRC else ''
    st.markdown(f"""
        <div style="text-align:center; padding: 1rem 0 1.5rem;">
            {logo_img}
            <h2 style="color:#ff6b6b; margin:8px 0 2px; font-size:22px; letter-spacing:2px;">SHIFA AI</h2>
            <p style="color:#90a4ae; font-size:12px; margin:0;">المنصة الطبية الذكية</p>
        </div>
    """, unsafe_allow_html=True)

    pages = [
        ("home",            "🏠", "الرئيسية"),
        ("chat",            "💬", "المحادثة الطبية"),
        ("vision",          "🔬", "تحليل الصور"),
        ("scanner",         "🩺", "فاحص الأعراض"),
        ("calculators",     "📊", "حاسبات طبية"),
        ("database",        "📚", "قاعدة المعرفة"),
        ("history",         "📜", "سجل الاستشارات"),
    ]
    for page_id, icon, label in pages:
        is_active = st.session_state.page == page_id
        if st.button(f"{icon} {label}", use_container_width=True,
                     type="primary" if is_active else "secondary",
                     key=f"nav_{page_id}"):
            st.session_state.page = page_id
            st.rerun()

    st.markdown("---")

    # ── Session Stats ──
    n_msgs = len(st.session_state.messages)
    n_user = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.markdown(f"""
        <div class="stat-badge">
            <span class="stat-num">{n_user}</span>
            <span class="stat-label">سؤال في هذه الجلسة</span>
        </div>
        <div class="stat-badge">
            <span class="stat-num" style="color: {'#4CAF50' if AI_STATUS else '#ef5350'};">\u25cf</span>
            <span class="stat-label">{'AI Engine: Connected' if AI_STATUS else 'AI Engine: Offline'}</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    use_whisper = os.getenv("USE_WHISPER", "").strip().lower() in ("1", "true", "yes")
    st.caption(f"Voice Input: {'Whisper' if use_whisper else 'Google'}")
    st.markdown("---")
    if st.session_state.page == "chat" and st.session_state.messages:
        if st.button("مسح المحادثة", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.quick_question = None
            st.rerun()

    st.info("للمعلومات الطبية فقط. لا يُغني عن استشارة الطبيب.")

# ─────────────────────────────────────────────────────────────
# PAGE: HOME (LANDING)
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown('<div class="shifa-title">شفاء AI</div>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">مساعدك الطبي الذكي — تشخيص، تحليل، إرشاد</p>', unsafe_allow_html=True)
    
    if LOGO_SRC:
        st.markdown(f'<div style="text-align:center; margin-bottom:3rem;"><img src="{LOGO_SRC}" style="width:180px; filter:drop-shadow(0 10px 20px rgba(229,57,53,0.3));"></div>', unsafe_allow_html=True)
    
    # ── Landing Cards ──
    cols = st.columns(5)
    
    cards = [
        ("🤖 محادثة طبية", "chat"),
        ("🔬 تحليل الصور", "vision"),
        ("🎙️ المساعد الصوتي", "chat"),
        ("💊 فحص الأدوية", "scanner"), # Using scanner as drug/symptom check
        ("📊 الحاسبات الطبية", "calculators")
    ]
    
    for i, (title, target) in enumerate(cards):
        with cols[i]:
            st.markdown(f"""
                <div class="landing-card">
                    <div class="card-title">{title}</div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("دخول", key=f"btn_home_{i}", use_container_width=True):
                st.session_state.page = target
                st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("💡 SHIFA AI يجمع بين قوة الذكاء الاصطناعي وخبرة قواعد البيانات الطبية الضخمة لتقديم أدق النتائج.")

# ─────────────────────────────────────────────────────────────
# PAGE: CHAT (MAIN)
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "chat":
    logo_tag = f'<img src="{LOGO_SRC}" style="width: 60px; margin-bottom: 10px;">' if LOGO_SRC else ''
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            {logo_tag}
            <h1 class="main-title">مساعدك الطبي الذكي</h1>
            <p style="color: #5A6072;">تحدث معي باللغة العربية حول أي موضوع صحي</p>
        </div>
    """, unsafe_allow_html=True)

    if not DB_STATUS:
        st.error("قاعدة المعرفة غير متاحة حالياً. يرجى إعداد النظام أولاً.")
        st.info(
            "يمكنك إعداد قاعدة المعرفة الآن (قد يستغرق وقتاً ويتطلب إنترنت)، "
            "أو تشغيل التطبيق بعد نسخ مجلد `models/` الصحيح."
        )

        with st.expander("إعداد قاعدة المعرفة"):
            max_samples = st.slider("حجم البيانات (max_samples)", 1000, 12000, 8000, step=500)
            st.caption("سيتم تنزيل بيانات طبية عربية من Hugging Face ثم بناء FAISS وتدريب مصنّف النوايا.")

            if st.button("بدء الإعداد", type="primary", use_container_width=True):
                try:
                    from data.knowledge_base import load_and_prepare_datasets
                    from engine.retriever import FAISSRetriever
                    from engine.classifier import IntentClassifier

                    with st.spinner("جاري تنزيل وتجهيز البيانات..."):
                        df = load_and_prepare_datasets(max_samples=max_samples)

                    with st.spinner("جاري بناء فهرس البحث..."):
                        r = FAISSRetriever()
                        embeddings = r.build_index(df, verbose=False)
                        r.save()

                    with st.spinner("جاري تدريب المصنّف..."):
                        c = IntentClassifier()
                        c.train(embeddings, df["intent"].tolist(), verbose=False)
                        c.save()

                    st.success("تم إعداد قاعدة المعرفة بنجاح.")
                    st.rerun()
                except Exception as e:
                    logger.error("KB setup failed", exc_info=True)
                    st.error("فشل الإعداد. يرجى التحقق من الاتصال بالإنترنت والمحاولة مرة أخرى.")

        st.stop()

    # ── Quick suggestions when chat is empty ──
    QUICK_QUESTIONS = [
        "ما هي أعراض نزلة البرد؟",
        "كيف أخفض ضغط الدم؟",
        "ما هي فوائد الكركم؟",
        "متى أذهب للطوارئ؟",
        "ما أسباب الصداع المتكرر؟",
        "كيف أحسّن نوعية النوم؟",
    ]

    if not st.session_state.messages:
        st.markdown("""
            <div class="welcome-card">
                <h3 style="color:#1E2028; font-weight:800; margin-bottom:8px;">كيف يمكنني مساعدتك اليوم؟</h3>
                <p style="color:#5A6072; margin-bottom:16px;">اسألني عن الأعراض، الأدوية، التغذية، أو أي موضوع صحي.</p>
                <p style="color:#888; font-size:13px; font-weight:600;">أسئلة سريعة:</p>
                <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:8px;">
        """, unsafe_allow_html=True)
        cols_q = st.columns(3)
        for i, q in enumerate(QUICK_QUESTIONS):
            with cols_q[i % 3]:
                if st.button(q, key=f"quick_{i}", use_container_width=True):
                    st.session_state.quick_question = q
                    st.rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Display Chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar=get_logo() if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])
            # TTS button for assistant messages
            if msg["role"] == "assistant":
                if st.button("🔊 استماع", key=f"tts_{i}", help="تشغيل الإجابة صوتياً"):
                    try:
                        from engine.audio import text_to_speech_arabic
                        audio_bytes = text_to_speech_arabic(msg["content"])
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                    except Exception:
                        logger.warning("TTS failed", exc_info=True)

    # Merge quick_question into user_q
    user_q = st.chat_input("اكتب استفسارك هنا...")
    if st.session_state.quick_question:
        user_q = st.session_state.quick_question
        st.session_state.quick_question = None

    # Voice input
    try:
        from audio_recorder_streamlit import audio_recorder
        cols = st.columns([0.08, 0.92])
        with cols[0]:
            raw_audio = audio_recorder(text="", recording_color="#E53935",
                                       neutral_color="#94A3B8", icon_size="1.2rem",
                                       key="mic_chat")
            if raw_audio and len(raw_audio) > 1000:
                with st.spinner("جاري معالجة الصوت..."):
                    wav = convert_audio_to_wav(raw_audio, src_format="wav")
                    text_v, err = speech_to_text_arabic(wav)
                    if text_v:
                        user_q = text_v
    except Exception:
        logger.info("Voice input unavailable", exc_info=True)

    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant", avatar=get_logo()):
            placeholder = st.empty()
            placeholder.markdown('<span class="typing-indicator">جاري التحليل...</span>',
                                  unsafe_allow_html=True)

            user_q_clean = user_q.strip()
            if len(user_q_clean) < 3:
                answer = "عفواً، لم أفهم استفسارك. يرجى توضيح السؤال الطبي أو الأعراض بمزيد من التفاصيل."
            else:
                try:
                    safe = guard.check(user_q)
                    
                    # CARDIAC EMERGENCY — bloc rouge animé + stop immédiat
                    if safe.get("emergency"):
                        placeholder.empty()
                        st.markdown("""
                        <div style="
                          background: rgba(220,38,38,0.15);
                          border: 2px solid #DC2626;
                          border-radius: 16px;
                          padding: 24px;
                          text-align: center;
                          direction: rtl;
                          animation: pulse 1s infinite;
                        ">
                        <h2 style="color:#DC2626; margin:0;">🚨 حالة طوارئ</h2>
                        <h3 style="color:#FCA5A5; margin:8px 0;">اتصل بالإسعاف فوراً — 📞 15</h3>
                        </div>
                        <style>
                        @keyframes pulse {
                          0%   { border-color: #DC2626; }
                          50%  { border-color: #FCA5A5; }
                          100% { border-color: #DC2626; }
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        answer = safe["override_response"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        save_history(st.session_state.messages)
                        st.stop()
                    
                    if safe["level"] in ("emergency", "boundary"):
                        answer = safe["override_response"]
                    else:
                        enc = retriever.encode_query(user_q)
                        try:
                            intent, _ = classifier.predict(enc)
                        except Exception:
                            intent = "general"

                        res = retriever.get_best_answer(user_q)
                        if res:
                            base, _, _, _ = res
                            if AI_STATUS:
                                # Pass conversation history for memory
                                history_ctx = st.session_state.messages[:-1]  # exclude current user msg
                                answer = ai_engine.generate_answer(user_q, base, intent,
                                                                    history=history_ctx)
                                if answer:
                                    answer = guard.post_check(answer)
                            else:
                                answer = format_response(base, intent)

                            if safe["level"] == "caution":
                                answer = guard.format_caution_response(answer)
                            answer = guard.add_disclaimer(answer)
                        else:
                            answer = "عذراً، لا أملك إجابة دقيقة حالياً. يرجى استشارة طبيب مختص."
                except Exception:
                    logger.error("Chat pipeline error", exc_info=True)
                    answer = "حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى."

            placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            save_history(st.session_state.messages)
            st.rerun()

# ─────────────────────────────────────────────────────────────
# PAGE: VISION
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "vision":
    st.markdown('<h1 class="main-title">🔬 تحليل الصور</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5A6072;">تحليل متقدم للصور الطبية باستخدام تقنيات الذكاء الاصطناعي (Dermato / X-Ray / Brain MRI). هذا النظام لا يعوض زيارة الطبيب المختص.</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    
    disclaimer_agreed = st.checkbox(
        "أوافق على أن هذه الأداة تجريبية للأغراض التعليمية فقط ولا تعطي تشخيصاً طبياً نهائياً. ⚠️ تنبيه مهم: هذه المعلومات للتوعية الصحية فقط ولا تغني عن استشارة الطبيب المختص.",
        key="vision_disclaimer"
    )
    
    if not disclaimer_agreed:
        st.info("يرجى الموافقة على الشروط أعلاه لتفعيل الفحص.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            vision_type_ar = st.selectbox(
                "نوع الصورة",
                ["🔴 فحص الجلد", "🫁 أشعة الصدر", "🧠 رنين الدماغ", "🩺 كشف السرطان", "🔬 كثافة الثدي"]
            )
            
            type_mapping = {
                "🔴 فحص الجلد": "dermato",
                "🫁 أشعة الصدر": "xray",
                "🧠 رنين الدماغ": "brain_mri",
                "🩺 كشف السرطان": "cancer",
                "🔬 كثافة الثدي": "breast"
            }
            vision_type = type_mapping[vision_type_ar]
            
            uploaded_img = st.file_uploader(
                "تحميل الصورة",
                type=["jpg", "jpeg", "png"]
            )
            
            if uploaded_img:
                st.image(uploaded_img, caption="الصورة الأصلية", use_container_width=True)
                
            analyze_btn = st.button("تحليل الصورة", type="primary", use_container_width=True)
            
        with col2:
            if analyze_btn and uploaded_img:
                with st.spinner("جاري التحليل..."):
                    img_pil = PILImage.open(uploaded_img)
                    try:
                        result = vision_router.analyze(img_pil, vision_type)
                        if result:
                            if not result.get("valid", True):
                                st.warning(result.get("recommendation_ar", "تحقق من جودة الصورة."))
                                if result.get("rejection_reason"):
                                    st.caption(f"Reason: {result['rejection_reason']}")
                            else:
                                conf = result.get("confidence", 0.0)
                                color_map = {
                                    "critique": "#DC3545",
                                    "élevée": "#FF9800",
                                    "modérée": "#FFC107",
                                    "faible": "#28A745",
                                    "indéfini": "#6C757D"
                                }
                                color = color_map.get(result.get("severity", "indéfini"), "#6C757D")
                                
                                st.markdown(f"""
                                <div style="text-align:center; padding:20px; border-radius:16px;
                                            background:{color}11; border:2px solid {color}; margin-bottom:10px;">
                                    <h3 style="color:{color}; margin:0;">{result.get('class', 'غير معروف')}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.metric("نسبة الثقة", f"{conf * 100:.1f}%")
                                
                                if conf < 0.60:
                                    st.warning("⚠️ نسبة الثقة منخفضة — يُنصح بإعادة التصوير أو استشارة الطبيب")
                                    
                                st.markdown(f"""
                                <div style="background:rgba(30,41,59,0.4); padding:15px; border-radius:12px; border-right:4px solid #38BDF8; margin-bottom:15px;">
                                    <h4 style="margin-top:0;">توصية طبية</h4>
                                    <span style="font-size:15px; color:#E2E8F0;">{result.get('recommendation_ar', '')}</span>
                                    <br><br>
                                    <small style="color:#94A3B8;">⚠️ تنبيه مهم: هذه المعلومات للتوعية الصحية فقط ولا تغني عن استشارة الطبيب المختص.</small>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("#### توزيع الاحتمالات")
                                probs_df = pd.DataFrame({
                                    "الاحتمال (%)": [p * 100 for p in result.get("all_probs", {}).values()],
                                    "الفئة": list(result.get("all_probs", {}).keys())
                                }).set_index("الفئة")
                                st.bar_chart(probs_df)
                                
                                if result.get("gradcam") is not None:
                                    st.markdown("#### خريطة التركيز (Grad-CAM)")
                                    try:
                                        import cv2
                                        import numpy as np
                                        cam_np = result["gradcam"]
                                        img_np = np.array(img_pil.convert('RGB').resize((cam_np.shape[1], cam_np.shape[0])))
                                        heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
                                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                                        overlay = img_np * 0.6 + heatmap * 0.4
                                        st.image(overlay.astype(np.uint8), caption="مناطق التنشيط (Heatmap)", use_container_width=True)
                                    except Exception as e:
                                        logger.error(f"GradCAM plot error: {e}")
                                        st.error("تعذر عرض خريطة التركيز.")
                    except Exception as e:
                        st.error(f"حدث خطأ أثناء التحليل: {str(e)}")
            
            elif analyze_btn and not uploaded_img:
                st.warning("يرجى تحميل الصورة أولاً.")
                
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: SYMPTOM SCANNER
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "scanner":
    st.markdown('<h1 class="main-title">فاحص الأعراض المتقدم</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5A6072;">يرجى ملء النموذج بدقة للحصول على تحليل أولي.</p>', unsafe_allow_html=True)
    
    with st.form("scanner_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("العمر", 0, 120, 30)
            gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
        with col2:
            duration = st.selectbox("مدة الأعراض", ["أقل من يوم", "1-3 أيام", "أسبوع", "أكثر من أسبوع"])
            severity = st.select_slider("شدة الألم/التعب", options=["خفيف", "متوسط", "شديد", "لا يطاق"])
            
        symptoms = st.text_area("وصف الأعراض بالتفصيل (مثال: سعال جاف مع ألم في الصدر)", height=100)
        history = st.text_input("هل تعاني من أمراض مزمنة؟ (اختياري)")
        
        submitted = st.form_submit_button("بدء التحليل الذكي", use_container_width=True)
        
    if submitted:
        if not symptoms:
            st.warning("يرجى وصف الأعراض أولاً.")
        else:
            with st.spinner("جاري تحليل الحالة..."):
                scan_prompt = f"""
                المريض: {gender}، العمر: {age}
                المدة: {duration}، الشدة: {severity}
                الأعراض الموصوفة: {symptoms}
                التاريخ المرضي: {history if history else 'لا يوجد'}
                """
                
                # Use retriever for context then LLM for analysis
                res = retriever.get_best_answer(symptoms)
                context = res[0] if res else "إرشادات عامة للمرض"
                
                if AI_STATUS:
                    analysis = ai_engine.generate_answer(
                        f"قم بتحليل هذه الأعراض وتقديم توجيهات أولية: {scan_prompt}", 
                        context, 
                        "وصف_أعراض"
                    )
                else:
                    analysis = None
                
                if not analysis:
                    analysis = f"""بناءً على الأعراض المذكرة:
                    
**الأعراض:** {symptoms}
**المدة:** {duration} | **الشدة:** {severity}

**التوصية:** يرجى مراجعة طبيب مختص في أقرب وقت لإجراء فحص سريري كامل.

**ملاحظة:** النظام الذكي غير متصل حالياً. هذه توصية عامة فقط."""
                
                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                st.subheader("نتيجة التحليل الأولي")
                st.write(analysis)
                st.markdown('</div>', unsafe_allow_html=True)
                st.info("تذكر: هذا التحليل آلي ولا يغني عن الفحص السريري.")


# ─────────────────────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "history":
    st.markdown('<h1 class="main-title">سجل الاستشارات الماضية</h1>', unsafe_allow_html=True)
    
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
            
        if not history:
            st.info("لا يوجد سجل استشارات حتى الآن.")
        else:
            for item in history:
                with st.expander(f"{item['date']} | {item['title'][:40]}..."):
                    for m in item['messages']:
                        role = "الطبيب الذكي" if m['role'] == "assistant" else "أنت"
                        st.markdown(f"**{role}:**")
                        st.write(m['content'])
                        st.markdown("---")
    else:
        st.info("سجل الاستشارات فارغ حالياً.")

# ─────────────────────────────────────────────────────────────
# PAGE: DATABASE
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "database":
    st.markdown('<h1 class="main-title">قاعدة المعرفة الطبية</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5A6072;">اطلع على قائمة التصنيفات والبيانات التي يعتمد عليها النظام.</p>', unsafe_allow_html=True)
    
    try:
        from engine.retriever import FAISSRetriever
        from data.knowledge_base import CATEGORY_KEYWORDS
        
        cols = st.columns(3)
        cats = list(CATEGORY_KEYWORDS.keys())
        for i, cat in enumerate(cats):
            with cols[i % 3]:
                st.markdown(f"""
                    <div style="background:#F8F9FA; padding:15px; border-radius:12px; margin-bottom:10px; border-right:4px solid #E53935;">
                        <h4 style="margin:0; font-size:14px;">{cat}</h4>
                    </div>
                """, unsafe_allow_html=True)
                
        st.markdown("### البحث في المصادر")
        search_term = st.text_input("ابحث عن موضوع طبي...")
        if search_term:
            res = retriever.get_best_answer(search_term)
            if res:
                st.markdown(f'<div class="data-card">{res[0]}</div>', unsafe_allow_html=True)
            else:
                st.info("لم نجد مراجع محددة لهذا البحث.")
    except:
        st.error("تعذر تحميل بيانات المصادر حالياً.")

# ─────────────────────────────────────────────────────────────
# GLOBAL FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.3);
font-size:0.75rem; padding:20px; border-top:1px solid
rgba(0,212,170,0.1); margin-top:40px;">
  شفاء AI · للمساعدة الطبية فقط · ليس بديلاً عن الطبيب
  <br>SHIFA AI v1.0 · 2025
</div>
""", unsafe_allow_html=True)


