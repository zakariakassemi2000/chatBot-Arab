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
    except:
        pass

# ─────────────────────────────────────────────────────────────
# IMPORT ENGINES
# ─────────────────────────────────────────────────────────────
from engine.retriever import FAISSRetriever
from engine.classifier import IntentClassifier, format_response
from engine.safety import SafetyGuard
from engine.llm import GroqGenerator
from engine.audio import speech_to_text_arabic, convert_audio_to_wav
from engine.cancer_detector import BreastCancerDetector
from engine.breast_density_detector import BreastDensityDetector
from engine.brain_tumor_detector import BrainTumorDetector
from engine.xray_analyzer import ChestXrayAnalyzer
from engine.derm_detector import DermDetector
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
    except:
        pass
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
        except: pass
    
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
        bc = BreastCancerDetector()
        bd = BreastDensityDetector()
        db_ok = r.load()
        if db_ok: c.load()
        return r, c, g, ai, bc, bd, db_ok
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None, None, None, None, None, None, False

retriever, classifier, guard, ai_engine, cancer_detector, density_detector, DB_STATUS = load_medical_system()

# Lazy-load heavy models only when their page is accessed
@st.cache_resource(show_spinner=False)
def load_brain_tumor_detector():
    try:
        return BrainTumorDetector()
    except Exception as e:
        st.error(f"Brain Tumor model error: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_xray_analyzer():
    try:
        return ChestXrayAnalyzer()
    except Exception as e:
        st.error(f"X-Ray model error: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_derm_detector():
    try:
        return DermDetector()
    except Exception as e:
        st.error(f"Derm model error: {e}")
        return None

AI_STATUS = bool(ai_engine and ai_engine.api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "chat"
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
        ("chat",            "", "المحادثة السريعة (القديمة)"),
        ("vision",          "", "🔬 تحليل الصور"),
        ("scanner",         "", "فاحص الأعراض"),
        ("cancer_scanner",  "", "كثافة الثدي (MONAI)"),
        ("calculators",     "", "حاسبات طبية"),
        ("database",        "", "قاعدة المعرفة"),
        ("history",         "", "سجل الاستشارات"),
    ]
    for page_id, icon, label in pages:
        is_active = st.session_state.page == page_id
        if st.button(f"{label}", use_container_width=True,
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
# PAGE: CHAT (MAIN)
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "chat":
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
                ["🔴 فحص الجلد", "🫁 أشعة الصدر", "🧠 رنين الدماغ"]
            )
            
            type_mapping = {
                "🔴 فحص الجلد": "dermato",
                "🫁 أشعة الصدر": "xray",
                "🧠 رنين الدماغ": "brain_mri"
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
                    
                    @st.cache_resource(show_spinner=False)
                    def get_vision_router():
                        from engine.vision_router import VisionRouter
                        return VisionRouter()
                        
                    try:
                        router = get_vision_router()
                        result = router.analyze(img_pil, vision_type)
                        
                        if result:
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
                        st.error(f"حدث خطأ أثناء تحميل النموذج: {str(e)}")
            
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
                        "וصف_أعراض"
                    )
                else:
                    analysis = "تحليل النظام غير متوفر حالياً بدون مفتاح LLM. يرجى مراجعة الطبيب."
                
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
# PAGE: BREAST DENSITY SCANNER (MONAI InceptionV3 — BI-RADS)
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "cancer_scanner":
    st.markdown('<h1 class="main-title">فاحص كثافة الثدي (MONAI)</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5A6072;">نموذج ذكاء اصطناعي من <b>MONAI / Hugging Face</b> مبني على <b>InceptionV3</b> لتصنيف كثافة أنسجة الثدي في صور الماموغرام وفق معايير <b>BI-RADS</b> الدولية (الفئات A إلى D).</p>', unsafe_allow_html=True)

    st.markdown('<div class="data-card">', unsafe_allow_html=True)

    # ── Model Status Banner ──
    if density_detector is None or density_detector.model is None:
        st.error("**نموذج كثافة الثدي غير متاح**")
        err_msg = density_detector.load_error if density_detector else "لم يتم تحميل الكاشف"
        st.warning(f"تنبيه تقني: {err_msg}")
        st.info("""
        **لتشغيل النموذج:**
        1. تأكد من أن المكتبات التالية مثبتة: `monai`, `torchvision`, `huggingface_hub`
        2. تأكد من وجود `HF_TOKEN` في ملف `.env`
        3. النموذج سيتم تحميله تلقائياً من Hugging Face عند أول استخدام (~101 ميجابايت)
        """)
    else:
        st.success("وحدة التحليل جاهزة — MONAI InceptionV3")

    st.markdown("---")

    # ── BI-RADS Info Card ──
    st.markdown("""
    <div style="background:linear-gradient(135deg, #f8f9fa, #fff5f5); border-radius:14px; padding:18px; margin-bottom:16px; border:1px solid #f0e0e0;">
        <h4 style="margin:0 0 10px; color:#1E2028;">تصنيفات كثافة الثدي (BI-RADS)</h4>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <div style="background:#e8f5e9; padding:10px 14px; border-radius:10px;">
                <b style="color:#2e7d32;">🟢 فئة A</b> — دهني بالكامل<br>
                <small style="color:#555;">كثافة منخفضة، سهولة قراءة الماموغرام</small>
            </div>
            <div style="background:#fff8e1; padding:10px 14px; border-radius:10px;">
                <b style="color:#f9a825;">🟡 فئة B</b> — ليفي غدي متناثر<br>
                <small style="color:#555;">كثافة متوسطة منخفضة</small>
            </div>
            <div style="background:#fff3e0; padding:10px 14px; border-radius:10px;">
                <b style="color:#ef6c00;">🟠 فئة C</b> — كثيف غير متجانس<br>
                <small style="color:#555;">قد يُخفي آفات صغيرة</small>
            </div>
            <div style="background:#ffebee; padding:10px 14px; border-radius:10px;">
                <b style="color:#c62828;">🔴 فئة D</b> — كثيف للغاية<br>
                <small style="color:#555;">يُقلل حساسية الماموغرام</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ──
    disclaimer_agreed = st.checkbox(
        "أوافق على أن هذه الأداة تجريبية للأغراض التعليمية فقط ولا تعطي تشخيصاً طبياً نهائياً.",
        key="cancer_disclaimer_v2"
    )

    if not disclaimer_agreed:
        st.info("يرجى الموافقة على الشروط أعلاه لتفعيل رفع صور الماموغرام.")
    else:
        uploaded_img = st.file_uploader(
            "تحميل صورة الماموغرام (Mammogram)",
            type=["jpg", "jpeg", "png"]
        )

        if st.button("تحليل كثافة الثدي", use_container_width=True, type="primary"):
            if not uploaded_img:
                st.warning("يرجى تحميل صورة ماموغرام أولاً.")
            elif density_detector is None or density_detector.model is None:
                st.error("النموذج غير متاح. تحقق من التثبيت.")
            else:
                with st.spinner("جاري تحليل كثافة الثدي..."):
                    img_pil = PILImage.open(uploaded_img)
                    prediction = None

                    # ── Image Gatekeeper ──
                    validation = validate_medical_image(img_pil, expected_type="mammogram")
                    if not validation["valid"]:
                        st.error(validation["reason"])
                        if validation.get("medical_score"):
                            st.caption(f"درجة الثقة الطبية: {validation['medical_score']*100:.0f}%")
                    else:
                        if not validation.get("type_match", True):
                            st.warning(validation["reason"])

                        prediction = density_detector.predict_image(img_pil)
                        label, explanation, risk_level, style = density_detector.interpret_density(prediction)

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(uploaded_img, caption="صورة الماموغرام المُحلَّلة", use_container_width=True)

                with col2:
                    if prediction:
                        color_map = {
                            "success": "#28A745",
                            "info": "#FFC107",
                            "warning": "#FF9800",
                            "danger": "#DC3545",
                            "error": "#6C757D",
                        }
                        color = color_map.get(style, "#6C757D")

                        st.markdown(f"""
                        <div style="text-align:center; padding:20px; border-radius:16px;
                                    background:{color}11; border:2px solid {color}; margin-top:10px;">
                            <h3 style="color:{color}; margin:0;">{label}</h3>
                            <p style="color:#1E2028; margin-top:10px; font-size:15px;">{explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability breakdown
                        st.markdown("#### توزيع الاحتمالات")
                        st.progress(prediction["prob_A"], text=f"🟢 A — دهني: {prediction['prob_A']*100:.1f}%")
                        st.progress(prediction["prob_B"], text=f"🟡 B — ليفي متناثر: {prediction['prob_B']*100:.1f}%")
                        st.progress(prediction["prob_C"], text=f"🟠 C — كثيف غير متجانس: {prediction['prob_C']*100:.1f}%")
                        st.progress(prediction["prob_D"], text=f"🔴 D — كثيف للغاية: {prediction['prob_D']*100:.1f}%")

                        # Risk-based recommendations
                        if risk_level == "high":
                            st.error("كثافة عالية: يُوصى بشدة بإجراء فحوصات تكميلية (MRI / أمواج فوق صوتية).")
                        elif risk_level == "moderate_high":
                            st.warning("كثافة متوسطة-عالية: يُنصح بمتابعة دورية وفحوصات إضافية حسب توجيه الطبيب.")
                        elif risk_level == "moderate_low":
                            st.info("كثافة متوسطة-منخفضة: يُنصح بالفحص الدوري المعتاد.")
                        else:
                            st.success("كثافة منخفضة: الماموغرام سهل القراءة. استمر في الفحص الدوري.")
                    else:
                        st.error("تعذّر تحليل الصورة. تأكد من أنها صورة ماموغرام صالحة.")

    # ── System Diagnostics ──
    with st.expander("تشخيص النظام"):
        if density_detector:
            status_color = "green" if density_detector.model else "red"
            model_status = "Connected (InceptionV3 MONAI)" if density_detector.model else "Unavailable"
            st.markdown(
                f"**Model Status:** <span style='color:{status_color};'>{model_status}</span>",
                unsafe_allow_html=True
            )
            if density_detector.load_error:
                st.warning(f"خطأ تقني: {density_detector.load_error}")
            device_info = str(density_detector.device) if density_detector.device else "غير محدد"
            st.caption(f"البنية: InceptionV3 (MONAI) | المصدر: Hugging Face | الفئات: A/B/C/D (BI-RADS) | الجهاز: {device_info}")
        else:
            st.error("لم يتم تهيئة كاشف كثافة الثدي.")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ⚖️ PAGE: MEDICAL CALCULATORS
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "calculators":
    st.markdown('<h1 class="main-title">حاسبات طبية ذكية</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5A6072;">مجموعة من الأدوات الدقيقة لتقييم صحتك ومؤشرات جسمك.</p>', unsafe_allow_html=True)

    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    tabs = st.tabs(["BMI", "ماء يومي", "BMR", "سعرات TDEE"])
    
    with tabs[0]:
        st.subheader("مؤشر كتلة الجسم (BMI)")
        st.caption("يقيس العلاقة بين وزنك وطولك.")
        c1, c2 = st.columns(2)
        weight = c1.number_input("الوزن (كجم)", 30.0, 300.0, 70.0, step=0.5, key="bmi_w")
        height = c2.number_input("الطول (سم)", 100.0, 250.0, 170.0, step=0.5, key="bmi_h")
        if st.button("احسب BMI", use_container_width=True, key="calc_bmi"):
            bmi = weight / ((height / 100) ** 2)
            if bmi < 18.5:   cat, icon, col = "نقص الوزن", "△", "#FF9800"
            elif bmi < 25:   cat, icon, col = "وزن مثالي", "●", "#4CAF50"
            elif bmi < 30:   cat, icon, col = "زيادة الوزن", "△", "#FF9800"
            else:            cat, icon, col = "سمنة", "■", "#F44336"
            st.markdown(f"""
                <div style="text-align:center; padding:20px; border-radius:16px;
                            background:{col}15; border:2px solid {col}; margin:10px 0;">
                    <div style="font-size:40px;">{icon}</div>
                    <div style="font-size:34px; font-weight:900; color:{col};">{bmi:.1f}</div>
                    <div style="font-size:14px; color:#555;">{cat}</div>
                </div>
            """, unsafe_allow_html=True)
                    
    with tabs[1]:
        st.subheader("الاحتياج اليومي للماء")
        weight_water = st.number_input("الوزن (كجم)", 30.0, 300.0, 70.0, key="water_w")
        activity = st.selectbox("مستوى النشاط", [
            "خفيف (عمل مكتبي، بدون رياضة)",
            "متوسط (رياضة 3-4 أيام/أسبوع)",
            "عالي (رياضة يومية أو عمل مجهد)"
        ])
        if st.button("احسب احتياج الماء", use_container_width=True, key="calc_water"):
            base = weight_water * 35
            if "متوسط" in activity: base += 500
            elif "عالي" in activity: base += 1000
            liters = base / 1000
            cups = round(base / 250)
            st.success(f"احتياجك اليومي: **{liters:.1f} لتر** ({cups} كوب تقريباً)")
            st.progress(min(liters / 4.0, 1.0), text=f"{liters:.1f}L / 4L")
            
    with tabs[2]:
        st.subheader("معدل الأيض الأساسي (BMR)")
        st.caption("السعرات التي يحرقها جسمك في حالة الراحة التامة.")
        gender_bmr = st.radio("الجنس", ["ذكر", "أنثى"], horizontal=True, key="bmr_gender")
        c1, c2, c3 = st.columns(3)
        weight_b = c1.number_input("الوزن (كجم)", 30.0, 300.0, 70.0, key="bmr_w")
        height_b = c2.number_input("الطول (سم)", 100.0, 250.0, 170.0, key="bmr_h")
        age_b    = c3.number_input("العمر", 10, 120, 30, key="bmr_a")
        if st.button("احسب BMR", use_container_width=True, key="calc_bmr"):
            if gender_bmr == "ذكر":
                bmr = 88.362 + (13.397 * weight_b) + (4.799 * height_b) - (5.677 * age_b)
            else:
                bmr = 447.593 + (9.247 * weight_b) + (3.098 * height_b) - (4.330 * age_b)
            st.success(f"معدل الأيض الأساسي: **{bmr:.0f} سعرة/يوم**")

    with tabs[3]:
        st.subheader("السعرات الكلية اليومية (TDEE)")
        st.caption("إجمالي ما تحرقه يومياً — الأساس لأي هدف غذائي.")
        gender_t = st.radio("الجنس", ["ذكر", "أنثى"], horizontal=True, key="tdee_g")
        ct1, ct2, ct3 = st.columns(3)
        wt = ct1.number_input("الوزن (كجم)", 30.0, 300.0, 70.0, key="tdee_w")
        ht = ct2.number_input("الطول (سم)", 100.0, 250.0, 170.0, key="tdee_h")
        at = ct3.number_input("العمر", 10, 100, 30, key="tdee_a")
        activity_t = st.selectbox("مستوى النشاط", [
            "مستقر (لا رياضة)",
            "خفيف (1-3 أيام/أسبوع)",
            "متوسط (3-5 أيام/أسبوع)",
            "عالي (6-7 أيام/أسبوع)",
            "شديد جداً (رياضة مكثفة + عمل بدني)"
        ], key="tdee_act")
        goal = st.selectbox("هدفك", ["الحفاظ على الوزن", "إنقاص الوزن", "زيادة الوزن"], key="tdee_goal")
        if st.button("احسب TDEE", use_container_width=True, key="calc_tdee"):
            if gender_t == "ذكر":
                bmr_t = 88.362 + (13.397 * wt) + (4.799 * ht) - (5.677 * at)
            else:
                bmr_t = 447.593 + (9.247 * wt) + (3.098 * ht) - (4.330 * at)
            factors = {"مستقر": 1.2, "خفيف": 1.375, "متوسط": 1.55, "عالي": 1.725, "شديد": 1.9}
            factor = next((v for k, v in factors.items() if k in activity_t), 1.55)
            tdee = bmr_t * factor
            if "إنقاص" in goal:   target = tdee - 500
            elif "زيادة" in goal: target = tdee + 400
            else:                  target = tdee
            st.markdown(f"""
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px;">
                    <div style="background:#fff3e0; border-radius:14px; padding:16px; text-align:center;">
                        <div style="font-size:12px; color:#888;">الحرق اليومي الكلي</div>
                        <div style="font-size:28px; font-weight:900; color:#FF9800;">{tdee:.0f}</div>
                        <div style="font-size:11px; color:#aaa;">سعرة/يوم</div>
                    </div>
                    <div style="background:#e8f5e9; border-radius:14px; padding:16px; text-align:center;">
                        <div style="font-size:12px; color:#888;">{goal}</div>
                        <div style="font-size:28px; font-weight:900; color:#4CAF50;">{target:.0f}</div>
                        <div style="font-size:11px; color:#aaa;">سعرة مستهدفة/يوم</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 🧠 PAGE: BRAIN TUMOR MRI SCANNER
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "brain_mri":
    st.markdown('<h1 class="main-title">فاحص أورام الدماغ (MRI)</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5A6072;">نموذج ذكاء اصطناعي <b>Swin Transformer</b> لتصنيف صور الرنين المغناطيسي وكشف أورام الدماغ — Glioma / Meningioma / Pituitary / No Tumor.</p>', unsafe_allow_html=True)

    st.markdown('<div class="data-card">', unsafe_allow_html=True)

    # Lazy load model only when page is accessed
    brain_detector = load_brain_tumor_detector()

    if brain_detector is None or brain_detector.model is None:
        st.error("**نموذج أورام الدماغ غير متاح**")
        err_msg = brain_detector.load_error if brain_detector else "لم يتم تحميل الكاشف"
        st.warning(f"تنبيه تقني: {err_msg}")
        st.info("النموذج يتم تحميله تلقائياً من Hugging Face عند أول استخدام (~110 ميجابايت)")
    else:
        st.success("وحدة التحليل جاهزة — Swin Transformer")

    st.markdown("---")

    # ── Tumor Types Info ──
    st.markdown("""
    <div style="background:linear-gradient(135deg, #f8f9fa, #f0f4ff); border-radius:14px; padding:18px; margin-bottom:16px; border:1px solid #e0e8f0;">
        <h4 style="margin:0 0 10px; color:#1E2028;">أنواع الأورام المكتشفة</h4>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <div style="background:#ffebee; padding:10px 14px; border-radius:10px;">
                <b style="color:#c62828;">🔴 ورم دبقي (Glioma)</b><br>
                <small style="color:#555;">ورم في الخلايا الدبقية — عادةً خبيث</small>
            </div>
            <div style="background:#fff3e0; padding:10px 14px; border-radius:10px;">
                <b style="color:#ef6c00;">🟠 ورم سحائي (Meningioma)</b><br>
                <small style="color:#555;">ورم في الأغشية السحائية — عادةً حميد</small>
            </div>
            <div style="background:#fff8e1; padding:10px 14px; border-radius:10px;">
                <b style="color:#f9a825;">🟡 ورم نخامي (Pituitary)</b><br>
                <small style="color:#555;">ورم في الغدة النخامية</small>
            </div>
            <div style="background:#e8f5e9; padding:10px 14px; border-radius:10px;">
                <b style="color:#2e7d32;">🟢 سليم (No Tumor)</b><br>
                <small style="color:#555;">لا توجد مؤشرات أورام</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    disclaimer_agreed = st.checkbox(
        "أوافق على أن هذه الأداة تجريبية للأغراض التعليمية فقط ولا تعطي تشخيصاً طبياً نهائياً.",
        key="brain_disclaimer"
    )

    if not disclaimer_agreed:
        st.info("يرجى الموافقة على الشروط أعلاه لتفعيل رفع صور MRI.")
    else:
        uploaded_img = st.file_uploader(
            "تحميل صورة الرنين المغناطيسي (MRI)",
            type=["jpg", "jpeg", "png"],
            key="brain_uploader"
        )

        if st.button("تحليل صورة MRI", use_container_width=True, type="primary", key="brain_analyze"):
            if not uploaded_img:
                st.warning("يرجى تحميل صورة MRI أولاً.")
            elif brain_detector is None or brain_detector.model is None:
                st.error("النموذج غير متاح.")
            else:
                with st.spinner("جاري تحليل صورة الرنين المغناطيسي..."):
                    img_pil = PILImage.open(uploaded_img)
                    prediction = None

                    # ── Image Gatekeeper ──
                    validation = validate_medical_image(img_pil, expected_type="mri")
                    if not validation["valid"]:
                        st.error(validation["reason"])
                        if validation.get("medical_score"):
                            st.caption(f"درجة الثقة الطبية: {validation['medical_score']*100:.0f}%")
                    else:
                        if not validation.get("type_match", True):
                            st.warning(validation["reason"])

                        prediction = brain_detector.predict_image(img_pil)
                        label, explanation, risk_level, style = brain_detector.interpret_result(prediction)

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(uploaded_img, caption="صورة MRI المُحلَّلة", use_container_width=True)
                    # Grad-CAM heatmap
                    if gradcam_available() and brain_detector.model and brain_detector.processor:
                        with st.spinner("توليد خريطة التنشيط (Grad-CAM)..."):
                            heatmap = generate_gradcam_heatmap(
                                brain_detector.model, brain_detector.processor,
                                img_pil, target_class_idx=prediction.get("class_name") and list(prediction.keys()).index("class_name") or None
                            )
                            if heatmap:
                                st.image(heatmap, caption="خريطة Grad-CAM — مناطق تركيز النموذج", use_container_width=True)

                with col2:
                    if prediction:
                        color_map = {"success": "#28A745", "warning": "#FF9800", "danger": "#DC3545", "error": "#6C757D"}
                        color = color_map.get(style, "#6C757D")
                        st.markdown(f"""
                        <div style="text-align:center; padding:20px; border-radius:16px;
                                    background:{color}11; border:2px solid {color}; margin-top:10px;">
                            <h3 style="color:{color}; margin:0;">{label}</h3>
                            <p style="color:#1E2028; margin-top:10px; font-size:15px;">{explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("#### توزيع الاحتمالات")
                        st.progress(prediction["prob_glioma"], text=f"🔴 Glioma: {prediction['prob_glioma']*100:.1f}%")
                        st.progress(prediction["prob_meningioma"], text=f"🟠 Meningioma: {prediction['prob_meningioma']*100:.1f}%")
                        st.progress(prediction["prob_no_tumor"], text=f"🟢 سليم: {prediction['prob_no_tumor']*100:.1f}%")
                        st.progress(prediction["prob_pituitary"], text=f"🟡 Pituitary: {prediction['prob_pituitary']*100:.1f}%")

                        if risk_level == "high":
                            st.error("يُرجى مراجعة أخصائي جراحة أعصاب فوراً.")
                        elif risk_level == "moderate":
                            st.warning("يُنصح بمراجعة طبيب أعصاب لتقييم الحالة.")
                        else:
                            st.success("لم تُرصد مؤشرات أورام. يُنصح بالمتابعة الدورية.")
                    else:
                        st.error("تعذّر تحليل الصورة.")

    with st.expander("تشخيص النظام"):
        if brain_detector:
            status_color = "green" if brain_detector.model else "red"
            model_status = "Connected (Swin Transformer)" if brain_detector.model else "Unavailable"
            st.markdown(f"**Model Status:** <span style='color:{status_color};'>{model_status}</span>", unsafe_allow_html=True)
            if brain_detector.load_error:
                st.warning(f"خطأ تقني: {brain_detector.load_error}")
            st.caption("البنية: Swin Transformer | المصدر: Hugging Face | الفئات: Glioma/Meningioma/Pituitary/No Tumor")
        else:
            st.error("لم يتم تهيئة الكاشف.")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 🩻 PAGE: CHEST X-RAY SCANNER
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "xray_scanner":
    st.markdown('<h1 class="main-title">تحليل الأشعة السينية (X-Ray)</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5A6072;">نموذج <b>Vision Transformer (ViT)</b> مُدرّب على بيانات <b>CheXpert (Stanford)</b> لتصنيف أمراض الصدر — دقة 98.46%.</p>', unsafe_allow_html=True)

    st.markdown('<div class="data-card">', unsafe_allow_html=True)

    # Lazy load
    xray_detector = load_xray_analyzer()

    if xray_detector is None or xray_detector.model is None:
        st.error("**نموذج الأشعة السينية غير متاح**")
        err_msg = xray_detector.load_error if xray_detector else "لم يتم تحميل الكاشف"
        st.warning(f"تنبيه تقني: {err_msg}")
        st.info("النموذج يتم تحميله تلقائياً من Hugging Face عند أول استخدام (~350 ميجابايت)")
    else:
        st.success("وحدة التحليل جاهزة — ViT Chest X-Ray (CheXpert)")

    st.markdown("---")

    # ── Conditions Info ──
    st.markdown("""
    <div style="background:linear-gradient(135deg, #f8f9fa, #f0f8ff); border-radius:14px; padding:18px; margin-bottom:16px; border:1px solid #e0e8f0;">
        <h4 style="margin:0 0 10px; color:#1E2028;">الحالات المكتشفة</h4>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <div style="background:#ffebee; padding:10px 14px; border-radius:10px;">
                <b style="color:#c62828;">🔴 التهاب رئوي (Pneumonia)</b><br>
                <small style="color:#555;">عدوى تصيب الحويصلات الهوائية</small>
            </div>
            <div style="background:#fce4ec; padding:10px 14px; border-radius:10px;">
                <b style="color:#ad1457;">🫀 تضخم القلب (Cardiomegaly)</b><br>
                <small style="color:#555;">تضخم في عضلة القلب</small>
            </div>
            <div style="background:#e3f2fd; padding:10px 14px; border-radius:10px;">
                <b style="color:#1565c0;">💧 وذمة رئوية (Edema)</b><br>
                <small style="color:#555;">تجمع سوائل في الرئتين</small>
            </div>
            <div style="background:#fff3e0; padding:10px 14px; border-radius:10px;">
                <b style="color:#ef6c00;">🫁 تصلب رئوي (Consolidation)</b><br>
                <small style="color:#555;">تصلب في أنسجة الرئة</small>
            </div>
            <div style="background:#e8f5e9; padding:12px 14px; border-radius:10px; grid-column: span 2;">
                <b style="color:#2e7d32;">🟢 سليم — No Finding</b><br>
                <small style="color:#555;">لا توجد مؤشرات مرضية في الأشعة</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    disclaimer_agreed = st.checkbox(
        "أوافق على أن هذه الأداة تجريبية للأغراض التعليمية فقط ولا تعطي تشخيصاً طبياً نهائياً.",
        key="xray_disclaimer"
    )

    if not disclaimer_agreed:
        st.info("يرجى الموافقة على الشروط أعلاه لتفعيل رفع صور الأشعة.")
    else:
        uploaded_img = st.file_uploader(
            "تحميل صورة الأشعة السينية (Chest X-Ray)",
            type=["jpg", "jpeg", "png"],
            key="xray_uploader"
        )

        if st.button("تحليل الأشعة السينية", use_container_width=True, type="primary", key="xray_analyze"):
            if not uploaded_img:
                st.warning("يرجى تحميل صورة أشعة سينية أولاً.")
            elif xray_detector is None or xray_detector.model is None:
                st.error("النموذج غير متاح.")
            else:
                with st.spinner("جاري تحليل صورة الأشعة السينية..."):
                    img_pil = PILImage.open(uploaded_img)
                    prediction = None

                    # ── Image Gatekeeper ──
                    validation = validate_medical_image(img_pil, expected_type="xray")
                    if not validation["valid"]:
                        st.error(validation["reason"])
                        if validation.get("medical_score"):
                            st.caption(f"درجة الثقة الطبية: {validation['medical_score']*100:.0f}%")
                    else:
                        if not validation.get("type_match", True):
                            st.warning(validation["reason"])

                        prediction = xray_detector.predict_image(img_pil)
                        label, explanation, risk_level, style = xray_detector.interpret_result(prediction)

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(uploaded_img, caption="صورة الأشعة المُحلَّلة", use_container_width=True)
                    # Grad-CAM heatmap
                    if gradcam_available() and xray_detector.model and xray_detector.processor:
                        with st.spinner("توليد خريطة التنشيط (Grad-CAM)..."):
                            heatmap = generate_gradcam_heatmap(
                                xray_detector.model, xray_detector.processor,
                                img_pil
                            )
                            if heatmap:
                                st.image(heatmap, caption="خريطة Grad-CAM — مناطق تركيز النموذج", use_container_width=True)

                with col2:
                    if prediction:
                        color_map = {"success": "#28A745", "warning": "#FF9800", "danger": "#DC3545", "error": "#6C757D"}
                        color = color_map.get(style, "#6C757D")
                        st.markdown(f"""
                        <div style="text-align:center; padding:20px; border-radius:16px;
                                    background:{color}11; border:2px solid {color}; margin-top:10px;">
                            <h3 style="color:{color}; margin:0;">{label}</h3>
                            <p style="color:#1E2028; margin-top:10px; font-size:15px;">{explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("#### توزيع الاحتمالات")
                        prob_items = [
                            ("🫀 تضخم القلب", prediction.get("prob_cardiomegaly", 0)),
                            ("💧 وذمة رئوية", prediction.get("prob_edema", 0)),
                            ("🫁 تصلب رئوي", prediction.get("prob_consolidation", 0)),
                            ("🔴 التهاب رئوي", prediction.get("prob_pneumonia", 0)),
                            ("🟢 سليم", prediction.get("prob_no_finding", 0)),
                        ]
                        for plabel, prob in prob_items:
                            st.progress(prob, text=f"{plabel}: {prob*100:.1f}%")

                        if risk_level == "high":
                            st.error("يُرجى مراجعة أخصائي أمراض صدرية فوراً.")
                        elif risk_level == "moderate":
                            st.warning("يُنصح بمراجعة طبيب لتقييم الحالة.")
                        else:
                            st.success("لم تُرصد مؤشرات مرضية. يُنصح بالمتابعة الدورية.")
                    else:
                        st.error("تعذّر تحليل الصورة.")

    with st.expander("تشخيص النظام"):
        if xray_detector:
            status_color = "green" if xray_detector.model else "red"
            model_status = "Connected (ViT CheXpert)" if xray_detector.model else "Unavailable"
            st.markdown(f"**Model Status:** <span style='color:{status_color};'>{model_status}</span>", unsafe_allow_html=True)
            if xray_detector.load_error:
                st.warning(f"خطأ تقني: {xray_detector.load_error}")
            st.caption("البنية: Vision Transformer (ViT) | المصدر: CheXpert (Stanford) | الفئات: 5 حالات صدرية | الدقة: 98.46%")
        else:
            st.error("لم يتم تهيئة الكاشف.")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 🩹 PAGE: DERMATOLOGY (SKIN LESIONS)
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "derm_scanner":
    st.markdown('<h1 class="main-title">فاحص الأمراض الجلدية</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5A6072;">نموذج <b>DinoV2</b> لتصنيف 31 حالة جلدية (ISIC 2018 + Atlas Dermatology) — دقة 95.57%. أداة تعليمية فقط، لا تُغني عن طبيب الجلد.</p>', unsafe_allow_html=True)

    st.markdown('<div class="data-card">', unsafe_allow_html=True)

    derm_detector = load_derm_detector()

    if derm_detector is None or derm_detector.model is None:
        st.error("**نموذج الجلد غير متاح**")
        err_msg = derm_detector.load_error if derm_detector else "لم يتم تحميل الكاشف"
        st.warning(f"تنبيه تقني: {err_msg}")
        st.info("النموذج يتم تحميله تلقائياً من Hugging Face عند أول استخدام.")
    else:
        st.success("وحدة التحليل جاهزة — DinoV2 Skin Disease (31 فئة)")

    st.markdown("---")
    st.markdown("""
    <div style="background:linear-gradient(135deg, #f8f9fa, #fff8f0); border-radius:14px; padding:18px; margin-bottom:16px; border:1px solid #f0e8e0;">
        <h4 style="margin:0 0 10px; color:#1E2028;">الحالات المكتشفة (31 فئة)</h4>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <div style="background:#ffebee; padding:10px 14px; border-radius:10px;"><b style="color:#c62828;">خطورة عالية</b><br><small>الميلانوما، BCC، SCC، الجذام، الذئبة، الفطار الفطراني</small></div>
            <div style="background:#fff3e0; padding:10px 14px; border-radius:10px;"><b style="color:#e65100;">خطورة متوسطة</b><br><small>الصدفية، الهربس، الحزاز المسطح، القوباء، التقرن السفعي</small></div>
            <div style="background:#e8f5e9; padding:10px 14px; border-radius:10px;"><b style="color:#2e7d32;">خطورة منخفضة</b><br><small>الشامات، التقرن الدهني، النخالية الوردية، آفات وعائية</small></div>
            <div style="background:#e3f2fd; padding:10px 14px; border-radius:10px;"><b style="color:#1565c0;">31 فئة إجمالية</b><br><small>ISIC 2018 + Atlas Dermatology | DinoV2 | 95.57%</small></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    disclaimer_agreed = st.checkbox(
        "أوافق على أن هذه الأداة تجريبية للأغراض التعليمية فقط ولا تعطي تشخيصاً طبياً نهائياً.",
        key="derm_disclaimer"
    )

    if not disclaimer_agreed:
        st.info("يرجى الموافقة على الشروط أعلاه لتفعيل رفع صور الجلد.")
    else:
        uploaded_img = st.file_uploader(
            "تحميل صورة لِسْيونة جلدية",
            type=["jpg", "jpeg", "png"],
            key="derm_uploader"
        )

        if st.button("تحليل لِسْيونة الجلد", use_container_width=True, type="primary", key="derm_analyze"):
            if not uploaded_img:
                st.warning("يرجى تحميل صورة أولاً.")
            elif derm_detector is None or derm_detector.model is None:
                st.error("النموذج غير متاح.")
            else:
                with st.spinner("جاري تحليل الصورة..."):
                    img_pil = PILImage.open(uploaded_img)
                    prediction = None
                    validation = validate_medical_image(img_pil, expected_type="derm")
                    if not validation.get("valid", True):
                        st.warning(validation.get("reason", "تحقق من صورة الجلد."))
                    prediction = derm_detector.predict_image(img_pil)
                    label, explanation, risk_level, style = derm_detector.interpret_result(prediction)

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(uploaded_img, caption="صورة الجلد المُحلَّلة", use_container_width=True)

                with col2:
                    if prediction:
                        color_map = {"success": "#28A745", "warning": "#FF9800", "danger": "#DC3545", "error": "#6C757D"}
                        color = color_map.get(style, "#6C757D")
                        st.markdown(f"""
                        <div style="text-align:center; padding:20px; border-radius:16px;
                                    background:{color}11; border:2px solid {color}; margin-top:10px;">
                            <h3 style="color:{color}; margin:0;">{label}</h3>
                            <p style="color:#1E2028; margin-top:10px; font-size:15px;">{explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("#### أعلى 5 احتمالات")
                        top5 = prediction.get("top5", [])
                        for item in top5:
                            p = item["probability"]
                            st.progress(p, text=f"{item['class_label_ar']}: {p*100:.1f}%")
                        if risk_level == "high":
                            st.error("يُرجى مراجعة طبيب جلدية فوراً.")
                        elif risk_level == "moderate":
                            st.warning("يُنصح بمراجعة طبيب جلدية.")
                        else:
                            st.success("يُنصح بالفحص الدوري عند طبيب الجلد.")
                    else:
                        st.error("تعذّر تحليل الصورة.")

    with st.expander("تشخيص النظام"):
        if derm_detector:
            status_color = "green" if derm_detector.model else "red"
            st.markdown(f"**Model Status:** <span style='color:{status_color};'>{'Connected' if derm_detector.model else 'Unavailable'}</span>", unsafe_allow_html=True)
            if derm_detector.load_error:
                st.warning(f"خطأ تقني: {derm_detector.load_error}")
        else:
            st.error("لم يتم تهيئة الكاشف.")

    st.markdown('</div>', unsafe_allow_html=True)
