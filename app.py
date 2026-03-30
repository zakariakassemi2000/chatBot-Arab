# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Enhanced Medical Platform (SaaS / Zellige Edition)
═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import io
import base64
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# ── Suppress TensorFlow info/warning messages ──
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import PIL.Image as PILImage

# ─────────────────────────────────────────────────────────────
# LOGGER CONFIGURATION
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shifa.app")

# ─────────────────────────────────────────────────────────────
# GLOBAL SAFETY & CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHIFA AI | المنصة الطبية الذكية",
    page_icon="⚜️",
    layout="wide",
    initial_sidebar_state="auto",
)

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

# ─────────────────────────────────────────────────────────────
# IMPORT AGENTS (modular architecture)
# ─────────────────────────────────────────────────────────────
try:
    from agents.orchestrator import Orchestrator
    from engine.audio import speech_to_text_arabic, convert_audio_to_wav
except ImportError as e:
    logger.error(f"Import error: {e}")
    st.error("خطأ في تحميل المكونات الأساسية. يرجى التحقق من التثبيت.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# CONSTANTS & UTILS
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "Stylized_Heart_and_Cross_Logo_for_SHIFA_AI__1_-removebg-preview.png"
PATTERN_PATH = BASE_DIR / "pattern.png"
HISTORY_FILE = BASE_DIR / "consultation_history.json"

@st.cache_data
def get_b64(path_str):
    """Convert image to base64 string (Cached for performance)"""
    path = Path(path_str)
    try:
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception as e:
        logger.error(f"Error loading logo: {e}")
    return None

LOGO_B64 = get_b64(str(LOGO_PATH))
LOGO_SRC = f"data:image/png;base64,{LOGO_B64}" if LOGO_B64 else ""

PATTERN_B64 = get_b64(str(PATTERN_PATH))
PATTERN_SRC = f"data:image/png;base64,{PATTERN_B64}" if PATTERN_B64 else ""

def save_history(messages):
    """Save consultation history"""
    if not messages:
        return
    
    if "local_history" not in st.session_state:
        st.session_state["local_history"] = []
        
    session_id = str(st.session_state.get("session_id", time.time()))
    
    existing_idx = next((idx for idx, s in enumerate(st.session_state["local_history"]) if s.get("id") == session_id), None)
    
    entry = {
        "id": session_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": messages[0]["content"][:50] + "..." if messages else "استشارة جديدة",
        "messages": messages
    }
    
    if existing_idx is not None:
        st.session_state["local_history"][existing_idx] = entry
    else:
        st.session_state["local_history"].insert(0, entry)

# ─────────────────────────────────────────────────────────────
# CSS FOR MODERN MOROCCAN SAAS UI / UX
# ─────────────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
    <style>
        /* Base Moroccan Typography */
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

        /* Theme Variables - Zellige & Deep Nights */
        :root {
            --z-green: #16a34a;
            --z-green-hover: #15803d;
            --z-green-light: rgba(22, 163, 74, 0.15);
            --z-red: #dc2626;
            --z-gold: #d4af37;
            --z-beige: #fef3c7;
            --z-bg: #0f172a;
            --z-card: #1e293b;
            --z-text: #f8fafc;
            --z-muted: #94a3b8;
        }

        /* RTL & Font Initialization (excluding icon fonts) */
        html, body {
            direction: rtl;
            text-align: right;
        }
        
        p, h1, h2, h3, h4, h5, h6, li, a {
            font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif;
            letter-spacing: 0.2px;
        }
        
        /* Protect Streamlit internal Material Icons */
        .material-symbols-rounded, 
        .material-symbols-outlined, 
        [data-testid="stIconMaterial"] {
            font-family: 'Material Symbols Rounded' !important;
            font-weight: normal;
        }
        
        /* Hide Default Sidebar Nav for controlled routing */
        [data-testid="stSidebarNav"] { display: none !important; }

        /* The subtle Zellige App Background (Infinite Resolution CSS) */
        .stApp {
            background-color: #0A1628;  /* Deep navy */
            background-image: 
                /* Soft Teal glow */
                radial-gradient(circle at 15% 50%, rgba(26, 139, 128, 0.08), transparent 40%),
                /* Deep Red glow */
                radial-gradient(circle at 85% 30%, rgba(160, 32, 47, 0.06), transparent 40%),
                /* Cobalt Blue glow */
                radial-gradient(circle at 50% 90%, rgba(26, 58, 128, 0.08), transparent 50%),
                /* Ambient Gradient */
                linear-gradient(135deg, rgba(10, 22, 40, 0) 0%, rgba(13, 30, 54, 0.9) 100%);
            background-size: 100% 100%, 100% 100%, 100% 100%, 100% 100%;
            background-attachment: fixed;
            color: var(--z-text);
        }
        [data-testid="stHeader"] { background: transparent !important; }

        /* Sidebar Glassmorphism */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.85) !important;
            border-left: 1px solid var(--z-green-light) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }

        /* Native Streamlit Container -> Zellige Card Styling */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--z-green-light) !important;
            background-color: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(8px);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        /* 
           Native Streamlit Buttons upgraded to Interactive Moroccan Cards 
           We target secondary buttons to act as dynamic grid cards!
        */
        div[data-testid="stButton"] > button[kind="secondary"] {
            background-color: var(--z-card);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            color: var(--z-text);
            min-height: 85px;
            font-size: 1.15rem;
            font-weight: 600;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            border-color: var(--z-green);
            background-color: rgba(22, 163, 74, 0.08);
            transform: translateY(-4px);
            box-shadow: 0 10px 20px -5px rgba(22, 163, 74, 0.2);
            color: var(--z-beige); /* Subtle gold text on hover */
        }
        div[data-testid="stButton"] > button[kind="secondary"] p {
            font-size: 1.1rem;
            margin: 0;
            white-space: pre-wrap; /* allow nice text flow */
        }

        /* Primary CTA Buttons */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, var(--z-green), var(--z-green-hover));
            color: #ffffff !important;
            border: none;
            border-radius: 12px;
            min-height: 60px;
            font-weight: 700;
            font-size: 1.2rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(22, 163, 74, 0.25);
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: linear-gradient(135deg, var(--z-green-hover), #14532d);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(22, 163, 74, 0.4);
        }

        /* Chat Input Field Focus Glow */
        [data-testid="stChatInput"] {
            border-radius: 16px;
            border: 1px solid var(--z-green-light) !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--z-green) !important;
            box-shadow: 0 0 0 2px rgba(22, 163, 74, 0.2) !important;
        }

        /* Chat Messages */
        [data-testid="stChatMessage"] {
            background: rgba(30,41,59,0.2) !important;
            border-radius: 12px;
            padding: 1.5rem !important;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.03);
        }
        /* Auth Screen Native Style */
        .auth-card {
            background-color: var(--z-card);
            border-top: 4px solid var(--z-green);
            border-radius: 16px;
            padding: 3rem;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
            text-align: center;
        }
        
    </style>
    """, unsafe_allow_html=True)
    
    if PATTERN_SRC:
        st.markdown(f"""
        <style>
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: #C9A855; /* Force tinted Gold for this specific pattern! */
            -webkit-mask-image: url("{PATTERN_SRC}");
            mask-image: url("{PATTERN_SRC}");
            -webkit-mask-size: 150px 150px;
            mask-size: 150px 150px;
            -webkit-mask-repeat: repeat;
            mask-repeat: repeat;
            opacity: 0.15;
            z-index: -1;
            pointer-events: none;
        }}
        </style>
        """, unsafe_allow_html=True)
    st.markdown("""
    <style>
        /* App Titles */
        .moroccan-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, var(--z-beige), #ffffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0px;
            padding-bottom: 5px;
        }
        .moroccan-subtitle {
            text-align: center;
            color: var(--z-muted);
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 3rem;
        }

        /* Minimal Zellige Emergency Alert */
        .zellige-alert {
            background: rgba(220, 38, 38, 0.05); /* very subtle red */
            border-right: 4px solid var(--z-red);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2.5rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .zellige-alert-title { color: #f8fafc; font-weight: 700; font-size: 1.2rem; display: flex; align-items: center; gap: 0.8rem; }
        .zellige-alert-text { color: var(--z-muted); font-size: 0.95rem; margin-top: 0.2rem; }
        .zellige-alert-numbers { display:flex; gap: 10px; }
        .zellige-alert-numbers span {
            background: rgba(220, 38, 38, 0.15);
            color: #fca5a5;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            font-weight: 700;
            border: 1px solid rgba(220, 38, 38, 0.2);
            letter-spacing: 0.5px;
        }
        
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# AUTHENTICATION GATE
# ─────────────────────────────────────────────────────────────
def _check_auth() -> bool:
    if st.session_state.get("_authenticated"):
        return True

    try:
        expected = st.secrets["APP_PASSWORD"]
    except Exception:
        expected = os.environ.get("APP_PASSWORD", "shifa2026")

    inject_custom_css()

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<div style='margin-bottom: 10vh;'></div>", unsafe_allow_html=True)
        with st.container(border=True): # Streamlit native card!
            if LOGO_SRC:
                st.markdown(f"<div style='text-align:center;'><img src='{LOGO_SRC}' style='height:85px; margin-bottom:15px;'><br><h2 style='margin-bottom:0;'>SHIFA AI</h2><p style='color:#94a3b8; margin-bottom: 2rem;'>المنصة الطبية الذكية · Accès Sécurisé</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:center;'><div style='font-size:3.5rem; color:#16a34a;'>⚜️</div><h2 style='margin-bottom:0;'>SHIFA AI</h2><p style='color:#94a3b8; margin-bottom: 2rem;'>المنصة الطبية الذكية · Accès Sécurisé</p></div>", unsafe_allow_html=True)
            password = st.text_input("كلمة المرور / Mot de passe", type="password", placeholder="••••••••", label_visibility="collapsed")
            if st.button("دخول / Connexion", type="primary", width="stretch"):
                if password == expected:
                    st.session_state["_authenticated"] = True
                    st.rerun()
                else:
                    st.error("❌ كلمة المرور خاطئة / Mot de passe incorrect")
            st.caption("<div style='text-align:center; margin-top:1rem; color:#64748b;'>للمحترفين الطبيين فقط</div>", unsafe_allow_html=True)
    return False

if not _check_auth():
    st.stop()

inject_custom_css()

# ─────────────────────────────────────────────────────────────
# LOAD SYSTEM (Agent-based orchestrator)
# ─────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_medical_system():
    try:
        return Orchestrator.load()
    except Exception as e:
        logger.error(f"Orchestrator load failed: {e}", exc_info=True)
        return None

orch = load_medical_system()

# ─────────────────────────────────────────────────────────────
# SESSION & RATE LIMITER
# ─────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "page": "home",
    "session_id": str(time.time()),
    "quick_question": None,
    "last_request_time": 0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

class RateLimiter:
    def __init__(self, max_requests=10, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_proceed(self):
        now = time.time()
        self.requests = [t for t in self.requests if now - t < self.time_window]
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

if "rate_limiter" not in st.session_state:
    st.session_state.rate_limiter = RateLimiter()

DB_STATUS = orch.db_ready if orch else False
AI_STATUS = orch.llm_ready if orch else False

# ─────────────────────────────────────────────────────────────
# TOP QUICK ACTIONS BAR (Visible when NOT on home)
# ─────────────────────────────────────────────────────────────
if st.session_state.page != "home":
    # Layout with columns for clean navigation
    cols_nav = st.columns([1.5, 1.5, 1.5, 1.5, 4]) # spacing
    nav_actions = [
        ("🏠 الرئيسية", "home", True),
        ("🔍 فحص مبدئي", "scanner", True),
        ("🎤 مساعد صوتي", "voice", True),
        ("📍 الرعاية", "pages/10_🏥_الرعاية_القريبة.py", False)
    ]
    
    for idx, (label, target, is_internal) in enumerate(nav_actions):
        with cols_nav[idx]:
            if st.button(label, key=f"top_nav_{idx}", width="stretch"):
                if is_internal:
                    st.session_state.page = target
                    st.rerun()
                else:
                    st.switch_page(target)
    st.markdown("<hr style='border:0; height:1px; background:linear-gradient(to right, transparent, #16a34a, transparent); opacity:0.3; margin-top:5px;'/>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO_SRC:
        st.markdown(f"""
            <div style="text-align:center; padding: 1rem 0;">
                <img src="{LOGO_SRC}" style="height:90px; margin-bottom:10px;">
                <h2 style="color:#d4af37; margin:8px 0 4px; font-family:'Cairo'; font-weight:800; font-size:1.6rem;">SHIFA AI</h2>
                <p style="color:#94a3b8; font-size:0.9rem; margin:0;">الذكاء الاصطناعي الطبي</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align:center; padding: 1rem 0;">
                <div style="font-size:3.5rem; color:#16a34a;">⚜️</div>
                <h2 style="color:#d4af37; margin:8px 0 4px; font-family:'Cairo'; font-weight:800; font-size:1.6rem;">SHIFA AI</h2>
                <p style="color:#94a3b8; font-size:0.9rem; margin:0;">الذكاء الاصطناعي الطبي</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.page != "home":
        if st.button("⬅️ العودة للرئيسية", width="stretch", type="primary"):
            st.session_state.page = "home"
            st.rerun()
    else:
        st.info("✓ أنت في الصفحة الرئيسية")
        
    st.markdown("<hr style='border-color: rgba(22, 163, 74, 0.2);'/>", unsafe_allow_html=True)
    
    if st.session_state.messages:
        st.caption(f"💬 المحادثة الحالية: {len(st.session_state.messages)} رسالة")
        if st.button("🗑️ حوار جديد", width="stretch"):
            st.session_state.messages = []
            st.session_state.session_id = str(time.time())
            st.rerun()
            
    # ── Link to Docteur+User Portal ──
    st.markdown("<hr style='border-color: rgba(22, 163, 74, 0.2);'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; margin-bottom:0.5rem;">
        <span style="color:#d4af37; font-size:1rem; font-weight:700;">🏥 فضاء الطبيب والمريض</span><br/>
        <span style="color:#94a3b8; font-size:0.8rem;">إدارة المرضى · المواعيد · التحاليل</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 تشغيل فضاء الطبيب", width="stretch"):
        import subprocess as _sp
        doctor_app = str(BASE_DIR / "partie Docteur+User" / "main.py")
        _sp.Popen(
            [sys.executable, "-m", "streamlit", "run", doctor_app, "--server.port", "8503"],
            cwd=str(BASE_DIR / "partie Docteur+User")
        )
        st.success("✅ تم التشغيل! افتح http://localhost:8503")
    st.link_button("🔗 فتح فضاء الطبيب", "http://localhost:8503", use_container_width=True)

    st.markdown("""
    <div style="background:rgba(22, 163, 74, 0.05); border-right:3px solid #16a34a; padding:12px; margin-top:2rem; border-radius:8px 0 0 8px;">
        <p style="color:#a7f3d0; font-size:0.8rem; margin:0; line-height:1.5;">
            <b>تنبيه إخلاء المسؤولية:</b><br/>
            المنصة توفر دعماً معلوماتياً. لا تغني أبدًا عن استشارة الطبيب المختص أو زيارة العيادة.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: HOME (LANDING DASHBOARD)
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown('<div class="moroccan-title">شفاء AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="moroccan-subtitle">مساعدك الطبي الذكي لتقييم الأعراض والتوجيه الصحي</div>', unsafe_allow_html=True)
        
    # ── Emergency Banner (Zellige Style) ──
    st.markdown("""
        <div class="zellige-alert">
            <div class="zellige-alert-title">
                <span style="font-size:1.8rem;">🚨</span>
                <div>
                    <div>تنبيه طوارئ طبية فعلية؟</div>
                    <div class="zellige-alert-text">تواصل فورا مع خدمات الطوارئ لإنقاذ الحياة. لا تنتظر التطبيق.</div>
                </div>
            </div>
            <div class="zellige-alert-numbers">
                <span>🚑 الإسعاف: 15</span>
                <span>🚓 الشرطة: 19</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # ── Main CTA ──
    col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
    with col_cta2:
        if st.button("🚀 ابدأ محادثة أو تشخيص جديد الآن", width="stretch", type="primary"):
            st.session_state.page = "chat"
            st.rerun()
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # ── System Metrics Panel (Native Elements) ──
    st.subheader("📊 حالة الجاهزية والنظام")
    cols_metrics = st.columns(4)
    
    with cols_metrics[0]:
        with st.container(border=True):
            st.markdown(f"<div style='text-align:center;'><h3>🧠</h3><p style='color:#94a3b8; margin:0;'>محرك التحليل</p><h4 style='color:{'#16a34a' if AI_STATUS else '#dc2626'}; margin:0;'>{'نشط ✔' if AI_STATUS else 'غير متصل ❌'}</h4></div>", unsafe_allow_html=True)
        
    with cols_metrics[1]:
        with st.container(border=True):
            st.markdown(f"<div style='text-align:center;'><h3>📚</h3><p style='color:#94a3b8; margin:0;'>قاعدة المعرفة</p><h4 style='color:{'#16a34a' if DB_STATUS else '#fbbf24'}; margin:0;'>{'محدثة بالكامل ✔' if DB_STATUS else 'جاري التحديث⏳'}</h4></div>", unsafe_allow_html=True)

    with cols_metrics[2]:
        msg_count = len(st.session_state.get("local_history", []))
        with st.container(border=True):
            st.markdown(f"<div style='text-align:center;'><h3>👥</h3><p style='color:#94a3b8; margin:0;'>استشاراتك</p><h4 style='color:#38bdf8; margin:0;'>{msg_count} محفوظ بنجاح</h4></div>", unsafe_allow_html=True)
        
    with cols_metrics[3]:
        with st.container(border=True):
            st.markdown("<div style='text-align:center;'><h3>⚡</h3><p style='color:#94a3b8; margin:0;'>الأداء والاستجابة</p><h4 style='color:#d4af37; margin:0;'>السرعة المستقرة</h4></div>", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    
    # ── Interactive Services Menu (Native Buttons styled via CSS) ──
    st.subheader("🛠️ باقة الخدمات الطبية المتقدمة")
    
    cards = [
        ("💬", "محادثة طبية", "chat", None),
        ("🎙️", "المساعد الصوتي", "voice", None),
        ("🔬", "تحليل الصور", "vision", None),
        ("🩺", "فاحص الأعراض", "scanner", None),
        ("🧮", "الحاسبات الطبية", "calculators", None),
        ("📚", "البحث المعرفي", "database", None),
        ("🏥", "الرعاية القريبة", None, "pages/10_🏥_الرعاية_القريبة.py"),
        ("💊", "التفاعلات", None, "pages/07_💊_التفاعلات_الدوائية.py"),
        ("📑", "ترتيب التقارير", None, "pages/08_📋_ترتيب_التقارير.py"),
        ("🦩", "ذكاء متعدد", None, "pages/06_🦩_المساعد_متعدد_الوسائط.py"),
        ("📊", "مقارنة النماذج", None, "pages/09_📊_مقارنة_النماذج.py"),
        ("📜", "الأرشيف", "history", None),
    ]

    for i in range(0, len(cards), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(cards):
                c = cards[i + j]
                with cols[j]:
                    # Primary rendering of the card as a simple styled button
                    if st.button(f"{c[0]}  {c[1]}", key=f"srv_btn_{i+j}", width="stretch"):
                        if c[2]:
                            st.session_state.page = c[2]
                            st.rerun()
                        elif c[3]:
                            st.switch_page(c[3])

# ─────────────────────────────────────────────────────────────
# PAGE: CHAT (MAIN)
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "chat":
    st.markdown('<div class="moroccan-title" style="font-size:2.8rem;">المساعد الطبي الذكي</div>', unsafe_allow_html=True)
    st.markdown("<div class='moroccan-subtitle'>اطرح أسئلتك وثق بمحرك SHIFA للإجابة الدقيقة والآمنة</div>", unsafe_allow_html=True)
    
    if not DB_STATUS:
        st.error("⚠️ قاعدة المعرفة غير متاحة حالياً. يرجى إعداد النظام أولاً. (يتم التنزيل لمرة واحدة)")
        if st.button("🔧 بناء قاعدة المعرفة الآن", type="primary"):
            try:
                with st.spinner("جاري تنزيل البيانات (قد يستغرق بضع دقائق)..."):
                    if orch:
                        orch.setup_knowledge_base(max_samples=5000)
                st.success("تم إعداد قاعدة المعرفة بنجاح!")
                st.rerun()
            except Exception as e:
                logger.error(f"KB setup failed: {e}")
                st.error("فشل الإعداد. يرجى التحقق من الاتصال بالإنترنت.")
        st.stop()
    
    # ── Welcome / Quick Suggestions ──
    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("<h4 style='color:var(--z-beige); margin-bottom:1rem;'>💡 اقتراحات للبدء:</h4>", unsafe_allow_html=True)
            QUICK_QUESTIONS = [
                "ما هي الأعراض المبكرة للسكري؟",
                "كيف أعالج ضغط الدم المرتفع طبيعياً؟",
                "متى يجب الذهاب للطوارئ عند ألم الصدر؟"
            ]
            cols_q = st.columns(3)
            for i, q in enumerate(QUICK_QUESTIONS):
                with cols_q[i]:
                    if st.button(q, key=f"quick_{i}", width="stretch"):
                        st.session_state.quick_question = q
                        st.rerun()
    
    # ── Chat Display ──
    for msg in st.session_state.messages:
        avatar_icon = "👤" if msg["role"] == "user" else ("⚜️" if not LOGO_SRC else "🩺")
        with st.chat_message(msg["role"], avatar=avatar_icon):
            st.markdown(msg["content"])
    
    # ── Chat Input ──
    user_input = st.chat_input("اكتب استفسارك الطبي هنا...")
    if st.session_state.quick_question:
        user_input = st.session_state.quick_question
        st.session_state.quick_question = None
    
    if user_input:
        if not st.session_state.rate_limiter.can_proceed():
            st.error("⚠️ لقد تجاوزت الحد المسموح. يرجى الانتظار دقيقة.")
            st.stop()
            
        st.session_state.last_request_time = time.time()
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        with st.chat_message("assistant", avatar=("⚜️" if not LOGO_SRC else "🩺")):
            with st.spinner("يعمل محرك الاستدلال على الإجابة..."):
                try:
                    if orch:
                        response = orch.handle(user_input, history=st.session_state.messages[:-1])
                        answer = response.answer if response else "عذراً، حدث خطأ داخلي في الخادم."
                    else:
                        answer = "النظام غير متصل لغياب مكون Orchestrator."
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    answer = "حدث خطأ غير متوقع. يرجى المحاولة لاحقاً."
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            save_history(st.session_state.messages)
        st.rerun()

# ─────────────────────────────────────────────────────────────
# PAGE: VOICE
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "voice":
    st.markdown('<div class="moroccan-title" style="font-size:2.8rem;">المساعد الصوتي</div>', unsafe_allow_html=True)
    st.markdown("<div class='moroccan-subtitle'>تحدث بصوتك مباشرة مع الذكاء الاصطناعي بدلاً من الكتابة</div>", unsafe_allow_html=True)
    
    try:
        import tempfile
        from audio_recorder_streamlit import audio_recorder
        from engine.audio import text_to_speech_arabic

        with st.container(border=True):
            st.markdown("<h3 style='text-align:center;'>🎙️ جهاز الاستقبال</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                audio_bytes = audio_recorder(text="انقر هنا لبدء التسجيل", recording_color="#dc2626", neutral_color="#16a34a")

        if audio_bytes and len(audio_bytes) > 2000:
            with st.spinner("التسجيل قيد المعالجة..."):
                try:
                    # Convert raw browser webm/ogg direct to WAV bytes
                    wav_data = convert_audio_to_wav(audio_bytes)
                    text, stt_err = speech_to_text_arabic(wav_data)
                except Exception as e:
                    text, stt_err = None, str(e)
                    
                if stt_err:
                    st.error(f"⚠️ خطأ في التعرف على الصوت: {stt_err}")
                elif text:
                    st.info(f"🎤 سمعتك تقول: **{text}**")

                    with st.spinner("🤖 جاري توليد إجابة مسموعة..."):
                        response = orch.handle(text, history=[]) if orch else None

                    if response:
                        if response.override_ui:
                            st.markdown(response.override_ui, unsafe_allow_html=True)
                        if getattr(response, "severity", None) in ["critique", "élevée"]:
                            from engine.nearby_care import render_nearby_care
                            render_nearby_care(response.severity)
                        answer = response.answer
                    else:
                        answer = "النظام لا يستطيع الإجابة حالياً."

                    st.success(f"النتيجة: {answer}")

                    audio_response = text_to_speech_arabic(answer)
                    if audio_response:
                        st.audio(audio_response, format="audio/mp3", autoplay=True)
                else:
                    st.warning("الصوت غير واضح تماماً. يرجى التحدث في بيئة هادئة.")
    except ImportError:
        st.error("⚠️ مكون التسجيل الصوتي غير مثبت (`audio_recorder_streamlit`).")

# ─────────────────────────────────────────────────────────────
# PAGE: VISION
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "vision":
    st.markdown('<div class="moroccan-title" style="font-size:2.8rem;">معمل تحليل الصور</div>', unsafe_allow_html=True)
    st.warning("🚨 **إخلاء مسؤولية تنظيمي:** هذا المعمل مخصص للأغراض الأكاديمية و المعرفية.")

    _VISION_MAP = {
        "🔴 الجلدية (Dermato / أمراض الجلد)": "dermato",
        "🫁 أشعة الصدر (X-Ray)": "xray",
        "🧠 رنين الدماغ المغناطيسي (MRI)": "brain_mri",
        "🩺 تحاليل الأنسجة السرطانية": "cancer",
        "🔬 تصوير الثدي الشعاعي": "breast",
    }
    
    with st.container(border=True):
        col_sel, col_up = st.columns([1, 1.5])
        with col_sel:
            vision_label = st.selectbox("حدد نوع الأشعة أو الصورة:", list(_VISION_MAP.keys()))
            vision_type = _VISION_MAP[vision_label]
        with col_up:
            uploaded_file = st.file_uploader("قم برفع الصورة هنا (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = PILImage.open(uploaded_file)
        
        with st.container(border=True):
            col_img, col_res = st.columns([1, 1.5])
            with col_img:
                st.image(image, caption="الصورة الأصلية", width="stretch", clamp=True)
            
            with col_res:
                if st.button("🚀 بدء فحص الشبكات العصبية", type="primary", width="stretch"):
                    with st.spinner("يتم مطابقة الأنماط البيولوجية..."):
                        try:
                            if orch:
                                result = orch.analyze_image(image, vision_type)
                                if result.success:
                                    conf = result.metadata.get("confidence", 0.0)
                                    cls  = result.metadata.get("class", "غير محدد")
                                    sev  = result.metadata.get("severity", "indéfini")
                                    
                                    # Mapped color logic based on Moroccan theme
                                    _COLOR = {"critique":"#dc2626","élevée":"#f97316","modérée":"#d4af37","faible":"#16a34a"}
                                    color = _COLOR.get(sev, "#94a3b8")
                                    
                                    st.markdown(f"""
                                    <div style="padding:1.5rem; border-radius:12px; background:rgba(15,23,42,0.8); 
                                                border-right: 6px solid {color}; margin-bottom:1rem;">
                                      <h3 style="color:{color}; margin-top:0;">التشخيص المحوسب: {cls}</h3>
                                      <div style="color:#d4af37; font-size:1.1rem;">درجة اليقين للخوارزمية: <b>{conf*100:.1f}%</b></div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.write(result.answer)
                                    
                                    if result.metadata.get("all_probs"):
                                        probs_df = pd.DataFrame({
                                            "نسبة التطابق (%)": [p*100 for p in result.metadata["all_probs"].values()],
                                            "الفئة المتوقعة": list(result.metadata["all_probs"].keys())
                                        }).set_index("الفئة المتوقعة")
                                        st.bar_chart(probs_df, height=250)
                                else:
                                    st.warning(result.answer or "لم تتمكن النماذج من تقييم الصورة بوضوح.")
                            else:
                                st.error("علاجات الذكاء الصناعي معطلة.")
                        except Exception as e:
                            logger.error(f"Vision error: {e}")
                            st.error("مشكلة برمجية أثناء التحليل.")

# ─────────────────────────────────────────────────────────────
# PAGE: SCANNER
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "scanner":
    st.markdown('<div class="moroccan-title" style="font-size:2.8rem;">نظام التقييم السريري الذكي</div>', unsafe_allow_html=True)
    st.markdown("<div class='moroccan-subtitle'>يقوم النظام الاستدلالي بالبحث عن ارتباطات الأعراض لاقتراح الحالات الممكنة</div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        with st.form("symptom_form"):
            st.subheader("بيانات المريض والأعراض الأساسية")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("عمر المريض", min_value=1, max_value=120, value=30)
                gender = st.selectbox("الجنس البيولوجي", ["ذكر", "أنثى"])
            with col2:
                duration = st.selectbox("المدة الزمنية للأعراض", ["أقل من 24 ساعة", "من يوم إلى 3 أيام", "حوالي أسبوع", "أكثر من أسبوع"])
                severity = st.select_slider("مدى حدة وقسوة الألم", options=["خفيف محتمل", "متوسط", "شديد ولا يطاق"])
            
            symptoms = st.text_area("أعطنا وصفاً مفصلاً (المكان، طبيعة الوجع، الشدة...)", height=120, placeholder="مثال: أشعر بصداع نصفي نابض مع غثيان عند التعرض للضوء...")
            history = st.text_input("الأمراض المزمنة أوالأدوية الحالية (إن وجد)")
            
            submitted = st.form_submit_button("إرسال للتحليل الذكي ✨", width="stretch")
        
    if submitted:
        if len(symptoms.strip()) < 5:
            st.warning("يرجى وصف الأعراض بدقة أكبر ليتمكن الذكاء من مساعدتك.")
        else:
            with st.spinner("يتم الآن دمج البيانات ومقارنتها بقاعدة بيانات التشخيصات..."):
                try:
                    if orch:
                        result = orch.scan_symptoms(
                            symptoms, age=age, gender=gender, duration=duration, 
                            severity=severity, medical_history=history
                        )
                        with st.container(border=True):
                            st.markdown("<h3 style='color:#16a34a;'>📋 التقرير المبدئي</h3>", unsafe_allow_html=True)
                            st.write(result.answer if result else "فشل تجميع التقرير.")
                    else:
                        st.error("النظام غير متصل.")
                except Exception as e:
                    logger.error(f"Scan error: {e}")
                    st.error("حدث خطأ تقني داخلي أثناء المسح.")

# ─────────────────────────────────────────────────────────────
# PAGE: CALCULATORS
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "calculators":
    st.markdown('<div class="moroccan-title" style="font-size:2.8rem;">دوال وحسابات القياسات الحيوية</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        calc_type = st.selectbox("اختر المعادلة الطبية المراد قياسها:", ["حاسبة مؤشر كتلة الجسم (BMI)", "حاسبة الاحتياج اليومي للسعرات", "مؤشر صحة القلب العام"])
        
        if "BMI" in calc_type:
            st.markdown("#### المتغيرات الحيوية:")
            col1, col2 = st.columns(2)
            with col1:
                weight = st.number_input("الوزن الإجمالي بالميزان (KG)", min_value=20.0, max_value=300.0, value=75.0)
            with col2:
                height = st.number_input("طول القامة (CM)", min_value=100.0, max_value=250.0, value=175.0)
            
            if height > 0:
                bmi = weight / ((height/100) ** 2)
                st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)
                
                c1, c2 = st.columns([1, 2.5])
                with c1:
                    st.metric("مؤشر الكتلة BMI", f"{bmi:.1f} kg/m²")
                with c2:
                    if bmi < 18.5:
                        st.info("نقص انحداري في الوزن. يوصى بمراجعة برنامج التغذية الخاص بك لضمان الحصول على المعادن الاساسية.")
                    elif bmi < 25:
                        st.success("الوزن ضمن النطاق الصحي والمثالي. حافظ على هذا المجهود الطيب!")
                    elif bmi < 30:
                        st.warning("زيادة في المؤشر. هذا جرس إنذار بسيط لتحسين اختيارات الأكل والحركة اليومية.")
                    else:
                        st.error("مؤشر يدل على السمنة المستوفاة. ترتبط السمنة بارتفاع فرص أمراض الضغط، ينصح بالمتابعة.")
        else:
            st.info("جاري تجميع الخوارزميات الحسابية لهذا القسم في التحديثات القادمة من المنصة.")

# ─────────────────────────────────────────────────────────────
# PAGE: DATABASE
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "database":
    st.markdown('<div class="moroccan-title" style="font-size:2.8rem;">المستكشف البحثي</div>', unsafe_allow_html=True)
    st.markdown("<div class='moroccan-subtitle'>بحث دقيق ومباشر في المراجع العلمية المؤرشفة الخاصة بالنظام</div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        search_term = st.text_input("أدخل المصطلح الطبي للتشريح الاسترشادي...", placeholder="مثال: التهاب الكبد الفيروسي، الفيروس المخلوي، الأسبرين...")
    
    if search_term:
        with st.spinner("جاري مسح المراجع واستخلاص النصوص الدقيقة..."):
            try:
                if orch and hasattr(orch, 'llm'):
                    # Prompt designed precisely to generate a highly professional encyclopedia entry
                    prompt = f"""بصفتك الذكاء الاصطناعي الطبي الأساسي SHIFA، ابحث في قاعدة معارفك عن '{search_term}'.
قم بتوفير تقرير طبي مرجعي دقيق جداً كالتالي (تجنب الإطالة):
- التعريف التفصيلي للمرض أو المصطلح
- الأسباب والأعراض
- مضاعفات محتملة
- طرق العلاج والمعايير السريرية

اكتب بلغة علمية دقيقة جداً ومباشرة. لا تقم بالترحيب ولا تختم بأي عبارة ودية، فقط المرجع الأكاديمي."""
                    
                    response = orch.llm.run(query=prompt, context={"kb_context": "", "intent": "database_search", "history": None})
                    
                    if response and response.success:
                        text_formatted = response.answer.replace('\\n', '<br/>').replace('\n', '<br/>')
                        specialty = "الطب العام والأبحاث (Groq AI Model)"
                        
                        with st.container(border=True):
                            st.markdown("<h3 style='color:#d4af37; margin-bottom:1.5rem;'>📑 المرجع الطبي المطابق (مُولّد الذكاء الاصطناعي):</h3>", unsafe_allow_html=True)
                            
                            bg_card = """
                            <div style="background:rgba(15,23,42,0.6); padding:1.5rem 2rem; border-radius:12px; border-right:4px solid #16a34a; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
                                <div style="display:flex; justify-content:space-between; margin-bottom:1rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:0.8rem;">
                                    <span style="color:#94a3b8; font-size:0.95rem;">التصنيف المرجعي: <b style="color:#a7f3d0;">{specialty}</b></span>
                                    <span style="color:#C9A855; font-size:0.95rem;">✦ SHIFA AI</span>
                                </div>
                                <div style="color:#f8fafc; line-height:1.9; font-size:1.05rem; white-space:pre-wrap;">
                                    {content}
                                </div>
                            </div>
                            """
                            st.markdown(bg_card.format(specialty=specialty, content=text_formatted), unsafe_allow_html=True)
                    else:
                        st.warning("تعذر على محرك Groq الذكي استخراج النص، يرجى المحاولة لاحقاً.")
                else:
                    st.error("الاستعلام المعرفي متوقف مؤقتاً.")
            except Exception as e:
                logger.error(f"Search error: {e}")
                st.error("فشل استخراج البيانات. قد يكون الملف مفقوداً.")

# ─────────────────────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "history":
    st.markdown('<div class="moroccan-title" style="font-size:2.8rem;">محفوظات الاستشارات</div>', unsafe_allow_html=True)
    
    history = st.session_state.get("local_history", [])
    
    if not history:
        st.info("سجلاتك فارغة حالياً نظيفة.")
    else:
        for i, item in enumerate(history):
            with st.expander(f"🕰️ الاستشارة رقم {i+1} : {item['date']} - {item['title']}", expanded=(i==0)):
                for msg in item['messages']:
                    if msg['role'] == 'user':
                        st.markdown(f"**أنت:** {msg['content']}")
                    else:
                        st.markdown(f"**المحرك المساعد (SHIFA):** {msg['content']}")
                st.markdown("---")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:var(--z-muted); font-size:0.85rem; padding:2.5rem 0; margin-top:4rem; border-top: 1px solid rgba(22, 163, 74, 0.2);">
  تمت البرمجة والتحسين والتصميم بواسطة فريق <b style="color:#d4af37;">SHIFA AI</b> © 2026<br/>
  <span style="font-size:0.75rem;">تنويه: النظام للاستخدامات الثقافية والتجريبية ولا يغني عن الطب البشري المعتمد.</span>
</div>
""", unsafe_allow_html=True)