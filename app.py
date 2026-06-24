# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Enhanced Medical Platform (SaaS / Zellige Edition)
═══════════════════════════════════════════════════════════════════════
"""

# ── Kill transformers verbose logging BEFORE any import ──
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── Suppress noisy deprecation warnings from transformers lazy-loading ──
import warnings
# Catch ALL warning categories (UserWarning, FutureWarning, DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*Accessing.*__path__.*")
warnings.filterwarnings("ignore", message=r".*Returning `__path__` instead.*")
warnings.filterwarnings("ignore", message=r".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", message=r".*is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*is deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"transformers.*")

import sys
import io
import base64
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# ── Auth module ──
try:
    from core.user_auth import register_user, login_user, init_db, guest_session
    init_db()
except Exception as _auth_err:
    logging.warning(f"[Auth] Module user_auth non chargé: {_auth_err}")
    register_user = login_user = guest_session = None

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
# Use transparent logo (no background) for white page
LOGO_PATH = BASE_DIR / "Stylized_Heart_and_Cross_Logo_for_SHIFA_AI__1_-removebg-preview.png"
if not LOGO_PATH.exists():
    LOGO_PATH = BASE_DIR / "logo.png"
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
        /* ═══════════════════════════════════════════════════════
           SHIFA AI — Premium Medical Design System v3
           Medical Teal + Trust Blue · Cairo Typography · RTL
           ═══════════════════════════════════════════════════════ */
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            /* ── Primary Medical Palette ── */
            --shifa-primary: #0891B2;
            --shifa-primary-hover: #0E7490;
            --shifa-primary-deep: #155E75;
            --shifa-primary-light: rgba(8, 145, 178, 0.08);
            --shifa-primary-glow: rgba(8, 145, 178, 0.15);
            --shifa-secondary: #2563EB;
            --shifa-secondary-hover: #1D4ED8;
            --shifa-accent: #10B981;
            --shifa-danger: #EF4444;
            --shifa-warning: #F59E0B;

            /* ── Surfaces ── */
            --shifa-bg: #F0FDFA;
            --shifa-bg-secondary: #F1F5F9;
            --shifa-card: #FFFFFF;

            /* ── Typography ── */
            --shifa-text: #134E4A;
            --shifa-text-secondary: #475569;
            --shifa-text-muted: #64748B;
            --shifa-text-on-primary: #FFFFFF;

            /* ── Borders & Shadows ── */
            --shifa-border: #E2E8F0;
            --shifa-border-hover: rgba(8, 145, 178, 0.3);
            --shifa-shadow-xs: 0 1px 2px rgba(0,0,0,0.03);
            --shifa-shadow-sm: 0 2px 6px rgba(8, 145, 178, 0.06);
            --shifa-shadow-md: 0 4px 14px rgba(8, 145, 178, 0.10);
            --shifa-shadow-lg: 0 10px 28px rgba(8, 145, 178, 0.14);
            --shifa-shadow-glow: 0 4px 20px rgba(8, 145, 178, 0.25);

            /* ── Spacing Scale (8px base) ── */
            --space-1: 8px;
            --space-2: 16px;
            --space-3: 24px;
            --space-4: 32px;
            --space-5: 48px;
            --space-6: 64px;

            /* ── Radii ── */
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 20px;
            --radius-pill: 9999px;

            /* ── Transitions ── */
            --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
            --transition-fast: 0.15s cubic-bezier(0.16, 1, 0.3, 1);
            --transition-normal: 0.25s cubic-bezier(0.16, 1, 0.3, 1);

            /* ── Legacy aliases (backward compat) ── */
            --z-green: var(--shifa-primary);
            --z-green-hover: var(--shifa-primary-hover);
            --z-green-light: var(--shifa-primary-light);
            --z-red: var(--shifa-danger);
            --z-gold: var(--shifa-primary);
            --z-beige: var(--shifa-text-secondary);
            --z-bg: var(--shifa-bg);
            --z-card: var(--shifa-card);
            --z-text: var(--shifa-text);
            --z-muted: var(--shifa-text-muted);
        }

        /* ── RTL & Font Foundation ── */
        html, body {
            direction: rtl;
            text-align: right;
            background-color: var(--shifa-bg);
            color: var(--shifa-text);
        }
        
        p, h1, h2, h3, h4, h5, h6, li, a, span, div {
            font-family: 'Cairo', 'Inter', sans-serif;
            letter-spacing: 0.01em;
        }
        
        /* Protect Streamlit internal Material Icons */
        .material-symbols-rounded, 
        .material-symbols-outlined, 
        [data-testid="stIconMaterial"] {
            font-family: 'Material Symbols Rounded' !important;
            font-weight: normal;
            direction: ltr !important;
        }
        
        /* Hide Default Sidebar Nav for controlled routing */
        [data-testid="stSidebarNav"] { display: none !important; }

        /* ── App Background (Medical Mesh Gradient) ── */
        .stApp {
            background-color: var(--shifa-bg);
            background-image: 
                radial-gradient(ellipse at 15% 20%, rgba(8, 145, 178, 0.06), transparent 50%),
                radial-gradient(ellipse at 85% 75%, rgba(16, 185, 129, 0.05), transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(37, 99, 235, 0.02), transparent 60%);
            background-attachment: fixed;
            color: var(--shifa-text);
        }

        /* ── Page Width Constraint ── */
        .block-container {
            max-width: 1100px !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }

        /* ═══════════════════════════════════════════════════════
           SIDEBAR — Clean Medical Panel
           ═══════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {
            background: var(--shifa-card) !important;
            border-left: 1px solid var(--shifa-border) !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button {
            border-radius: var(--radius-md) !important;
            margin: 5px 0 !important;
            padding: 10px 16px !important;
            min-height: 46px !important;
            font-size: 0.92rem !important;
            font-family: 'Cairo', sans-serif !important;
            transition: all var(--transition-normal) !important;
            cursor: pointer !important;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"] {
            background-color: transparent !important;
            border: 1px solid var(--shifa-border) !important;
            color: var(--shifa-text-secondary) !important;
            box-shadow: none !important;
            width: 100% !important;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background-color: var(--shifa-primary-light) !important;
            color: var(--shifa-primary) !important;
            border-color: var(--shifa-border-hover) !important;
            transform: translateX(3px) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button:active {
            transform: scale(0.97) !important;
        }

        /* ═══════════════════════════════════════════════════════
           CARDS — Unified Medical Card System
           ═══════════════════════════════════════════════════════ */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--shifa-border) !important;
            background-color: var(--shifa-card) !important;
            border-radius: var(--radius-lg) !important;
            padding: var(--space-3) !important;
            box-shadow: var(--shifa-shadow-xs) !important;
            transition: transform var(--transition-normal),
                        box-shadow var(--transition-normal),
                        border-color var(--transition-normal);
        }

        /* ═══════════════════════════════════════════════════════
           SECONDARY BUTTONS — Interactive Service Cards
           ═══════════════════════════════════════════════════════ */
        div[data-testid="stButton"] > button[kind="secondary"] {
            background-color: var(--shifa-card) !important;
            border: 1px solid var(--shifa-border) !important;
            border-radius: var(--radius-lg) !important;
            color: var(--shifa-text) !important;
            min-height: 88px !important;
            font-size: 0.93rem !important;
            font-weight: 600 !important;
            transition: all var(--transition-normal) !important;
            box-shadow: var(--shifa-shadow-xs) !important;
            padding: 20px 24px !important;
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
            line-height: 1.6 !important;
            cursor: pointer !important;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            border-color: var(--shifa-border-hover) !important;
            background-color: var(--shifa-primary-light) !important;
            transform: translateY(-3px) !important;
            box-shadow: var(--shifa-shadow-md) !important;
            color: var(--shifa-primary-deep) !important;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:active {
            transform: translateY(0) scale(0.98) !important;
        }
        div[data-testid="stButton"] > button[kind="secondary"] p {
            font-size: 0.93rem !important;
            font-weight: 600 !important;
            margin: 0 !important;
            line-height: 1.6 !important;
            white-space: pre-wrap;
        }

        /* ═══════════════════════════════════════════════════════
           PRIMARY CTA — Medical Teal Gradient
           ═══════════════════════════════════════════════════════ */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, var(--shifa-primary), var(--shifa-primary-hover)) !important;
            color: var(--shifa-text-on-primary) !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            min-height: 54px !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            transition: all var(--transition-normal) !important;
            box-shadow: var(--shifa-shadow-glow) !important;
            cursor: pointer !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: linear-gradient(135deg, var(--shifa-primary-hover), var(--shifa-primary-deep)) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 28px rgba(8, 145, 178, 0.35) !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:active {
            transform: translateY(0) !important;
        }

        /* ── Hero Secondary CTA (tagged via JS) ── */
        button.hero-secondary-btn {
            background: transparent !important;
            color: var(--shifa-primary) !important;
            border: 2px solid var(--shifa-primary) !important;
            border-radius: var(--radius-md) !important;
            min-height: 54px !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            transition: all var(--transition-normal) !important;
            box-shadow: none !important;
            width: 100% !important;
            cursor: pointer !important;
        }
        button.hero-secondary-btn:hover {
            background: var(--shifa-primary-light) !important;
            border-color: var(--shifa-primary-hover) !important;
            color: var(--shifa-primary-hover) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 14px rgba(8, 145, 178, 0.12) !important;
        }
        button.hero-secondary-btn:active {
            transform: translateY(0) !important;
            scale: 0.98 !important;
        }

        /* ═══════════════════════════════════════════════════════
           CHAT INPUT — Teal Focus Ring
           ═══════════════════════════════════════════════════════ */
        [data-testid="stChatInput"] {
            border-radius: var(--radius-lg);
            background: var(--shifa-card) !important;
            border: 1px solid var(--shifa-border) !important;
            box-shadow: var(--shifa-shadow-sm) !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--shifa-primary) !important;
            box-shadow: 0 0 0 3px var(--shifa-primary-glow) !important;
        }
        [data-testid="stChatInput"] textarea {
            color: var(--shifa-text) !important;
        }

        /* ── Chat Messages ── */
        [data-testid="stChatMessage"] {
            background: var(--shifa-card) !important;
            border-radius: var(--radius-lg);
            padding: var(--space-3) !important;
            margin-bottom: var(--space-2);
            border: 1px solid var(--shifa-border) !important;
            box-shadow: var(--shifa-shadow-xs) !important;
        }

        /* ── Auth Card ── */
        .auth-card {
            background-color: var(--shifa-card);
            border-top: 4px solid var(--shifa-primary);
            border-radius: var(--radius-lg);
            padding: var(--space-4);
            box-shadow: var(--shifa-shadow-md);
            text-align: center;
        }

        /* ═══════════════════════════════════════════════════════
           ACCESSIBILITY — Reduced Motion
           ═══════════════════════════════════════════════════════ */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
        
    </style>
    """, unsafe_allow_html=True)
    
    # We remove the background PATTERN_SRC to keep the minimal SaaS look
    
    st.markdown("""
    <style>
        /* ═══════════════════════════════════════════════════════
           TYPOGRAPHY — Medical Visual Hierarchy
           ═══════════════════════════════════════════════════════ */
        .moroccan-title {
            font-size: 3.5rem !important;
            font-weight: 900 !important;
            background: linear-gradient(135deg, #0891B2 0%, #2563EB 50%, #0E7490 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            text-align: center !important;
            margin-bottom: 0px !important;
            padding-bottom: 8px !important;
            letter-spacing: -1px !important;
            font-family: 'Cairo', 'Inter', sans-serif !important;
            line-height: 1.2 !important;
        }
        .moroccan-subtitle {
            text-align: center !important;
            color: var(--shifa-text-muted) !important;
            font-size: 1.05rem !important;
            font-weight: 400 !important;
            margin-bottom: var(--space-6) !important;
            line-height: 1.7 !important;
            font-family: 'Cairo', sans-serif !important;
            max-width: 600px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .home-section-title {
            margin-top: var(--space-6) !important;
            margin-bottom: var(--space-3) !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            text-align: center !important;
            color: var(--shifa-text) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
            font-family: 'Cairo', sans-serif !important;
        }
        .home-section-title span {
            color: var(--shifa-primary) !important;
            font-size: 1.3rem !important;
        }

        /* ═══════════════════════════════════════════════════════
           EMERGENCY BANNER — Compact Professional Alert
           ═══════════════════════════════════════════════════════ */
        @keyframes pulseAlert {
            0% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.15); }
            70% { box-shadow: 0 0 0 8px rgba(245, 158, 11, 0); }
            100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
        }
        .zellige-alert {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border: 1px solid #fcd34d;
            border-right: 4px solid #f59e0b;
            border-radius: var(--radius-lg);
            padding: 1rem 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--space-4);
            animation: pulseAlert 3s infinite;
            flex-wrap: wrap;
            gap: 12px;
        }
        .zellige-alert-title { color: #92400e; font-weight: 700; font-size: 1rem; display: flex; align-items: center; gap: 0.6rem; }
        .zellige-alert-text { color: #b45309; font-size: 0.85rem; margin-top: 0.15rem; font-weight: 500; line-height: 1.4; }
        .zellige-alert-numbers { display: flex; gap: 8px; flex-wrap: wrap; }
        .zellige-alert-numbers span {
            background: rgba(245,158,11,0.08) !important;
            color: #92400e !important;
            padding: 6px 14px !important;
            border-radius: var(--radius-sm) !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            border: 1px solid rgba(245,158,11,0.2) !important;
            white-space: nowrap !important;
        }
        
    </style>
    
    <!-- ترجمة النصوص المضمنة في Streamlit إلى العربية وإدارة الموافقة -->
    <script>
    const _shifaInit = () => {
        const doc = window.parent.document || document;
        
        // 1. Translations & Tagging Hero Secondary Button
        doc.querySelectorAll('button').forEach(btn => {
            if (btn.textContent.trim() === 'Browse files') btn.textContent = 'تصفح الملفات';
            if (btn.textContent.includes('📸')) {
                btn.classList.add('hero-secondary-btn');
            }
        });
        doc.querySelectorAll('p, span, div, small, label').forEach(el => {
            if (el.childElementCount === 0) {
                let t = el.textContent.trim();
                if (t === 'Drag and drop file here') el.textContent = 'اسحب وأفلت الملف هنا';
                if (t.startsWith('Limit') || (t.includes('Limit') && t.includes('per file'))) {
                    el.textContent = 'الحد الأقصى 10 ميغا • JPG, JPEG, PNG';
                }
            }
        });

        // 2. Auto-accept Consent if stored in localStorage
        if (localStorage.getItem('shifa_consent_accepted') === 'true') {
            doc.querySelectorAll('button').forEach(btn => {
                if (btn.textContent.trim().includes("أوافق وأفهم")) {
                    btn.click();
                }
            });
        }
    };

    const doc = window.parent.document || document;
    doc.addEventListener('click', (e) => {
        if (e.target && e.target.textContent && e.target.textContent.includes("أوافق وأفهم")) {
            localStorage.setItem('shifa_consent_accepted', 'true');
        }
    });

    const _obs = new MutationObserver(_shifaInit);
    _obs.observe(window.parent.document.body || document.body, {childList: true, subtree: true});
    setTimeout(_shifaInit, 500);
    setTimeout(_shifaInit, 2000);
    </script>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# AUTHENTICATION GATE  (Login · Register · Guest)
# ─────────────────────────────────────────────────────────────
def _check_auth() -> bool:
    """Retourne True si l'utilisateur est authentifié (compte ou invité)."""
    if st.session_state.get("_authenticated"):
        return True

    inject_custom_css()

    # ── Extra CSS for premium auth card ──
    st.markdown("""
    <style>
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    
    .auth-wrapper {
        animation: fadeSlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    /* Card styling: soft medical, no heavy shadows */
    .auth-wrapper [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 20px !important;
        padding: 32px !important;
        border: 1px solid var(--shifa-border) !important;
        background-color: var(--shifa-card) !important;
        box-shadow: var(--shifa-shadow-lg) !important;
    }

    /* Typography & Hierarchy */
    .brand-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        margin-bottom: 32px;
        gap: 12px;
    }
    .brand-logo {
        height: 100px;
        width: auto;
        object-fit: contain;
        transition: transform 0.3s ease;
        filter: drop-shadow(0 2px 8px rgba(8, 145, 178, 0.15));
    }
    .brand-logo:hover {
        transform: scale(1.03);
    }
    .brand-title {
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #0891B2, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-tagline {
        font-size: 0.95rem;
        color: var(--shifa-text-muted);
        margin: 0;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Segmented equal-width tabs */
    div[data-baseweb="tab-list"] {
        display: flex !important;
        width: 100% !important;
        background-color: var(--shifa-bg-secondary) !important;
        border-radius: 14px !important;
        padding: 6px !important;
        gap: 6px !important;
        border: 1px solid var(--shifa-border) !important;
        margin-bottom: 24px !important;
    }
    div[data-baseweb="tab-list"] button[data-baseweb="tab"] {
        flex: 1 1 0% !important;
        text-align: center !important;
        justify-content: center !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 12px 16px !important;
        color: var(--shifa-text-secondary) !important;
        background-color: transparent !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    div[data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {
        color: var(--shifa-text) !important;
        background-color: rgba(255, 255, 255, 0.4) !important;
    }
    div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff !important;
        color: var(--shifa-primary) !important;
        box-shadow: 0 4px 10px rgba(8, 145, 178, 0.05) !important;
        font-weight: 700 !important;
    }

    /* Form input styling */
    .auth-wrapper [data-testid="stTextInput"] label, 
    .auth-wrapper [data-testid="stPasswordInput"] label {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: var(--shifa-text-secondary) !important;
        margin-bottom: 8px !important;
    }
    .auth-wrapper input {
        border-radius: 10px !important;
        border: 1px solid var(--shifa-border) !important;
        background-color: #ffffff !important;
        color: var(--shifa-text) !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    .auth-wrapper input:focus {
        border-color: var(--shifa-primary) !important;
        box-shadow: 0 0 0 3px var(--shifa-primary-glow) !important;
    }

    /* Form Primary CTA styling */
    .auth-cta-container {
        margin-top: 24px;
    }
    .auth-cta-container div[data-testid="stButton"] button {
        height: 52px !important;
        border-radius: 12px !important;
        background-color: var(--shifa-primary) !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }
    .auth-cta-container div[data-testid="stButton"] button:hover {
        background-color: var(--shifa-primary-hover) !important;
        box-shadow: 0 4px 12px rgba(8, 145, 178, 0.2) !important;
        transform: translateY(-1px);
    }
    .auth-cta-container div[data-testid="stButton"] button:active {
        transform: translateY(0);
    }

    /* Guest Card Layout */
    .guest-feature-card {
        background: #ffffff;
        border: 1px solid var(--shifa-border);
        border-radius: 16px;
        padding: 20px;
        margin: 8px 0 24px 0;
    }
    .guest-card-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .guest-badge {
        display: inline-block;
        background-color: var(--shifa-primary-light);
        color: var(--shifa-primary);
        font-size: 0.8rem;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }
    .guest-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--shifa-text);
        margin: 0 0 6px 0;
    }
    .guest-subtitle {
        font-size: 0.88rem;
        color: var(--shifa-text-muted);
        margin: 0;
        font-weight: 400;
        line-height: 1.5;
    }
    .guest-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-top: 16px;
        text-align: right;
    }
    @media (max-width: 768px) {
        .guest-grid {
            grid-template-columns: 1fr;
            gap: 16px;
        }
    }
    .guest-col {
        background: var(--shifa-bg-secondary);
        border-radius: 12px;
        padding: 14px;
        border: 1px solid var(--shifa-border);
    }
    .guest-col-available {
        border-top: 3px solid var(--shifa-accent);
    }
    .guest-col-unavailable {
        border-top: 3px solid var(--shifa-danger);
    }
    .guest-col-title {
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0 0 12px 0;
    }
    .guest-col-available .guest-col-title {
        color: #065f46;
    }
    .guest-col-unavailable .guest-col-title {
        color: #991b1b;
    }
    .guest-list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .guest-list li {
        font-size: 0.85rem;
        color: var(--shifa-text-secondary);
        display: flex;
        align-items: center;
        gap: 8px;
        line-height: 1.4;
    }
    .icon-check {
        color: var(--shifa-accent);
        font-weight: bold;
    }
    .icon-cross {
        color: var(--shifa-danger);
        font-weight: bold;
    }

    /* Guest CTA Button (Strongest Element) */
    .guest-cta-container {
        margin-top: 16px;
    }
    .guest-cta-container div[data-testid="stButton"] button {
        height: 54px !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, var(--shifa-primary), var(--shifa-primary-hover)) !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(8, 145, 178, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    .guest-cta-container div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, var(--shifa-primary-hover), var(--shifa-primary-deep)) !important;
        box-shadow: 0 8px 22px rgba(8, 145, 178, 0.35) !important;
        transform: translateY(-2px) !important;
    }
    .guest-cta-container div[data-testid="stButton"] button:active {
        transform: translateY(0) !important;
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.5, 2.0, 0.5])
    with col2:
        st.markdown("<div style='margin-bottom: 4vh;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='auth-wrapper'>", unsafe_allow_html=True)

        with st.container(border=True):
            # ── Logo + header ──
            logo_html = (
                f"<img src='{LOGO_SRC}' class='brand-logo'>"
                if LOGO_SRC else
                "<div style='font-size:3.5rem;'>⚜️</div>"
            )
            st.markdown(f"""
            <div class="brand-section">
                {logo_html}
                <h1 class="brand-title">SHIFA AI</h1>
                <p class="brand-tagline">
                    المنصة الطبية الذكية للمحترفين &nbsp;·&nbsp; Plateforme Médicale IA Premium
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── 3 Tabs ──
            tab_login, tab_register, tab_guest = st.tabs([
                "🔑 تسجيل الدخول",
                "📝 إنشاء حساب",
                "👤 وضع الزائر / Invité"
            ])

            # ════════════════════════════════════
            # TAB 1 – LOGIN
            # ════════════════════════════════════
            with tab_login:
                st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
                login_username = st.text_input(
                    "اسم المستخدم / Nom d'utilisateur",
                    placeholder="ex: dr_hassan",
                    key="login_username"
                )
                login_password = st.text_input(
                    "كلمة المرور / Mot de passe",
                    type="password",
                    placeholder="••••••••",
                    key="login_password"
                )
                st.markdown("<div class='auth-cta-container'>", unsafe_allow_html=True)
                if st.button("🔑 دخول / Se connecter", type="primary", key="btn_login", width="stretch"):
                    if not login_username or not login_password:
                        st.error("⚠️ Veuillez remplir tous les champs.")
                    elif login_user is None:
                        st.error("Module auth non disponible.")
                    else:
                        result = login_user(login_username, login_password)
                        if result["success"]:
                            st.session_state["_authenticated"] = True
                            st.session_state["_user"] = result["user"]
                            st.session_state["_is_guest"] = False
                            st.success(f"✅ Bienvenue {result['user']['full_name']} !")
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div style='text-align:center; margin-top:1.25rem;'>"
                    "<span style='color:#64748b; font-size:0.85rem; font-weight: 500;'>للمحترفين الطبيين فقط / Réservé aux professionnels de santé</span></div>",
                    unsafe_allow_html=True
                )

            # ════════════════════════════════════
            # TAB 2 – REGISTER
            # ════════════════════════════════════
            with tab_register:
                st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
                reg_full_name = st.text_input(
                    "الاسم الكامل / Nom complet",
                    placeholder="Dr. Hassan Alaoui",
                    key="reg_full_name"
                )
                reg_username = st.text_input(
                    "اسم المستخدم / Nom d'utilisateur *",
                    placeholder="ex: dr_hassan",
                    key="reg_username"
                )
                reg_email = st.text_input(
                    "البريد الإلكتروني / Email (facultatif)",
                    placeholder="hassan@example.com",
                    key="reg_email"
                )
                reg_password = st.text_input(
                    "كلمة المرور / Mot de passe * (min. 6 car.)",
                    type="password",
                    placeholder="••••••••",
                    key="reg_password"
                )
                reg_password2 = st.text_input(
                    "تأكيد كلمة المرور / Confirmer",
                    type="password",
                    placeholder="••••••••",
                    key="reg_password2"
                )
                st.markdown("<div class='auth-cta-container'>", unsafe_allow_html=True)
                if st.button("📝 إنشاء الحساب / Créer", type="primary", key="btn_register", width="stretch"):
                    if not reg_username or not reg_password:
                        st.error("⚠️ Les champs marqués * sont obligatoires.")
                    elif reg_password != reg_password2:
                        st.error("❌ Les mots de passe ne correspondent pas.")
                    elif register_user is None:
                        st.error("Module auth non disponible.")
                    else:
                        result = register_user(
                            username=reg_username,
                            password=reg_password,
                            email=reg_email,
                            full_name=reg_full_name
                        )
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                            st.info("👆 Cliquez maintenant sur l'onglet \"تسجيل الدخول\" pour vous connecter.")
                        else:
                            st.error(f"❌ {result['message']}")
                st.markdown("</div>", unsafe_allow_html=True)

            # ════════════════════════════════════
            # TAB 3 – GUEST
            # ════════════════════════════════════
            with tab_guest:
                st.markdown("""<div class="guest-feature-card">
<div class="guest-card-header">
<span class="guest-badge">وضع الزائر · Mode Invité</span>
<h3 class="guest-title">استخدام SHIFA AI بدون إنشاء حساب</h3>
<p class="guest-subtitle">استكشف القدرات الأساسية للذكاء الاصطناعي الطبي فوراً وبسرعة.</p>
</div>
<div class="guest-grid">
<div class="guest-col guest-col-available">
<h4 class="guest-col-title">✓ ce que vous pouvez utiliser</h4>
<ul class="guest-list">
<li><span class="icon-check">✓</span> المحادثة الطبية الذكية</li>
<li><span class="icon-check">✓</span> فحص الأعراض الفوري</li>
<li><span class="icon-check">✓</span> تحليل الصور والتقارير</li>
<li><span class="icon-check">✓</span> الحاسبات الطبية الحيوية</li>
</ul>
</div>
<div class="guest-col guest-col-unavailable">
<h4 class="guest-col-title">✕ nécessite un compte</h4>
<ul class="guest-list">
<li><span class="icon-cross">✕</span> حفظ تاريخ الاستشارات</li>
<li><span class="icon-cross">✕</span> التقارير الطبية المفصلة</li>
<li><span class="icon-cross">✕</span> المفضلة والمستندات</li>
<li><span class="icon-cross">✕</span> مزامنة البيانات السحابية</li>
</ul>
</div>
</div>
</div>""", unsafe_allow_html=True)

                st.markdown("<div class='guest-cta-container'>", unsafe_allow_html=True)
                if st.button("👤 المتابعة كزائر / Continuer en invité", key="btn_guest", width="stretch"):
                    st.session_state["_authenticated"] = True
                    st.session_state["_user"] = guest_session() if guest_session else {
                        "id": None, "username": "Invité", "role": "guest"
                    }
                    st.session_state["_is_guest"] = True
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    return False

if not _check_auth():
    st.stop()

inject_custom_css()

# ─────────────────────────────────────────────────────────────
# موافقة إخلاء المسؤولية الطبية (تظهر مرة واحدة فقط)
# يتم تخزين الموافقة في localStorage + session_state
# ─────────────────────────────────────────────────────────────
def _check_consent():
    """التحقق من موافقة المستخدم على إخلاء المسؤولية الطبية."""
    if "consent_accepted" not in st.session_state:
        st.session_state["consent_accepted"] = False

    if not st.session_state["consent_accepted"]:
        st.markdown("""
        <style>
        .consent-wrapper [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 20px !important;
            padding: 32px !important;
            border: 1px solid #e2e8f0 !important;
            border-top: 5px solid #ef4444 !important;
            background-color: #ffffff !important;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<div style='margin-top: 10vh;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='consent-wrapper'>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"""<div style='text-align:center; padding: 0.5rem;'>
{'<img src=\"' + LOGO_SRC + '\" style=\"height:64px; margin-bottom:12px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.05));\">' if LOGO_SRC else '<div style=\"font-size:3rem;\">⚜️</div>'}
<h2 style='color:#0f172a; margin:4px 0 8px 0; font-size:1.75rem; font-weight:800; font-family:\"Cairo\", sans-serif;'>SHIFA AI</h2>
<div style='background:rgba(239,68,68,0.04); border:1px solid rgba(239,68,68,0.15); border-radius:14px; padding:1.5rem; margin:1.5rem 0; text-align:right;'>
<h3 style='color:#dc2626; font-size:1.15rem; font-weight:700; margin:0 0 1rem 0; font-family:\"Cairo\", sans-serif; display:flex; align-items:center; gap:8px;'>
<span>⚕️ إخلاء المسؤولية الطبية / Clause de Non-responsabilité</span>
</h3>
<p style='color:#334155; font-size:0.92rem; line-height:1.75; margin:0; font-family:\"Cairo\", sans-serif;'>
هذه المنصة توفر <b style='color:#dc2626; font-weight:700;'>دعماً معلوماتياً فقط</b> ولا تُعتبر بديلاً عن الاستشارة الطبية المهنية أو التشخيص أو العلاج.
<br/><br/>
⚠️ لا تستخدم هذا التطبيق في حالات الطوارئ الطبية. في حالة الطوارئ، اتصل بالإسعاف فوراً على الرقم <b style='color:#dc2626; font-weight:700;'>15</b>.
<br/><br/>
🔒 جميع البيانات المدخلة تُعالج محلياً ولا يتم مشاركتها مع أي طرف ثالث.
<br/><br/>
النتائج والتحليلات مخصصة للأغراض <b style='color:#2563eb; font-weight:700;'>التعليمية والأكاديمية</b> فقط.
</p>
</div>
</div>""", unsafe_allow_html=True)

                if st.button("✅ أوافق وأفهم", type="primary", key="consent_btn", width="stretch"):
                    st.session_state["consent_accepted"] = True
                    st.rerun()

                st.markdown("""<p style='color:#64748b; font-size:0.8rem; text-align:center; margin-top:0.75rem; font-family:"Cairo", sans-serif; line-height:1.4;'>
بالنقر على "أوافق وأفهم"، أنت تقر بقراءة وفهم إخلاء المسؤولية أعلاه.
</p>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        return False
    return True

if not _check_consent():
    st.stop()


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
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO_SRC:
        st.markdown(f"""
            <div style="text-align:center; padding: 1rem 0;">
                <img src="{LOGO_SRC}" style="height:90px; margin-bottom:10px;">
                <h2 style="color:var(--shifa-primary); margin:8px 0 4px; font-family:'Cairo'; font-weight:800; font-size:1.6rem;">SHIFA AI</h2>
                <p style="color:var(--shifa-text-secondary); font-size:0.9rem; margin:0; font-family:'Cairo'; font-weight:500;">الذكاء الاصطناعي الطبي</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align:center; padding: 1rem 0;">
                <div style="font-size:3.5rem; color:var(--shifa-primary);">⚜️</div>
                <h2 style="color:var(--shifa-primary); margin:8px 0 4px; font-family:'Cairo'; font-weight:800; font-size:1.6rem;">SHIFA AI</h2>
                <p style="color:var(--shifa-text-secondary); font-size:0.9rem; margin:0; font-family:'Cairo'; font-weight:500;">الذكاء الاصطناعي الطبي</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ── User Profile Badge ──
    _current_user = st.session_state.get("_user", {})
    _is_guest     = st.session_state.get("_is_guest", False)
    _uname        = _current_user.get("username", "مستخدم")
    _ufull        = _current_user.get("full_name", _uname)

    if _is_guest:
        st.markdown("""
        <div style="background:#fffbeb; border:1px solid #fcd34d;
                    border-radius:10px; padding:0.6rem 1rem; text-align:center; margin-bottom:0.5rem;">
            <span style="color:#b45309; font-weight:700; font-size:0.85rem; font-family:'Cairo';">👤 زائر / حساب مؤقت</span><br>
            <span style="color:#64748b; font-size:0.75rem; font-family:'Cairo';">الميزات محدودة. أنشئ حساباً للحفظ.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#ecfdf5; border:1px solid #a7f3d0;
                    border-radius:10px; padding:0.6rem 1rem; text-align:center; margin-bottom:0.5rem;">
            <span style="color:#065f46; font-weight:700; font-size:0.85rem; font-family:'Cairo';">✅ {_ufull}</span><br>
            <span style="color:#64748b; font-size:0.75rem; font-family:'Cairo';">@{_uname}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚪 تسجيل الخروج", width="stretch", key="sidebar_logout"):
        for k in ["_authenticated", "_user", "_is_guest", "messages", "local_history"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("<hr style='border-color: rgba(8, 145, 178, 0.1);'/>", unsafe_allow_html=True)
    
    if st.session_state.page != "home":
        if st.button("⬅️ العودة للرئيسية", width="stretch", type="primary"):
            st.session_state.page = "home"
            st.rerun()
    else:
        st.info("✓ أنت في الصفحة الرئيسية")
        
    st.markdown("<hr style='border-color: rgba(8, 145, 178, 0.2);'/>", unsafe_allow_html=True)
    
    if st.session_state.messages:
        st.caption(f"💬 المحادثة الحالية: {len(st.session_state.messages)} رسالة")
        if st.button("🗑️ حوار جديد", width="stretch"):
            st.session_state.messages = []
            st.session_state.session_id = str(time.time())
            st.rerun()
            
    # Removed Sidebar Links: Mental Health & Doctor Portal have been moved to main dashboard for clarity


    st.markdown("""
    <div style="background: var(--shifa-primary-light); border-right: 3px solid var(--shifa-primary); padding: 12px; margin-top: 2rem; border-radius: 8px 0 0 8px;">
        <p style="color: var(--shifa-text); font-size: 0.8rem; margin: 0; line-height: 1.5; font-family: 'Cairo'; font-weight: 500;">
            <b>تنبيه إخلاء المسؤولية:</b><br/>
            المنصة توفر دعماً معلوماتياً. لا تغني أبدًا عن استشارة الطبيب المختص أو زيارة العيادة.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: HOME (LANDING DASHBOARD)
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown('<div class="moroccan-title">SHIFA AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="moroccan-subtitle">مساعدك الطبي الذكي لتقييم الأعراض والتوجيه الصحي</div>', unsafe_allow_html=True)
        
    # ── Emergency Banner (Zellige Style) ──
    st.markdown("""
        <div class="zellige-alert">
            <div class="zellige-alert-title">
                <span class="material-symbols-rounded" style="font-size:1.8rem; color:#d97706; display:flex; align-items:center; justify-content:center;">emergency</span>
                <div>
                    <div style="font-weight: 700; font-size: 1rem;">تنبيه طوارئ طبية فعلية؟</div>
                    <div class="zellige-alert-text">تواصل فورا مع خدمات الطوارئ لإنقاذ الحياة. لا تنتظر التطبيق.</div>
                </div>
            </div>
            <div class="zellige-alert-numbers">
                <span>🚑 الإسعاف: 15</span>
                <span>🚓 الشرطة: 19</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # ── Main CTA (Onboarding / 2 Actions) ──
    st.markdown("<p style='text-align:center; color:#64748b; font-size:1.15rem; margin-bottom:1.5rem;'>👋 مرحباً بك! يرجى اختيار الخدمة المناسبة لحالتك للبدء.</p>", unsafe_allow_html=True)
    col_cta1, col_cta2, col_cta3, col_cta4 = st.columns([0.5, 2, 2, 0.5])
    with col_cta2:
        if st.button("💬 صف أعراضك للطبيب الذكي", width="stretch", type="primary"):
            st.session_state.page = "chat"
            st.rerun()
    with col_cta3:
        # User hierarchy recommendation: Make second button a secondary/outline CTA
        if st.button("📸 رفع ومسح صورة طبية / وصفة", width="stretch", type="secondary"):
            st.session_state.page = "vision"
            st.rerun()
            


    
    # ── Interactive Services Menu (Categorized with Distinct UI Layout) ──
    st.markdown("<div style='margin-top: var(--space-6);'></div>", unsafe_allow_html=True)

    # 1) Category: Medical AI Tools (Diagnostics)
    st.markdown("<h3 class='home-section-title'><span class='material-symbols-rounded'>smart_toy</span> الذكاء الاصطناعي التشخيصي</h3>", unsafe_allow_html=True)
    col_ai1, col_ai2 = st.columns(2)
    with col_ai1:
        if st.button("💬 المحادثة الطبية\nتفاصيل الأعراض والتقييم المباشر", key="srv_chat", width="stretch"):
            st.session_state.page = "chat"
            st.rerun()
    with col_ai2:
        if st.button("🩺 فاحص الأعراض\nتحليل ذكي يعتمد على البيانات السريرية", key="srv_scanner", width="stretch"):
            st.session_state.page = "scanner"
            st.rerun()

    # 2) Category: Advanced Diagnostics Tools
    st.markdown("<h3 class='home-section-title'><span class='material-symbols-rounded'>science</span> أدوات الفحص المتقدمة</h3>", unsafe_allow_html=True)
    col_adv1, col_adv2, col_adv3 = st.columns(3)
    with col_adv1:
        if st.button("🔬 مختبر الصور (Vision)\nحلل الرنين المغناطيسي والأشعة", key="srv_vision", width="stretch"):
            st.session_state.page = "vision"
            st.rerun()
    with col_adv2:
        if st.button("📋 ماسح الوصفات (OCR)\nاستخرج أسماء الأدوية بسرعة", key="srv_ordo", width="stretch"):
            st.session_state.page = "ordonnance"
            st.rerun()
    with col_adv3:
        if st.button("🧮 حاسبات سريرية\nقياسات حيوية ومعدلات الخطورة", key="srv_calc", width="stretch"):
            st.session_state.page = "calculators"
            st.rerun()

    # 3) Category: Specialized Services & Modules
    st.markdown("<h3 class='home-section-title'><span class='material-symbols-rounded'>health_and_safety</span> خدمات وموديولات متخصصة</h3>", unsafe_allow_html=True)
    col_gen1, col_gen2, col_gen3 = st.columns(3)
    with col_gen1:
        if st.button("🧠 وحدة الصحة النفسية\nدعم نفسي ومعرفي مخصص", key="srv_mental", width="stretch"):
            st.session_state.page = "mental"
            st.rerun()
    with col_gen2:
        if st.button("🏥 العيادات والرعاية\nتوجيه للإسعاف والتخصصات القريبة", key="srv_care", width="stretch"):
            st.switch_page("pages/10_🏥_الرعاية_القريبة.py")
    with col_gen3:
        if st.button("📚 المستكشف البحثي\nمحرك أبحاث الكتب والمقالات", key="srv_db", width="stretch"):
            st.session_state.page = "database"
            st.rerun()

    # 4) Secondary Tools
    st.markdown("<div style='margin-top: var(--space-6);'></div>", unsafe_allow_html=True)
    col_sec1, col_sec2, col_sec3 = st.columns(3)
    with col_sec1:
        if st.button("🎙️ المساعد الصوتي السريع\nتحدث مباشرة ليتم فهمك فوراً", key="srv_voice", width="stretch"):
            st.session_state.page = "voice"
            st.rerun()
    with col_sec2:
        if st.button("🦩 المساعد متعدد الوسائط\nرفع ملفات متنوعة مع نصوص", key="srv_flamingo", width="stretch"):
            st.switch_page("pages/06_🦩_المساعد_متعدد_الوسائط.py")
    with col_sec3:
        if st.button("📜 الأرشيف والسجلات\nمراجعة جلساتك الاستشارية السابقة", key="srv_hist", width="stretch"):
            st.session_state.page = "history"
            st.rerun()
    
    # ── Doctor Space launcher — robust engineering solution ──────────────────
    st.markdown("<hr style='opacity:0.1; margin-top: var(--space-6); margin-bottom: var(--space-4);'/>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    if st.button("👨‍⚕️ فضاء الطبيب والإدارة الطبية الخاصة", key="srv_doc",
                 help="الانتقال إلى تطبيق إدارة المستشفى والمرضى"):

        import socket as _socket
        import subprocess as _sp
        import time as _time

        _DOCTOR_PORT = 8503
        _DOCTOR_APP  = str(BASE_DIR / "partie Docteur+User" / "main.py")
        _DOCTOR_CWD  = str(BASE_DIR / "partie Docteur+User")

        def _port_responds(port: int, timeout: float = 1.5) -> bool:
            """Return True if something is already listening on *port*."""
            try:
                with _socket.create_connection(("127.0.0.1", port), timeout=timeout):
                    return True
            except OSError:
                return False

        def _kill_port(port: int) -> None:
            """Kill any process occupying *port* (Windows + Unix safe)."""
            try:
                if sys.platform == "win32":
                    # netstat → find PID → kill
                    out = _sp.check_output(
                        ["netstat", "-ano"],
                        stderr=_sp.DEVNULL, text=True
                    )
                    for line in out.splitlines():
                        if f":{port}" in line and "LISTENING" in line:
                            pid = line.strip().split()[-1]
                            _sp.call(
                                ["taskkill", "/F", "/PID", pid],
                                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL
                            )
                else:
                    _sp.call(
                        ["fuser", "-k", f"{port}/tcp"],
                        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL
                    )
            except Exception:
                pass  # silent — best-effort

        import webbrowser as _wb
        import streamlit.components.v1 as _stcomp

        _DOCTOR_URL = f"http://localhost:{_DOCTOR_PORT}"

        # Step 1: ensure the server is running
        if not _port_responds(_DOCTOR_PORT):
            _kill_port(_DOCTOR_PORT)
            _time.sleep(0.4)

            _sp.Popen(
                [sys.executable, '-m', 'streamlit', 'run', _DOCTOR_APP,
                 '--server.port', str(_DOCTOR_PORT),
                 '--server.headless', 'true'],
                cwd=_DOCTOR_CWD,
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
            )
            for _ in range(20):
                _time.sleep(0.5)
                if _port_responds(_DOCTOR_PORT):
                    break

        # Step 2a: Python-native tab open (server on user machine)
        _wb.open_new_tab(_DOCTOR_URL)

        # Step 2b: JS window.open as secondary insurance
        _stcomp.html(
            f'<script>window.open("{_DOCTOR_URL}", "_blank");</script>',
            height=0,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: CHAT (MAIN)
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "chat":
    import re

    # ── Medical Report Formatting Helpers ──
    def md_to_html(text: str) -> str:
        # Convert bold **text** to <strong>text</strong>
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        # Convert bold *text* to <em>text</em>
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        return text

    def format_medical_report(text: str) -> str:
        # 1. Detect Risk Level
        risk_level = "low"
        risk_badge_text = "🟢 خطورة منخفضة / Risque Faible"
        text_lower = text.lower()
        
        if any(k in text_lower for k in ["طوارئ", "urgence", "critical", "خطير جدا"]):
            risk_level = "emergency"
            risk_badge_text = "🚨 طوارئ فورية / Urgence Médicale"
        elif any(k in text_lower for k in ["مرتفع", "عالية", "grave", "high risk", "خطورة عالية"]):
            risk_level = "high"
            risk_badge_text = "🔴 خطورة عالية / Risque Élevé"
        elif any(k in text_lower for k in ["متوسط", "modéré", "medium risk", "خطورة متوسطة"]):
            risk_level = "medium"
            risk_badge_text = "🟡 خطورة متوسطة / Risque Modéré"
        elif any(k in text_lower for k in ["منخفض", "faible", "low risk", "خطورة منخفضة"]):
            risk_level = "low"
            risk_badge_text = "🟢 خطورة منخفضة / Risque Faible"

        # Define section matchers
        lines = text.split("\n")
        
        sections = {
            "diagnosis": {"title": "🔍 التشخيص المحتمل / Diagnostic Probable", "lines": [], "icon": "🔍"},
            "recommendations": {"title": "📋 التوصيات والإرشادات / Recommandations", "lines": [], "icon": "📋"},
            "warning_doctor": {"title": "⚠️ علامات الخطر ومراجعة الطبيب / Signes d'Alerte", "lines": [], "icon": "⚠️"},
            "risk_level": {"title": "🏥 مستوى الخطورة / Niveau de Risque", "lines": [], "icon": "🏥"},
            "medical_notice": {"title": "📌 تنبيه طبي هام / Note Médicale", "lines": [], "icon": "📌"},
            "general": {"title": "ℹ️ تفاصيل إضافية / Informations", "lines": [], "icon": "ℹ️"}
        }
        
        current_section = "general"
        
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue
                
            lower_line = line_strip.lower()
            is_header = False
            
            # Diagnosis header detection
            if any(k in lower_line for k in ["تشخيص", "diagnosis", "diagnostic", "الحالة المتوقعة"]):
                current_section = "diagnosis"
                is_header = True
            # Recommendations header detection
            elif any(k in lower_line for k in ["توصيات", "recommendation", "العلاج", "conseil", "الإجراءات"]):
                current_section = "recommendations"
                is_header = True
            # Warning doctor / emergency detection
            elif any(k in lower_line for k in ["مراجعة الطبيب", "علامات الخطر", "استشارة الطبيب", "consulter"]):
                current_section = "warning_doctor"
                is_header = True
            # Risk level detection
            elif any(k in lower_line for k in ["الخطورة", "gravité", "niveau de risque", "risk level"]):
                current_section = "risk_level"
                is_header = True
            # Medical notice detection
            elif any(k in lower_line for k in ["تنبيه", "ملاحظة", "note médicale", "avertissement"]):
                current_section = "medical_notice"
                is_header = True
                
            if is_header:
                continue
                
            # Clean markdown symbols for display
            cleaned = line_strip.lstrip("#* -•").strip()
            if cleaned:
                cleaned = md_to_html(cleaned)
                if line_strip.startswith(("-", "*", "•")):
                    sections[current_section]["lines"].append(f"<li>{cleaned}</li>")
                else:
                    sections[current_section]["lines"].append(f"<p>{cleaned}</p>")

        # Build the HTML output
        has_medical_sections = any(len(sections[k]["lines"]) > 0 for k in ["diagnosis", "recommendations", "warning_doctor", "risk_level", "medical_notice"])
        
        if not has_medical_sections:
            return f"<div class='conversational-chat-bubble'>{md_to_html(text.replace(chr(10), '<br>'))}</div>"

        html = []
        html.append(f"<div class='medical-report-card border-{risk_level}'>")
        html.append("<div class='report-card-header'>")
        html.append("<span class='report-card-badge'>📋 تقرير طبي / Rapport Médical</span>")
        html.append(f"<span class='risk-badge badge-{risk_level}'>{risk_badge_text}</span>")
        html.append("</div>")
        html.append("<div class='report-card-divider'></div>")
        
        order = ["risk_level", "diagnosis", "recommendations", "warning_doctor", "medical_notice", "general"]
        
        for key in order:
            sec = sections[key]
            if not sec["lines"]:
                continue
                
            html.append(f"<div class='report-section section-{key}'>")
            html.append(f"<h4 class='report-section-title'>{sec['title']}</h4>")
            html.append("<div class='report-section-divider'></div>")
            html.append("<div class='report-section-content'>")
            
            in_list = False
            for item in sec["lines"]:
                if item.startswith("<li>"):
                    if not in_list:
                        html.append("<ul class='report-list'>")
                        in_list = True
                    html.append(item)
                else:
                    if in_list:
                        html.append("</ul>")
                        in_list = False
                    html.append(item)
                    
            if in_list:
                html.append("</ul>")
                
            html.append("</div>")
            html.append("</div>")
            
        html.append("</div>")
        return "\n".join(html)

    # ── Local CSS Injections for enhanced Chat UX ──
    st.markdown("""
    <style>
    /* Chat Container Styles */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    
    .msg-container {
        display: flex;
        margin-bottom: 20px;
        gap: 12px;
        width: 100%;
        align-items: flex-start;
    }

    /* User Message: Right Aligned, Soft Teal Background */
    .msg-user {
        flex-direction: row-reverse; /* Avatar on the far right in RTL/LTR */
        text-align: right;
    }
    .msg-user .msg-bubble {
        background-color: var(--shifa-primary-light); /* Soft Light Teal */
        color: var(--shifa-text) !important; /* Deep Teal Text */
        border: 1px solid rgba(8, 145, 178, 0.15);
        border-radius: 16px 4px 16px 16px; /* Chat bubble corner on top-right */
        padding: 14px 18px;
        max-width: 75%;
        font-size: 0.95rem;
        font-weight: 500;
        line-height: 1.6;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .msg-user .msg-avatar {
        width: 38px;
        height: 38px;
        background-color: var(--shifa-primary-glow);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        color: var(--shifa-primary);
        flex-shrink: 0;
    }

    /* Assistant Message: Left Aligned, Premium White Card */
    .msg-assistant {
        flex-direction: row; /* Avatar on the left */
        text-align: right;
    }
    .msg-assistant .msg-bubble {
        background-color: #ffffff;
        color: var(--shifa-text) !important;
        border: 1px solid var(--shifa-border);
        border-right: 4px solid var(--shifa-primary); /* Right accent border for assistant */
        border-radius: 4px 16px 16px 16px; /* Chat bubble corner on top-left */
        padding: 24px;
        width: 100%;
        max-width: 85%;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(8, 145, 178, 0.05);
    }
    .msg-assistant .msg-avatar {
        width: 38px;
        height: 38px;
        background-color: #f1f5f9;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        border: 1px solid var(--shifa-border);
        overflow: hidden;
        flex-shrink: 0;
    }
    .assistant-avatar-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    /* Premium Medical Report Card inside Assistant Bubble */
    .medical-report-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        text-align: right;
        border: 1px solid var(--shifa-border);
        border-top: 4px solid transparent;
        position: relative;
        box-shadow: var(--shifa-shadow-sm);
        margin-top: 12px;
    }
    .medical-report-card::before {
        content: '';
        position: absolute;
        top: -4px;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--shifa-primary), var(--shifa-secondary));
        border-radius: 12px 12px 0 0;
    }

    .report-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 12px;
        margin-bottom: 16px;
        gap: 12px;
    }

    .report-card-badge {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--shifa-text);
        font-family: 'Cairo', sans-serif;
    }

    /* Compact Risk Badges */
    .risk-badge {
        font-size: 0.8rem;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 9999px;
        display: inline-block;
        font-family: 'Cairo', sans-serif;
    }

    .badge-low {
        background-color: rgba(16, 185, 129, 0.08);
        color: #10b981;
    }

    .badge-medium {
        background-color: rgba(245, 158, 11, 0.08);
        color: #d97706;
    }

    .badge-high {
        background-color: rgba(249, 115, 22, 0.08);
        color: #ea580c;
    }

    .badge-emergency {
        background-color: rgba(239, 68, 68, 0.08);
        color: #ef4444;
    }

    /* Section design: title, divider, content */
    .report-section {
        margin-bottom: 20px;
    }
    .report-section:last-child {
        margin-bottom: 0;
    }

    .report-section-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--shifa-text);
        margin: 0 0 6px 0;
        font-family: 'Cairo', sans-serif;
    }

    .report-section-divider {
        height: 1px;
        background-color: var(--shifa-border);
        margin-bottom: 10px;
    }

    .report-section-content {
        color: var(--shifa-text-secondary);
        font-size: 0.92rem;
        line-height: 1.7;
    }

    .report-section-content p {
        margin: 0 0 8px 0;
    }

    .report-list {
        margin: 0;
        padding-right: 20px; /* Arabic RTL indent */
        list-style-type: disc;
    }

    .report-list li {
        margin-bottom: 6px;
        color: var(--shifa-text-secondary);
    }

    /* Accent borders on the container card depending on risk level */
    .border-low {
        border-right: 4px solid #10b981 !important;
    }
    .border-medium {
        border-right: 4px solid #f59e0b !important;
    }
    .border-high {
        border-right: 4px solid #f97316 !important;
    }
    .border-emergency {
        border-right: 4px solid #ef4444 !important;
    }
    
    .conversational-chat-bubble {
        font-size: 0.95rem;
        line-height: 1.6;
        color: var(--shifa-text-secondary);
    }

    /* Typing Pulse Indicator Animation */
    .shifa-pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--shifa-primary);
        animation: shifaPulse 1.4s infinite ease-in-out both;
        margin: 0 2px;
    }
    .shifa-pulse-dot:nth-child(1) { animation-delay: -0.32s; }
    .shifa-pulse-dot:nth-child(2) { animation-delay: -0.16s; }

    @keyframes shifaPulse {
        0%, 80%, 100% { transform: scale(0); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

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
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "user":
            st.markdown(f"""
            <div class="msg-container msg-user">
                <div class="msg-avatar">👤</div>
                <div class="msg-bubble">
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            formatted_report = format_medical_report(msg["content"])
            avatar_html = (
                f'<img src="{LOGO_SRC}" class="assistant-avatar-img">'
                if LOGO_SRC else
                '🩺'
            )
            st.markdown(f"""
            <div class="msg-container msg-assistant">
                <div class="msg-avatar">{avatar_html}</div>
                <div class="msg-bubble">
                    {formatted_report}
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
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
        
        # Show new user bubble immediately
        st.markdown(f"""
        <div class="msg-container msg-user">
            <div class="msg-avatar">👤</div>
            <div class="msg-bubble">
                {user_input}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show assistant spinner bubble
        avatar_html = (
            f'<img src="{LOGO_SRC}" class="assistant-avatar-img">'
            if LOGO_SRC else
            '🩺'
        )
        
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
            
            # عداد الثواني أثناء التسجيل + أنيميشن النبض
            import streamlit.components.v1 as _mic_components
            _mic_components.html("""
            <div id="mic-status" style="text-align:center; min-height:60px;"></div>
            <script>
            (function() {
                let timer = null;
                let seconds = 0;
                const statusEl = document.getElementById('mic-status');
                
                const checkRecording = () => {
                    let isRecording = false;
                    const doc = window.parent.document || document;
                    
                    // Search main document first
                    const recBtn = doc.querySelector('[data-testid="stAudioRecorder"] button, .audio-recorder-btn');
                    if (recBtn && (recBtn.getAttribute('aria-label') === 'Stop' || recBtn.classList.contains('recording') || recBtn.style.color === 'rgb(220, 38, 38)')) {
                        isRecording = true;
                    }
                    
                    // Search all iframes
                    const iframes = doc.querySelectorAll('iframe');
                    iframes.forEach(iframe => {
                        try {
                            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                            const btn = iframeDoc.querySelector('button, svg');
                            if (btn) {
                                // Inject keyframes definition inside iframe if missing
                                if (!iframeDoc.getElementById('mic-pulse-style')) {
                                    const styleEl = iframeDoc.createElement('style');
                                    styleEl.id = 'mic-pulse-style';
                                    styleEl.textContent = `
                                        @keyframes mic-pulse {
                                            0%   { box-shadow: 0 0 0 0 rgba(220,38,38,0.6); transform: scale(1); }
                                            50%  { box-shadow: 0 0 0 15px rgba(220,38,38,0); transform: scale(1.1); }
                                            100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); transform: scale(1); }
                                        }
                                    `;
                                    iframeDoc.head.appendChild(styleEl);
                                }
                                
                                const style = window.getComputedStyle(btn);
                                const isRed = style.color === 'rgb(220, 38, 38)' || style.fill === 'rgb(220, 38, 38)';
                                if (isRed || btn.getAttribute('aria-label') === 'Stop' || btn.classList.contains('recording') || btn.classList.contains('active')) {
                                    isRecording = true;
                                    btn.style.animation = 'mic-pulse 1.5s ease-in-out infinite';
                                } else {
                                    btn.style.animation = 'none';
                                }
                            }
                        } catch (e) {}
                    });
                    
                    if (isRecording) {
                        if (!timer) {
                            seconds = 0;
                            timer = setInterval(() => {
                                seconds++;
                                const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
                                const secs = (seconds % 60).toString().padStart(2, '0');
                                statusEl.innerHTML = `
                                    <div class="mic-timer" style="text-align:center; font-size:2rem; font-weight:800; color:#dc2626; font-family:'Cairo',monospace;">${mins}:${secs}</div>
                                    <div style="text-align:center; color:#fca5a5; font-size:0.85rem; font-family:'Cairo'; font-weight:700;">🔴 جاري التسجيل...</div>
                                `;
                            }, 1000);
                        }
                    } else {
                        if (timer) {
                            clearInterval(timer);
                            timer = null;
                            if (seconds > 0) {
                                statusEl.innerHTML = '<div style="text-align:center; color:#16a34a; font-size:0.9rem; font-family:\'Cairo\'; font-weight:700;">✅ تم التسجيل بنجاح</div>';
                            } else {
                                statusEl.innerHTML = '';
                            }
                        }
                    }
                };
                setInterval(checkRecording, 300);
            })();
            </script>
            """, height=80)

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
    st.markdown('<div class="moroccan-title" style="font-size:2.8rem;">تحليل الصور</div>', unsafe_allow_html=True)
    st.warning("🚨 **إخلاء مسؤولية تنظيمي:** هذا القسم مخصص للأغراض الأكاديمية و المعرفية.")

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
            st.markdown("<h3 style='text-align: center; color: var(--shifa-text); font-family: \"Cairo\", sans-serif; margin-bottom: 1.5rem;'>بيانات المريض والأعراض الأساسية</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("عمر المريض *", min_value=1, max_value=120, value=30)
                gender = st.selectbox("الجنس البيولوجي *", ["ذكر", "أنثى"])
            with col2:
                duration = st.selectbox("المدة الزمنية للأعراض *", ["أقل من 24 ساعة", "من يوم إلى 3 أيام", "حوالي أسبوع", "أكثر من أسبوع"])
                # Slider de douleur (1 à 10)
                severity_num = st.slider("مدى حدة وقسوة الألم * (1-10)", min_value=1, max_value=10, value=5, step=1)
                
                # Real-time display of numeric value and descriptive text
                _pain_labels = {
                    1: "😊 خفيف جداً", 2: "🙂 خفيف", 3: "😐 خفيف محتمل",
                    4: "😕 متوسط خفيف", 5: "😟 متوسط", 6: "😣 متوسط شديد",
                    7: "😖 شديد", 8: "😫 شديد جداً", 9: "😱 مؤلم للغاية",
                    10: "🔴 لا يُطاق"
                }
                _pain_color = "#16a34a" if severity_num <= 3 else "#f97316" if severity_num <= 6 else "#dc2626"
                st.markdown(f"<div style='text-align:center; padding:6px; background:rgba({','.join(str(int(c)) for c in [int(_pain_color[1:3],16), int(_pain_color[3:5],16), int(_pain_color[5:7],16)])},0.15); border-radius:8px; margin-top:4px;'><span style='font-size:1.3rem; font-weight:800; color:{_pain_color};'>{severity_num}/10</span> <span style='color:#cbd5e1;'>{_pain_labels[severity_num]}</span></div>", unsafe_allow_html=True)
                
                # Convert numeric to original categorical values for orchestrator compatibility
                severity = "خفيف محتمل" if severity_num <= 3 else "متوسط" if severity_num <= 6 else "شديد ولا يطاق"
            
            symptoms = st.text_area("أعطنا وصفاً مفصلاً (المكان، طبيعة الوجع، الشدة...) *", height=120, placeholder="مثال: أشعر بصداع نصفي نابض مع غثيان عند التعرض للضوء...")
            
            # Real-time validation inside the form container
            if not symptoms:
                st.warning("⚠️ حقل الوصف إلزامي.")
            elif len(symptoms.strip()) < 5:
                st.warning("⚠️ يرجى وصف الأعراض بدقة أكبر (يجب أن يحتوي على 5 أحرف على الأقل).")
            else:
                st.success("✅ وصف الأعراض جاهز للتحليل.")

            history = st.text_input("الأمراض المزمنة أوالأدوية الحالية (إن وجد)")
            
            submitted = st.form_submit_button("إرسال للتحليل الذكي ✨", width="stretch")
        
    if submitted:
        if not symptoms or len(symptoms.strip()) < 5:
            st.error("⚠️ يرجى تصحيح الأخطاء قبل إرسال النموذج.")
        else:
            with st.spinner("جاري التحليل..."):
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
    st.markdown('<div class="moroccan-title" style="font-size:2.8rem;">🧮 دوال وحسابات القياسات الحيوية</div>', unsafe_allow_html=True)
    st.markdown("<div class='moroccan-subtitle'>أدوات حسابية طبية دقيقة مبنية على معادلات سريرية معتمدة دولياً</div>", unsafe_allow_html=True)

    CALCULATORS = [
        "⚖️ مؤشر كتلة الجسم (BMI)",
        "🔥 الاحتياج اليومي من السعرات (Harris-Benedict)",
        "❤️ مؤشر صحة القلب العام",
        "🫀 خطر الأمراض القلبية الوعائية (Framingham)",
        "🩸 تقدير كلسترول LDL (Friedewald)",
        "💉 تصفية الكرياتينين (Cockcroft-Gault)",
        "📊 تقدير HbA1c من متوسط السكر",
        "🩺 ضغط الدم وتفسيره",
    ]

    with st.container(border=True):
        calc_type = st.selectbox("اختر المعادلة الطبية المراد قياسها:", CALCULATORS)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 1. BMI ─────────────────────────────────────────────────
    if "BMI" in calc_type:
        with st.container(border=True):
            st.markdown("### ⚖️ حاسبة مؤشر كتلة الجسم")
            st.caption("المعادلة: BMI = الوزن (kg) ÷ الطول² (m)")
            col1, col2, col3 = st.columns(3)
            with col1:
                weight = st.number_input("الوزن (KG)", min_value=20.0, max_value=300.0, value=75.0, step=0.5)
            with col2:
                height = st.number_input("الطول (CM)", min_value=100.0, max_value=250.0, value=175.0, step=0.5)
            with col3:
                age_bmi = st.number_input("العمر", min_value=2, max_value=120, value=30)

            if height > 0:
                bmi = weight / ((height / 100) ** 2)
                # Ideal weight range
                ideal_min = 18.5 * (height/100)**2
                ideal_max = 24.9 * (height/100)**2

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("مؤشر BMI", f"{bmi:.1f} kg/m²")
                with c2:
                    st.metric("الوزن المثالي", f"{ideal_min:.1f} – {ideal_max:.1f} kg")
                with c3:
                    diff = weight - ((ideal_min + ideal_max) / 2)
                    st.metric("الفرق عن المثالي", f"{abs(diff):.1f} kg {'زيادة' if diff > 0 else 'نقص'}")

                if bmi < 16:
                    st.error("🔴 **نقص حاد في الوزن (Severe Thinness)** — يستدعي تدخلاً غذائياً وطبياً فورياً.")
                elif bmi < 18.5:
                    st.warning("🟡 **نقص في الوزن (Underweight)** — يوصى بمراجعة أخصائي التغذية.")
                elif bmi < 25:
                    st.success("🟢 **وزن طبيعي ومثالي (Normal)** — حافظ على نمط حياتك الصحي!")
                elif bmi < 30:
                    st.warning("🟠 **زيادة في الوزن (Overweight)** — خطر متزايد لأمراض القلب والسكري.")
                elif bmi < 35:
                    st.error("🔴 **سمنة من الدرجة الأولى (Obese I)** — يُنصح بمتابعة طبية منتظمة.")
                elif bmi < 40:
                    st.error("🔴 **سمنة من الدرجة الثانية (Obese II)** — خطر مرتفع جداً.")
                else:
                    st.error("⛔ **سمنة مرضية (Obese III / Morbid)** — خطر حرج. استشر طبيبك فوراً.")

    # ── 2. Harris-Benedict ──────────────────────────────────────
    elif "Harris-Benedict" in calc_type:
        with st.container(border=True):
            st.markdown("### 🔥 حاسبة الاحتياج اليومي من السعرات الحرارية")
            st.caption("معادلة Harris-Benedict المُحدَّثة (Mifflin-St Jeor)")
            col1, col2 = st.columns(2)
            with col1:
                w = st.number_input("الوزن (KG)", 20.0, 300.0, 70.0, 0.5)
                h = st.number_input("الطول (CM)", 100.0, 250.0, 170.0, 0.5)
            with col2:
                a = st.number_input("العمر (سنة)", 10, 100, 30)
                g = st.selectbox("الجنس", ["ذكر", "أنثى"])

            activity = st.select_slider(
                "مستوى النشاط البدني:",
                options=["جالس (بدون رياضة)", "خفيف (1-3 أيام/أسبوع)", "متوسط (3-5 أيام)", "نشط (6-7 أيام)", "رياضي مكثف"],
            )
            act_map = {
                "جالس (بدون رياضة)": 1.2,
                "خفيف (1-3 أيام/أسبوع)": 1.375,
                "متوسط (3-5 أيام)": 1.55,
                "نشط (6-7 أيام)": 1.725,
                "رياضي مكثف": 1.9
            }
            factor = act_map[activity]

            # Mifflin-St Jeor
            if g == "ذكر":
                bmr = 10 * w + 6.25 * h - 5 * a + 5
            else:
                bmr = 10 * w + 6.25 * h - 5 * a - 161

            tdee = bmr * factor
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("معدل الأيض الأساسي (BMR)", f"{bmr:.0f} kcal/day")
            with c2:
                st.metric("الاحتياج اليومي الكلي (TDEE)", f"{tdee:.0f} kcal/day")
            with c3:
                st.metric("لخسارة 0.5kg/أسبوع", f"{tdee - 500:.0f} kcal/day")

            st.info(f"""
**💡 توزيع المغذيات الكبرى (Macros) المقترح:**
- 🥩 البروتين: **{tdee*0.25/4:.0f}g** ({tdee*0.25:.0f} kcal · 25%)
- 🌾 الكربوهيدرات: **{tdee*0.45/4:.0f}g** ({tdee*0.45:.0f} kcal · 45%)
- 🫒 الدهون: **{tdee*0.30/9:.0f}g** ({tdee*0.30:.0f} kcal · 30%)
""")

    # ── 3. مؤشر صحة القلب العام ────────────────────────────────
    elif "صحة القلب العام" in calc_type:
        with st.container(border=True):
            st.markdown("### ❤️ مؤشر صحة القلب العام")
            st.caption("يحسب نسبة VO₂ Max التقديرية ومؤشر الإجهاد القلبي")
            col1, col2 = st.columns(2)
            with col1:
                age_h = st.number_input("العمر (سنة)", 10, 100, 35)
                rhr = st.number_input("النبض في الراحة (bpm)", 40, 120, 65)
                gender_h = st.selectbox("الجنس", ["ذكر", "أنثى"])
            with col2:
                weight_h = st.number_input("الوزن (KG)", 20.0, 200.0, 70.0)
                smoker = st.checkbox("مدخن")
                diabetic = st.checkbox("مصاب بالسكري")

            # Max HR (Tanaka formula)
            hr_max = 208 - (0.7 * age_h)
            # VO2 Max estimate (Uth–Sørensen–Overgaard–Pedersen)
            vo2max = 15 * (hr_max / rhr)
            # Cardiac stress index
            stress_idx = (rhr / hr_max) * 100

            # Adjustments for risk factors
            score = 100
            if smoker:
                score -= 15
                vo2max *= 0.85
            if diabetic:
                score -= 10
            if rhr > 80:
                score -= 10
            elif rhr < 60:
                score += 5
            if age_h > 60:
                score -= 10
            score = max(0, min(100, score))

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("معدل النبض الأقصى", f"{hr_max:.0f} bpm")
            with c2:
                st.metric("VO₂ Max (تقدير)", f"{vo2max:.1f} ml/kg/min")
            with c3:
                st.metric("مؤشر الإجهاد القلبي", f"{stress_idx:.1f}%")
            with c4:
                st.metric("نقاط الصحة القلبية", f"{score}/100")

            # VO2 Max classification
            if vo2max >= 55:
                st.success("🏆 **ممتاز** — لياقة قلبية استثنائية (رياضي).")
            elif vo2max >= 43:
                st.success("🟢 **جيد جداً** — فوق المتوسط.")
            elif vo2max >= 35:
                st.info("🔵 **متوسط** — لياقة مقبولة.")
            elif vo2max >= 25:
                st.warning("🟠 **دون المتوسط** — يوصى بزيادة النشاط البدني.")
            else:
                st.error("🔴 **ضعيف** — خطر قلبي مرتفع. استشر طبيبك.")

    # ── 4. Framingham Risk Score ────────────────────────────────
    elif "Framingham" in calc_type:
        with st.container(border=True):
            st.markdown("### 🫀 خطر الأمراض القلبية الوعائية (نموذج Framingham)")
            st.caption("يقدر احتمالية النوبة القلبية خلال 10 سنوات")
            col1, col2 = st.columns(2)
            with col1:
                age_f = st.number_input("العمر", 20, 79, 45)
                gender_f = st.selectbox("الجنس", ["ذكر", "أنثى"])
                total_chol = st.number_input("الكوليسترول الكلي (mg/dL)", 100, 400, 200)
                hdl = st.number_input("HDL الكوليسترول الجيد (mg/dL)", 20, 100, 50)
            with col2:
                sbp = st.number_input("ضغط الدم الانقباضي (mmHg)", 90, 200, 120)
                bp_treated = st.checkbox("تحت علاج ضغط الدم")
                smoker_f = st.checkbox("مدخن حالياً")
                diabetic_f = st.checkbox("مصاب بالسكري")

            # Simplified Framingham scoring
            if gender_f == "ذكر":
                pts = 0
                # Age
                if age_f < 35: pts -= 1
                elif age_f < 40: pts += 0
                elif age_f < 45: pts += 1
                elif age_f < 50: pts += 2
                elif age_f < 55: pts += 3
                elif age_f < 60: pts += 4
                elif age_f < 65: pts += 5
                elif age_f < 70: pts += 6
                else: pts += 7
                # Cholesterol
                if total_chol < 160: pts += 0
                elif total_chol < 200: pts += 1
                elif total_chol < 240: pts += 2
                elif total_chol < 280: pts += 3
                else: pts += 4
                # HDL
                if hdl >= 60: pts -= 1
                elif hdl >= 50: pts += 0
                elif hdl >= 40: pts += 1
                else: pts += 2
                # BP
                if sbp < 120: pts += 0
                elif sbp < 130: pts += 1 if not bp_treated else 2
                elif sbp < 140: pts += 1 if not bp_treated else 2
                elif sbp < 160: pts += 1 if not bp_treated else 2
                else: pts += 2 if not bp_treated else 3
                if smoker_f: pts += 2
                if diabetic_f: pts += 2

                risk_map = {-3:1,-2:1,-1:1,0:1,1:1,2:2,3:2,4:2,5:3,6:4,7:5,8:6,9:8,10:10,11:12,12:14,13:16,14:18,15:20,16:25,17:30}
                risk_10yr = risk_map.get(max(-3, min(17, pts)), 30)
            else:
                pts = 0
                if age_f < 35: pts -= 7
                elif age_f < 40: pts -= 3
                elif age_f < 45: pts += 0
                elif age_f < 50: pts += 3
                elif age_f < 55: pts += 6
                elif age_f < 60: pts += 8
                elif age_f < 65: pts += 10
                elif age_f < 70: pts += 12
                else: pts += 14
                if total_chol < 160: pts -= 2
                elif total_chol < 200: pts += 0
                elif total_chol < 240: pts += 1
                elif total_chol < 280: pts += 2
                else: pts += 3
                if hdl >= 60: pts -= 2
                elif hdl >= 50: pts -= 1
                elif hdl >= 40: pts += 0
                else: pts += 2
                if sbp < 120: pts -= 3
                elif sbp < 130: pts += 0 if not bp_treated else 3
                elif sbp < 140: pts += 1 if not bp_treated else 4
                elif sbp < 160: pts += 2 if not bp_treated else 5
                else: pts += 4 if not bp_treated else 7
                if smoker_f: pts += 2
                if diabetic_f: pts += 4

                risk_map_f = {-1:1,0:1,1:1,2:1,3:1,4:1,5:2,6:2,7:3,8:4,9:5,10:6,11:8,12:10,13:12,14:14,15:16,16:20,17:24,18:27,19:30}
                risk_10yr = risk_map_f.get(max(-1, min(19, pts)), 30)

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("نقاط Framingham", f"{pts} نقاط")
            with c2:
                st.metric("خطر السكتة لـ 10 سنوات", f"{risk_10yr}%")

            if risk_10yr < 10:
                st.success(f"🟢 **خطر منخفض ({risk_10yr}%)** — حافظ على نمط حياتك الصحي.")
            elif risk_10yr < 20:
                st.warning(f"🟠 **خطر متوسط ({risk_10yr}%)** — يُنصح بتعديل عوامل الخطر.")
            else:
                st.error(f"🔴 **خطر مرتفع ({risk_10yr}%)** — استشر طبيب القلب فوراً.")

    # ── 5. LDL Friedewald ──────────────────────────────────────
    elif "LDL" in calc_type or "كلسترول" in calc_type:
        with st.container(border=True):
            st.markdown("### 🩸 حساب LDL بمعادلة Friedewald")
            st.caption("LDL = الكوليسترول الكلي − HDL − (الدهون الثلاثية ÷ 5)")
            col1, col2 = st.columns(2)
            with col1:
                tc = st.number_input("الكوليسترول الكلي (mg/dL)", 100, 500, 200)
                hdl_l = st.number_input("HDL (mg/dL)", 10, 150, 50)
            with col2:
                tg = st.number_input("الدهون الثلاثية / TG (mg/dL)", 30, 800, 120)

            if tg >= 400:
                st.warning("⚠️ لا يمكن تطبيق معادلة Friedewald إذا كانت TG ≥ 400 mg/dL")
            else:
                ldl = tc - hdl_l - (tg / 5)
                vldl = tg / 5
                non_hdl = tc - hdl_l

                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("LDL (الكوليسترول الضار)", f"{ldl:.0f} mg/dL")
                with c2: st.metric("HDL (الكوليسترول الجيد)", f"{hdl_l} mg/dL")
                with c3: st.metric("VLDL", f"{vldl:.0f} mg/dL")
                with c4: st.metric("Non-HDL", f"{non_hdl:.0f} mg/dL")

                if ldl < 100:
                    st.success("🟢 LDL مثالي (< 100 mg/dL)")
                elif ldl < 130:
                    st.info("🔵 LDL قريب من المثالي (100–129)")
                elif ldl < 160:
                    st.warning("🟡 LDL حدّي مرتفع (130–159) — راقب نظامك الغذائي")
                elif ldl < 190:
                    st.error("🟠 LDL مرتفع (160–189) — استشر طبيبك")
                else:
                    st.error("🔴 LDL مرتفع جداً (≥ 190) — علاج دوائي قد يكون ضرورياً")

    # ── 6. Cockcroft-Gault ──────────────────────────────────────
    elif "Cockcroft" in calc_type or "كرياتينين" in calc_type:
        with st.container(border=True):
            st.markdown("### 💉 تصفية الكرياتينين — Cockcroft-Gault")
            st.caption("تُقدِّر معدل ترشيح الكبيبات (eGFR) وتقييم وظائف الكلى")
            col1, col2 = st.columns(2)
            with col1:
                age_cg = st.number_input("العمر", 18, 100, 45)
                weight_cg = st.number_input("الوزن (KG)", 30.0, 200.0, 70.0)
            with col2:
                creatinine = st.number_input("الكرياتينين في الدم (mg/dL)", 0.3, 15.0, 1.0, 0.1)
                gender_cg = st.selectbox("الجنس", ["ذكر", "أنثى"])

            egfr = ((140 - age_cg) * weight_cg) / (72 * creatinine)
            if gender_cg == "أنثى":
                egfr *= 0.85

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("eGFR (تصفية الكرياتينين)", f"{egfr:.1f} ml/min")
            with c2:
                if egfr >= 90:
                    stage, color = "G1 — كلى سليمة", "success"
                elif egfr >= 60:
                    stage, color = "G2 — قصور خفيف", "info"
                elif egfr >= 45:
                    stage, color = "G3a — قصور خفيف-متوسط", "warning"
                elif egfr >= 30:
                    stage, color = "G3b — قصور متوسط-شديد", "warning"
                elif egfr >= 15:
                    stage, color = "G4 — قصور شديد", "error"
                else:
                    stage, color = "G5 — فشل كلوي (قصور تام)", "error"
                st.metric("مرحلة وظيفة الكلى", stage)

            if color == "success": st.success(f"🟢 {stage} — وظائف كلى طبيعية.")
            elif color == "info": st.info(f"🔵 {stage} — متابعة دورية.")
            elif color == "warning": st.warning(f"🟠 {stage} — مراجعة أخصائي الكلى.")
            else: st.error(f"🔴 {stage} — تدخل طبي عاجل مطلوب.")

    # ── 7. HbA1c ───────────────────────────────────────────────
    elif "HbA1c" in calc_type:
        with st.container(border=True):
            st.markdown("### 📊 تقدير HbA1c من متوسط سكر الدم")
            st.caption("HbA1c(%) = (متوسط السكر + 46.7) ÷ 28.7  —  معادلة Nathan 2008")
            col1, col2 = st.columns(2)
            with col1:
                avg_glucose = st.number_input("متوسط سكر الدم خلال 3 أشهر (mg/dL)", 60, 400, 120)
            with col2:
                hba1c_input = st.number_input("أو أدخل HbA1c% مباشرة (لحساب متوسط السكر)", 4.0, 15.0, 5.5, 0.1)

            hba1c_est = (avg_glucose + 46.7) / 28.7
            avg_from_hba1c = (hba1c_input * 28.7) - 46.7

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("HbA1c المُقدَّر", f"{hba1c_est:.1f}%")
                if hba1c_est < 5.7:
                    st.success("🟢 طبيعي (< 5.7%) — لا سكري")
                elif hba1c_est < 6.5:
                    st.warning("🟡 مقدمات السكري (5.7–6.4%) — نمط حياة صحي ضروري")
                else:
                    st.error("🔴 سكري (≥ 6.5%) — متابعة طبية منتظمة")
            with c2:
                st.metric("متوسط السكر من HbA1c", f"{avg_from_hba1c:.0f} mg/dL")
                if avg_from_hba1c < 100:
                    st.success("🟢 سكر صيام طبيعي")
                elif avg_from_hba1c < 126:
                    st.warning("🟡 مقدمات السكري")
                else:
                    st.error("🔴 مستوى سكري")

    # ── 8. Blood Pressure ───────────────────────────────────────
    elif "ضغط الدم" in calc_type:
        with st.container(border=True):
            st.markdown("### 🩺 قياس وتفسير ضغط الدم")
            st.caption("وفق معايير الجمعية الأمريكية لأمراض القلب ACC/AHA 2017")
            col1, col2, col3 = st.columns(3)
            with col1:
                sbp_bp = st.number_input("الضغط الانقباضي (Systolic / mmHg)", 60, 250, 120)
            with col2:
                dbp_bp = st.number_input("الضغط الانبساطي (Diastolic / mmHg)", 40, 150, 80)
            with col3:
                pulse_bp = st.number_input("النبض (bpm)", 40, 200, 72)

            pp = sbp_bp - dbp_bp  # Pulse pressure
            map_val = dbp_bp + (pp / 3)  # Mean arterial pressure

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("ضغط النبض (PP)", f"{pp} mmHg")
            with c2: st.metric("الضغط الشرياني الوسطي (MAP)", f"{map_val:.0f} mmHg")
            with c3: st.metric("النبض", f"{pulse_bp} bpm")

            # Classification ACC/AHA 2017
            if sbp_bp < 120 and dbp_bp < 80:
                st.success("🟢 **ضغط طبيعي** — ممتاز! حافظ على نمط حياتك الصحي.")
            elif sbp_bp < 130 and dbp_bp < 80:
                st.info("🔵 **ضغط مرتفع حدّياً (Elevated)** — تحول نمط الحياة مطلوب.")
            elif sbp_bp < 140 or dbp_bp < 90:
                st.warning("🟠 **ارتفاع ضغط المرحلة 1 (Hypertension Stage 1)** — راجع طبيبك.")
            elif sbp_bp < 180 or dbp_bp < 120:
                st.error("🔴 **ارتفاع ضغط المرحلة 2 (Hypertension Stage 2)** — أدوية مطلوبة.")
            else:
                st.error("⛔ **أزمة ارتفاع الضغط (Hypertensive Crisis ≥ 180/120)** — طوارئ فورية! اتصل بالإسعاف.")

            if pp > 60:
                st.warning("⚠️ **ضغط النبض مرتفع (PP > 60)** — قد يشير لتصلب الشرايين.")
            if map_val < 60:
                st.error("⛔ **MAP منخفض جداً** — خطر نقص التروية الدموية.")



# ─────────────────────────────────────────────────────────────
# PAGE: MENTAL HEALTH (شفاء-نفس)
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "mental":
    try:
        from mental_module import render_mental_module
        _or_key = st.secrets.get("OPENROUTER_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
        render_mental_module(api_key=_or_key)
    except ImportError as e:
        logger.error(f"Mental module import error: {e}")
        st.error("⚠️ Module الصحة النفسية غير متاح حالياً.")
    except Exception as e:
        logger.error(f"Mental module error: {e}")
        st.error(f"خطأ في تحميل وحدة الصحة النفسية: {e}")

# ─────────────────────────────────────────────────────────────
# PAGE: ORDONNANCE SCANNER
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "ordonnance":
    try:
        from ui.ordonnance_ui import render_ordonnance_page
        render_ordonnance_page()
    except ImportError as e:
        logger.error(f"Ordonnance module import error: {e}")
        st.error("⚠️ Module scanner d'ordonnance non disponible.")
        st.info("Installez les dépendances : `pip install pytesseract rapidfuzz`")
    except Exception as e:
        logger.error(f"Ordonnance page error: {e}")
        st.error(f"Erreur lors du chargement du scanner : {e}")
# ─────────────────────────────────────────────────────────────
# PAGE: DATABASE  — المستكشف البحثي الذكي (Research Explorer)
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "database":
    st.markdown('<div class="moroccan-title" style="font-size:2.8rem;">المستكشف البحثي الذكي</div>', unsafe_allow_html=True)
    st.markdown("<div class='moroccan-subtitle'>محرك بحث ذكي يقترح كتباً ومقالات وفيديوهات مرتبة حسب الصلة بموضوعك الطبي</div>", unsafe_allow_html=True)

    # ── Extra CSS for research cards ──
    st.markdown("""
    <style>
    @keyframes fadeSlideUpCard {
        from { opacity: 0; transform: translateY(20px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    .research-topic-badge {
        display: inline-flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, rgba(22,163,74,0.2), rgba(212,175,55,0.15));
        border: 1px solid rgba(212,175,55,0.4);
        border-radius: 30px;
        padding: 0.6rem 1.5rem;
        margin: 0.4rem;
        font-size: 1rem;
        color: #fef3c7;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        backdrop-filter: blur(8px);
        transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.3s;
    }
    .research-topic-badge:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 20px rgba(212,175,55,0.3);
    }
    
    .research-content-card {
        background: linear-gradient(145deg, rgba(30,41,59,0.7), rgba(15,23,42,0.9));
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        animation: fadeSlideUpCard 0.6s ease-out forwards;
        opacity: 0;
    }
    
    /* Staggered animation effect for child cards */
    .research-content-card:nth-child(1) { animation-delay: 0.1s; }
    .research-content-card:nth-child(2) { animation-delay: 0.2s; }
    .research-content-card:nth-child(3) { animation-delay: 0.3s; }
    .research-content-card:nth-child(4) { animation-delay: 0.4s; }
    
    .research-content-card:hover {
        border-color: rgba(22,163,74,0.6);
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.4), 0 0 20px rgba(22,163,74,0.2);
    }
    
    .research-content-card::before {
        content: "";
        position: absolute;
        top: 0; right: 0;
        width: 5px; height: 100%;
        border-radius: 0 16px 16px 0;
        transition: width 0.3s ease;
    }
    .research-content-card:hover::before {
        width: 8px;
    }
    
    .research-card-book::before { background: linear-gradient(180deg, #10b981, #047857); }
    .research-card-article::before { background: linear-gradient(180deg, #3b82f6, #1d4ed8); }
    .research-card-video::before { background: linear-gradient(180deg, #ef4444, #b91c1c); }
    
    .research-score-badge {
        display: inline-flex; align-items: center;
        background: linear-gradient(135deg, rgba(22,163,74,0.25), rgba(22,163,74,0.1));
        color: #6ee7b7;
        border: 1px solid rgba(22,163,74,0.3);
        border-radius: 12px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 800;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        letter-spacing: 0.5px;
    }
    
    .research-link-btn {
        display: inline-flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, rgba(22,163,74,0.15), rgba(22,163,74,0.05));
        border: 1px solid rgba(22,163,74,0.4);
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        color: #86efac !important;
        text-decoration: none !important;
        font-size: 0.95rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .research-link-btn:hover {
        background: linear-gradient(135deg, rgba(22,163,74,0.35), rgba(22,163,74,0.15));
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(22,163,74,0.3);
        color: #ffffff !important;
    }
    
    .research-btn-primary {
        display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 12px; padding: 0.7rem 1rem;
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: #ffffff !important; text-decoration: none !important;
        border-radius: 12px; font-weight: 800; font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(22,163,74,0.4);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .research-btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(22,163,74,0.5);
    }
    
    .research-section-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 2.5rem 0 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid rgba(255,255,255,0.05);
        position: relative;
    }
    .research-section-header::after {
        content: ""; position: absolute; bottom: -2px; right: 0; width: 80px; height: 3px;
        background: linear-gradient(90deg, var(--z-green), transparent);
        border-radius: 3px;
    }
    .research-section-header h3 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 800;
        text-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }
    
    .research-section-count {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.2);
        color: #ffffff;
        border-radius: 14px;
        padding: 5px 14px;
        font-size: 0.9rem;
        font-weight: 800;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .img-hover-zoom {
        transition: transform 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    .research-content-card:hover .img-hover-zoom {
        transform: scale(1.08);
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Search Bar ──
    with st.container(border=True):
        search_col1, search_col2 = st.columns([5, 1])
        with search_col1:
            search_term = st.text_input(
                "🔍 ابحث عن أي موضوع طبي أو علمي...",
                placeholder="مثال: سرطان الثدي، السكري من النوع الثاني، ضغط الدم المرتفع...",
                key="research_search_input",
            )
        with search_col2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            search_clicked = st.button("🔎 بحث", type="primary", width="stretch", key="research_btn")

    if search_term and (search_clicked or search_term):
        try:
            from engine.research_explorer import ContentSearchEngine

            engine = ContentSearchEngine()

            # ── Phase 1: Topic Detection ──
            with st.spinner("🧠 تحليل الموضوع وتحسين كلمات البحث..."):
                topic_info = engine.detect_topic(search_term)

            # Topic badge
            st.markdown(f"""
            <div style="text-align:center; margin:1rem 0;">
                <span class="research-topic-badge">🧠 الموضوع: {topic_info.get('topic', search_term)}</span>
                <span class="research-topic-badge">📂 التصنيف: {topic_info.get('category', 'عام')}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Phase 2: Multi-source search ──
            with st.spinner("🔍 جاري البحث في الكتب والمقالات والفيديوهات..."):
                results = engine.search_all(
                    search_term,
                    max_books=5,
                    max_articles=3,
                    max_videos=3,
                )

            books = results.get("books", [])
            articles = results.get("articles", [])
            videos = results.get("videos", [])
            total = len(books) + len(articles) + len(videos)

            if total == 0:
                st.warning("❌ لم يتم العثور على نتائج. حاول إعادة صياغة بحثك بمصطلحات مختلفة.")
            else:
                # Proper Arabic pluralization for "Results" (نتيجة / نتائج / نتيجتان)
                if total == 1:
                    t_text = "نتيجة واحدة"
                elif total == 2:
                    t_text = "نتيجتان"
                elif 3 <= total <= 10:
                    t_text = f"**{total}** نتائج"
                else:
                    t_text = f"**{total}** نتيجة"
                st.success(f"✅ تم العثور على {t_text} مرتبة حسب الصلة")

                # ═══════════════════ BOOKS ═══════════════════
                if books:
                    # Proper Arabic pluralization for Books (كتاب / كتب / كتابان)
                    b_count = len(books)
                    if b_count == 1:
                        b_text = "كتاب واحد"
                    elif b_count == 2:
                        b_text = "كتابان"
                    elif 3 <= b_count <= 10:
                        b_text = f"{b_count} كتب"
                    else:
                        b_text = f"{b_count} كتاباً"

                    st.markdown(f"""
                    <div class="research-section-header">
                        <h3 style="color:#86efac;">📚 الكتب الموصى بها</h3>
                        <span class="research-section-count">{b_text}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    import re

                    # Display in 2-column grid
                    for row_start in range(0, len(books), 2):
                        book_cols = st.columns(2)

                        for col_idx in range(2):
                            b_idx = row_start + col_idx
                            if b_idx >= len(books):
                                break

                            b = books[b_idx]

                            with book_cols[col_idx]:

                                # ===== DATA =====
                                authors = "، ".join(b.get("authors", ["مؤلف غير معروف"]))
                                score = b.get("relevance_score", 0) * 100

                                desc = re.sub(r'<[^>]+>', '', b.get("description") or "")
                                desc = desc[:200]

                                thumb = b.get("thumbnail", "").replace("http://", "https://")

                                rating = b.get("average_rating", 0) or 0
                                ratings_count = b.get("ratings_count", 0) or 0

                                pages = b.get("page_count", 0) or 0
                                publisher = b.get("publisher", "")
                                pub_date = b.get("published_date", "")

                                preview_link = b.get("preview_link", "")
                                info_link = b.get("info_link", "")

                                # ✅ FIX LINK
                                read_link = (preview_link or info_link or "#").replace("http://", "https://")

                                # ===== STARS =====
                                if rating > 0:
                                    full_stars = int(rating)
                                    half_star = "½" if (rating - full_stars) >= 0.3 else ""
                                    stars_html = f'<span style="color:#fbbf24;">{"⭐" * full_stars}{half_star}</span> <span style="color:#94a3b8; font-size:0.8rem;">({rating}/5 · {ratings_count})</span>'
                                else:
                                    stars_html = '<span style="color:#64748b;">بدون تقييمات</span>'

                                # ===== COVER =====
                                if thumb:
                                    cover_html = f'<div style="flex-shrink:0; margin-left:15px;"><img src="{thumb}" style="width:105px; height:145px; object-fit:cover; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.4);" /></div>'
                                else:
                                    cover_html = '<div style="flex-shrink:0; margin-left:15px; width:105px; height:145px; display:flex; align-items:center; justify-content:center; border-radius:10px; background:#1e293b;">📕</div>'

                                # ===== META =====
                                meta_parts = []
                                if pages: meta_parts.append(f"📄 {pages} صفحة")
                                if publisher: meta_parts.append(f"🏢 {publisher}")
                                if pub_date: meta_parts.append(f"📅 {pub_date}")
                                meta_html = " · ".join(meta_parts)

                                # ===== DESC & BUTTON =====
                                desc_html = ""
                                if desc.strip():
                                    desc_html = f'<div style="margin-top:12px; color:#cbd5e1; font-size:0.88rem; line-height:1.7; border-top:1px solid rgba(255,255,255,0.08); padding-top:10px; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;">{desc}...</div>'

                                btn_html = ""
                                if read_link != "#":
                                    btn_html = f'<div style="margin-top:15px; display:flex; gap:10px;"><a href="{read_link}" target="_blank" class="research-link-btn" style="flex:1;">📖 اقرأ الكتاب</a></div>'

                                # ===== CARD =====
                                card_html = f"""<div class="research-content-card research-card-book"><div style="display:flex;">{cover_html}<div style="flex:1;"><div style="font-weight:800; color:#ffffff; font-size:1.15rem; margin-bottom:6px; line-height:1.4;">{b.get('title', 'بدون عنوان')}</div><div style="color:#a7f3d0; font-size:0.92rem; margin-bottom:6px; font-weight:500;">✍️ {authors}</div><div style="margin-top:4px; display:flex; align-items:center; gap:10px; flex-wrap:wrap;">{stars_html}<span class="research-score-badge">✨ {score:.0f}% تطابق</span></div><div style="color:#94a3b8; font-size:0.82rem; margin-top:8px; font-weight:500;">{meta_html}</div></div></div>{desc_html}{btn_html}</div>"""
                                st.markdown(card_html, unsafe_allow_html=True)

                # ═══════════════════ ARTICLES ═══════════════════
                if articles:
                    # Proper Arabic pluralization for Articles (مقال / مقالات / مقالان)
                    a_count = len(articles)
                    if a_count == 1:
                        a_text = "مقال واحد"
                    elif a_count == 2:
                        a_text = "مقالان"
                    elif 3 <= a_count <= 10:
                        a_text = f"{a_count} مقالات"
                    else:
                        a_text = f"{a_count} مقالاً"

                    st.markdown(f"""
                    <div class="research-section-header">
                        <h3 style="color:#60a5fa;">🌐 مقالات ذات صلة</h3>
                        <span class="research-section-count">{a_text}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    for a in articles:
                        score = a.get("relevance_score", 0) * 100
                        summary = a.get("summary") or ""
                        import re
                        summary = re.sub(r'<[^>]+>', '', summary)[:350]
                        thumb = a.get("thumbnail", "")
                        thumb_html = f'<div style="overflow:hidden; border-radius:8px; margin-left:12px; float:right; box-shadow:0 2px 10px rgba(0,0,0,0.3);"><img src="{thumb}" class="img-hover-zoom" style="width:110px; height:75px; object-fit:cover; display:block;" /></div>' if thumb else ""
                        link_html = f'<a href="{a.get("url", "#")}" target="_blank" class="research-link-btn" style="border-color:rgba(59,130,246,0.3); color:#93c5fd !important; margin-top:0.5rem;">🔗 اقرأ المقال بالكامل</a>' if a.get("url") else ""

                        st.markdown(f"""
                        <div class="research-content-card research-card-article">{thumb_html}
                            <div style="font-weight:800; color:#ffffff; font-size:1.1rem; margin-bottom:6px; text-shadow:0 1px 2px rgba(0,0,0,0.5);">
                                {a.get('title', 'بدون عنوان')}
                            </div>
                            <div style="color:#94a3b8; font-size:0.9rem; margin-bottom:6px; font-weight:600;">
                                📰 <span style="color:#cbd5e1;">{a.get('source', 'Wikipedia')}</span> &nbsp;·&nbsp;
                                <span class="research-score-badge" style="background:rgba(59,130,246,0.2); color:#93c5fd; border-color:rgba(59,130,246,0.4);">✨ {score:.0f}%</span>
                            </div>
                            <div style="color:#cbd5e1; font-size:0.88rem; line-height:1.7; margin-bottom:12px; clear:both; padding-top:4px;">
                                {summary}{'...' if len(summary) >= 350 else ''}
                            </div>{link_html}
                        </div>
                        """, unsafe_allow_html=True)

                # ═══════════════════ VIDEOS ═══════════════════
                if videos:
                    # Proper Arabic pluralization for Videos (فيديو / فيديوهات / فيديوهان)
                    v_count = len(videos)
                    if v_count == 1:
                        v_text = "فيديو واحد"
                    elif v_count == 2:
                        v_text = "فيديوهان"
                    elif 3 <= v_count <= 10:
                        v_text = f"{v_count} فيديوهات"
                    else:
                        v_text = f"{v_count} فيديو"

                    st.markdown(f"""
                    <div class="research-section-header">
                        <h3 style="color:#f87171;">🎥 فيديوهات مقترحة</h3>
                        <span class="research-section-count">{v_text}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    vid_cols = st.columns(min(len(videos), 3))
                    for v_idx, v in enumerate(videos[:3]):
                        with vid_cols[v_idx]:
                            score = v.get("relevance_score", 0) * 100
                            views = int(v.get("view_count", 0))
                            views_str = f"{views:,}" if views else "—"
                            thumb = v.get("thumbnail", "")
                            thumb_html = f'<div style="overflow:hidden; border-radius:12px 12px 0 0;"><img src="{thumb}" class="img-hover-zoom" style="width:100%; height:150px; object-fit:cover; display:block;" /></div>' if thumb else '<div style="width:100%;height:150px;background:linear-gradient(135deg, rgba(220,38,38,0.15), rgba(185,28,28,0.05));border-radius:12px 12px 0 0;display:flex;align-items:center;justify-content:center;font-size:3rem;">🎬</div>'
                            link_html = f'<a href="{v.get("url", "#")}" target="_blank" class="research-link-btn" style="border-color:rgba(220,38,38,0.4); color:#fca5a5 !important; width:100%; text-align:center; display:block; margin-top:8px;">▶️ شاهد على يوتيوب</a>' if v.get("url") and v.get("url") != "#" else ""

                            st.markdown(f"""
                            <div class="research-content-card research-card-video" style="padding:0; border-radius:12px;">
                                {thumb_html}
                                <div style="padding:1.2rem;">
                                    <div style="font-weight:800; color:#ffffff; font-size:0.95rem; margin-bottom:6px; line-height:1.5; text-shadow:0 1px 2px rgba(0,0,0,0.5);">
                                        {v.get('title', 'بدون عنوان')[:85]}...
                                    </div>
                                    <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:8px; font-weight:600;">
                                        📺 <span style="color:#cbd5e1;">{v.get('channel_title', '')}</span><br/>👁️ {views_str} مشاهدة
                                    </div>
                                    <div style="display:flex; gap:6px; margin-bottom:12px;">
                                        <span class="research-score-badge" style="background:rgba(220,38,38,0.2); color:#fca5a5; border-color:rgba(220,38,38,0.4);">✨ {score:.0f}%</span>
                                    </div>{link_html}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # ═══════════════════ AI SUMMARY ═══════════════════
                st.markdown("""
                <div class="research-section-header">
                    <h3 style="color:#d4af37;">🧠 ملخص الذكاء الاصطناعي</h3>
                </div>
                """, unsafe_allow_html=True)

                with st.spinner("🤖 جاري توليد ملخص ذكي..."):
                    try:
                        if orch and hasattr(orch, 'llm') and orch.llm:
                            prompt = f"""بصفتك الذكاء الاصطناعي الطبي SHIFA، قدم ملخصاً طبياً مرجعياً موجزاً عن '{search_term}'.
اكتب بلغة عربية علمية دقيقة ومباشرة (5-8 أسطر). اذكر:
- التعريف
- الأسباب الرئيسية
- أبرز الأعراض
- خيارات العلاج الأساسية
لا تقم بالترحيب ولا تختم بأي عبارة ودية."""

                            ai_response = orch.llm.run(query=prompt, context={"kb_context": "", "intent": "database_search", "history": None})

                            if ai_response and ai_response.success:
                                text_formatted = ai_response.answer.replace('\\n', '<br/>').replace('\n', '<br/>')
                                st.markdown(f"""
                                <div style="background:rgba(15,23,42,0.6); padding:1.5rem 2rem; border-radius:12px;
                                            border-right:4px solid #d4af37; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
                                    <div style="display:flex; justify-content:space-between; margin-bottom:1rem;
                                                border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:0.8rem;">
                                        <span style="color:#94a3b8; font-size:0.95rem;">الملخص الطبي المرجعي: <b style="color:#a7f3d0;">SHIFA AI</b></span>
                                        <span style="color:#C9A855; font-size:0.95rem;">✦ Groq LLM</span>
                                    </div>
                                    <div style="color:#f8fafc; line-height:1.9; font-size:1.05rem; white-space:pre-wrap;">
                                        {text_formatted}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("تعذر توليد ملخص ذكي لهذا الموضوع.")
                        else:
                            st.info("وحدة الذكاء الاصطناعي غير متوفرة حالياً لتوليد الملخص.")
                    except Exception as e:
                        logger.error(f"AI summary error: {e}")
                        st.info("تعذر توليد ملخص ذكي حالياً.")

                # ── Disclaimer ──
                st.markdown("""
                <div style="background:rgba(220,38,38,0.05); border-right:3px solid #dc2626;
                            padding:1rem 1.2rem; border-radius:0 8px 8px 0; margin-top:1.5rem;">
                    <p style="color:#fca5a5; font-size:0.85rem; margin:0;">
                        ⚠️ <b>تنبيه:</b> هذه النتائج لأغراض معلوماتية فقط. المحتوى مستخرج من مصادر خارجية (Google Books, Wikipedia, YouTube).
                        لا تُغني هذه المعلومات عن استشارة طبيب مختص.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        except ImportError as ie:
            logger.error(f"Research explorer import error: {ie}")
            st.error("⚠️ وحدة المستكشف البحثي غير متوفرة. تحقق من تثبيت المكونات.")
        except Exception as e:
            logger.error(f"Research search error: {e}")
            st.error("حدث خطأ أثناء البحث. يرجى المحاولة مرة أخرى.")

# ─────────────────────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "history":
    st.markdown('<div class="moroccan-title" style="font-size:2.8rem;">محفوظات الاستشارات</div>', unsafe_allow_html=True)
    
    history = st.session_state.get("local_history", [])
    
    if not history:
        # صفحة فارغة مع زر CTA للانتقال للمحادثة
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; margin:2rem 0;">
            <div style="font-size:4rem; margin-bottom:1rem; opacity:0.6;">💬</div>
            <h3 style="color:var(--z-text); font-size:1.4rem; margin-bottom:0.5rem; font-family:'Cairo';">لا توجد استشارات سابقة</h3>
            <p style="color:var(--z-muted); font-size:0.95rem; margin-bottom:1.5rem; font-family:'Cairo';">سجلاتك فارغة حالياً. ابدأ أول استشارة طبية مع المساعد الذكي.</p>
        </div>
        """, unsafe_allow_html=True)
        col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
        with col_cta2:
            if st.button("💬 ابدأ استشارتك الأولى", type="primary", width="stretch", key="history_cta"):
                st.session_state.page = "chat"
                st.rerun()
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
<div style="text-align:center; color:var(--z-muted); font-size:0.75rem; padding:2.5rem 0; margin-top:4rem; border-top: 1px solid rgba(22, 163, 74, 0.2);">
  <div style="margin-bottom:1rem; display:flex; justify-content:center; gap:2rem; flex-wrap:wrap; font-family:'Cairo';">
    <a href="#" style="color:var(--z-green); text-decoration:none; font-weight:600; font-size:0.85rem;">سياسة الخصوصية</a>
    <span style="color:var(--z-muted); opacity:0.3;">|</span>
    <a href="#" style="color:var(--z-green); text-decoration:none; font-weight:600; font-size:0.85rem;">شروط الاستخدام</a>
    <span style="color:var(--z-muted); opacity:0.3;">|</span>
    <a href="#" style="color:var(--z-green); text-decoration:none; font-weight:600; font-size:0.85rem;">اتصل بنا</a>
  </div>
  تمت البرمجة والتحسين والتصميم بواسطة فريق <b style="color:#d4af37;">SHIFA AI</b> © 2026<br/>
  <span style="font-size:0.75rem;">تنويه: النظام للاستخدامات الثقافية والتجريبية ولا يغني عن الطب البشري المعتمد.</span>
</div>
""", unsafe_allow_html=True)