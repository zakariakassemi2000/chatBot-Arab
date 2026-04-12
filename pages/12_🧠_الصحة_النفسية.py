"""
SHIFA-Mental — Page Streamlit multipage
Wrappeur pour render_mental_module() avec clé OpenRouter.
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="SHIFA-Mental — شفاء-نفس",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

try:
    from mental_module import render_mental_module
    _api_key = st.secrets.get("OPENROUTER_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
    render_mental_module(api_key=_api_key)
except ImportError as e:
    st.error(f"⚠️ Impossible de charger le module: {e}")
except Exception as e:
    st.error(f"Erreur: {e}")
