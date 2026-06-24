# -*- coding: utf-8 -*-
"""
SHIFA AI — 📋 مسح الوصفة الطبية
═══════════════════════════════════
Analyse intelligente d'ordonnances via Vision AI (Gemini 3 Flash OpenRouter)
Centralized page rendering to maintain UI consistency and avoid code duplication.
"""

import streamlit as st
from ui.ordonnance_ui import render_ordonnance_page

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="SHIFA AI — مسح الوصفة",
    page_icon="📋",
    layout="wide"
)

# Render the shared, redesigned prescription scanner page
render_ordonnance_page()
