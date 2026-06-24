# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Light Medical SaaS Theme v2
  ─────────────────────────────────────────────────────────────────────
  Centralized CSS for the entire app.
  Import `inject_theme()` once per page to apply consistent styling.

  Design System:
    Primary   : #2563EB (Medical Blue)
    Secondary : #10B981 (Teal/Green)
    Danger    : #EF4444 (Soft Red)
    Warning   : #F59E0B (Amber)
    Surface   : Light, clean medical cards
═══════════════════════════════════════════════════════════════════════
"""

import streamlit as st

# ── Color Tokens ──────────────────────────────────────────────────────────────
COLORS = {
    "bg_primary":    "#F0FDFA",         # Teal-tinted background
    "bg_secondary":  "#F1F5F9",         # Slate 100
    "bg_card":       "#FFFFFF",         # Surface
    "bg_glass":      "#FFFFFF",         # Surface
    "accent":        "#0891B2",         # Primary Color (Medical Teal)
    "accent_blue":   "#2563EB",         # Secondary Color (Trust Blue)
    "accent_teal":   "#14B8A6",         # Accent Color
    "accent_green":  "#10B981",         # Success Color
    "accent_amber":  "#F59E0B",         # Warning Color
    "accent_red":    "#EF4444",         # Danger Color
    "text_primary":  "#134E4A",         # Primary Text (Deep Teal)
    "text_secondary":"#475569",         # Secondary Text
    "text_muted":    "#64748B",         # Muted Text
    "border_subtle": "#E2E8F0",         # Border Color
}


def inject_theme(*, page_font: str = "Cairo") -> None:
    """
    Inject the full SHIFA light theme CSS into the current Streamlit page.

    Call once at the top of each page, after `st.set_page_config()`.
    """
    st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family={page_font}:wght@300;400;600;700;900&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');

  /* ── Anti-Overlap Foundation ── */
  *, *::before, *::after {{
    box-sizing: border-box !important;
  }}

  html, body, [class*="css"], [class*="st-"] {{
    font-family: '{page_font}', 'Tajawal', sans-serif !important;
    direction: rtl;
    text-align: right;
    line-height: 1.7;
  }}

  /* Keep Streamlit/Material icon ligatures rendered as icons, not literal text */
  .stIconMaterial,
  [data-testid="stIconMaterial"],
  .material-icons,
  .material-icons-round,
  .material-icons-outlined,
  .material-icons-sharp,
  .material-symbols-outlined,
  .material-symbols-rounded,
  .material-symbols-sharp,
  [data-testid="stExpanderToggleIcon"] span,
  [data-testid="stSidebarCollapseButton"] span {{
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    direction: ltr !important;
    text-align: center !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-transform: none !important;
    letter-spacing: normal !important;
  }}

  /* Prevent Arabic text from overflowing containers */
  p, span, div, label, h1, h2, h3, h4, h5, h6,
  .stMarkdown, .shifa-card, .stat-card, .hero-sub,
  .type-badge, .rank-badge, .search-title {{
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    hyphens: auto;
  }}

  .stApp {{
    background: linear-gradient(160deg, {COLORS['bg_primary']} 0%, {COLORS['bg_secondary']} 60%, #e2e8f0 100%) !important;
    color: {COLORS['text_primary']} !important;
  }}

  [data-testid="stHeader"] {{
    background: transparent !important;
  }}

  /* ════════════════════════════════════════════════════════
     HERO BANNER — Premium medical gradient with shimmer (Light Theme)
     ════════════════════════════════════════════════════════ */
  @keyframes shimmer {{
    0% {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
  }}

  .hero-banner {{
    background: linear-gradient(135deg, #f0fdfa 0%, #e0f2fe 50%, #f5f3ff 100%);
    border: 1px solid rgba(8, 145, 178, 0.15);
    border-radius: 20px;
    padding: 44px 40px 36px;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(8, 145, 178, 0.05);
  }}
  .hero-banner::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg,
      transparent 0%,
      rgba(8, 145, 178, 0.03) 25%,
      rgba(16, 185, 129, 0.04) 50%,
      rgba(8, 145, 178, 0.03) 75%,
      transparent 100%);
    background-size: 200% 100%;
    animation: shimmer 6s linear infinite;
    pointer-events: none;
  }}
  .hero-banner::after {{
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(8, 145, 178, 0.08) 0%, transparent 65%);
    pointer-events: none;
  }}
  .hero-title {{
    font-size: 2.4rem;
    font-weight: 900;
    color: #1e293b;
    margin: 0 0 10px;
    position: relative;
    z-index: 1;
    letter-spacing: -0.02em;
  }}
  .hero-sub {{
    font-size: 1.05rem;
    color: {COLORS['text_secondary']};
    position: relative;
    z-index: 1;
    line-height: 1.7;
    max-width: 550px;
    margin: 0 auto;
  }}
  .hero-badge {{
    display: inline-block;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    color: #059669;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 16px;
    position: relative;
    z-index: 1;
  }}

  /* ════════════════════════════════════════════════════════
     SEARCH PANEL — Light theme card
     ════════════════════════════════════════════════════════ */
  .search-panel {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
  }}
  .search-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {COLORS['accent']};
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  /* ════════════════════════════════════════════════════════
     CITY PILLS — Compact horizontal buttons
     ════════════════════════════════════════════════════════ */
  .city-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 16px 0 8px;
    justify-content: center;
  }}
  .city-pill {{
    background: rgba(8, 145, 178, 0.05);
    border: 1px solid rgba(8, 145, 178, 0.15);
    color: {COLORS['accent']};
    padding: 8px 18px;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.25s ease;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .city-pill:hover {{
    background: rgba(8, 145, 178, 0.1);
    border-color: rgba(8, 145, 178, 0.3);
    transform: translateY(-2px);
    color: #0e7490;
  }}

  /* ════════════════════════════════════════════════════════
     FACILITY / DATA CARDS — Light Theme Card
     ════════════════════════════════════════════════════════ */
  .shifa-card {{
    direction: rtl;
    background: {COLORS['bg_card']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: 16px;
    padding: 20px 22px;
    margin: 10px 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    transition: transform 0.25s ease, border-color 0.3s ease, box-shadow 0.3s ease;
  }}
  .shifa-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(8, 145, 178, 0.25);
    box-shadow: 0 10px 20px rgba(8, 145, 178, 0.1);
  }}

  .shifa-card-row {{
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }}
  .shifa-card-info {{
    flex: 1 1 280px;
    min-width: 0;
    overflow: hidden;
  }}
  .shifa-card-info span,
  .shifa-card-info div {{
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .shifa-card-actions {{
    flex: 0 0 auto;
    min-width: 140px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    justify-content: center;
    align-self: center;
  }}

  /* ── Rank Badge ── */
  .rank-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 32px;
    height: 32px;
    padding: 0 8px;
    border-radius: 999px;
    background: rgba(8, 145, 178, 0.05);
    color: {COLORS['accent']};
    font-weight: 700;
    font-size: 0.85rem;
    border: 1px solid rgba(8, 145, 178, 0.15);
  }}

  /* ── Type Badge ── */
  .type-badge {{
    display: inline-block;
    max-width: 100%;
    background: #f5f3ff;
    border: 1px solid #ddd6fe;
    color: #6d28d9;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  /* ── Action Buttons ── */
  .shifa-btn {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    text-align: center;
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 0.84rem;
    font-weight: 700;
    text-decoration: none;
    transition: all 0.2s ease;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    width: 100%;
  }}
  .shifa-btn:hover {{
    transform: translateY(-1px);
  }}
  .shifa-btn-green {{
    border: 1px solid rgba(16, 185, 129, 0.25);
    background: rgba(16, 185, 129, 0.06);
    color: #059669;
  }}
  .shifa-btn-green:hover {{
    background: rgba(16, 185, 129, 0.15);
  }}
  .shifa-btn-blue {{
    border: 1px solid rgba(8, 145, 178, 0.25);
    background: rgba(8, 145, 178, 0.06);
    color: {COLORS['accent']};
  }}
  .shifa-btn-blue:hover {{
    background: rgba(8, 145, 178, 0.15);
  }}

  /* ════════════════════════════════════════════════════════
     STATS ROW — Dashboard metrics
     ════════════════════════════════════════════════════════ */
  .stats-row {{
    display: flex;
    gap: 14px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }}
  .stat-card {{
    flex: 1 1 130px;
    min-width: 130px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    transition: transform 0.25s ease, border-color 0.25s ease;
  }}
  .stat-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(8, 145, 178, 0.25);
  }}
  .stat-number {{ font-size: 1.8rem; font-weight: 800; color: {COLORS['accent']}; }}
  .stat-label  {{ font-size: 0.8rem; color: {COLORS['text_muted']}; margin-top: 4px; font-weight: 500; }}

  /* ════════════════════════════════════════════════════════
     PRIMARY CTA — Large gradient search button
     ════════════════════════════════════════════════════════ */
  .cta-search {{
    display: block;
    width: 100%;
    padding: 16px 32px;
    background: linear-gradient(135deg, #0891B2 0%, #0e7490 60%, #155E75 100%);
    color: #FFFFFF;
    font-size: 1.15rem;
    font-weight: 800;
    font-family: '{page_font}', sans-serif;
    border: none;
    border-radius: 14px;
    cursor: pointer;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(8, 145, 178, 0.25);
    letter-spacing: 0.02em;
  }}
  .cta-search:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(8, 145, 178, 0.4);
  }}

  /* ════════════════════════════════════════════════════════
     EMPTY STATE — No results illustration
     ════════════════════════════════════════════════════════ */
  .empty-state {{
    text-align: center;
    padding: 48px 24px;
    background: #ffffff;
    border: 1px dashed #cbd5e1;
    border-radius: 20px;
    margin: 24px 0;
  }}
  .empty-state-icon {{
    font-size: 4rem;
    margin-bottom: 16px;
    opacity: 0.6;
  }}
  .empty-state-title {{
    font-size: 1.3rem;
    font-weight: 700;
    color: #64748b;
    margin-bottom: 8px;
  }}
  .empty-state-text {{
    font-size: 0.95rem;
    color: {COLORS['text_muted']};
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.7;
  }}

  /* ════════════════════════════════════════════════════════
     SUCCESS STATE ──
     ════════════════════════════════════════════════════════ */
  @keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  .success-banner {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 14px;
    color: #047857;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 20px;
    animation: fadeInUp 0.4s ease-out;
  }}

  /* ════════════════════════════════════════════════════════
     EMERGENCY BANNER — Danger state
     ════════════════════════════════════════════════════════ */
  .emergency-banner {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px;
    background: #fef2f2;
    border: 1px solid #fca5a5;
    border-radius: 14px;
    color: #b91c1c;
    font-weight: 600;
    font-size: 0.92rem;
    margin-top: 16px;
    direction: rtl;
  }}

  /* ── Legend ── */
  .legend-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 14px 0;
    padding: 12px 16px;
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    font-size: 0.84rem;
    color: {COLORS['text_secondary']};
  }}

  /* ════════════════════════════════════════════════════════
     STREAMLIT OVERRIDES (Light Theme)
     ════════════════════════════════════════════════════════ */

  /* Primary Buttons */
  .stButton > button[kind="primary"],
  .stButton > button {{
    background: linear-gradient(135deg, #0891B2, #0e7490) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-family: '{page_font}', 'Tajawal', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(8, 145, 178, 0.25) !important;
    width: 100% !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
  }}
  .stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(8, 145, 178, 0.4) !important;
  }}

  /* Input Labels */
  div[data-testid="stNumberInput"] label,
  div[data-testid="stSlider"] label,
  div[data-testid="stSelectSlider"] label,
  div[data-testid="stMultiSelect"] label,
  div[data-testid="stTextInput"] label,
  div[data-testid="stSelectbox"] label {{
    color: #475569 !important;
    font-family: '{page_font}', 'Tajawal', sans-serif !important;
    font-weight: 600 !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    display: block !important;
    max-width: 100% !important;
  }}

  /* Inputs */
  div[data-testid="stNumberInput"] input,
  div[data-testid="stTextInput"] input,
  .stSelectbox select {{
    width: 100% !important;
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
    color: #1e293b !important;
    padding: 10px 14px !important;
  }}
  div[data-testid="stNumberInput"] input:focus,
  div[data-testid="stTextInput"] input:focus {{
    border-color: #0891B2 !important;
    box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.15) !important;
  }}

  /* Markdown */
  .stMarkdown p {{
    color: #334155 !important;
    line-height: 1.7 !important;
  }}

  /* Expander */
  [data-testid="stExpander"] {{
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
  }}
  [data-testid="stExpander"] summary {{
    color: {COLORS['text_primary']} !important;
    font-weight: 600 !important;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background-color: transparent !important;
    gap: 8px !important;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px 12px 0 0 !important;
    font-weight: 600 !important;
    color: {COLORS['text_secondary']} !important;
    padding: 10px 20px !important;
  }}
  .stTabs [aria-selected="true"] {{
    background: rgba(8, 145, 178, 0.08) !important;
    color: {COLORS['accent']} !important;
    border-top: 2px solid {COLORS['accent']} !important;
  }}

  /* Metrics */
  div[data-testid="stMetric"] {{
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 16px !important;
  }}
  div[data-testid="stMetric"] label {{
    color: {COLORS['text_muted']} !important;
  }}
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: #0891B2 !important;
  }}

  /* Divider */
  hr {{
    border-color: #e2e8f0 !important;
    margin: 20px 0 !important;
  }}

  /* Footer */
  .footer-bar {{
    display: flex;
    justify-content: space-around;
    align-items: center;
    padding: 12px 20px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    margin-top: 24px;
    flex-wrap: wrap;
    gap: 12px;
  }}
</style>
""", unsafe_allow_html=True)
