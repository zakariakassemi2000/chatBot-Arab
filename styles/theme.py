# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Dark Medical SaaS Theme v2
  ─────────────────────────────────────────────────────────────────────
  Centralized CSS for the entire app.
  Import `inject_theme()` once per page to apply consistent styling.

  Design System:
    Primary   : #2563EB (soft blue)
    Secondary : #10B981 (teal/green)
    Danger    : #EF4444 (soft red)
    Warning   : #F59E0B (amber)
    Surface   : glassmorphism dark cards
═══════════════════════════════════════════════════════════════════════
"""

import streamlit as st

# ── Color Tokens ──────────────────────────────────────────────────────────────
COLORS = {
    "bg_primary":    "#0B0E14",
    "bg_secondary":  "#0F172A",
    "bg_card":       "rgba(15, 23, 42, 0.66)",
    "bg_glass":      "rgba(30, 41, 59, 0.5)",
    "accent":        "#E53935",
    "accent_blue":   "#2563EB",
    "accent_green":  "#10B981",
    "accent_purple": "#8B5CF6",
    "accent_amber":  "#F59E0B",
    "accent_red":    "#EF4444",
    "text_primary":  "#E2E8F0",
    "text_secondary":"#94A3B8",
    "text_muted":    "#64748B",
    "border_subtle": "rgba(255,255,255,0.08)",
}


def inject_theme(*, page_font: str = "Cairo") -> None:
    """
    Inject the full SHIFA dark theme CSS into the current Streamlit page.

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
    background: linear-gradient(160deg, {COLORS['bg_primary']} 0%, {COLORS['bg_secondary']} 60%, #0B1628 100%) !important;
    color: {COLORS['text_primary']} !important;
  }}

  [data-testid="stHeader"] {{
    background: transparent !important;
  }}

  /* ════════════════════════════════════════════════════════
     HERO BANNER — Premium medical gradient with shimmer
     ════════════════════════════════════════════════════════ */
  @keyframes shimmer {{
    0% {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
  }}

  .hero-banner {{
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 40%, #0F3460 70%, #0F172A 100%);
    border: 1px solid rgba(37, 99, 235, 0.2);
    border-radius: 20px;
    padding: 44px 40px 36px;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }}
  .hero-banner::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg,
      transparent 0%,
      rgba(37, 99, 235, 0.06) 25%,
      rgba(16, 185, 129, 0.08) 50%,
      rgba(37, 99, 235, 0.06) 75%,
      transparent 100%);
    background-size: 200% 100%;
    animation: shimmer 6s linear infinite;
    pointer-events: none;
  }}
  .hero-banner::after {{
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(37, 99, 235, 0.12) 0%, transparent 65%);
    pointer-events: none;
  }}
  .hero-title {{
    font-size: 2.4rem;
    font-weight: 900;
    color: #F1F5F9;
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
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: {COLORS['accent_green']};
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 16px;
    position: relative;
    z-index: 1;
  }}

  /* ════════════════════════════════════════════════════════
     SEARCH PANEL — Glassmorphism card
     ════════════════════════════════════════════════════════ */
  .search-panel {{
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(37, 99, 235, 0.15);
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
  }}
  .search-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #93C5FD;
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
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.2);
    color: #93C5FD;
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
    background: rgba(37, 99, 235, 0.18);
    border-color: rgba(37, 99, 235, 0.4);
    transform: translateY(-2px);
    color: #BFDBFE;
  }}

  /* ════════════════════════════════════════════════════════
     FACILITY / DATA CARDS — Flexbox with micro-animation
     ════════════════════════════════════════════════════════ */
  .shifa-card {{
    direction: rtl;
    background: {COLORS['bg_card']};
    border: 1px solid {COLORS['border_subtle']};
    border-radius: 16px;
    padding: 20px 22px;
    margin: 10px 0;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    transition: transform 0.25s ease, border-color 0.3s ease, box-shadow 0.3s ease;
  }}
  .shifa-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(37, 99, 235, 0.25);
    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.12);
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
    min-width: 0;        /* ← prevents flex child blowout */
    overflow: hidden;    /* ← clips any overflow */
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
    background: rgba(37, 99, 235, 0.15);
    color: #93C5FD;
    font-weight: 700;
    font-size: 0.85rem;
  }}

  /* ── Type Badge ── */
  .type-badge {{
    display: inline-block;
    max-width: 100%;
    background: rgba(165, 180, 252, 0.08);
    border: 1px solid rgba(165, 180, 252, 0.18);
    color: #A5B4FC;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  /* ── Action Buttons (link-styled) ── */
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
    border: 1px solid rgba(16, 185, 129, 0.3);
    background: rgba(16, 185, 129, 0.1);
    color: #34D399;
  }}
  .shifa-btn-green:hover {{
    background: rgba(16, 185, 129, 0.22);
  }}
  .shifa-btn-blue {{
    border: 1px solid rgba(37, 99, 235, 0.3);
    background: rgba(37, 99, 235, 0.1);
    color: #93C5FD;
  }}
  .shifa-btn-blue:hover {{
    background: rgba(37, 99, 235, 0.22);
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
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    transition: transform 0.25s ease, border-color 0.25s ease;
  }}
  .stat-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(37, 99, 235, 0.25);
  }}
  .stat-number {{ font-size: 1.8rem; font-weight: 800; color: #60A5FA; }}
  .stat-label  {{ font-size: 0.8rem; color: {COLORS['text_muted']}; margin-top: 4px; font-weight: 500; }}

  /* ════════════════════════════════════════════════════════
     PRIMARY CTA — Large gradient search button
     ════════════════════════════════════════════════════════ */
  .cta-search {{
    display: block;
    width: 100%;
    padding: 16px 32px;
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 60%, #1E40AF 100%);
    color: #FFFFFF;
    font-size: 1.15rem;
    font-weight: 800;
    font-family: '{page_font}', sans-serif;
    border: none;
    border-radius: 14px;
    cursor: pointer;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.3);
    letter-spacing: 0.02em;
  }}
  .cta-search:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(37, 99, 235, 0.45);
  }}
  .cta-search:active {{
    transform: translateY(0);
  }}

  /* ════════════════════════════════════════════════════════
     EMPTY STATE — No results illustration
     ════════════════════════════════════════════════════════ */
  .empty-state {{
    text-align: center;
    padding: 48px 24px;
    background: rgba(15, 23, 42, 0.5);
    border: 1px dashed rgba(148, 163, 184, 0.2);
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
    color: #94A3B8;
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
     SUCCESS STATE — Search complete banner
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
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 14px;
    color: #34D399;
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
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 14px;
    color: #FCA5A5;
    font-weight: 600;
    font-size: 0.92rem;
    margin-top: 16px;
    direction: rtl;
  }}

  /* ── Search Panel (duplicate override removed) ── */

  /* ── Legend ── */
  .legend-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 14px 0;
    padding: 12px 16px;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 0.84rem;
    color: {COLORS['text_secondary']};
  }}

  /* ════════════════════════════════════════════════════════
     STREAMLIT OVERRIDES
     ════════════════════════════════════════════════════════ */

  /* Primary Buttons — full-width, no overflow */
  .stButton > button[kind="primary"],
  .stButton > button {{
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-family: '{page_font}', 'Tajawal', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.2) !important;
    width: 100% !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
  }}
  .stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(37, 99, 235, 0.35) !important;
  }}

  /* Input Labels — truncate long Arabic text */
  div[data-testid="stNumberInput"] label,
  div[data-testid="stSlider"] label,
  div[data-testid="stSelectSlider"] label,
  div[data-testid="stMultiSelect"] label,
  div[data-testid="stTextInput"] label,
  div[data-testid="stSelectbox"] label {{
    color: #CBD5E1 !important;
    font-family: '{page_font}', 'Tajawal', sans-serif !important;
    font-weight: 600 !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    display: block !important;
    max-width: 100% !important;
  }}

  /* All Streamlit inputs — full-width, rounded, no bleed */
  div[data-testid="stNumberInput"],
  div[data-testid="stTextInput"],
  div[data-testid="stSelectbox"],
  div[data-testid="stMultiSelect"],
  div[data-testid="stSelectSlider"],
  div[data-testid="stSlider"] {{
    width: 100% !important;
    max-width: 100% !important;
  }}

  div[data-testid="stNumberInput"] input,
  div[data-testid="stTextInput"] input,
  .stSelectbox select {{
    width: 100% !important;
    border-radius: 10px !important;
    border-color: rgba(37, 99, 235, 0.15) !important;
    background: rgba(15, 23, 42, 0.6) !important;
    color: #E2E8F0 !important;
    padding: 10px 14px !important;
  }}
  div[data-testid="stNumberInput"] input:focus,
  div[data-testid="stTextInput"] input:focus {{
    border-color: rgba(37, 99, 235, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
  }}

  /* Markdown */
  .stMarkdown p {{
    color: #CBD5E1 !important;
    line-height: 1.7 !important;
  }}

  /* Expander */
  [data-testid="stExpander"] {{
    background: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
  }}
  [data-testid="stExpander"] summary {{
    color: {COLORS['text_primary']} !important;
    font-weight: 600 !important;
  }}
  [data-testid="stExpander"] summary p,
  [data-testid="stExpander"] summary span {{
    overflow: visible !important;
    white-space: normal !important;
    text-overflow: initial !important;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background-color: transparent !important;
    gap: 8px !important;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px 12px 0 0 !important;
    font-weight: 600 !important;
    color: {COLORS['text_secondary']} !important;
    padding: 10px 20px !important;
  }}
  .stTabs [aria-selected="true"] {{
    background: rgba(37, 99, 235, 0.1) !important;
    color: {COLORS['accent_blue']} !important;
    border-top: 2px solid {COLORS['accent_blue']} !important;
  }}

  /* Metrics */
  div[data-testid="stMetric"] {{
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    padding: 16px !important;
  }}
  div[data-testid="stMetric"] label {{
    color: {COLORS['text_muted']} !important;
  }}
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: #60A5FA !important;
  }}

  /* Divider */
  hr {{
    border-color: rgba(255, 255, 255, 0.06) !important;
    margin: 20px 0 !important;
  }}

  /* Footer */
  .footer-bar {{
    display: flex;
    justify-content: space-around;
    align-items: center;
    padding: 12px 20px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 14px;
    margin-top: 24px;
    flex-wrap: wrap;
    gap: 12px;
  }}
  .footer-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    color: {COLORS['text_muted']};
    font-size: 0.82rem;
    font-weight: 500;
  }}

  /* ── Streamlit Column containers — prevent overflow ── */
  [data-testid="column"] {{
    overflow: hidden !important;
    min-width: 0 !important;
  }}

  /* ── Block container max-width ── */
  .block-container {{
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 1rem 2rem !important;
  }}

  /* Mobile Responsiveness */
  @media (max-width: 768px) {{
    .hero-title {{ font-size: 1.6rem !important; }}
    .hero-sub {{ font-size: 0.9rem !important; }}
    .hero-banner {{ padding: 28px 20px 24px !important; }}
    .search-panel {{ padding: 20px 16px !important; }}
    .stats-row {{ gap: 8px; }}
    .stat-card {{ min-width: 90px; padding: 12px 8px; }}
    .stat-number {{ font-size: 1.3rem; }}
    .city-pills {{ gap: 6px; }}
    .city-pill {{ padding: 6px 12px; font-size: 0.8rem; }}
    .shifa-card {{ padding: 14px 16px !important; }}
    .shifa-card-actions {{ min-width: 120px; }}
    .shifa-btn {{ padding: 10px 12px !important; font-size: 0.8rem !important; }}
    .block-container {{ padding: 0.5rem 1rem !important; }}
    .footer-bar {{ flex-direction: column; gap: 8px; text-align: center; }}
  }}
</style>
""", unsafe_allow_html=True)
