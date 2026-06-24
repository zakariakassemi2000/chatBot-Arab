# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Reusable Card Components v3
  ─────────────────────────────────────────────────────────────────────
  Clean, self-contained HTML card renderers for Streamlit.
  Each card uses CSS classes from `styles.theme` — no inline HTML soup.
  Buttons are ALWAYS inside the card boundary.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import html as html_mod
import streamlit as st
from typing import Dict


def render_facility_card(rank: int, place: Dict) -> None:
    """
    Render a single medical facility as a premium card.

    Layout:
      ┌──────────────────────────────────────────────────┐
      │  #rank  Name                                      │
      │  [Type Badge]                                     │
      │  📏 Distance                                      │
      │  📍 Address  ⚕️ Specialty  📞 Phone  🕐 Hours    │
      │  [🗺️ Google Maps]   [🌍 OpenStreetMap]           │
      └──────────────────────────────────────────────────┘
    """
    # ── Escape all user-facing strings ────────────────────────────────────
    name       = html_mod.escape(str(place.get("name", "مرفق طبي")))
    type_label = html_mod.escape(str(place.get("type_label", "🏥 مرفق صحي")))
    distance   = html_mod.escape(str(place.get("distance_str", "")))
    address    = html_mod.escape(str(place.get("address", "")))
    phone      = html_mod.escape(str(place.get("phone", "")))
    specialty  = html_mod.escape(str(place.get("specialty", "")))
    opening    = html_mod.escape(str(place.get("opening_hours", "")))
    gmaps_url  = html_mod.escape(str(place.get("gmaps_url", "#")), quote=True)
    osm_url    = html_mod.escape(str(place.get("osm_url", "#")), quote=True)

    # ── Conditional detail rows ───────────────────────────────────────────
    address_row = (
        f'<div class="shifa-meta-row"><span class="shifa-meta-icon">📍</span>'
        f'<span class="shifa-meta-text">{address}</span></div>'
        if address else ""
    )
    specialty_row = (
        f'<div class="shifa-meta-row" style="color:#6d28d9;"><span class="shifa-meta-icon">⚕️</span>'
        f'<span class="shifa-meta-text">{specialty}</span></div>'
        if specialty else ""
    )
    phone_row = (
        f'<div class="shifa-meta-row"><a href="tel:{phone}" class="shifa-phone-link">'
        f'<span class="shifa-meta-icon">📞</span>'
        f'<span class="shifa-meta-text">{phone}</span></a></div>'
        if phone else ""
    )
    opening_row = (
        f'<div class="shifa-meta-row" style="color:#64748b;"><span class="shifa-meta-icon">🕐</span>'
        f'<span class="shifa-meta-text">{opening}</span></div>'
        if opening else ""
    )

    st.markdown(f"""
    <div class="shifa-card">
      <!-- Header: rank + name -->
      <div class="shifa-card-header">
        <span class="rank-badge">#{rank}</span>
        <span class="shifa-card-name">{name}</span>
      </div>

      <!-- Type badge + distance -->
      <div class="shifa-card-meta-top">
        <span class="type-badge">{type_label}</span>
        <span class="shifa-distance-badge">
          <span>📏</span> {distance}
        </span>
      </div>

      <!-- Detail rows: address / specialty / phone / hours -->
      <div class="shifa-card-details">
        {address_row}
        {specialty_row}
        {phone_row}
        {opening_row}
      </div>

      <!-- Action buttons always at bottom -->
      <div class="shifa-card-btns">
        <a href="{gmaps_url}" target="_blank" rel="noopener noreferrer"
           class="shifa-btn shifa-btn-green">🗺️ خرائط جوجل</a>
        <a href="{osm_url}" target="_blank" rel="noopener noreferrer"
           class="shifa-btn shifa-btn-blue">🌍 خريطة مفتوحة</a>
      </div>
    </div>

    <style>
      /* ── Card v3 overrides (scoped, won't break other pages) ── */
      .shifa-card {{
        direction: rtl;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px 22px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.25s ease, border-color 0.3s ease, box-shadow 0.3s ease;
      }}
      .shifa-card:hover {{
        transform: translateY(-3px);
        border-color: rgba(37, 99, 235, 0.3);
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.1);
      }}

      /* Header row: rank badge + name */
      .shifa-card-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
        flex-wrap: wrap;
      }}
      .shifa-card-name {{
        color: #1e293b;
        font-weight: 700;
        font-size: 1.05rem;
        flex: 1 1 0;
        min-width: 0;
        word-break: break-word;
      }}

      /* Type + distance row */
      .shifa-card-meta-top {{
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 10px;
      }}
      .shifa-distance-badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        color: #2563eb;
        font-weight: 700;
        font-size: 0.9rem;
        background: rgba(37,99,235,0.07);
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid rgba(37,99,235,0.15);
      }}

      /* Detail rows */
      .shifa-card-details {{
        display: flex;
        flex-direction: column;
        gap: 5px;
        margin-bottom: 14px;
      }}
      .shifa-meta-row {{
        display: flex;
        align-items: flex-start;
        gap: 6px;
        font-size: 0.84rem;
        color: #475569;
        word-break: break-word;
      }}
      .shifa-meta-icon {{
        flex-shrink: 0;
        opacity: 0.75;
        margin-top: 1px;
      }}
      .shifa-meta-text {{
        flex: 1;
        min-width: 0;
        word-break: break-word;
      }}
      .shifa-phone-link {{
        color: #059669;
        text-decoration: none;
        display: flex;
        align-items: center;
        gap: 6px;
      }}
      .shifa-phone-link:hover {{ text-decoration: underline; }}

      /* Action buttons row — always at bottom */
      .shifa-card-btns {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 4px;
      }}
      .shifa-card-btns .shifa-btn {{
        flex: 1 1 140px;
        min-width: 0;
      }}

      /* Rank Badge */
      .rank-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 32px;
        height: 32px;
        padding: 0 8px;
        border-radius: 999px;
        background: #eff6ff;
        color: #2563eb;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #bfdbfe;
        flex-shrink: 0;
      }}

      /* Type Badge */
      .type-badge {{
        display: inline-block;
        background: #f5f3ff;
        border: 1px solid #ddd6fe;
        color: #6d28d9;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        white-space: nowrap;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
      }}

      /* Action Buttons */
      .shifa-btn {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        text-align: center;
        padding: 10px 14px;
        border-radius: 12px;
        font-size: 0.84rem;
        font-weight: 700;
        text-decoration: none;
        transition: all 0.2s ease;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .shifa-btn:hover {{ transform: translateY(-1px); }}
      .shifa-btn-green {{
        border: 1px solid rgba(16, 185, 129, 0.25);
        background: rgba(16, 185, 129, 0.07);
        color: #059669;
      }}
      .shifa-btn-green:hover {{ background: rgba(16, 185, 129, 0.15); }}
      .shifa-btn-blue {{
        border: 1px solid rgba(37, 99, 235, 0.25);
        background: rgba(37, 99, 235, 0.07);
        color: #2563eb;
      }}
      .shifa-btn-blue:hover {{ background: rgba(37, 99, 235, 0.15); }}

      /* Mobile */
      @media (max-width: 600px) {{
        .shifa-card-btns {{ flex-direction: column; }}
        .shifa-card-btns .shifa-btn {{ flex: 1 1 100%; }}
      }}
    </style>
    """, unsafe_allow_html=True)


def render_stats_row(
    total: int,
    hospitals: int,
    clinics: int,
    pharmacies: int,
    nearest_distance: str,
) -> None:
    """Render a responsive stats summary row."""
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-number">{total}</div>
        <div class="stat-label">إجمالي المرافق</div>
      </div>
      <div class="stat-card">
        <div class="stat-number" style="color:#EF4444">{hospitals}</div>
        <div class="stat-label">🏥 مستشفى</div>
      </div>
      <div class="stat-card">
        <div class="stat-number" style="color:#F59E0B">{clinics}</div>
        <div class="stat-label">🏨 عيادة</div>
      </div>
      <div class="stat-card">
        <div class="stat-number" style="color:#8B5CF6">{pharmacies}</div>
        <div class="stat-label">💊 صيدلية</div>
      </div>
      <div class="stat-card">
        <div class="stat-number" style="color:#10B981">{nearest_distance}</div>
        <div class="stat-label">أقرب مرفق</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_map_legend() -> None:
    """Render a color-coded map legend."""
    st.markdown("""
    <div class="legend-row">
      <span>🔴 مستشفى</span>
      <span>🟠 عيادة / طبيب</span>
      <span>🟢 مركز صحي</span>
      <span>🟣 صيدلية</span>
      <span>🔵 موقعك</span>
    </div>
    """, unsafe_allow_html=True)
