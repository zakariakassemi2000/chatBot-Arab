# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Reusable Card Components v2
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
    Render a single medical facility as a premium flexbox card.

    Layout:
      ┌──────────────────────────────────────────────────┐
      │  #rank  Name                    [Google Maps]    │
      │  Type Badge                     [OpenStreetMap]  │
      │  📏 Distance                                     │
      │  📍 Address  ⚕️ Specialty  📞 Phone  🕐 Hours    │
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
        f'<div style="color:#94A3B8;font-size:0.84rem;margin-top:4px;'
        f'display:flex;align-items:center;gap:6px;">'
        f'<span style="opacity:0.7;">📍</span> {address}</div>'
        if address else ""
    )
    specialty_row = (
        f'<div style="color:#A78BFA;font-size:0.84rem;margin-top:4px;'
        f'display:flex;align-items:center;gap:6px;">'
        f'<span style="opacity:0.7;">⚕️</span> {specialty}</div>'
        if specialty else ""
    )
    phone_row = (
        f'<div style="margin-top:6px;">'
        f'<a href="tel:{phone}" style="color:#34D399;text-decoration:none;'
        f'font-size:0.84rem;display:flex;align-items:center;gap:6px;">'
        f'<span style="opacity:0.7;">📞</span> {phone}</a></div>'
        if phone else ""
    )
    opening_row = (
        f'<div style="color:#64748B;font-size:0.8rem;margin-top:4px;'
        f'display:flex;align-items:center;gap:6px;">'
        f'<span style="opacity:0.7;">🕐</span> {opening}</div>'
        if opening else ""
    )

    st.markdown(f"""
    <div class="shifa-card">
      <div class="shifa-card-row">
        <!-- Info column -->
        <div class="shifa-card-info">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap;">
            <span class="rank-badge">#{rank}</span>
            <span style="color:#E2E8F0;font-weight:700;font-size:1.08rem;">{name}</span>
          </div>
          <div class="type-badge">{type_label}</div>
          <div style="color:#60A5FA;font-weight:700;font-size:0.95rem;margin:6px 0;
                      display:flex;align-items:center;gap:6px;">
            <span>📏</span> {distance}
          </div>
          {address_row}
          {specialty_row}
          {phone_row}
          {opening_row}
        </div>

        <!-- Actions column -->
        <div class="shifa-card-actions">
          <a href="{gmaps_url}" target="_blank" rel="noopener noreferrer"
             class="shifa-btn shifa-btn-green">🗺️ خرائط جوجل</a>
          <a href="{osm_url}" target="_blank" rel="noopener noreferrer"
             class="shifa-btn shifa-btn-blue">🌍 خريطة مفتوحة</a>
        </div>
      </div>
    </div>
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
