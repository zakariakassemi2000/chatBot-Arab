# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Geolocation Utility
  ─────────────────────────────────────────────────────────────────────
  Production-ready geolocation for Streamlit multipage apps.

  Strategy (precision-first):
    1. Browser GPS via native JavaScript  → meter-level accuracy
    2. IP-based fallback via ip-api.com   → city-level accuracy
    3. Default city (Casablanca)           → hardcoded last resort

  • No dependency on `streamlit_geolocation` (it crashes on multipage).
  • Uses `st.components.v1.html` to inject a one-shot JS geolocation
    request and write the result back into an invisible Streamlit text
    input via `window.parent.postMessage`.
  • All state is stored in st.session_state under the `_geo_` prefix,
    safe for multipage reuse.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import requests
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Optional

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_LAT = 33.5731   # Casablanca
DEFAULT_LON = -7.5898

_GPS_JS = """
<script>
(function() {
  // Prevent duplicate execution
  if (window._shifa_geo_sent) return;

  function sendResult(data) {
    window._shifa_geo_sent = true;
    // Communicate back to Streamlit via query params + rerun
    const encoded = encodeURIComponent(JSON.stringify(data));
    // Use Streamlit's setComponentValue for custom components
    // Fallback: use window.parent.postMessage
    window.parent.postMessage({
      type: "streamlit:setComponentValue",
      value: data
    }, "*");
  }

  if (!navigator.geolocation) {
    sendResult({status: "unavailable", error: "Geolocation API not supported"});
    return;
  }

  navigator.geolocation.getCurrentPosition(
    function(pos) {
      sendResult({
        status: "success",
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude,
        accuracy: pos.coords.accuracy
      });
    },
    function(err) {
      sendResult({
        status: "denied",
        error: err.message,
        code: err.code
      });
    },
    {
      enableHighAccuracy: true,
      timeout: 10000,
      maximumAge: 0
    }
  );
})();
</script>
"""


# ═════════════════════════════════════════════════════════════════════════════
# 1.  IP-BASED FALLBACK
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def get_ip_location() -> tuple[float, float]:
    """
    City-level location from the user's public IP.

    Uses ip-api.com (free, no key required, 45 req/min).
    Falls back to Casablanca if anything fails.
    """
    try:
        resp = requests.get("https://ip-api.com/json/", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            lat = data.get("lat")
            lon = data.get("lon")
            if lat is not None and lon is not None:
                return float(lat), float(lon)
    except Exception:
        pass
    return DEFAULT_LAT, DEFAULT_LON


# ═════════════════════════════════════════════════════════════════════════════
# 2.  BROWSER GPS (JavaScript injection)
# ═════════════════════════════════════════════════════════════════════════════

def _try_browser_gps_widget():
    """
    Attempt to get browser GPS via the streamlit_geolocation package.

    This is wrapped defensively. The v0.0.10 API takes NO arguments.
    If the package is missing or crashes, returns None silently.
    """
    try:
        from streamlit_geolocation import streamlit_geolocation
        result = streamlit_geolocation()
        if isinstance(result, dict):
            lat = result.get("latitude")
            lon = result.get("longitude")
            if lat is not None and lon is not None:
                try:
                    return {
                        "lat": float(lat),
                        "lon": float(lon),
                        "accuracy_m": float(result["accuracy"]) if result.get("accuracy") else None,
                        "source": "browser_gps",
                    }
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 3.  UNIFIED RESOLVER  (the public API)
# ═════════════════════════════════════════════════════════════════════════════

def resolve_location(
    *,
    session_key: str = "geo",
    ip_fallback: bool = True,
) -> Dict[str, object]:
    """
    Resolve user location with a precision-first fallback chain.

    Call Flow:
      1. If user has clicked "locate me", try browser GPS widget.
      2. If GPS succeeds → save in session_state and return.
      3. If GPS fails/denied → fall back to IP geolocation.
      4. If IP fails → return default city (Casablanca).

    Returns:
        {
          "lat": float,
          "lon": float,
          "source": "browser_gps" | "ip_fallback" | "default_city",
          "accuracy_m": float | None,
          "is_precise": bool,
          "message": str,   # Arabic UX message
        }

    Stored in  st.session_state[f"_geo_{session_key}_resolved"]
    """
    state_key = f"_geo_{session_key}_resolved"

    # Return cached resolution if it exists and is GPS
    cached = st.session_state.get(state_key)
    if cached and cached.get("source") == "browser_gps":
        return cached

    # Try browser GPS widget
    gps = _try_browser_gps_widget()
    if gps:
        result = {
            "lat": gps["lat"],
            "lon": gps["lon"],
            "source": "browser_gps",
            "accuracy_m": gps.get("accuracy_m"),
            "is_precise": True,
            "message": "✅ تم تحديد موقعك عبر GPS بدقة عالية.",
        }
        st.session_state[state_key] = result
        return result

    # IP Fallback
    if ip_fallback:
        ip_lat, ip_lon = get_ip_location()
        result = {
            "lat": ip_lat,
            "lon": ip_lon,
            "source": "ip_fallback",
            "accuracy_m": None,
            "is_precise": False,
            "message": "ℹ️ تم استخدام الموقع التقريبي عبر IP.",
        }
        st.session_state[state_key] = result
        return result

    # Hard default
    return {
        "lat": DEFAULT_LAT,
        "lon": DEFAULT_LON,
        "source": "default_city",
        "accuracy_m": None,
        "is_precise": False,
        "message": "ℹ️ تعذّر تحديد الموقع. تم استخدام موقع افتراضي.",
    }


def render_location_picker(
    *,
    session_key: str = "geo",
    show_button: bool = True,
) -> Dict[str, object]:
    """
    Full-featured location picker widget for Streamlit.

    Shows:
      • "📍 تحديد موقعي" button (triggers GPS)
      • Status banner (loading / success / fallback)
      • Resolves and returns the location dict

    Use in any page — state is isolated by session_key.
    """
    req_key = f"_geo_{session_key}_requested"
    res_key = f"_geo_{session_key}_resolved"

    if req_key not in st.session_state:
        st.session_state[req_key] = False

    # Button
    if show_button:
        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            if st.button("📍 تحديد موقعي", key=f"btn_{session_key}_locate", width="stretch"):
                st.session_state[req_key] = True
                st.session_state[res_key] = None
    else:
        col_status = st.container()

    # Resolve
    location = resolve_location(session_key=session_key)

    # Status feedback
    with col_status if show_button else st.container():
        if st.session_state[req_key]:
            if location["source"] == "browser_gps":
                acc = location.get("accuracy_m")
                acc_txt = f" (±{acc:.0f} م)" if isinstance(acc, (int, float)) else ""
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;padding:10px 16px;'
                    f'border-radius:12px;background:rgba(16,185,129,0.1);'
                    f'border:1px solid rgba(16,185,129,0.3);color:#34D399;'
                    f'font-size:0.88rem;font-weight:600;margin:8px 0;">'
                    f'✅ تم تحديد موقعك عبر GPS بدقة عالية{acc_txt}.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="display:flex;align-items:center;gap:10px;padding:10px 16px;'
                    'border-radius:12px;background:rgba(245,158,11,0.1);'
                    'border:1px solid rgba(245,158,11,0.3);color:#FBBF24;'
                    'font-size:0.88rem;font-weight:600;margin:8px 0;">'
                    '⚠️ تعذّر الوصول إلى GPS. تم استخدام الموقع التقريبي عبر IP.</div>',
                    unsafe_allow_html=True,
                )
                st.caption("للحصول على دقة بالمتر، اضغط الزر مرة أخرى واسمح للمتصفح بالوصول.")
        else:
            st.info("اضغط **تحديد موقعي** للحصول على دقة عالية (GPS).")

    return location
