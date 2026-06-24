# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Nearby Care Finder  (Free & Open Source)
  ─────────────────────────────────────────────────────────────────────
  • Overpass API  → No API key required, 100 % free (OpenStreetMap)
  • Haversine     → Accurate distance calculation in meters
  • Folium        → Interactive map rendered inside Streamlit
═══════════════════════════════════════════════════════════════════════
"""

import math
import html
import requests
import streamlit as st
from typing import List, Dict, Optional

try:
    import folium
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────
# Primary + fallback Overpass API mirrors (tried in order)
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",          # primary
    "https://overpass.kumi.systems/api/interpreter",    # mirror 1
    "https://overpass.openstreetmap.fr/api/interpreter",# mirror 2
]
OVERPASS_TIMEOUT = 25          # seconds per mirror attempt
OSM_QUERY_TIMEOUT = 20         # seconds declared inside the QL query

# Amenity → human-readable Arabic label
AMENITY_LABELS: Dict[str, str] = {
    "hospital":       "🏥 مستشفى",
    "clinic":         "🏨 عيادة",
    "doctors":        "👨‍⚕️ طبيب خاص",
    "health_centre":  "🏥 مركز صحي",
    "pharmacy":       "💊 صيدلية",
}

# Folium icon palette per amenity
AMENITY_ICONS: Dict[str, Dict] = {
    "hospital":       {"color": "red",    "icon": "plus-square",  "prefix": "fa"},
    "clinic":         {"color": "orange", "icon": "stethoscope",  "prefix": "fa"},
    "doctors":        {"color": "orange", "icon": "user-md",      "prefix": "fa"},
    "health_centre":  {"color": "green",  "icon": "hospital-o",   "prefix": "fa"},
    "pharmacy":       {"color": "purple", "icon": "medkit",       "prefix": "fa"},
}

# Healthcare speciality key mappings (OSM tag → Arabic)
SPECIALTY_MAP: Dict[str, str] = {
    "general":          "طب عام",
    "general_practice": "طب عام",
    "cardiology":       "أمراض القلب",
    "dermatology":      "الأمراض الجلدية",
    "gynaecology":      "أمراض النساء",
    "ophthalmology":    "طب العيون",
    "paediatrics":      "طب الأطفال",
    "orthopaedics":     "جراحة العظام",
    "neurology":        "طب الأعصاب",
    "oncology":         "الأورام",
    "radiology":        "الأشعة",
    "surgery":          "الجراحة",
    "dentistry":        "طب الأسنان",
    "psychiatry":       "الطب النفسي",
    "urology":          "المسالك البولية",
    "endocrinology":    "الغدد الصماء",
    "gastroenterology": "الجهاز الهضمي",
    "nephrology":       "أمراض الكلى",
    "emergency":        "طوارئ",
}


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DISTANCE CALCULATION  (Haversine formula)
# ═════════════════════════════════════════════════════════════════════════════

def calculate_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> int:
    """
    Return the great-circle distance between two points in **meters**.

    Uses the Haversine formula which is accurate to within ~0.3 % for
    distances up to a few hundred kilometers — perfect for city-scale queries.

    Args:
        lat1, lon1: Origin coordinates (user location).
        lat2, lon2: Destination coordinates (facility location).

    Returns:
        Distance in whole meters (int).
    """
    R = 6_371_000  # Earth's mean radius in meters

    phi1, phi2     = math.radians(lat1), math.radians(lat2)
    d_phi          = math.radians(lat2 - lat1)
    d_lambda       = math.radians(lon2 - lon1)

    a = (math.sin(d_phi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2)

    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return int(R * c)


def _format_distance(meters: int) -> str:
    """Return a human-readable distance string (e.g. '1.3 كم' or '850 م')."""
    if meters >= 1000:
        return f"{meters / 1000:.1f} كم"
    return f"{meters} م"


# ═════════════════════════════════════════════════════════════════════════════
# 2.  OVERPASS API  — fetch nearby medical facilities
# ═════════════════════════════════════════════════════════════════════════════

def _build_overpass_query(user_lat: float, user_lon: float, radius: int) -> str:
    """
    Build an Overpass QL query that fetches all medical POIs within *radius* meters.

    We query both **node** and **way** geometries and ask for `out center` so
    that ways (polygon buildings) are returned with a single centroid coordinate.
    """
    amenities = ["hospital", "clinic", "doctors", "health_centre", "pharmacy"]
    node_blocks = "\n".join(
        f'  node["amenity"="{a}"](around:{radius},{user_lat},{user_lon});'
        for a in amenities
    )
    way_blocks = "\n".join(
        f'  way["amenity"="{a}"](around:{radius},{user_lat},{user_lon});'
        for a in amenities
    )

    return f"""
[out:json][timeout:{OSM_QUERY_TIMEOUT}];
(
{node_blocks}
{way_blocks}
);
out center tags;
"""


def _parse_element(
    element: dict,
    user_lat: float,
    user_lon: float
) -> Optional[Dict]:
    """
    Parse a single Overpass element into our internal dict format.

    Returns None if the element lacks valid coordinates.
    """
    el_type = element.get("type")
    tags    = element.get("tags", {})

    # ── Coordinates ──────────────────────────────────────────────────────
    if el_type == "node":
        lat = element.get("lat")
        lon = element.get("lon")
    elif el_type == "way":
        center = element.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")
    else:
        return None

    if lat is None or lon is None:
        return None

    # ── Name (prefer Arabic) ─────────────────────────────────────────────
    name = (
        tags.get("name:ar")
        or tags.get("name")
        or tags.get("name:en")
        or tags.get("name:fr")
        or "مرفق طبي"
    )

    # ── Specialty ────────────────────────────────────────────────────────
    raw_specialty = (
        tags.get("healthcare:speciality")
        or tags.get("speciality")
        or tags.get("healthcare")
        or ""
    ).lower().strip()

    # Only show specialty if it has an Arabic translation in SPECIALTY_MAP
    # and it's not the same as the amenity type (to avoid duplicate info).
    amenity_raw = tags.get("amenity", "").lower()
    if raw_specialty and raw_specialty != amenity_raw and raw_specialty in SPECIALTY_MAP:
        specialty_ar = SPECIALTY_MAP[raw_specialty]
    else:
        specialty_ar = ""

    # ── Amenity type ─────────────────────────────────────────────────────
    amenity      = tags.get("amenity", "clinic")
    type_label   = AMENITY_LABELS.get(amenity, "🏥 مرفق صحي")
    icon_cfg     = AMENITY_ICONS.get(amenity, AMENITY_ICONS["clinic"])

    # ── Distance ─────────────────────────────────────────────────────────
    distance_m   = calculate_distance(user_lat, user_lon, lat, lon)

    # ── Contact info ─────────────────────────────────────────────────────
    phone   = tags.get("phone") or tags.get("contact:phone") or ""
    website = tags.get("website") or tags.get("contact:website") or ""
    address = (
        " ".join(filter(None, [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:city", ""),
        ])).strip()
    )
    opening_hours = tags.get("opening_hours", "")

    return {
        "osm_id":       element.get("id"),
        "name":         name,
        "amenity":      amenity,
        "type_label":   type_label,
        "specialty":    specialty_ar,
        "lat":          lat,
        "lon":          lon,
        "distance_m":   distance_m,
        "distance_str": _format_distance(distance_m),
        "phone":        phone,
        "website":      website,
        "address":      address,
        "opening_hours": opening_hours,
        "icon_cfg":     icon_cfg,
        "gmaps_url":    f"https://www.google.com/maps/search/?api=1&query={lat},{lon}",
        "osm_url":      f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=17/{lat}/{lon}",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_nearby_doctors(
    user_lat: float,
    user_lon: float,
    radius_meters: int = 5000,
) -> List[Dict]:
    """
    Fetch medical facilities near *user_lat/user_lon* within *radius_meters*.

    Data source: OpenStreetMap via the free Overpass API — no API key needed.

    Args:
        user_lat:      Latitude of the user.
        user_lon:      Longitude of the user.
        radius_meters: Search radius in meters (default 5 km).

    Returns:
        List of facility dicts, sorted by distance (nearest first).
        Each dict contains: name, type_label, specialty, lat, lon,
        distance_m, distance_str, phone, address, opening_hours, gmaps_url.
    """
    query = _build_overpass_query(user_lat, user_lon, radius_meters)

    data = None
    last_error: str = ""

    for mirror_url in OVERPASS_MIRRORS:
        try:
            response = requests.post(
                mirror_url,
                data={"data": query},
                timeout=OVERPASS_TIMEOUT,
                headers={"User-Agent": "SHIFA-AI/1.0 (medical chatbot; educational)"},
            )
            response.raise_for_status()
            data = response.json()
            break  # success — stop trying mirrors
        except requests.exceptions.Timeout:
            last_error = f"⏱️ انتهت مهلة الاتصال بـ {mirror_url}"
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
        except ValueError:
            last_error = "استجابة غير صالحة من الخادم"

    if data is None:
        st.warning(
            "⚠️ تعذّر الاتصال بخوادم OpenStreetMap (تم المحاولة على 3 خوادم).\n\n"
            f"السبب: {last_error}\n\n"
            "💡 **نصائح:** وسّع النطاق، أزل الفلاتر، أو حاول مجدداً بعد دقيقة."
        )
        return []

    results: List[Dict] = []
    seen_ids = set()          # deduplicate: same OSM object can appear as node + way

    for element in data.get("elements", []):
        el_id = (element.get("type"), element.get("id"))
        if el_id in seen_ids:
            continue
        seen_ids.add(el_id)

        parsed = _parse_element(element, user_lat, user_lon)
        if parsed:
            results.append(parsed)

    results.sort(key=lambda x: x["distance_m"])
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MAP RENDERING  (Folium + streamlit-folium)
# ═════════════════════════════════════════════════════════════════════════════

def render_doctors_map(
    user_lat: float,
    user_lon: float,
    doctors_list: List[Dict],
    use_clustering: bool = True,
    height: int = 480,
    zoom_start: int = 14,
) -> None:
    """
    Render an interactive Folium map inside the current Streamlit context.

    Features:
      • CartoDB Positron base layer
      • Blue marker for user location
      • Color-coded markers per amenity type
      • Rich RTL Arabic popups with distance, phone, and links
      • Optional MarkerCluster for densely populated areas
    """
    if not FOLIUM_AVAILABLE:
        st.info("📦 يرجى تثبيت `folium` و `streamlit-folium` لعرض الخريطة التفاعلية.")
        return

    # ── Base map ──────────────────────────────────────────────────────────
    m = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # ── User location ─────────────────────────────────────────────────────
    folium.Marker(
        location=[user_lat, user_lon],
        popup=folium.Popup(
            "<div style='font-family:sans-serif;text-align:center'>"
            "<b style='color:#2563EB'>📍 موقعك الحالي</b>"
            "</div>",
            max_width=180,
        ),
        tooltip="📍 أنت هنا",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    # Radius circle
    if doctors_list:
        max_dist = max(d["distance_m"] for d in doctors_list)
        folium.Circle(
            location=[user_lat, user_lon],
            radius=max_dist,
            color="#3B82F6",
            fill=True,
            fill_color="#3B82F6",
            fill_opacity=0.05,
            weight=1,
        ).add_to(m)

    # ── Facility markers ──────────────────────────────────────────────────
    target = MarkerCluster().add_to(m) if use_clustering else m

    for doc in doctors_list[:50]:
        safe_name = html.escape(str(doc.get("name", "مرفق طبي")))
        safe_type = html.escape(str(doc.get("type_label", "🏥 مرفق صحي")))
        safe_dist = html.escape(str(doc.get("distance_str", "")))
        safe_phone = html.escape(str(doc.get("phone", "")))
        safe_specialty = html.escape(str(doc.get("specialty", "")))
        safe_opening = html.escape(str(doc.get("opening_hours", "")))
        safe_gmaps = html.escape(str(doc.get("gmaps_url", "#")), quote=True)
        safe_osm = html.escape(str(doc.get("osm_url", "#")), quote=True)

        phone_html = (
            f'<a href="tel:{safe_phone}" style="color:#059669">📞 {safe_phone}</a><br>'
            if safe_phone else ""
        )
        specialty_html = (
            f'<span style="color:#7C3AED">⚕️ {safe_specialty}</span><br>'
            if safe_specialty else ""
        )
        opening_html = (
            f'<span style="font-size:0.78rem;color:#6B7280">🕐 {safe_opening}</span><br>'
            if safe_opening else ""
        )

        popup_html = f"""
        <div style="direction:rtl;font-family:sans-serif;
                    min-width:200px;max-width:280px;text-align:right;
                    line-height:1.6">
          <b style="color:#DC2626;font-size:14px">{safe_name}</b><br>
          <span style="color:#6B7280;font-size:0.82rem">{safe_type}</span><br>
          {specialty_html}
          <b style="color:#0284C7">📏 {safe_dist}</b><br>
          {phone_html}
          {opening_html}
          <div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap">
            <a href="{safe_gmaps}" target="_blank"
               style="background:#10B981;color:#fff;padding:3px 8px;
                      border-radius:4px;text-decoration:none;font-size:0.75rem">
              Google Maps
            </a>
            <a href="{safe_osm}" target="_blank"
               style="background:#3B82F6;color:#fff;padding:3px 8px;
                      border-radius:4px;text-decoration:none;font-size:0.75rem">
              OSM
            </a>
          </div>
        </div>
        """

        cfg = doc["icon_cfg"]
        folium.Marker(
            location=[doc["lat"], doc["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{safe_type} · {safe_name} ({safe_dist})",
            icon=folium.Icon(
                color=cfg["color"],
                icon=cfg["icon"],
                prefix=cfg["prefix"],
            ),
        ).add_to(target)

    st_folium(m, width="stretch", height=height, returned_objects=[])


# ═════════════════════════════════════════════════════════════════════════════
# 4.  STREAMLIT UI WIDGET — REFACTORED
# ═════════════════════════════════════════════════════════════════════════════
#
# Geolocation and card rendering now live in centralized modules:
#   • utils.geolocation  → resolve_location(), render_location_picker()
#   • components.card    → render_facility_card(), render_stats_row()
#
# The functions below are kept for backward compatibility with app.py and
# other pages that call render_nearby_care() directly.
# ═════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=3600, show_spinner=False)
def get_approximate_location():
    """Fallback location via IP lookup (city-level, not meter-accurate).

    .. deprecated:: Use ``utils.geolocation.get_ip_location()`` instead.
    """
    try:
        resp = requests.get("http://ip-api.com/json/", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            lat = data.get("lat")
            lon = data.get("lon")
            if lat is not None and lon is not None:
                return float(lat), float(lon)
    except Exception:
        pass
    return 33.5731, -7.5898  # Fallback to Casablanca


def _extract_browser_coords(location_payload: Optional[dict]) -> Optional[Dict[str, float]]:
    """Normalize payload returned by streamlit_geolocation.

    .. deprecated:: Use ``utils.geolocation`` resolver chain instead.
    """
    if not isinstance(location_payload, dict):
        return None
    lat = location_payload.get("latitude")
    lon = location_payload.get("longitude")
    accuracy = location_payload.get("accuracy")
    try:
        if lat is None or lon is None:
            return None
        lat = float(lat)
        lon = float(lon)
        accuracy_m = float(accuracy) if accuracy is not None else None
        return {"lat": lat, "lon": lon, "accuracy_m": accuracy_m}
    except (TypeError, ValueError):
        return None


def resolve_user_location(
    browser_location: Optional[dict],
    *,
    ip_fallback: bool = True,
) -> Dict[str, object]:
    """Resolve user location (legacy API, kept for backward compat).

    .. deprecated:: Use ``utils.geolocation.resolve_location()`` instead.
    """
    parsed = _extract_browser_coords(browser_location)
    if parsed:
        return {
            "lat": parsed["lat"],
            "lon": parsed["lon"],
            "source": "browser_gps",
            "accuracy_m": parsed.get("accuracy_m"),
            "is_precise": True,
            "message": "✅ تم تحديد موقعك عبر GPS بدقة عالية.",
        }
    if ip_fallback:
        ip_lat, ip_lon = get_approximate_location()
        return {
            "lat": ip_lat,
            "lon": ip_lon,
            "source": "ip_fallback",
            "accuracy_m": None,
            "is_precise": False,
            "message": "ℹ️ تم استخدام الموقع التقريبي عبر IP.",
        }
    return {
        "lat": 33.5731,
        "lon": -7.5898,
        "source": "default_city",
        "accuracy_m": None,
        "is_precise": False,
        "message": "ℹ️ تعذّر تحديد الموقع. تم استخدام موقع افتراضي.",
    }


def render_nearby_care(severity: str) -> None:
    """
    Display the Nearby Care widget inside the main chatbot page.

    Uses the centralized geolocation utility and card components.
    Only visible when triage severity is 'critique' or 'élevée'.
    """
    if severity not in ("critique", "élevée"):
        return

    st.markdown("""
    <div style="
      background: rgba(37,99,235,0.08);
      border: 1px solid rgba(37,99,235,0.25);
      border-radius: 14px;
      padding: 18px 22px;
      margin: 18px 0 10px;
      direction: rtl;
    ">
      <div style="font-size:1.05rem;font-weight:700;color:#93C5FD;margin-bottom:4px">
        🏥 أقرب مراكز الرعاية الصحية
      </div>
      <div style="font-size:0.82rem;color:#64748B">
        <br><i>اضغط الزر أدناه واسمح بالوصول إلى الموقع للحصول على أفضل دقة.</i>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Use centralized geolocation ───────────────────────────────────────
    try:
        from utils.geolocation import render_location_picker
        location = render_location_picker(session_key="emergency")
    except ImportError:
        # Fallback if new module not available yet
        location = resolve_user_location(None, ip_fallback=True)

    lat, lon = float(location["lat"]), float(location["lon"])
    _run_search(lat, lon, 5000)


def _run_search(lat: float, lon: float, radius_m: int) -> None:
    """Execute the search and render results using centralized components."""
    with st.spinner("🔄 جاري الاستعلام عن OpenStreetMap..."):
        places = get_nearby_doctors(lat, lon, radius_m)

    if not places:
        st.warning("⚠️ لم يُعثر على أي مرفق صحي في هذا النطاق.")
        st.error("📞 في حالات الطوارئ، اتصل بالإسعاف مباشرة: **15**")
        return

    st.success(f"✅ تم العثور على **{len(places)}** مرفق صحي ضمن نطاق {radius_m // 1000} كم.")

    # Map
    if FOLIUM_AVAILABLE:
        st.markdown("### 🗺️ خريطة الرعاية القريبة")
        render_doctors_map(lat, lon, places)
    else:
        st.info("لتفعيل الخريطة التفاعلية:\n```\npip install folium streamlit-folium\n```")

    # Top-5 list using centralized card component
    st.markdown("### 📋 أقرب 5 مراكز صحية")
    try:
        from components.card import render_facility_card
        for i, place in enumerate(places[:5], 1):
            render_facility_card(i, place)
    except ImportError:
        for i, place in enumerate(places[:5], 1):
            _render_facility_card(i, place)


def _render_facility_card(rank: int, place: Dict) -> None:
    """Inline fallback card renderer (used if components.card is unavailable)."""
    name       = html.escape(str(place.get("name", "مرفق طبي")))
    type_label = html.escape(str(place.get("type_label", "🏥 مرفق صحي")))
    distance   = html.escape(str(place.get("distance_str", "")))
    address    = html.escape(str(place.get("address", "")))
    phone      = html.escape(str(place.get("phone", "")))
    gmaps_url  = html.escape(str(place.get("gmaps_url", "#")), quote=True)
    osm_url    = html.escape(str(place.get("osm_url", "#")), quote=True)

    phone_html = (
        f'<div style="color:#34D399;font-size:0.82rem;margin-top:4px;">📞 {phone}</div>'
        if phone else ""
    )
    address_html = (
        f'<div style="color:#94A3B8;font-size:0.82rem;margin-top:2px;">📍 {address}</div>'
        if address else ""
    )

    st.markdown(f"""
    <div style="
      direction: rtl;
      background: rgba(15, 23, 42, 0.66);
      border: 1px solid rgba(148, 163, 184, 0.15);
      border-radius: 16px;
      padding: 18px 20px;
      margin: 10px 0;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    ">
      <div style="display:flex;align-items:stretch;justify-content:space-between;gap:16px;flex-wrap:wrap;">
        <div style="flex:1 1 280px;min-width:220px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
            <span style="display:inline-flex;align-items:center;justify-content:center;
              min-width:30px;height:30px;border-radius:999px;
              background:rgba(59,130,246,0.18);color:#93C5FD;
              font-weight:700;font-size:0.82rem;">#{rank}</span>
            <span style="color:#E2E8F0;font-weight:700;font-size:1.05rem;">{name}</span>
          </div>
          <div style="color:#A5B4FC;font-size:0.84rem;font-weight:600;margin-bottom:6px;">{type_label}</div>
          <div style="color:#FCA5A5;font-weight:700;font-size:0.92rem;margin-bottom:4px;">📏 {distance}</div>
          {address_html}
          {phone_html}
        </div>
        <div style="flex:0 0 auto;min-width:140px;display:flex;flex-direction:column;gap:8px;justify-content:center;align-self:center;">
          <a href="{gmaps_url}" target="_blank" rel="noopener noreferrer" style="
            display:block;text-align:center;padding:10px 14px;border-radius:10px;
            border:1px solid rgba(16,185,129,0.35);background:rgba(16,185,129,0.12);
            color:#34D399;font-size:0.82rem;font-weight:700;text-decoration:none;
          ">🗺️ Google Maps</a>
          <a href="{osm_url}" target="_blank" rel="noopener noreferrer" style="
            display:block;text-align:center;padding:10px 14px;border-radius:10px;
            border:1px solid rgba(59,130,246,0.35);background:rgba(59,130,246,0.12);
            color:#93C5FD;font-size:0.82rem;font-weight:700;text-decoration:none;
          ">🌍 OpenStreetMap</a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
