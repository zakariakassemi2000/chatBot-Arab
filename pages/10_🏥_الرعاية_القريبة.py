# -*- coding: utf-8 -*-
"""
Nearby Care page for SHIFA AI.

Design goals:
  - Cleaner hierarchy for the hero and search panel
  - Faster location shortcuts without clutter
  - Better mobile spacing and label wrapping
  - Stronger visual continuity between setup and results
"""

from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd
import streamlit as st

from components.card import (
    render_facility_card,
    render_map_legend,
    render_stats_row,
)
from engine.nearby_care import (
    AMENITY_LABELS,
    FOLIUM_AVAILABLE,
    _format_distance,
    get_nearby_doctors,
    render_doctors_map,
)
from styles.theme import inject_theme
from utils.geolocation import render_location_picker


PAGE_TITLE = "SHIFA AI · الرعاية الصحية القريبة"
PAGE_ICON = "🏥"
CACHE_DURATION = 3600
MAX_RESULTS_DISPLAY = 100
DEFAULT_RADIUS_KM = 5
QUICK_CITY_SHORTCUTS: List[Tuple[str, float, float, str]] = [
    ("الدار البيضاء", 33.5731, -7.5898, "وصول سريع لمدن المغرب"),
    ("القاهرة", 30.0444, 31.2357, "وصول سريع لمصر"),
    ("الجزائر", 36.7372, 3.0871, "وصول سريع للجزائر"),
    ("الرياض", 24.7136, 46.6753, "وصول سريع للسعودية"),
]


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      [data-testid="stSidebarNav"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initialize session state variables used by the page."""
    defaults = {
        "last_search_time": 0,
        "cached_places": [],
        "search_coords": (None, None),
        "search_radius": DEFAULT_RADIUS_KM,
        "page_lat": None,
        "page_lon": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_nearby_care_page_styles() -> None:
    """Apply page-specific visual overrides without affecting app logic."""
    st.markdown(
        """
        <style>
          .block-container {
            max-width: 1240px !important;
            padding-top: 1.25rem !important;
          }

          div[data-testid="stNumberInput"] label,
          div[data-testid="stSlider"] label,
          div[data-testid="stSelectSlider"] label,
          div[data-testid="stMultiSelect"] label,
          div[data-testid="stTextInput"] label,
          div[data-testid="stSelectbox"] label {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: initial !important;
            line-height: 1.5 !important;
          }

          .stButton > button[kind="primary"],
          .stButton > button {
            min-height: 3rem !important;
            white-space: normal !important;
            line-height: 1.35 !important;
          }

          .search-panel {
            padding: 0 !important;
            overflow: hidden;
          }

          .nearby-search-shell {
            display: flex;
            flex-direction: column;
            gap: 18px;
            padding: 28px 30px;
          }

          .nearby-panel-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
          }

          .nearby-panel-copy {
            flex: 1 1 420px;
          }

          .nearby-inline-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.22);
            color: #BFDBFE;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 10px;
          }

          .nearby-panel-title {
            color: #F8FAFC;
            font-size: 1.12rem;
            font-weight: 800;
            margin-bottom: 6px;
          }

          .nearby-panel-text {
            color: #94A3B8;
            font-size: 0.95rem;
            line-height: 1.8;
            max-width: 760px;
          }

          .nearby-help-box {
            flex: 0 1 300px;
            padding: 16px 18px;
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.72), rgba(2, 6, 23, 0.62));
            border: 1px solid rgba(148, 163, 184, 0.14);
          }

          .nearby-help-title {
            color: #E2E8F0;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 8px;
          }

          .nearby-help-text {
            color: #94A3B8;
            font-size: 0.84rem;
            line-height: 1.7;
          }

          .nearby-quick-title {
            color: #CBD5E1;
            font-size: 0.92rem;
            font-weight: 700;
            margin-top: 2px;
            margin-bottom: -4px;
          }

          .nearby-snapshot {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
          }

          .nearby-snapshot-card {
            padding: 14px 16px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
          }

          .nearby-snapshot-label {
            color: #94A3B8;
            font-size: 0.8rem;
            margin-bottom: 6px;
          }

          .nearby-snapshot-value {
            color: #F8FAFC;
            font-size: 0.96rem;
            font-weight: 700;
            line-height: 1.5;
          }

          .nearby-snapshot-sub {
            color: #64748B;
            font-size: 0.78rem;
            margin-top: 4px;
            line-height: 1.5;
          }

          .nearby-cta-note {
            color: #93C5FD;
            font-size: 0.84rem;
            font-weight: 600;
            padding: 12px 14px;
            border-radius: 14px;
            background: rgba(37, 99, 235, 0.08);
            border: 1px dashed rgba(37, 99, 235, 0.24);
          }

          .nearby-result-anchor {
            color: #E2E8F0;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 6px 0 12px;
          }

          @media (max-width: 768px) {
            .nearby-search-shell {
              padding: 20px 16px;
            }

            .nearby-snapshot {
              grid-template-columns: 1fr;
            }

            .nearby-panel-title {
              font-size: 1rem;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def infer_active_location(
    default_lat: float,
    default_lon: float,
    current_lat: float,
    current_lon: float,
    location_source: str,
) -> Tuple[str, str]:
    """Infer the human-readable location mode shown in the search summary."""
    tolerance = 1e-4

    for city_name, city_lat, city_lon, city_hint in QUICK_CITY_SHORTCUTS:
        if abs(current_lat - city_lat) < tolerance and abs(current_lon - city_lon) < tolerance:
            return city_name, f"تم اختيار مدينة جاهزة: {city_hint}"

    if abs(current_lat - default_lat) > tolerance or abs(current_lon - default_lon) > tolerance:
        return "إحداثيات مخصصة", "تم إدخال الموقع يدوياً عبر الإعدادات المتقدمة."

    if location_source == "browser_gps":
        return "موقعك الحالي", "تم تحديد الموقع بدقة عبر GPS."

    if location_source == "ip_fallback":
        return "موقع تقريبي", "تم استخدام الموقع التقريبي عبر IP."

    return "موقع افتراضي", "تعذر تحديد الموقع بدقة وتم استخدام قيمة افتراضية."


def fetch_places_with_cache(lat: float, lon: float, radius_m: int) -> List[Dict]:
    """Fetch nearby places with session caching to reduce repeated requests."""
    current_time = time.time()

    if (
        st.session_state.cached_places
        and st.session_state.search_coords == (lat, lon)
        and st.session_state.search_radius == radius_m / 1000
        and current_time - st.session_state.last_search_time < CACHE_DURATION
    ):
        return st.session_state.cached_places

    with st.spinner("🔄 جاري البحث عن المنشآت الصحية القريبة..."):
        try:
            places = get_nearby_doctors(lat, lon, radius_m)
        except Exception:
            st.error("تعذر الاتصال بالخدمة، حاول لاحقاً")
            return []

    st.session_state.cached_places = places
    st.session_state.search_coords = (lat, lon)
    st.session_state.last_search_time = current_time
    return places


with st.sidebar:
    st.page_link("app.py", label="الرجوع للرئيسية", icon="🏠")

    with st.expander("ℹ️ معلومات التطبيق", expanded=False):
        st.info(
            """
            **المصادر:**
            • خرائط OpenStreetMap المفتوحة
            • تحديد الموقع عبر GPS أو IP
            • تحديث البيانات كل ساعة

            **الميزات:**
            • بحث حتى 20 كم
            • 5 أنواع من المنشآت الصحية
            • تصدير النتائج بصيغة CSV أو Excel
            """
        )


inject_theme()
inject_nearby_care_page_styles()
init_session_state()


st.markdown("""
<style>
* { box-sizing: border-box; }
[data-testid="stHorizontalBlock"] { gap: 12px; }
.stMarkdown { word-wrap: break-word; }

[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    display: flex;
    align-items: center;
    gap: 8px;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary span,
[data-testid="stSidebar"] .stPageLink a,
[data-testid="stSidebar"] .stAlert {
    overflow: visible !important;
    white-space: normal !important;
    text-overflow: initial !important;
    line-height: 1.6 !important;
}

.hero-banner { padding: 30px; border-radius: 16px; text-align: right; margin-bottom: 20px; }
.hero-title { font-size: 28px; font-weight: bold; }
.hero-sub { opacity: 0.8; margin-top: 10px; }

.stButton button {
    background: linear-gradient(135deg, #2563EB, #10B981);
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 16px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🏥 الرعاية الصحية القريبة منك</div>
        <div class="hero-sub">
            اكتشف أقرب المستشفيات والعيادات والصيدليات والأطباء إليك بسهولة، مع واجهة أوضح
            وترتيب أفضل للبحث والنتائج.
        </div>
        <div class="hero-badge">⚡ مجاني 100% · بيانات مفتوحة · بدون مفتاح API</div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.container():
    st.markdown("### 📍 حدد موقعك")

    location = render_location_picker(session_key="nearby_care")
    default_lat = float(location["lat"])
    default_lon = float(location["lon"])

    with st.expander("⚙️ إعدادات متقدمة", expanded=False):
        user_lat = st.number_input("📍 خط العرض", value=default_lat)
        user_lon = st.number_input("📍 خط الطول", value=default_lon)

    st.session_state.page_lat = user_lat
    st.session_state.page_lon = user_lon
    
    # Radius + filter
    col1, col2 = st.columns([1,2])

    with col1:
        radius_km = st.slider(
            "🔍 نطاق البحث (كم)",
            1, 20,
            int(st.session_state.search_radius) if hasattr(st.session_state, 'search_radius') else 5
        )
        st.session_state.search_radius = radius_km

    with col2:
        amenity_filter = st.multiselect(
            "🏥 نوع المنشأة",
            list(AMENITY_LABELS.values())
        )

    # CTA INSIDE panel
    do_search = st.button(
        "🚀 ابدأ البحث الآن",
        use_container_width=True
    )


places = None

if do_search:
    radius_m = radius_km * 1000
    all_places = fetch_places_with_cache(user_lat, user_lon, radius_m)

    if amenity_filter:
        places = [place for place in all_places if place["type_label"] in amenity_filter]
    else:
        places = all_places

    if not places:
        st.warning("لم يتم العثور على نتائج")

        with st.expander("💡 نصائح للحصول على نتائج أفضل", expanded=True):
            st.markdown(
                """
                - 🗺️ **وسّع نطاق البحث** — جرب 10 أو 15 كم بدلاً من 5 كم
                - 🏥 **أزل الفلاتر** — اترك حقل التصفية فارغاً لعرض جميع النتائج
                - 📍 **غيّر الموقع** — استخدم مدينة قريبة من الاختصارات أعلاه
                - 🔄 **أعد التحميل** — حدّث الصفحة وجرب مرة أخرى
                """
            )

        st.markdown(
            """
            <div class="emergency-banner">
                📞 <b>في حالة طوارئ:</b> اتصل بالإسعاف —
                <b>15</b> (المغرب) ·
                <b>1400</b> (مصر) ·
                <b>115</b> (الجزائر) ·
                <b>17</b> (فرنسا)
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown('<div class="nearby-result-anchor">نتائج البحث القريبة</div>', unsafe_allow_html=True)
    st.success(f"تم العثور على {len(places)} منشأة")

    hospitals = sum(1 for place in places if "مستشفى" in place["type_label"])
    clinics = sum(1 for place in places if "عيادة" in place["type_label"])
    pharmacies = sum(1 for place in places if "صيدلية" in place["type_label"])
    nearest_m = places[0]["distance_m"] if places else 0

    render_stats_row(
        total=len(places),
        hospitals=hospitals,
        clinics=clinics,
        pharmacies=pharmacies,
        nearest_distance=_format_distance(nearest_m),
    )

    render_map_legend()

    tab_map, tab_list, tab_data, tab_insights = st.tabs(
        [
            "🗺️ الخريطة",
            "📋 القائمة",
            "📊 البيانات",
            "📈 تحليلات",
        ]
    )

    with tab_map:
        if FOLIUM_AVAILABLE:
            render_doctors_map(
                user_lat,
                user_lon,
                places,
                use_clustering=True,
                height=550,
                zoom_start=13,
            )
        else:
            st.markdown(
                """
                <div class="empty-state" style="padding:32px 20px;">
                    <div class="empty-state-icon">🗺️</div>
                    <div class="empty-state-title">الخريطة غير متاحة</div>
                    <div class="empty-state-text">
                        لعرض الخريطة التفاعلية، قم بتثبيت المكتبات المطلوبة:
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code("pip install folium streamlit-folium", language="bash")

    with tab_list:
        sort_by = st.selectbox(
            "📊 ترتيب حسب",
            ["المسافة (الأقرب أولاً)", "المسافة (الأبعد أولاً)", "الاسم", "النوع"],
            key="sort_by",
        )

        sorted_places = places.copy()
        if sort_by == "المسافة (الأقرب أولاً)":
            sorted_places.sort(key=lambda item: item["distance_m"])
        elif sort_by == "المسافة (الأبعد أولاً)":
            sorted_places.sort(key=lambda item: item["distance_m"], reverse=True)
        elif sort_by == "الاسم":
            sorted_places.sort(key=lambda item: item.get("name", ""))
        elif sort_by == "النوع":
            sorted_places.sort(key=lambda item: item.get("type_label", ""))

        max_results = min(MAX_RESULTS_DISPLAY, len(sorted_places))
        if max_results <= 5:
            display_count = max_results
            if max_results > 0:
                st.info(f"عرض {max_results} نتائج متاحة.")
        else:
            display_count = st.slider(
                "عدد النتائج المعروضة",
                5,
                max_results,
                min(10, max_results),
                key="top_n",
            )

        for idx, place in enumerate(sorted_places[:display_count], start=1):
            render_facility_card(idx, place)
            st.markdown("<br>", unsafe_allow_html=True)

    with tab_data:
        df_cols = [
            "name",
            "type_label",
            "specialty",
            "distance_str",
            "distance_m",
            "phone",
            "address",
            "opening_hours",
            "lat",
            "lon",
        ]

        df = pd.DataFrame(places)[df_cols].rename(
            columns={
                "name": "الاسم",
                "type_label": "النوع",
                "specialty": "التخصص",
                "distance_str": "المسافة",
                "distance_m": "المسافة (متر)",
                "phone": "الهاتف",
                "address": "العنوان",
                "opening_hours": "ساعات العمل",
                "lat": "خط العرض",
                "lon": "خط الطول",
            }
        )

        df["المسافة (كم)"] = (df["المسافة (متر)"] / 1000).round(2)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "المسافة (كم)": st.column_config.NumberColumn(format="%.2f كم"),
                "المسافة (متر)": st.column_config.NumberColumn(format="%.0f م"),
            },
        )

        col_dl_1, col_dl_2 = st.columns(2)
        with col_dl_1:
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ تحميل CSV",
                data=csv,
                file_name=(
                    f"shifa_nearby_{user_lat:.4f}_{user_lon:.4f}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                ),
                mime="text/csv",
                key="download_csv",
                use_container_width=True,
            )

        with col_dl_2:
            try:
                import io

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="المنشآت الصحية", index=False)
                excel_data = buffer.getvalue()
                st.download_button(
                    label="📊 تحميل Excel",
                    data=excel_data,
                    file_name=f"shifa_nearby_{user_lat:.4f}_{user_lon:.4f}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel",
                    use_container_width=True,
                )
            except ImportError:
                st.info("💡 قم بتثبيت `openpyxl` لتفعيل تصدير Excel")

    with tab_insights:
        st.subheader("📈 تحليلات متقدمة")

        if len(places) > 1:
            distances_km = [place["distance_m"] / 1000 for place in places]
            col_ch_1, col_ch_2 = st.columns(2)

            with col_ch_1:
                st.markdown("#### توزيع المسافات")
                bins = [0, 1, 2, 3, 5, 7, 10, 15, 20]
                hist_data = np.histogram(distances_km, bins=bins)

                try:
                    import plotly.graph_objects as go

                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=[f"{bins[i]}-{bins[i + 1]} كم" for i in range(len(bins) - 1)],
                                y=hist_data[0],
                                marker_color="#3B82F6",
                                marker_line_width=0,
                            )
                        ]
                    )
                    fig.update_layout(
                        title="توزيع المنشآت حسب المسافة",
                        xaxis_title="المسافة",
                        yaxis_title="عدد المنشآت",
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94A3B8"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("💡 قم بتثبيت `plotly` لعرض الرسوم البيانية")

            with col_ch_2:
                st.markdown("#### إحصائيات سريعة")
                col_s_1, col_s_2 = st.columns(2)
                with col_s_1:
                    st.metric("📏 أقرب منشأة", f"{min(distances_km):.2f} كم")
                    st.metric("📐 أبعد منشأة", f"{max(distances_km):.2f} كم")
                with col_s_2:
                    st.metric("📊 متوسط المسافة", f"{np.mean(distances_km):.2f} كم")
                    st.metric("📉 الانحراف المعياري", f"{np.std(distances_km):.2f} كم")

        st.markdown("#### توزيع حسب النوع")
        type_counts = pd.Series([place["type_label"] for place in places]).value_counts()

        try:
            import plotly.graph_objects as go

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=type_counts.index.tolist(),
                        values=type_counts.values.tolist(),
                        hole=0.4,
                        marker_colors=["#EF4444", "#10B981", "#3B82F6", "#8B5CF6", "#F59E0B"],
                    )
                ]
            )
            fig_pie.update_layout(
                title="نسبة كل نوع من المنشآت الصحية",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        except ImportError:
            st.info("💡 قم بتثبيت `plotly` لعرض الرسوم البيانية")

        st.info(f"⚡ البيانات مخزنة مؤقتاً لمدة {CACHE_DURATION // 60} دقيقة لتقليل طلبات API")


if do_search and places:
    st.markdown(
        f"""
        <div class="footer-bar">
            <div class="footer-item">📍 الموقع: {user_lat:.4f}, {user_lon:.4f}</div>
            <div class="footer-item">🔍 نطاق البحث: {radius_km} كم</div>
            <div class="footer-item">🏥 عدد النتائج: {len(places)} منشأة</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
