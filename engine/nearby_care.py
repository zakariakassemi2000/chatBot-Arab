import requests
import streamlit as st

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def get_nearby_hospitals(lat: float, lng: float, 
                          radius: int = 5000) -> list:
    """
    OpenStreetMap Overpass API — 100% gratuit
    Cherche hôpitaux + cliniques + médecins
    """
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"]
        (around:{radius},{lat},{lng});
      node["amenity"="clinic"]
        (around:{radius},{lat},{lng});
      node["amenity"="doctors"]
        (around:{radius},{lat},{lng});
      way["amenity"="hospital"]
        (around:{radius},{lat},{lng});
    );
    out body center;
    """
    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=30
        )
        data = response.json()
        
        results = []
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            
            # Coordonnées
            if el["type"] == "way":
                lat_p = el.get("center", {}).get("lat", lat)
                lng_p = el.get("center", {}).get("lon", lng)
            else:
                lat_p = el.get("lat", lat)
                lng_p = el.get("lon", lng)
            
            name = (
                tags.get("name:ar") or
                tags.get("name") or
                "مركز صحي"
            )
            
            amenity_label = {
                "hospital" : "🏥 مستشفى",
                "clinic"   : "🏨 عيادة",
                "doctors"  : "👨⚕️ طبيب"
            }.get(tags.get("amenity",""), "🏥")
            
            phone = tags.get("phone") or tags.get("contact:phone", "")
            
            maps_url = (
                f"https://www.openstreetmap.org/?mlat={lat_p}"
                f"&mlon={lng_p}#map=17/{lat_p}/{lng_p}"
            )
            
            gmaps_url = (
                f"https://www.google.com/maps/search/"
                f"?api=1&query={lat_p},{lng_p}"
            )
            
            results.append({
                "name"      : name,
                "type"      : amenity_label,
                "address"   : tags.get("addr:street", ""),
                "phone"     : phone,
                "lat"       : lat_p,
                "lng"       : lng_p,
                "osm_url"   : maps_url,
                "gmaps_url" : gmaps_url
            })
        
        return results[:6]
    
    except Exception as e:
        return []


def render_nearby_care(severity: str):
    if severity not in ["critique", "élevée"]:
        return
    
    st.markdown("""
    <div style="
      background:rgba(45,125,210,0.08);
      border:1px solid rgba(45,125,210,0.25);
      border-radius:12px;
      padding:16px 20px;
      margin:16px 0;
      direction:rtl;
    ">
      <div style="font-size:1rem;font-weight:600;
                  color:#93C5FD;margin-bottom:4px">
        🏥 أقرب مراكز الرعاية الصحية
      </div>
      <div style="font-size:0.8rem;color:#64748B">
        بيانات OpenStreetMap · مجانية ومفتوحة
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input(
            "خط العرض (Latitude)",
            value=33.9716,
            format="%.4f",
            key="user_lat"
        )
    with col2:
        lng = st.number_input(
            "خط الطول (Longitude)",
            value=-6.8498,
            format="%.4f",
            key="user_lng"
        )
    
    radius = st.select_slider(
        "نطاق البحث",
        options=[1,2,3,5,10,15,20],
        value=5,
        format_func=lambda x: f"{x} كم",
        key="search_radius"
    )
    
    if st.button("🔍 ابحث عن أقرب مستشفى", key="search_hospitals"):
        with st.spinner("جاري البحث في OpenStreetMap..."):
            places = get_nearby_hospitals(lat, lng, radius * 1000)
        
        if not places:
            st.warning("لم يتم العثور على مراكز صحية في هذا النطاق")
            st.error("📞 اتصل بـ SAMU مباشرة : **15**")
            return
        
        st.success(f"تم العثور على {len(places)} مرافق صحية")
        
        for i, place in enumerate(places, 1):
            phone_html = (
                f'<a href="tel:{place["phone"]}" style="color:#00C9A7">'
                f'📞 {place["phone"]}</a>'
            ) if place["phone"] else ""
            
            st.markdown(f"""
            <div style="
              background:rgba(255,255,255,0.03);
              border:1px solid rgba(255,255,255,0.07);
              border-radius:10px;
              padding:14px 16px;
              margin:6px 0;
              direction:rtl;
            ">
              <div style="display:flex;justify-content:space-between;
                          align-items:flex-start;gap:12px">
                <div style="flex:1">
                  <div style="font-weight:600;color:#F1F5F9;margin-bottom:3px">
                    {place['type']} {i}. {place['name']}
                  </div>
                  <div style="font-size:0.82rem;color:#94A3B8">
                    {place['address']}
                  </div>
                  <div style="margin-top:4px">{phone_html}</div>
                </div>
                <div style="display:flex;flex-direction:column;gap:6px">
                  <a href="{place['gmaps_url']}" target="_blank" style="
                    background:rgba(0,201,167,0.12);
                    border:1px solid rgba(0,201,167,0.25);
                    color:#00C9A7;padding:5px 12px;
                    border-radius:7px;font-size:0.78rem;
                    text-decoration:none;text-align:center;
                    white-space:nowrap;
                  ">🗺️ Google Maps</a>
                  <a href="{place['osm_url']}" target="_blank" style="
                    background:rgba(255,255,255,0.05);
                    border:1px solid rgba(255,255,255,0.1);
                    color:#94A3B8;padding:5px 12px;
                    border-radius:7px;font-size:0.78rem;
                    text-decoration:none;text-align:center;
                    white-space:nowrap;
                  ">🌍 OSM</a>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
