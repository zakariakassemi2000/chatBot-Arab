import streamlit as st
import sys
import os
from PIL import Image as PILImage

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.derm_detector import DermDetector
from utils.image_validator import validate_medical_image

st.set_page_config(page_title="الأمراض الجلدية", page_icon="🩹", layout="wide")

st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #1E2028;
    margin-bottom: 5px;
    font-family: 'Inter', sans-serif;
}
.data-card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    border: 1px solid #f0f2f6;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">فاحص الأمراض الجلدية</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#5A6072;">نموذج <b>DinoV2</b> لتصنيف 31 حالة جلدية (ISIC 2018 + Atlas Dermatology) — دقة 95.57%. أداة تعليمية فقط، لا تُغني عن طبيب الجلد.</p>', unsafe_allow_html=True)

st.markdown('<div class="data-card">', unsafe_allow_html=True)

@st.cache_resource
def get_derm_analyzer():
    # Will use the model_manager internally to offload when idle
    return DermDetector()

try:
    with st.spinner("جاري تهيئة نموذج DinoV2..."):
        derm_detector = get_derm_analyzer()
        if derm_detector.model is None:
            raise Exception(derm_detector.load_error)
    st.success("وحدة التحليل جاهزة — DinoV2 Skin Disease (31 فئة)")
except Exception as e:
    st.error("**نموذج الجلد غير متاح**")
    st.warning(f"تنبيه تقني: {e}")
    st.info("النموذج يتم تحميله تلقائياً من Hugging Face عند أول استخدام.")
    st.stop()

st.markdown("---")
st.markdown("""
<div style="background:linear-gradient(135deg, #f8f9fa, #fff8f0); border-radius:14px; padding:18px; margin-bottom:16px; border:1px solid #f0e8e0;">
    <h4 style="margin:0 0 10px; color:#1E2028;">الحالات المكتشفة (31 فئة)</h4>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
        <div style="background:#ffebee; padding:10px 14px; border-radius:10px;"><b style="color:#c62828;">خطورة عالية</b><br><small>الميلانوما، BCC، SCC، الجذام، الذئبة، الفطار الفطراني</small></div>
        <div style="background:#fff3e0; padding:10px 14px; border-radius:10px;"><b style="color:#e65100;">خطورة متوسطة</b><br><small>الصدفية، الهربس، الحزاز المسطح، القوباء، التقرن السفعي</small></div>
        <div style="background:#e8f5e9; padding:10px 14px; border-radius:10px;"><b style="color:#2e7d32;">خطورة منخفضة</b><br><small>الشامات، التقرن الدهني، النخالية الوردية، آفات وعائية</small></div>
        <div style="background:#e3f2fd; padding:10px 14px; border-radius:10px;"><b style="color:#1565c0;">31 فئة إجمالية</b><br><small>ISIC 2018 + Atlas Dermatology | DinoV2 | 95.57%</small></div>
    </div>
</div>
""", unsafe_allow_html=True)

disclaimer_agreed = st.checkbox(
    "أوافق على أن هذه الأداة تجريبية للأغراض التعليمية فقط ولا تعطي تشخيصاً طبياً نهائياً.",
    key="derm_disclaimer"
)

if not disclaimer_agreed:
    st.info("يرجى الموافقة على الشروط أعلاه لتفعيل رفع صور الجلد.")
else:
    uploaded_img = st.file_uploader(
        "تحميل صورة لِسْيونة جلدية",
        type=["jpg", "jpeg", "png"],
        key="derm_uploader"
    )

    if st.button("تحليل لِسْيونة الجلد", use_container_width=True, type="primary"):
        if not uploaded_img:
            st.warning("يرجى تحميل صورة أولاً.")
        else:
            with st.spinner("جاري تحليل الصورة..."):
                img_pil = PILImage.open(uploaded_img)
                validation = validate_medical_image(img_pil, expected_type="derm")
                if not validation.get("valid", True):
                    st.warning(validation.get("reason", "تحقق من صورة الجلد."))
                
                prediction = derm_detector.predict_image(img_pil)
                label, explanation, risk_level, style = derm_detector.interpret_result(prediction)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(uploaded_img, caption="صورة الجلد المُحلَّلة", use_container_width=True)

            with col2:
                if prediction:
                    color_map = {"success": "#28A745", "warning": "#FF9800", "danger": "#DC3545", "error": "#6C757D"}
                    color = color_map.get(style, "#6C757D")
                    st.markdown(f"""
                    <div style="text-align:center; padding:20px; border-radius:16px;
                                background:{color}11; border:2px solid {color}; margin-top:10px;">
                        <h3 style="color:{color}; margin:0;">{label}</h3>
                        <p style="color:#1E2028; margin-top:10px; font-size:15px;">{explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("#### أعلى 5 احتمالات")
                    top5 = prediction.get("top5", [])
                    for item in top5:
                        p = item["probability"]
                        st.progress(p, text=f"{item['class_label_ar']}: {p*100:.1f}%")
                    if risk_level == "high":
                        st.error("يُرجى مراجعة طبيب جلدية فوراً.")
                    elif risk_level == "moderate":
                        st.warning("يُنصح بمراجعة طبيب جلدية.")
                    else:
                        st.success("يُنصح بالفحص الدوري عند طبيب الجلد.")
                else:
                    st.error("تعذّر تحليل الصورة.")
