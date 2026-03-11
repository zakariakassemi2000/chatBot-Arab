import streamlit as st
import sys
import os
from PIL import Image as PILImage

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.imaging.mri import BrainTumorAnalyzer
from utils.image_validator import validate_medical_image

st.set_page_config(page_title="تصوير الدماغ", page_icon="🧠", layout="wide")

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

st.markdown('<h1 class="main-title">تحليل أورام الدماغ (MRI)</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#5A6072;">نموذج <b>Vision Transformer (ViT)</b> لتصنيف أورام الدماغ من صور الرنين المغناطيسي.</p>', unsafe_allow_html=True)

st.markdown('<div class="data-card">', unsafe_allow_html=True)

@st.cache_resource
def get_mri_analyzer():
    # Will use ModelManager internally to offload when idle
    return BrainTumorAnalyzer()

try:
    with st.spinner("جاري تهيئة نموذج ViT Brain MRI..."):
        mri_detector = get_mri_analyzer()
    st.success("وحدة التحليل جاهزة — ahmedhamdy/brain-tumor-vit")
except Exception as e:
    st.error("**نموذج تصوير الدماغ غير متاح**")
    st.warning(f"تنبيه تقني: {e}")
    st.info("النموذج يتم تحميله تلقائياً من Hugging Face عند أول استخدام.")
    st.stop()

st.markdown("---")

# ── Conditions Info ──
st.markdown("""
<div style="background:linear-gradient(135deg, #f8f9fa, #f0f8ff); border-radius:14px; padding:18px; margin-bottom:16px; border:1px solid #e0e8f0;">
    <h4 style="margin:0 0 10px; color:#1E2028;">الحالات المكتشفة</h4>
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:8px;">
        <div style="background:#ffebee; padding:10px 14px; border-radius:10px;">
            <b style="color:#c62828;">🔴 ورم دبقي (Glioma)</b>
        </div>
        <div style="background:#fce4ec; padding:10px 14px; border-radius:10px;">
            <b style="color:#ad1457;">🟣 ورم سحائي (Meningioma)</b>
        </div>
        <div style="background:#e3f2fd; padding:10px 14px; border-radius:10px;">
            <b style="color:#1565c0;">🔵 ورم الغدة النخامية (Pituitary)</b>
        </div>
        <div style="background:#e8f5e9; padding:10px 14px; border-radius:10px;">
            <b style="color:#2e7d32;">🟢 سليم (No Tumor)</b>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

disclaimer_agreed = st.checkbox(
    "أوافق على أن هذه الأداة تجريبية للأغراض التعليمية فقط ولا تعطي تشخيصاً طبياً نهائياً.",
    key="mri_disclaimer"
)

if not disclaimer_agreed:
    st.info("يرجى الموافقة على الشروط أعلاه لتفعيل رفع صور الدماغ.")
else:
    uploaded_img = st.file_uploader(
        "تحميل صورة الرنين المغناطيسي للدماغ (Brain MRI)",
        type=["jpg", "jpeg", "png"],
        key="mri_uploader"
    )

    if st.button("تحليل الرنين المغناطيسي", use_container_width=True, type="primary"):
        if not uploaded_img:
            st.warning("يرجى تحميل صورة أولاً.")
        else:
            with st.spinner("جاري تحليل الصورة..."):
                img_pil = PILImage.open(uploaded_img)
                validation = validate_medical_image(img_pil, expected_type="mri")
                
                if not validation.get("valid", True):
                    st.warning(validation.get("reason", "تحقق من صورة الدماغ."))
                
                try:
                    result = mri_detector.analyze(img_pil)
                except Exception as e:
                    st.error(f"حدث خطأ أثناء التحليل: {e}")
                    st.stop()

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(uploaded_img, caption="صورة الدماغ المُحلَّلة", use_container_width=True)

            with col2:
                risk_level = result["risk_level"]
                label_en = result["top_prediction"]
                
                # Simple translation dict
                ar_map = {
                    "glioma_tumor": "ورم دبقي (Glioma Tumors)",
                    "meningioma_tumor": "ورم سحائي (Meningioma Tumors)",
                    "pituitary_tumor": "ورم الغدة النخامية (Pituitary Tumors)",
                    "no_tumor": "لا يوجد ورم (No Tumor)"
                }
                label_ar = ar_map.get(label_en.lower(), label_en)

                if risk_level == "high":
                    style, exp = "#DC3545", "يُرجى مراجعة جراح مخ وأعصاب أو طبيب أورام فوراً."
                elif risk_level == "moderate":
                    style, exp = "#FF9800", "يُنصح بمراجعة طبيب لتقييم الحالة."
                else:
                    style, exp = "#28A745", "لم تُرصد مؤشرات لأورام. يُنصح بالمتابعة الدورية."

                st.markdown(f"""
                <div style="text-align:center; padding:20px; border-radius:16px;
                            background:{style}11; border:2px solid {style}; margin-top:10px;">
                    <h3 style="color:{style}; margin:0;">{label_ar}</h3>
                    <p style="color:#1E2028; margin-top:10px; font-size:15px;">{exp}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### توزيع الاحتمالات")
                for item in result["predictions"]:
                    p = item["probability"]
                    text = ar_map.get(item["tumor_type"].lower(), item["tumor_type"])
                    st.progress(p, text=f"{text}: {p*100:.1f}%")

st.markdown('</div>', unsafe_allow_html=True)
