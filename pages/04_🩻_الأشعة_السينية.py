import streamlit as st
import sys
import os
from PIL import Image as PILImage

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.imaging.xray import ChestXrayAnalyzer
from utils.image_validator import validate_medical_image

st.set_page_config(page_title="الأشعة السينية", page_icon="🩻", layout="wide")

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

st.markdown('<h1 class="main-title">تحليل الأشعة السينية (X-Ray)</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#5A6072;">نموذج <b>StanfordAIMI/CheXNet</b> لتصنيف 14 مرضاً صدرياً بصيغة متعددة التسميات (Multi-label).</p>', unsafe_allow_html=True)

st.markdown('<div class="data-card">', unsafe_allow_html=True)

@st.cache_resource
def get_xray_analyzer():
    # Will use the ModelManager internally to offload inactive models from VRAM
    return ChestXrayAnalyzer()

try:
    with st.spinner("جاري تهيئة نموذج CheXNet..."):
        analyzer = get_xray_analyzer()
    st.success("وحدة التحليل جاهزة — StanfordAIMI/CheXNet 14-Disease Model")
except Exception as e:
    st.error("**نموذج الأشعة السينية غير متاح**")
    st.warning(f"تنبيه تقني: {e}")
    st.info("النموذج يتم تحميله تلقائياً من Hugging Face عند أول استخدام.")
    st.stop()

st.markdown("---")

# ── Conditions Info ──
st.markdown("""
<div style="background:linear-gradient(135deg, #f8f9fa, #f0f8ff); border-radius:14px; padding:18px; margin-bottom:16px; border:1px solid #e0e8f0;">
    <h4 style="margin:0 0 10px; color:#1E2028;">أبرز الحالات المكتشفة</h4>
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:8px;">
        <div style="background:#ffebee; padding:10px 14px; border-radius:10px;">
            <b style="color:#c62828;">🔴 التهاب رئوي (Pneumonia)</b>
        </div>
        <div style="background:#fce4ec; padding:10px 14px; border-radius:10px;">
            <b style="color:#ad1457;">🫀 تضخم القلب (Cardiomegaly)</b>
        </div>
        <div style="background:#e3f2fd; padding:10px 14px; border-radius:10px;">
            <b style="color:#1565c0;">💧 وذمة رئوية (Edema)</b>
        </div>
        <div style="background:#e2e8f0; padding:10px 14px; border-radius:10px;">
            <b style="color:#475569;">🩺 استرواح الصدر (Pneumothorax)</b>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

disclaimer_agreed = st.checkbox(
    "أوافق على أن هذه الأداة تجريبية للأغراض التعليمية فقط ولا تعطي تشخيصاً طبياً نهائياً.",
    key="xray_disclaimer"
)

if not disclaimer_agreed:
    st.info("يرجى الموافقة على الشروط أعلاه لتفعيل رفع صور الأشعة.")
else:
    uploaded_img = st.file_uploader(
        "تحميل صورة الأشعة السينية (Chest X-Ray)",
        type=["jpg", "jpeg", "png"],
        key="xray_uploader"
    )

    if st.button("تحليل الأشعة السينية", use_container_width=True, type="primary"):
        if not uploaded_img:
            st.warning("يرجى تحميل صورة أشعة سينية أولاً.")
        else:
            with st.spinner("جاري تحليل صورة الأشعة السينية..."):
                img_pil = PILImage.open(uploaded_img)
                
                # Validation
                validation = validate_medical_image(img_pil, expected_type="xray")
                if not validation["valid"]:
                    st.error(validation["reason"])
                    if validation.get("medical_score"):
                        st.caption(f"درجة الثقة الطبية: {validation['medical_score']*100:.0f}%")
                    st.stop()
                elif not validation.get("type_match", True):
                    st.warning(validation["reason"])

                # Run Inference
                try:
                    result = analyzer.analyze(img_pil)
                except Exception as e:
                    st.error(f"حدث خطأ أثناء التحليل: {e}")
                    st.stop()

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(uploaded_img, caption="صورة الأشعة المُحلَّلة", use_container_width=True)

            with col2:
                risk_level = result["highest_risk"]
                
                if risk_level == "high":
                    style, label, exp = "#DC3545", "مؤشرات مرضية عالية", "يُرجى مراجعة أخصائي أمراض صدرية فوراً."
                elif risk_level == "moderate":
                    style, label, exp = "#FF9800", "مؤشرات مرضية متوسطة", "يُنصح بمراجعة طبيب لتقييم الحالة."
                else:
                    style, label, exp = "#28A745", "سليم (غالباً)", "لم تُرصد مؤشرات مرضية عالية. يُنصح بالمتابعة الدورية."
                    
                st.markdown(f"""
                <div style="text-align:center; padding:20px; border-radius:16px;
                            background:{style}11; border:2px solid {style}; margin-top:10px;">
                    <h3 style="color:{style}; margin:0;">{label}</h3>
                    <p style="color:#1E2028; margin-top:10px; font-size:15px;">{exp}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### توزيع احتمالات الـ 14 فئة (اعلى الاحتمالات):")
                
                # Sort the 14 diseases by probability
                sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
                
                for disease, prob in sorted_probs[:6]: # Show top 6
                    st.progress(prob, text=f"{disease}: {prob*100:.1f}%")

st.markdown('</div>', unsafe_allow_html=True)
