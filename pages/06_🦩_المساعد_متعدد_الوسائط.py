import streamlit as st
import sys
import os
from PIL import Image as PILImage

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.multimodal.med_flamingo import MedFlamingoAnalyzer

st.set_page_config(page_title="المساعد الفلامنجو (Med-Flamingo)", page_icon="🦩", layout="wide")

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

st.markdown('<h1 class="main-title">المساعد متعدد الوسائط (Med-Flamingo)</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#5A6072;">نموذج <b>9 مليار بارامتر</b> مدرب خصيصاً للتفكير المنطقي حول الصور الطبية (Medical VQA).</p>', unsafe_allow_html=True)

st.markdown('<div class="data-card">', unsafe_allow_html=True)

@st.cache_resource
def get_flamingo_analyzer():
    # ModelManager will automatically evict other models (X-Ray, Skin) 
    # from VRAM to make room for this massive 9B model!
    return MedFlamingoAnalyzer()

try:
    with st.spinner("جاري تهيئة نموذج Med-Flamingo (هذا قد يستغرق وقتاً طويلاً)..."):
        flamingo_detector = get_flamingo_analyzer()
    st.success("وحدة التحليل جاهزة — med-flamingo/med-flamingo-9b")
except Exception as e:
    st.error("**نموذج فلامنجو غير متاح أو تنقصه المكتبات**")
    st.warning("يتطلب هذا النموذج تثبيت open-flamingo و bitsandbytes. راجع السجلات للتفاصيل.")
    st.info(f"الخطأ: {e}")
    st.stop()

st.markdown("---")

col_img, col_chat = st.columns([1, 2])

with col_img:
    uploaded_img = st.file_uploader(
        "رفع صورة طبية (أشعة، رنين، جلدية...)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_img:
        st.image(uploaded_img, caption="الصورة المرفوعة", use_container_width=True)

with col_chat:
    st.markdown("#### اسأل المساعد عن الصورة")
    if "flamingo_history" not in st.session_state:
        st.session_state.flamingo_history = []
        
    for msg in st.session_state.flamingo_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("ما هو التشخيص المحتمل لهذه الصورة؟"):
        if not uploaded_img:
            st.warning("يرجى تحميل صورة أولاً ليبدأ المساعد في تحليلها.")
            st.stop()
            
        st.session_state.flamingo_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("جاري التحليل واستنتاج الإجابة..."):
                img_pil = PILImage.open(uploaded_img)
                # Format specific to Flamingo: it requires <image> tags matching the input images
                flamingo_prompt = f"<image>Question: {prompt} Answer:"
                
                try:
                    # Analyze text and image
                    answer = flamingo_detector.analyze([img_pil], flamingo_prompt)
                    
                    # Clean up Flamingo's raw output which includes the prompt
                    answer_clean = answer.split("Answer:")[-1].strip()
                    if "<|endofchunk|>" in answer_clean:
                        answer_clean = answer_clean.split("<|endofchunk|>")[0]
                        
                    response_placeholder.markdown(answer_clean)
                    st.session_state.flamingo_history.append({"role": "assistant", "content": answer_clean})
                    
                except Exception as e:
                    response_placeholder.error(f"حدث خطأ أثناء معالجة السؤال: {e}")

st.markdown('</div>', unsafe_allow_html=True)
