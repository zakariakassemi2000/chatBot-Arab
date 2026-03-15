import streamlit as st
import sys
import os
import base64
from PIL import Image as PILImage

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.llm import GroqVision

st.set_page_config(page_title="المساعد البصري المتطور", page_icon="👁️", layout="wide")

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

st.markdown('<h1 class="main-title">المساعد البصري التفاعلي (Groq Vision)</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#5A6072;">نموذج ذكاء اصطناعي فائق السرعة <b>Llama-3.2 Vision</b> قادر على التفكير المنطقي حول الصور الطبية.</p>', unsafe_allow_html=True)

st.markdown('<div class="data-card">', unsafe_allow_html=True)

@st.cache_resource
def get_vision_analyzer():
    return GroqVision()

vision_analyzer = get_vision_analyzer()
if not vision_analyzer.client:
    st.error("❌ مفتاح `GROQ_API_KEY` مفقود أو غير صالح. يرجى إضافته في الإعدادات أو ملف `.env`.")
    st.stop()
else:
    st.success("✅ وحدة التحليل البصري جاهزة — Llama-3.2-90b-Vision")

st.markdown("---")

col_img, col_chat = st.columns([1, 2])

with col_img:
    uploaded_img = st.file_uploader(
        "رفع صورة (أشعة، رنين، تقرير، إلخ...)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_img:
        st.image(uploaded_img, caption="الصورة المرفوعة", use_container_width=True)

with col_chat:
    st.markdown("#### اسأل المساعد عن الصورة")
    if "vision_history" not in st.session_state:
        st.session_state.vision_history = []
        
    for msg in st.session_state.vision_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("ما هو التشخيص المحتمل أو ماذا ترى في هذه الصورة؟"):
        if not uploaded_img:
            st.warning("⚠️ يرجى تحميل صورة أولاً ليبدأ المساعد في تحليلها.")
            st.stop()
            
        st.session_state.vision_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("جاري التحليل واستنتاج الإجابة بسرعة فائقة..."):
                try:
                    # Convert uploaded image to base64
                    bytes_data = uploaded_img.getvalue()
                    base64_img = base64.b64encode(bytes_data).decode('utf-8')
                    
                    # Ensure the AI responds in Arabic and acts as a medical assistant
                    system_prompt = f"أنت مساعد طبي ذكي. تأمل الصورة بعناية وأجب باللغة العربية الفصحى على هذا السؤال: {prompt}"
                    
                    # Analyze text and image
                    answer = vision_analyzer.analyze_image(base64_image=base64_img, prompt=system_prompt)
                    
                    response_placeholder.markdown(answer)
                    st.session_state.vision_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    response_placeholder.error(f"حدث خطأ أثناء معالجة الصورة أو الاتصال بخادم Groq: {e}")

st.markdown('</div>', unsafe_allow_html=True)

