import streamlit as st
import sys
import os

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.nlp.biomistral import BioMistralChatbot

st.set_page_config(page_title="المحادثة الطبية", page_icon="💬", layout="wide")

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

st.markdown('<h1 class="main-title">المساعد الطبي (BioMistral)</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#5A6072;">نموذج <b>BioMistral-7B</b> مخصص للقطاع الطبي للإجابة بذكاء على الاستشارات الطبية بشكل تحليلي.</p>', unsafe_allow_html=True)

st.markdown('<div class="data-card">', unsafe_allow_html=True)

@st.cache_resource
def get_biomistral_chatbot():
    # ModelManager will automatically evict other models (X-Ray, Skin) 
    # from VRAM to make room for this 7B model!
    return BioMistralChatbot()

try:
    with st.spinner("جاري تهيئة نموذج BioMistral (يتطلب BitsAndBytes 8-bit)..."):
        biomistral = get_biomistral_chatbot()
        if biomistral.model is None:
            raise Exception("Failed to load model weights.")
    st.success("المساعد الذكي جاهز — BioMistral/BioMistral-7B (8-bit Quantized)")
except Exception as e:
    st.error("**نموذج المحادثة غير متاح أو تنقصه المكتبات**")
    st.warning("يتطلب هذا النموذج تثبيت bitsandbytes و accelerate بنجاح.")
    st.info(f"الخطأ: {e}")
    st.stop()

st.markdown("---")

# Session state initialization
if "biomistral_messages" not in st.session_state:
    st.session_state.biomistral_messages = []
    
for msg in st.session_state.biomistral_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
if prompt := st.chat_input("تحدث مع المساعد الطبي عن أي استشارة أو دواء..."):
    st.session_state.biomistral_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("جاري صياغة الإجابة..."):
            try:
                # Add basic context context if available from standard RAG (skipped for pure LLM test)
                context = ""
                answer = biomistral.generate_answer(prompt, context=context)
                
                response_placeholder.markdown(answer)
                st.session_state.biomistral_messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                response_placeholder.error(f"حدث خطأ أثناء معالجة السؤال: {e}")

st.markdown('</div>', unsafe_allow_html=True)
