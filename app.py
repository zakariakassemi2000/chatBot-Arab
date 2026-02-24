# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Enhanced Medical Platform
═══════════════════════════════════════════════════════════════════════
"""

import os, sys, io, base64, time, json
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# 1️⃣ GLOBAL SAFETY & CONFIG
# ─────────────────────────────────────────────────────────────
st.set_option('client.showErrorDetails', False)
load_dotenv()

# Windows UTF8 Fix
if sys.platform == 'win32' and not os.environ.get('_UTF8_FIX_APPLIED'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        os.environ['_UTF8_FIX_APPLIED'] = '1'
    except:
        pass

# ─────────────────────────────────────────────────────────────
# 2️⃣ IMPORT ENGINES
# ─────────────────────────────────────────────────────────────
from engine.retriever import FAISSRetriever
from engine.classifier import IntentClassifier, format_response
from engine.safety import SafetyGuard
from engine.llm import GroqGenerator
from engine.audio import speech_to_text_arabic, convert_audio_to_wav

# ─────────────────────────────────────────────────────────────
# 3️⃣ CONSTANTS & UTILS
# ─────────────────────────────────────────────────────────────
LOGO_PATH = r"c:\Users\zakar\chatBot Arab\Stylized_Heart_and_Cross_Logo_for_SHIFA_AI__1_-removebg-preview.png"
HISTORY_FILE = "consultation_history.json"

def get_b64(path):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return None

LOGO_B64 = get_b64(LOGO_PATH)
LOGO_SRC = f"data:image/png;base64,{LOGO_B64}" if LOGO_B64 else ""

def get_logo():
    if os.path.exists(LOGO_PATH):
        return LOGO_PATH
    return "🩺"

def save_history(messages):
    if not messages: return
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except: pass
    
    # Check if last session is already saved
    session_id = str(st.session_state.get("session_id", time.time()))
    existing = next((idx for idx, s in enumerate(history) if s.get("id") == session_id), None)
    
    entry = {
        "id": session_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": messages[0]["content"][:50] + "...",
        "messages": messages
    }
    
    if existing is not None:
        history[existing] = entry
    else:
        history.insert(0, entry)
        
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[:50], f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────
# 4️⃣ PAGE CONFIG & UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHIFA AI | المنصة الطبية الشاملة",
    page_icon=get_logo(),
    layout="centered",
)

# ── ADVANCED UI CSS ──
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&display=swap');
    
    html, body, [class*="st-"] {{
        font-family: 'Cairo', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }}
    
    .stApp {{
        background-color: #FFFFFF !important;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: #F8F9FA !important;
        border-left: 1px solid #E9ECEF !important;
    }}
    
    [data-testid="stSidebarNav"] {{
        display: none !important;
    }}

    .nav-item {{
        padding: 12px 20px;
        margin: 8px 0;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s;
        display: flex;
        align-items: center;
        gap: 15px;
        color: #495057;
        font-weight: 600;
        text-decoration: none;
    }}
    
    .nav-item:hover {{
        background-color: #FEEEEE;
        color: #E53935;
    }}
    
    .nav-item.active {{
        background-color: #E53935;
        color: white;
    }}

    /* Global Tweaks */
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
        max-width: 850px !important;
    }}
    
    .main-title {{
        color: #E53935;
        font-size: 38px;
        font-weight: 900;
        margin: 0;
        text-transform: uppercase;
    }}
    
    .welcome-card, .data-card {{
        background: white;
        border-radius: 20px;
        padding: 25px;
        border: 1px solid #F1F3F5;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        margin-bottom: 20px;
    }}

    /* Chat Styling */
    [data-testid="stChatMessage"] {{
        animation: fadeInUp 0.4s ease-out;
    }}
    
    [data-testid="stChatMessageAvatarContainer"] img, 
    [data-testid="stChatMessageAvatarContainer"] div {{
        border: 2px solid #E53935 !important;
    }}

    [data-testid="stChatMessage"]:not(:has([data-testid="chatAvatarIcon-user"])) div[data-testid="stChatMessageContent"] {{
        background: #FFFFFF !important;
        border-right: 4px solid #E53935 !important;
        border-radius: 4px 20px 20px 20px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        color: #1E2028 !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"] {{
        background: #FEEEEE !important;
        border-right: 4px solid #C62828 !important;
        border-radius: 20px 4px 20px 20px !important;
        color: #1A202C !important;
    }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 5️⃣ LOAD SYSTEM
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    try:
        r = FAISSRetriever()
        c = IntentClassifier()
        g = SafetyGuard()
        ai = GroqGenerator()
        db_ok = r.load()
        if db_ok: c.load()
        return r, c, g, ai, db_ok
    except:
        return None, None, None, None, False

retriever, classifier, guard, ai_engine, DB_STATUS = load_system()
AI_STATUS = bool(ai_engine and ai_engine.api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "page" not in st.session_state:
    st.session_state.page = "chat"
if "session_id" not in st.session_state:
    st.session_state.session_id = str(time.time())

# ─────────────────────────────────────────────────────────────
# 6️⃣ SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <img src="{LOGO_SRC}" style="width: 80px; margin-bottom: 10px;">
            <h3 style="color: #E53935; margin: 0;">SHIFA AI</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("💬 المحادثة الطبية", use_container_width=True, type="primary" if st.session_state.page == "chat" else "secondary"):
        st.session_state.page = "chat"
        st.rerun()
        
    if st.button("🔍 فاحص الأعراض", use_container_width=True, type="primary" if st.session_state.page == "scanner" else "secondary"):
        st.session_state.page = "scanner"
        st.rerun()
        
    if st.button("📜 سجل الاستشارات", use_container_width=True, type="primary" if st.session_state.page == "history" else "secondary"):
        st.session_state.page = "history"
        st.rerun()
        
    if st.button("📚 قاعدة المعرفة", use_container_width=True, type="primary" if st.session_state.page == "database" else "secondary"):
        st.session_state.page = "database"
        st.rerun()

    st.markdown("---")
    st.info("⚠️ هذا التطبيق للأبحاث والمعلومات الطبية فقط. لا يحل محل الطبيب.")

# ─────────────────────────────────────────────────────────────
# 7️⃣ PAGE: CHAT (MAIN)
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "chat":
    logo_tag = f'<img src="{LOGO_SRC}" style="width: 60px; margin-bottom: 10px;">' if LOGO_SRC else ''
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            {logo_tag}
            <h1 class="main-title">مساعدك الطبي الذكي</h1>
            <p style="color: #5A6072;">تحدث معي باللغة العربية حول أي موضوع صحي</p>
        </div>
    """, unsafe_allow_html=True)

    if not DB_STATUS:
        st.error("⚠️ قاعدة البيانات غير متاحة حالياً.")
        st.stop()

    if not st.session_state.messages:
        st.markdown("""
            <div class="welcome-card">
                <h3 style="color:#1E2028; font-weight:800; margin-bottom:12px;">كيف يمكنني مساعدتك اليوم؟</h3>
                <p style="color:#5A6072;">يمكنك سؤالي عن الأعراض، الأدوية، أو طلب نصائح عامة.</p>
                <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:15px;">
                    <span style="background:#FEEEEE; color:#C62828; padding:5px 12px; border-radius:8px; font-size:13px;">👨‍⚕️ استشارة</span>
                    <span style="background:#FEEEEE; color:#C62828; padding:5px 12px; border-radius:8px; font-size:13px;">💊 أدوية</span>
                    <span style="background:#FEEEEE; color:#C62828; padding:5px 12px; border-radius:8px; font-size:13px;">🍎 نصائح تغذية</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Display Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=get_logo() if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Input handling
    user_q = st.chat_input("اكتب استفسارك هنا...")
    
    # Voice (simple integration)
    try:
        from audio_recorder_streamlit import audio_recorder
        cols = st.columns([0.1, 0.9])
        with cols[0]:
            raw_audio = audio_recorder(text="", recording_color="#E53935", neutral_color="#94A3B8", icon_size="1.2rem", key="mic_chat")
            if raw_audio and len(raw_audio) > 1000:
                with st.spinner("⏳..."):
                    wav = convert_audio_to_wav(raw_audio, src_format="wav")
                    text, err = speech_to_text_arabic(wav)
                    if text: user_q = text
    except: pass

    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_q)
            
        with st.chat_message("assistant", avatar=get_logo()):
            placeholder = st.empty()
            placeholder.markdown("⏳ جاري التحليل...")
            
            try:
                safe = guard.check(user_q)
                if safe["level"] in ("emergency", "boundary"):
                    answer = safe["override_response"]
                else:
                    enc = retriever.encode_query(user_q)
                    try: intent, _ = classifier.predict(enc)
                    except: intent = "general"
                    
                    res = retriever.get_best_answer(user_q)
                    if res:
                        base, _, _, _ = res
                        if AI_STATUS:
                            answer = ai_engine.generate_answer(user_q, base, intent)
                        else:
                            answer = format_response(base, intent)
                        
                        if safe["level"] == "caution":
                            answer = guard.format_caution_response(answer)
                        answer = guard.add_disclaimer(answer)
                    else:
                        answer = "عذراً، لا أملك إجابة دقيقة حالياً. يرجى استشارة طبيب مختص."
            except:
                answer = "⚠️ حدث خطأ غير متوقع."
            
            placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            save_history(st.session_state.messages)
            st.rerun()

# ─────────────────────────────────────────────────────────────
# 8️⃣ PAGE: SYMPTOM SCANNER
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "scanner":
    st.markdown('<h1 class="main-title">🔍 فاحص الأعراض المتقدم</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5A6072;">يرجى ملء النموذج بدقة للحصول على تحليل أولي.</p>', unsafe_allow_html=True)
    
    with st.form("scanner_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("العمر", 0, 120, 30)
            gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
        with col2:
            duration = st.selectbox("مدة الأعراض", ["أقل من يوم", "1-3 أيام", "أسبوع", "أكثر من أسبوع"])
            severity = st.select_slider("شدة الألم/التعب", options=["خفيف", "متوسط", "شديد", "لا يطاق"])
            
        symptoms = st.text_area("وصف الأعراض بالتفصيل (مثال: سعال جاف مع ألم في الصدر)", height=100)
        history = st.text_input("هل تعاني من أمراض مزمنة؟ (اختياري)")
        
        submitted = st.form_submit_button("بدء التحليل الذكي", use_container_width=True)
        
    if submitted:
        if not symptoms:
            st.warning("يرجى وصف الأعراض أولاً.")
        else:
            with st.spinner("⏳ جاري تحليل الحالة..."):
                scan_prompt = f"""
                المريض: {gender}، العمر: {age}
                المدة: {duration}، الشدة: {severity}
                الأعراض الموصوفة: {symptoms}
                التاريخ المرضي: {history if history else 'لا يوجد'}
                """
                
                # Use retriever for context then LLM for analysis
                res = retriever.get_best_answer(symptoms)
                context = res[0] if res else "إرشادات عامة للمرض"
                
                if AI_STATUS:
                    analysis = ai_engine.generate_answer(
                        f"قم بتحليل هذه الأعراض وتقديم توجيهات أولية: {scan_prompt}", 
                        context, 
                        "וصف_أعراض"
                    )
                else:
                    analysis = "تحليل النظام غير متوفر حالياً بدون مفتاح LLM. يرجى مراجعة الطبيب."
                
                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                st.subheader("📋 نتيجة التحليل الأولي")
                st.write(analysis)
                st.markdown('</div>', unsafe_allow_html=True)
                st.info("⚠️ تذكر: هذا التحليل آلي ولا يغني عن الفحص السريري.")

# ─────────────────────────────────────────────────────────────
# 9️⃣ PAGE: HISTORY
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "history":
    st.markdown('<h1 class="main-title">📜 سجل الاستشارات الماضية</h1>', unsafe_allow_html=True)
    
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
            
        if not history:
            st.info("لا يوجد سجل استشارات حتى الآن.")
        else:
            for item in history:
                with st.expander(f"📅 {item['date']} | {item['title'][:40]}..."):
                    for m in item['messages']:
                        role = "👨‍⚕️ الطبيب الذكي" if m['role'] == "assistant" else "👤 أنت"
                        st.markdown(f"**{role}:**")
                        st.write(m['content'])
                        st.markdown("---")
    else:
        st.info("سجل الاستشارات فارغ حالياً.")

# ─────────────────────────────────────────────────────────────
# 🔟 PAGE: DATABASE
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "database":
    st.markdown('<h1 class="main-title">📚 قاعدة المعرفة الطبية</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5A6072;">اطلع على قائمة التصنيفات والبيانات التي يعتمد عليها النظام.</p>', unsafe_allow_html=True)
    
    try:
        from engine.retriever import FAISSRetriever
        from data.knowledge_base import CATEGORY_KEYWORDS
        
        cols = st.columns(3)
        cats = list(CATEGORY_KEYWORDS.keys())
        for i, cat in enumerate(cats):
            with cols[i % 3]:
                st.markdown(f"""
                    <div style="background:#F8F9FA; padding:15px; border-radius:12px; margin-bottom:10px; border-right:4px solid #E53935;">
                        <h4 style="margin:0; font-size:14px;">{cat}</h4>
                    </div>
                """, unsafe_allow_html=True)
                
        st.markdown("### 🔍 البحث في المصادر")
        search_term = st.text_input("ابحث عن موضوع طبي...")
        if search_term:
            res = retriever.get_best_answer(search_term)
            if res:
                st.markdown(f'<div class="data-card">{res[0]}</div>', unsafe_allow_html=True)
            else:
                st.info("لم نجد مراجع محددة لهذا البحث.")
    except:
        st.error("تعذر تحميل بيانات المصادر حالياً.")

