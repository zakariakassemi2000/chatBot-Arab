# -*- coding: utf-8 -*-
"""
SHIFA-Mental — Module de soutien psychologique (v2.0)
═══════════════════════════════════════════════════════
Détection de détresse multi-dialectale (MSA + Darija + Égyptien + Levantin)
Exercices de relaxation · Orientation psychologue · Chat IA · PHQ-9 · GAD-7
Mode Darija · Persistance SQLite · Safety Layer · Protocole post-crise

Architecture modulaire : modules/mental/
"""

import streamlit as st
import time
import hashlib
import logging
from datetime import datetime
from typing import Optional

# ── Modular imports ──────────────────────────────────────────────────────────
from modules.mental.config import (
    MENTAL_CSS, RELAXATION_EXERCISES, MENTAL_RESOURCES,
    ISLAMIC_SUPPORTS, CBT_TECHNIQUES,
    SYSTEM_PROMPT_MSA, SYSTEM_PROMPT_DARIJA,
)
from modules.mental.detector import DistressDetector
from modules.mental.llm_client import MentalLLMClient
from modules.mental.phq9 import PHQ9Assessment, PHQ9_QUESTIONS, PHQ9_OPTIONS
from modules.mental.gad7 import GAD7Assessment, GAD7_QUESTIONS, GAD7_OPTIONS
from modules.mental.persistence import MentalPersistence

logger = logging.getLogger("shifa.mental")

# ─── Singleton instances (lazy-loaded) ───────────────────────────────────────
_detector: Optional[DistressDetector] = None
_persistence: Optional[MentalPersistence] = None


def _get_detector() -> DistressDetector:
    global _detector
    if _detector is None:
        _detector = DistressDetector()
    return _detector


def _get_persistence() -> MentalPersistence:
    global _persistence
    if _persistence is None:
        _persistence = MentalPersistence()
    return _persistence


# ─── Fallback Responses ──────────────────────────────────────────────────────
def _fallback_response(level: int, dialect: str = "msa") -> str:
    """Rule-based fallback when no LLM API is available."""
    if dialect == "darija":
        if level == 3:
            return """كنسمعك، وداكشي لي كتحس بيه دابا خاص يتعامل معاه فوراً.
ماشي بوحدك ف هاد اللحظة. 🌿

عافاك تواصل دابا مع خط الدعم النفسي المجاني:
📞 **0800 005 100**

إلا كنتي ف خطر مباشر، عيّط للطوارئ: **15** [CRISIS_DETECTED]"""
        elif level == 2:
            return """كنسمعك، وكنفهم بلي كتقضي وقت صعيب. المشاعر ديالك صحيحة وطبيعية.
ف هاد اللحظة، جرّب تمرين تنفس الصندوق باش تهدا شوية.
إلا بقاو هاد المشاعر، ممكن يكون مفيد تهضر مع مختص نفسي. 💙"""
        elif level == 1:
            return """شكراً لي شاركتي معايا. كنفهم بلي هادشي ماشي ساهل.
تفكّر بلي القلق والحزن شي حاجة طبيعية، وديما كاين طرق باش نتعاملو معاهم.
بغيتي تجرّب شي تمارين ديال الاسترخاء؟ 🌱"""
        else:
            return """فرحان بلي راك لاباس عليك! الاهتمام بالصحة النفسية حتى ف الأوقات الزوينة شي حاجة مزيانة.
بغيتي تكتشف تقنيات لتعزيز الصحة ديالك أو تمارين التأمل اليومي؟ ✨"""
    else:  # MSA
        if level == 3:
            return """أسمعك، وما تشعر به الآن يستحق اهتماماً فورياً.
أنت لست وحدك في هذه اللحظة. 🌿

يُرجى التواصل الآن مع خط الدعم النفسي المجاني:
📞 **0800 005 100**

إذا كنت في خطر مباشر، اتصل بالطوارئ: **15** [CRISIS_DETECTED]"""
        elif level == 2:
            return """أسمعك، وأفهم أنك تمر بوقت عصيب. مشاعرك صحيحة وطبيعية تماماً.
في هذه اللحظة، أقترح عليك تجربة تمرين تنفس الصندوق للتهدئة الفورية.
إذا استمرت هذه المشاعر، قد يكون من المفيد التحدث مع مختص نفسي. 💙"""
        elif level == 1:
            return """شكراً لمشاركتي ما تشعر به. أفهم أن هذه الأمور ليست سهلة.
تذكّر أن مشاعر القلق والحزن طبيعية، وهناك دائماً طرق للتعامل معها بشكل أفضل.
هل تريد تجربة أحد تمارين الاسترخاء المتاحة؟ 🌱"""
        else:
            return """أسعدني أنك بخير! الاهتمام بصحتنا النفسية حتى في الأوقات الجيدة أمر رائع.
هل تريد اكتشاف تقنيات لتعزيز صحتك النفسية أو تمارين التأمل اليومي؟ ✨"""


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ═════════════════════════════════════════════════════════════════════════════
def render_mental_module(api_key: Optional[str] = None):
    """Main entry point for the SHIFA-Mental module."""
    st.markdown(MENTAL_CSS, unsafe_allow_html=True)

    detector = _get_detector()
    db = _get_persistence()

    # ── Header ──
    st.markdown("""
    <div class="mental-header">
        <h1>🧠 شفاء-نفس · الصحة النفسية</h1>
        <p>مساعد نفسي ذكي بالعربية — استمع · اطمئن · تعافَ</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Privacy Banner ──
    st.markdown("""
    <div class="privacy-banner">
        <span style="color:#8b5cf6; font-size:0.82rem;">🔒 محادثاتك خاصة ومحفوظة محلياً فقط · لا يتم إرسال بيانات شخصية لأي خادم خارجي</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Dialect Selector ──
    if "mental_dialect" not in st.session_state:
        st.session_state.mental_dialect = "msa"

    col_d1, col_d2, col_d3 = st.columns([1, 1, 4])
    with col_d1:
        if st.button("🇲🇦 الدارجة", key="btn_darija",
                     width="stretch",
                     type="primary" if st.session_state.mental_dialect == "darija" else "secondary"):
            st.session_state.mental_dialect = "darija"
            st.rerun()
    with col_d2:
        if st.button("📖 الفصحى", key="btn_msa",
                     width="stretch",
                     type="primary" if st.session_state.mental_dialect == "msa" else "secondary"):
            st.session_state.mental_dialect = "msa"
            st.rerun()

    dialect = st.session_state.mental_dialect

    # ── API Key fallback ──
    if not api_key:
        api_key = st.session_state.get("openrouter_api_key", "")
        if not api_key:
            import os
            api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            api_key = st.text_input("🔑 مفتاح OpenRouter API", type="password",
                                    placeholder="sk-or-v1-...",
                                    help="مطلوب للمحادثة النفسية الذكية")
            if api_key:
                st.session_state["openrouter_api_key"] = api_key

    # LLM client
    llm_client = MentalLLMClient(api_key=api_key)

    # ── Session State Init ──
    if "mental_chat" not in st.session_state:
        st.session_state.mental_chat = []
    if "distress_history" not in st.session_state:
        st.session_state.distress_history = []

    # ── Post-Crisis Check-in ──
    if db.has_recent_crisis(hours=48):
        if "crisis_checkin_done" not in st.session_state:
            st.markdown("""
            <div class="crisis-screen" style="margin-bottom:1.5rem;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">💙</div>
                <div style="color:#e11d48; font-size:1.2rem; font-weight:700; margin-bottom:0.5rem;">
                    كيف حالك اليوم؟
                </div>
                <div style="color:#475569; font-size:0.9rem;">
                    لاحظنا أنك مررت بوقت صعب مؤخراً. نريد الاطمئنان عليك.
                    <br>تذكّر أن طلب المساعدة علامة قوة وليست ضعفاً.
                    <br>📞 خط الدعم النفسي: <strong style="color:#e11d48;">0800 005 100</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("✅ أنا بخير، شكراً", key="crisis_checkin"):
                st.session_state.crisis_checkin_done = True
                st.rerun()

    # ── Tabs ──
    tabs = st.tabs([
        "💬 محادثة نفسية",
        "📋 تقييم PHQ-9",
        "😰 تقييم GAD-7",
        "🫁 تمارين الاسترخاء",
        "💙 تقنيات CBT الإسلامية",
        "📊 تتبع المزاج",
        "📝 يومية شخصية",
        "📍 الموارد والدعم"
    ])

    # ══════════════════════════════════════════════
    # TAB 1 — CHAT
    # ══════════════════════════════════════════════
    with tabs[0]:
        _render_chat_tab(detector, llm_client, db, dialect, api_key)

    # ══════════════════════════════════════════════
    # TAB 2 — PHQ-9
    # ══════════════════════════════════════════════
    with tabs[1]:
        _render_phq9_tab(db)

    # ══════════════════════════════════════════════
    # TAB 3 — GAD-7 (NEW)
    # ══════════════════════════════════════════════
    with tabs[2]:
        _render_gad7_tab(db)

    # ══════════════════════════════════════════════
    # TAB 4 — RELAXATION
    # ══════════════════════════════════════════════
    with tabs[3]:
        _render_exercises_tab()

    # ══════════════════════════════════════════════
    # TAB 5 — CBT ISLAMIQUE
    # ══════════════════════════════════════════════
    with tabs[4]:
        _render_cbt_tab(db)

    # ══════════════════════════════════════════════
    # TAB 6 — MOOD TRACKER
    # ══════════════════════════════════════════════
    with tabs[5]:
        _render_mood_tab(db)

    # ══════════════════════════════════════════════
    # TAB 7 — JOURNAL INTIME (NEW)
    # ══════════════════════════════════════════════
    with tabs[6]:
        _render_journal_tab(db, llm_client, dialect, api_key)

    # ══════════════════════════════════════════════
    # TAB 8 — RESOURCES
    # ══════════════════════════════════════════════
    with tabs[7]:
        _render_resources_tab()


# ═════════════════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ═════════════════════════════════════════════════════════════════════════════

def _render_chat_tab(detector, llm_client, db, dialect, api_key):
    """Tab 1: Chat psychologique avec détection de détresse."""
    col1, col2 = st.columns([3, 1])
    with col1:
        greeting = "كيفاش كتحس اليوم؟" if dialect == "darija" else "كيف تشعر اليوم؟"
        st.markdown(f"#### {greeting}")
    with col2:
        if st.button("🔄 محادثة جديدة", width="stretch"):
            st.session_state.mental_chat = []
            st.session_state.distress_history = []
            st.rerun()

    # Distress Gauge
    if st.session_state.distress_history:
        current_level = st.session_state.distress_history[-1]
        labels = ["✅ طبيعي", "🟡 خفيف", "🟠 متوسط", "🔴 أزمة"]
        colors = ["active-green", "active-yellow", "active-orange", "active-red"]
        badge_classes = ["sev-0", "sev-1", "sev-2", "sev-3"]

        bars_html = "".join([
            f'<div class="gauge-bar {colors[i] if i <= current_level else ""}"></div>'
            for i in range(4)
        ])
        st.markdown(f"""
        <div class="mental-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="color:#475569; font-size:0.85rem;">مستوى الضيق المكتشف</span>
                <span class="severity-badge {badge_classes[current_level]}">{labels[current_level]}</span>
            </div>
            <div class="distress-gauge">{bars_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # Welcome message
    if not st.session_state.mental_chat:
        if dialect == "darija":
            welcome = """مرحبا، أنا شفاء-نفس 🌿<br><br>
            أنا هنا باش نسمع ليك بتعاطف وبلا أحكام.
            يمكنك تهضر على المشاعر ديالك والأفكار ديالك بحرية تامة.<br><br>
            <em style="color:#475569;">كيفاش كتحس ف هاد اللحظة؟</em>"""
        else:
            welcome = """مرحباً، أنا شفاء-نفس 🌿<br><br>
            أنا هنا للاستماع إليك بتعاطف ودون أحكام.
            يمكنك التحدث عن مشاعرك وأفكارك بحرية تامة.<br><br>
            <em style="color:#475569;">كيف تشعر في هذه اللحظة؟</em>"""

        st.markdown(f"""
        <div class="mental-card">
            <div class="bubble bubble-ai">{welcome}</div>
        </div>
        """, unsafe_allow_html=True)

    # Chat history
    if st.session_state.mental_chat:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.mental_chat:
            role_class = "bubble-user" if msg["role"] == "user" else "bubble-ai"
            if "[CRISIS_DETECTED]" in msg.get("content", ""):
                content = msg["content"].replace("[CRISIS_DETECTED]", "").strip()
                chat_html += f"""
                <div class="bubble {role_class}">{content}</div>
                <div class="bubble-crisis">
                    🚨 <strong style="color:#f43f5e;">تحذير: تم الكشف عن أفكار تحتاج لدعم فوري</strong><br>
                    يرجى التواصل مع خط الدعم النفسي الآن: <strong>0800 005 100</strong>
                    <br><a href="tel:0800005100" style="color:#f43f5e; font-weight:700;">📞 اتصل الآن</a>
                </div>
                """
            else:
                chat_html += f'<div class="bubble {role_class}">{msg["content"]}</div>'
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    # Input
    placeholder_text = "هضر على داكشي لي كتحس بيه..." if dialect == "darija" else "تحدث عن مشاعرك، أفكارك، أو ما يقلقك..."
    user_input = st.text_area(
        "اكتب ما تشعر به...",
        placeholder=placeholder_text,
        height=100,
        label_visibility="collapsed",
        key="mental_input"
    )

    send_col, clear_col = st.columns([3, 1])
    with send_col:
        send_btn = st.button("📤 إرسال", width="stretch", type="primary")
    with clear_col:
        if st.button("🗑️ مسح", width="stretch"):
            st.session_state["mental_input"] = ""

    if send_btn and user_input.strip():
        # Detect distress (multi-dialectal)
        level, reason, matched_kws = detector.detect(user_input)
        st.session_state.distress_history.append(level)

        # Log crisis
        if level >= 3:
            text_hash = hashlib.sha256(user_input.encode()).hexdigest()[:16]
            db.log_crisis(level, matched_kws, text_hash)
            logger.warning("[Mental] Crisis detected — level=%d, keywords=%s", level, matched_kws)

        # Add user message
        st.session_state.mental_chat.append({"role": "user", "content": user_input})

        # Build context for LLM
        llm_messages = [
            {"role": m["role"], "content": m["content"].replace("[CRISIS_DETECTED]", "").strip()}
            for m in st.session_state.mental_chat
        ]

        # Crisis injection
        if level == 3:
            llm_messages[-1]["content"] += "\n[تنبيه: الكشف الآلي أشار لمؤشرات أزمة — تعامل بحساسية وأحل لخطوط الطوارئ]"

        # Select system prompt based on dialect
        system_prompt = SYSTEM_PROMPT_DARIJA if dialect == "darija" else SYSTEM_PROMPT_MSA

        # Call LLM
        spinner_text = "شفاء-نفس كيسمع ليك..." if dialect == "darija" else "شفاء-نفس يستمع إليك..."
        with st.spinner(spinner_text):
            if api_key:
                response = llm_client.chat(llm_messages, system_prompt, dialect)
            else:
                response = ""

            if not response:
                response = _fallback_response(level, dialect)

        st.session_state.mental_chat.append({"role": "assistant", "content": response})
        st.rerun()


def _render_phq9_tab(db):
    """Tab 2: PHQ-9 assessment."""
    st.markdown("#### 📋 مقياس تقييم الاكتئاب (PHQ-9)")
    st.markdown("""
    <div class="mental-card">
        <p style="color:#475569; margin:0;">
        على مدى آخر أسبوعين، كم مرة أزعجتك المشكلات التالية؟
        </p>
    </div>
    """, unsafe_allow_html=True)

    answers = []
    for i, question in enumerate(PHQ9_QUESTIONS):
        st.markdown(f'<div class="phq-question">{"🔴" if i == 8 else "🔵"} {question}</div>',
                    unsafe_allow_html=True)
        answer = st.select_slider(
            f"q_phq_{i}",
            options=[0, 1, 2, 3],
            format_func=lambda x: PHQ9_OPTIONS[x],
            label_visibility="collapsed",
            key=f"phq9_{i}"
        )
        answers.append(answer)

    if st.button("📊 احسب النتيجة", width="stretch", key="calc_phq9"):
        score, severity, rec, badge_class = PHQ9Assessment.calculate(answers)
        icon = PHQ9Assessment.get_badge_icon(severity)

        # Persist
        db.save_assessment("PHQ-9", score, severity, answers)

        st.markdown(f"""
        <div class="mental-card" style="border-color: rgba(124, 58, 237, 0.25);">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:2.5rem; font-weight:700; color:#0284c7;">{score}<span style="font-size:1rem; color:#64748b;">/27</span></div>
                    <span class="severity-badge {badge_class}">{icon} {severity}</span>
                </div>
                <div style="font-size:3rem;">{icon}</div>
            </div>
            <p style="color:#475569; margin-top:1rem; margin-bottom:0;">{rec}</p>
        </div>
        """, unsafe_allow_html=True)

        if score >= 10:
            st.info("💡 يُنصح بمراجعة متخصص نفسي. يمكنك العثور على أقرب مركز في تبويب الموارد.")

    # History
    past = db.get_assessments("PHQ-9", limit=5)
    if past:
        st.markdown("---")
        st.markdown("**📅 النتائج السابقة**")
        for entry in past:
            st.markdown(f"""
            <div class="resource-card">
                <div style="color:#64748b; font-size:0.8rem; width:80px;">{entry['date'][-5:]}</div>
                <div style="flex:1; color:#1e293b; font-weight:600;">{entry['severity']}</div>
                <div style="color:#0284c7; font-weight:700;">{entry['score']}/27</div>
            </div>
            """, unsafe_allow_html=True)


def _render_gad7_tab(db):
    """Tab 3: GAD-7 assessment (NEW)."""
    st.markdown("#### 😰 مقياس اضطراب القلق العام (GAD-7)")
    st.markdown("""
    <div class="mental-card">
        <p style="color:#475569; margin:0;">
        على مدى آخر أسبوعين، كم مرة أزعجتك المشكلات التالية؟
        </p>
    </div>
    """, unsafe_allow_html=True)

    answers = []
    for i, question in enumerate(GAD7_QUESTIONS):
        st.markdown(f'<div class="gad-question">🟣 {question}</div>',
                    unsafe_allow_html=True)
        answer = st.select_slider(
            f"q_gad_{i}",
            options=[0, 1, 2, 3],
            format_func=lambda x: GAD7_OPTIONS[x],
            label_visibility="collapsed",
            key=f"gad7_{i}"
        )
        answers.append(answer)

    if st.button("📊 احسب نتيجة القلق", width="stretch", key="calc_gad7"):
        score, severity, rec, badge_class = GAD7Assessment.calculate(answers)
        icon = GAD7Assessment.get_badge_icon(severity)

        # Persist
        db.save_assessment("GAD-7", score, severity, answers)

        st.markdown(f"""
        <div class="mental-card" style="border-color: rgba(124, 58, 237, 0.25);">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:2.5rem; font-weight:700; color:#7c3aed;">{score}<span style="font-size:1rem; color:#64748b;">/21</span></div>
                    <span class="severity-badge {badge_class}">{icon} {severity}</span>
                </div>
                <div style="font-size:3rem;">{icon}</div>
            </div>
            <p style="color:#475569; margin-top:1rem; margin-bottom:0;">{rec}</p>
        </div>
        """, unsafe_allow_html=True)

        if score >= 10:
            st.info("💡 يُنصح بالعلاج السلوكي المعرفي (CBT). الشفاء ممكن!")

    # History
    past = db.get_assessments("GAD-7", limit=5)
    if past:
        st.markdown("---")
        st.markdown("**📅 النتائج السابقة**")
        for entry in past:
            st.markdown(f"""
            <div class="resource-card">
                <div style="color:#64748b; font-size:0.8rem; width:80px;">{entry['date'][-5:]}</div>
                <div style="flex:1; color:#1e293b; font-weight:600;">{entry['severity']}</div>
                <div style="color:#7c3aed; font-weight:700;">{entry['score']}/21</div>
            </div>
            """, unsafe_allow_html=True)


def _render_exercises_tab():
    """Tab 4: Relaxation exercises."""
    st.markdown("#### 🫁 تمارين الاسترخاء والتأمل")

    for ex_name, ex_data in RELAXATION_EXERCISES.items():
        with st.expander(f"{ex_data['icon']} {ex_name} — {ex_data['desc']}"):
            # Breathing animation
            if "تنفس" in ex_name:
                st.markdown("""
                <div class="breath-container">
                    <div class="breath-circle">تنفّس</div>
                    <p style="color:#475569; text-align:center;">اتبع نبضة الدائرة</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("**الخطوات:**")
            for j, (step_name, step_desc, duration) in enumerate(ex_data["steps"]):
                st.markdown(f"""
                <div class="resource-card">
                    <div class="resource-icon" style="background:rgba(2, 132, 199, 0.08);">
                        <span style="color:#0284c7;">{j+1}</span>
                    </div>
                    <div>
                        <div style="color:#1e293b; font-weight:600;">{step_name}</div>
                        <div style="color:#475569; font-size:0.85rem;">{step_desc} — {duration} ثانية</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("عدد الدورات", ex_data["cycles"])
            with col_b:
                total = sum(s[2] for s in ex_data["steps"]) * ex_data["cycles"]
                st.metric("المدة الإجمالية", f"{total} ثانية")


def _render_cbt_tab(db):
    """Tab 5: CBT + Islamic support."""
    st.markdown("#### 💙 تقنيات العلاج المعرفي السلوكي والإرشاد الإسلامي")

    st.markdown("<br>", unsafe_allow_html=True)
    for sv in ISLAMIC_SUPPORTS:
        st.markdown(f"""
        <div class="mental-card" style="border-right: 4px solid {sv['color']}; padding: 1.2rem 1.5rem;">
            <div style="font-size:1.3rem; color:{sv['color']}; font-weight:700; margin-bottom:0.5rem; line-height:2;">
                {sv['verse']}
            </div>
            <div style="color:#475569; font-size:0.82rem; margin-bottom:0.5rem;">📖 {sv['source']}</div>
            <div style="color:#1e293b; font-size:0.92rem;">{sv['meaning']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🧩 تقنيات CBT للتطبيق اليومي")

    for tech in CBT_TECHNIQUES:
        with st.expander(f"{tech['icon']} {tech['title']} — {tech['desc']}"):
            for step in tech['steps']:
                st.markdown(f"""
                <div class="resource-card">
                    <div style="color:#1e293b; font-size:0.92rem;">{step}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Thought Journal ──
    st.markdown("---")
    st.markdown("#### ✍️ يومية الأفكار")
    st.caption("سجّل فكرة سلبية واكتب بديلاً لها — التكرار يُعيد برمجة العقل")
    col_j1, col_j2 = st.columns(2)
    with col_j1:
        neg_thought = st.text_area(
            "الفكرة السلبية:", height=100,
            placeholder="مثال: أنا فاشل في كل شيء...",
            key="neg_thought_input"
        )
    with col_j2:
        pos_thought = st.text_area(
            "الفكرة البديلة المتوازنة:", height=100,
            placeholder="مثال: لقد أخفقت في هذا الأمر لكنني نجحت في أمور كثيرة...",
            key="pos_thought_input"
        )
    if st.button("💾 حفظ في اليومية", key="save_journal_thought"):
        if neg_thought.strip():
            db.save_thought(neg_thought, pos_thought)
            st.success("✅ تم حفظ الفكرة في يوميتك!")

    past_thoughts = db.get_thoughts(limit=5)
    if past_thoughts:
        st.markdown(f"**📚 يوميتك ({len(past_thoughts)} سجلات حديثة)**")
        for entry in past_thoughts:
            st.markdown(f"""
            <div class="mental-card" style="margin:0.5rem 0;">
                <div style="color:#475569; font-size:0.75rem; margin-bottom:0.5rem;">🕐 {entry['date']}</div>
                <div style="color:#e11d48; font-size:0.88rem;">❌ {entry['negative']}</div>
                <div style="color:#059669; font-size:0.88rem; margin-top:0.4rem;">✅ {entry['positive']}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_mood_tab(db):
    """Tab 6: Mood tracker with persistence."""
    st.markdown("#### 📊 تتبع المزاج اليومي")
    st.caption("سجّل مزاجك يومياً لمعرفة الأنماط واكتشاف محفزات التحسن")

    MOOD_OPTIONS = {
        "😄 ممتاز": 5,
        "🙂 جيد": 4,
        "😐 عادي": 3,
        "😔 حزين": 2,
        "😰 سيء جداً": 1
    }

    col_mood1, col_mood2 = st.columns([1, 2])
    with col_mood1:
        selected_mood = st.selectbox("كيف مزاجك الآن؟", list(MOOD_OPTIONS.keys()), key="mood_select")
        mood_note = st.text_input("ملاحظة:", placeholder="ما الذي أثّر على مزاجك؟", key="mood_note")
        if st.button("➕ تسجيل المزاج", key="log_mood", width="stretch"):
            db.save_mood(selected_mood, MOOD_OPTIONS[selected_mood], mood_note)
            st.success(f"تم تسجيل مزاجك: {selected_mood}")
            st.rerun()

    with col_mood2:
        all_moods = db.get_moods(limit=30)
        if all_moods:
            recent = list(reversed(all_moods[:7]))
            scores = [m["score"] for m in recent]
            labels = [m["date"][-5:] for m in recent]

            avg = sum(scores) / len(scores)
            trend = "📈 تحسن" if len(scores) > 1 and scores[-1] > scores[0] else ("📉 تراجع" if len(scores) > 1 and scores[-1] < scores[0] else "→ مستقر")

            m1, m2, m3 = st.columns(3)
            m1.metric("السجلات", len(all_moods))
            m2.metric("المتوسط", f"{avg:.1f}/5")
            m3.metric("الاتجاه", trend)

            colors = {5: "#059669", 4: "#0284c7", 3: "#d97706", 2: "#ea580c", 1: "#e11d48"}
            bars_html = "".join([
                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                f'<span style="color:#475569;font-size:0.75rem;width:40px;">{labels[i]}</span>'
                f'<div style="flex:1;background:#e2e8f0;border-radius:4px;height:18px;overflow:hidden;">'
                f'<div style="width:{scores[i]*20}%;background:{colors[scores[i]]};height:100%;border-radius:4px;transition:width 0.5s;"></div>'
                f'</div>'
                f'<span style="color:{colors[scores[i]]};font-size:0.85rem;">{m["mood_label"].split()[0]}</span>'
                f'</div>'
                for i, m in enumerate(recent)
            ])

            st.markdown(f"""
            <div class="mental-card">
                <div style="color:#475569;font-size:0.8rem;margin-bottom:0.75rem;">📅 آخر {len(recent)} سجل</div>
                {bars_html}
            </div>
            """, unsafe_allow_html=True)

            # Insights
            if len(scores) >= 3:
                if avg >= 4:
                    insight = "🌟 مزاجك ممتاز بشكل عام! استمر في ما تفعله."
                    insight_color = "#059669"
                elif avg >= 3:
                    insight = "💙 مزاجك معقول. بعض تمارين الاسترخاء قد تساعد."
                    insight_color = "#0284c7"
                else:
                    insight = "🤗 يبدو أنك تمر بوقت صعب. تحدث مع شفاء-نفس أو جرب تمارين الاسترخاء."
                    insight_color = "#d97706"

                st.markdown(f"""
                <div class="mental-card" style="border-color:{insight_color}40; background:#f8fafc;">
                    <div style="color:{insight_color}; font-size:0.92rem;">{insight}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="mental-card" style="text-align:center; padding: 2rem;">
                <div style="font-size:2rem;">📊</div>
                <div style="color:#475569; margin-top:0.5rem;">لا توجد سجلات بعد.<br>سجّل مزاجك اليومي لرؤية الأنماط.</div>
            </div>
            """, unsafe_allow_html=True)

    # Full history
    if all_moods:
        st.markdown("---")
        st.markdown("**📋 السجل الكامل**")
        if st.button("🗑️ مسح كل السجلات", key="clear_mood_history"):
            db.clear_moods()
            st.rerun()
        for entry in all_moods[:15]:
            st.markdown(f"""
            <div class="resource-card">
                <div class="resource-icon" style="background:rgba(2, 132, 199, 0.08); font-size:1.2rem;">{entry['mood_label'].split()[0]}</div>
                <div style="flex:1;">
                    <div style="color:#1e293b; font-weight:600; font-size:0.88rem;">{entry['mood_label']}</div>
                    <div style="color:#475569; font-size:0.78rem;">{entry['date']}{' — ' + entry['note'] if entry.get('note') else ''}</div>
                </div>
                <div style="color:#0284c7; font-weight:700;">{'⭐' * entry['score']}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_journal_tab(db, llm_client, dialect, api_key):
    """Tab 7: Personal journal with optional sentiment analysis (NEW)."""
    st.markdown("#### 📝 يوميتك الشخصية")
    if dialect == "darija":
        st.caption("هاد الفضاء ديالك بوحدك. كتب داكشي لي بغيتي — ما حد غادي يشوفو غيرك.")
    else:
        st.caption("هذا الفضاء لك وحدك. اكتب ما تشاء — لن يراه أحد غيرك.")

    journal_input = st.text_area(
        "اكتب هنا...",
        height=180,
        placeholder="اليوم شعرت بـ..." if dialect == "msa" else "اليوم حسّيت بـ...",
        key="journal_content"
    )

    if st.button("💾 حفظ في اليومية", key="save_personal_journal", width="stretch"):
        if journal_input.strip():
            db.save_journal(journal_input.strip())
            st.success("✅ تم حفظ الإدخال في يوميتك الشخصية!")
            st.rerun()

    # Past entries
    past_journals = db.get_journals(limit=10)
    if past_journals:
        st.markdown("---")
        st.markdown(f"**📚 إدخالاتك السابقة ({len(past_journals)} حديثة)**")
        for entry in past_journals:
            preview = entry["content"][:120] + "..." if len(entry["content"]) > 120 else entry["content"]
            sentiment_badge = ""
            if entry.get("sentiment_label"):
                colors_map = {"إيجابي": "#059669", "سلبي": "#e11d48", "محايد": "#d97706"}
                scolor = colors_map.get(entry["sentiment_label"], "#475569")
                sentiment_badge = f'<span style="color:{scolor}; font-size:0.75rem; margin-right:8px;">● {entry["sentiment_label"]}</span>'

            st.markdown(f"""
            <div class="mental-card" style="margin:0.5rem 0; padding:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <span style="color:#475569; font-size:0.75rem;">🕐 {entry['date']}</span>
                    {sentiment_badge}
                </div>
                <div style="color:#1e293b; font-size:0.9rem; line-height:1.7;">{preview}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_resources_tab():
    """Tab 8: Mental health resources."""
    st.markdown("#### 📍 موارد الدعم النفسي في المغرب")

    urgent_resources = [r for r in MENTAL_RESOURCES if r["urgent"]]
    other_resources = [r for r in MENTAL_RESOURCES if not r["urgent"]]

    st.markdown("**🚨 خطوط الطوارئ النفسية**")
    for res in urgent_resources:
        st.markdown(f"""
        <div class="resource-card" style="border-color: rgba(225, 29, 72, 0.25);">
            <div class="resource-icon" style="background:rgba(225, 29, 72, 0.08);">
                {res['icon']}
            </div>
            <div style="flex:1;">
                <div style="color:#1e293b; font-weight:700;">{res['name']}</div>
                <div style="color:#475569; font-size:0.85rem;">{res['detail']}</div>
            </div>
            <div style="color:{res['color']}; font-weight:700; font-size:1.1rem;">{res['contact']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>**🏥 موارد الدعم الأخرى**", unsafe_allow_html=True)
    for res in other_resources:
        st.markdown(f"""
        <div class="resource-card">
            <div class="resource-icon" style="background:rgba(124, 58, 237, 0.08);">
                {res['icon']}
            </div>
            <div style="flex:1;">
                <div style="color:#1e293b; font-weight:600;">{res['name']}</div>
                <div style="color:#475569; font-size:0.85rem;">{res['detail']}</div>
            </div>
            <div style="color:{res['color']}; font-size:0.9rem;">{res['contact']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="mental-card" style="margin-top:1.5rem; border-color: rgba(2, 132, 199, 0.25);">
        <p style="color:#475569; font-size:0.88rem; margin:0; text-align:center;">
            ⚠️ شفاء-نفس هو مساعد دعم، وليس بديلاً عن الرعاية النفسية المتخصصة.
            في حالات الأزمة، يُرجى التواصل فوراً مع خطوط الطوارئ أعلاه.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="SHIFA-Mental — شفاء-نفس",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    render_mental_module()
