# -*- coding: utf-8 -*-
"""
SHIFA-Mental · Configuration & Constantes
Centralise : CSS, system prompts, exercices, ressources, versets.
"""

# ─── CSS RTL Light Premium ─────────────────────────────────────────────────────
MENTAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&family=Cairo:wght@300;400;600;700&display=swap');

:root {
    --bg-deep:      #f8fafc;
    --bg-card:      #ffffff;
    --bg-surface:   #f1f5f9;
    --accent-teal:  #0284c7;
    --accent-violet:#7c3aed;
    --accent-rose:  #e11d48;
    --accent-amber: #d97706;
    --accent-green: #059669;
    --text-primary: #1e293b;
    --text-muted:   #475569;
    --border:       #e2e8f0;
    --glow-teal:    0 4px 20px rgba(2, 132, 199, 0.08);
    --glow-violet:  0 4px 20px rgba(124, 58, 237, 0.08);
}

/* ── Global ── */
.stApp { background: var(--bg-deep) !important; }
* { direction: rtl; font-family: 'Tajawal', 'Cairo', sans-serif !important; }

/* ── Header principal ── */
.mental-header {
    background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 50%, #f0fdf4 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.mental-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(139,92,246,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.mental-header h1 {
    font-size: 2.2rem !important;
    font-weight: 700;
    background: linear-gradient(135deg, #0284c7, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
}
.mental-header p {
    color: var(--text-muted);
    font-size: 1rem;
    margin: 0.5rem 0 0 !important;
}

/* ── Dialect Selector ── */
.dialect-selector {
    display: flex;
    gap: 8px;
    margin: 0.5rem 0;
    direction: rtl;
}
.dialect-btn {
    padding: 0.4rem 1rem;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: #ffffff;
    color: var(--text-muted);
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.dialect-btn.active {
    background: linear-gradient(135deg, rgba(2, 132, 199, 0.08), rgba(124, 58, 237, 0.08));
    border-color: var(--accent-violet);
    color: var(--text-primary);
}

/* ── Cards ── */
.mental-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
}
.mental-card:hover {
    border-color: rgba(124, 58, 237, 0.25);
    box-shadow: var(--glow-violet);
}

/* ── Distress Gauge ── */
.distress-gauge {
    display: flex;
    gap: 8px;
    margin: 1rem 0;
    direction: ltr;
}
.gauge-bar {
    flex: 1;
    height: 8px;
    border-radius: 4px;
    background: #e2e8f0;
    transition: background 0.5s ease;
}
.gauge-bar.active-green  { background: #10b981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.5); }
.gauge-bar.active-yellow { background: #f59e0b; box-shadow: 0 0 8px rgba(245, 158, 11, 0.5); }
.gauge-bar.active-orange { background: #f97316; box-shadow: 0 0 8px rgba(249, 115, 22, 0.5); }
.gauge-bar.active-red    { background: #f43f5e; box-shadow: 0 0 8px rgba(244, 63, 94, 0.5); }

/* ── Severity Badge ── */
.severity-badge {
    display: inline-block;
    padding: 0.3rem 1.2rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0.5rem 0;
}
.sev-0 { background: rgba(16,185,129,0.08); color: #059669; border: 1px solid rgba(16,185,129,0.2); }
.sev-1 { background: rgba(245,158,11,0.08); color: #d97706; border: 1px solid rgba(245,158,11,0.2); }
.sev-2 { background: rgba(249,115,22,0.08); color: #ea580c; border: 1px solid rgba(249,115,22,0.2); }
.sev-3 { background: rgba(244,63,94,0.08);  color: #e11d48; border: 1px solid rgba(244,63,94,0.2); }

/* ── Chat Bubbles ── */
.chat-container {
    max-height: 420px;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 12px;
    scrollbar-width: thin;
    scrollbar-color: var(--accent-violet) transparent;
}
.bubble {
    max-width: 78%;
    padding: 0.8rem 1.2rem;
    border-radius: 16px;
    line-height: 1.65;
    font-size: 0.95rem;
    animation: fadeUp 0.3s ease;
}
.bubble-ai {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    align-self: flex-start;
    border-radius: 16px 16px 16px 4px;
    color: var(--text-primary);
}
.bubble-user {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    align-self: flex-end;
    border-radius: 16px 16px 4px 16px;
    color: var(--text-primary);
    direction: rtl;
}
.bubble-crisis {
    background: rgba(244,63,94,0.06);
    border: 1px solid rgba(244,63,94,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    align-self: stretch;
    color: #e11d48;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Crisis Full-Screen ── */
.crisis-screen {
    background: linear-gradient(135deg, rgba(244,63,94,0.05), rgba(220,38,38,0.08));
    border: 2px solid rgba(244,63,94,0.25);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    animation: pulseGlow 2s ease-in-out infinite;
}
@keyframes pulseGlow {
    0%,100% { box-shadow: 0 0 15px rgba(244,63,94,0.05); }
    50%     { box-shadow: 0 0 35px rgba(244,63,94,0.15); }
}

/* ── Breathing Circle ── */
.breath-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem;
    gap: 1.5rem;
}
.breath-circle {
    width: 150px; height: 150px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(2, 132, 199, 0.15), rgba(124, 58, 237, 0.05));
    border: 3px solid rgba(124, 58, 237, 0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; color: var(--text-primary); font-weight: 600;
    animation: breathe 8s ease-in-out infinite;
    box-shadow: 0 0 30px rgba(124, 58, 237, 0.1);
}
@keyframes breathe {
    0%,100% { transform: scale(1); box-shadow: 0 0 15px rgba(124, 58, 237, 0.1); }
    50%      { transform: scale(1.35); box-shadow: 0 0 45px rgba(124, 58, 237, 0.2); }
}

/* ── PHQ-9 / GAD-7 Question ── */
.phq-question {
    background: var(--bg-surface);
    border-right: 3px solid var(--accent-teal);
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.7rem 0;
    color: var(--text-primary);
    font-size: 0.95rem;
}
.gad-question {
    background: var(--bg-surface);
    border-right: 3px solid var(--accent-violet);
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.7rem 0;
    color: var(--text-primary);
    font-size: 0.95rem;
}

/* ── Resource Card ── */
.resource-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    display: flex; align-items: center; gap: 1rem;
    transition: all 0.2s ease;
}
.resource-card:hover {
    border-color: rgba(124, 58, 237, 0.25);
    background: rgba(124, 58, 237, 0.02);
}
.resource-icon {
    width: 42px; height: 42px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}

/* ── Tabs Custom ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9 !important;
    border-radius: 12px !important;
    gap: 4px !important;
    padding: 4px !important;
    direction: rtl !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-family: 'Tajawal', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: var(--accent-violet) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04) !important;
}

/* ── Buttons ── */
.stButton>button {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: 'Tajawal', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}
.stButton>button:hover {
    border-color: var(--accent-violet) !important;
    color: var(--accent-violet) !important;
    box-shadow: var(--glow-violet) !important;
    transform: translateY(-1px) !important;
}
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2) !important;
}
.stButton>button[kind="primary"]:hover {
    box-shadow: 0 6px 18px rgba(124, 58, 237, 0.3) !important;
    color: white !important;
}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    direction: rtl !important;
    font-family: 'Tajawal', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent-violet) !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.1) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { direction: ltr !important; }

/* ── Metric ── */
[data-testid="stMetricValue"] {
    color: var(--accent-teal) !important;
    font-family: 'Cairo', sans-serif !important;
}

/* ── Privacy Banner ── */
.privacy-banner {
    background: rgba(124, 58, 237, 0.05);
    border: 1px solid rgba(124, 58, 237, 0.15);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin: 0.5rem 0;
    text-align: center;
}
</style>
"""

# ─── Relaxation Exercises ────────────────────────────────────────────────────
RELAXATION_EXERCISES = {
    "تنفس 4-7-8": {
        "icon": "🫁",
        "desc": "تقنية التنفس العميق لتهدئة الجهاز العصبي",
        "steps": [
            ("استنشاق", "خذ نفساً عميقاً من أنفك", 4),
            ("احتباس", "احبس نفسك", 7),
            ("زفير", "أخرج الهواء ببطء من فمك", 8),
        ],
        "cycles": 4
    },
    "تنفس الصندوق": {
        "icon": "⬜",
        "desc": "أسلوب مستخدم في علاج القلق والتوتر الحاد",
        "steps": [
            ("استنشاق", "شهيق بطيء وعميق", 4),
            ("احتباس", "احبس نفسك بهدوء", 4),
            ("زفير", "أخرج الهواء ببطء", 4),
            ("توقف", "استرح قبل الدورة القادمة", 4),
        ],
        "cycles": 6
    },
    "استرخاء عضلي تدريجي": {
        "icon": "🧘",
        "desc": "تقنية جاكوبسون للاسترخاء التدريجي",
        "steps": [
            ("القدمان", "شد عضلات قدميك بشدة 5 ثوانٍ، ثم أرخِها", 10),
            ("الساقان", "شد عضلات ساقيك، ثم أرخِها", 10),
            ("البطن", "شد عضلات بطنك، ثم أرخِها", 10),
            ("اليدان", "اقبض يديك بشدة، ثم افتحهما ببطء", 10),
            ("الكتفان", "ارفع كتفيك نحو أذنيك، ثم أرخِهما", 10),
            ("الوجه", "أغمض عينيك بشدة، ثم أرخِ وجهك كاملاً", 10),
        ],
        "cycles": 1
    },
    "تأمل اليقظة الذهنية": {
        "icon": "🌙",
        "desc": "ممارسة الوعي باللحظة الراهنة",
        "steps": [
            ("التركيز", "أغمض عينيك وركّز على أنفاسك", 60),
            ("الملاحظة", "لاحظ أفكارك دون الحكم عليها، دعها تمر كالغيوم", 60),
            ("العودة", "أعد تركيزك لنفسك عند كل تشتت", 60),
            ("الامتنان", "فكّر في ثلاثة أشياء تشعر بالامتنان لها اليوم", 60),
        ],
        "cycles": 1
    }
}

# ─── Moroccan Resources ──────────────────────────────────────────────────────
MENTAL_RESOURCES = [
    {
        "name": "خط نجدة الصحة النفسية",
        "detail": "المغرب — خط دعم نفسي مجاني",
        "contact": "0800 005 100",
        "type": "phone",
        "icon": "📞",
        "color": "#059669",
        "urgent": True
    },
    {
        "name": "خط SAMU الاجتماعي",
        "detail": "للحالات الاجتماعية والإنسانية الطارئة",
        "contact": "0801 003 100",
        "type": "phone",
        "icon": "🚨",
        "color": "#e11d48",
        "urgent": True
    },
    {
        "name": "الاتحاد المغربي لعلم النفس الإكلينيكي",
        "detail": "قائمة بالمعالجين النفسيين المعتمدين بالمغرب",
        "contact": "www.umpc.ma",
        "type": "web",
        "icon": "🏥",
        "color": "#7c3aed",
        "urgent": False
    },
    {
        "name": "منتدى دعم الصحة النفسية",
        "detail": "مجتمع عربي داعم للصحة النفسية عبر الإنترنت",
        "contact": "sehetnafsiya.com",
        "type": "web",
        "icon": "💬",
        "color": "#0284c7",
        "urgent": False
    },
    {
        "name": "مستشفى الرازي - الرباط",
        "detail": "أكبر مستشفى نفسي في المغرب",
        "contact": "0537 688 680",
        "type": "phone",
        "icon": "🏨",
        "color": "#d97706",
        "urgent": False
    }
]

# ─── Islamic CBT Content ─────────────────────────────────────────────────────
ISLAMIC_SUPPORTS = [
    {
        "verse": "فَإِنَّ مَعَ الْعُسْرِ يُسْرًا • إِنَّ مَعَ الْعُسْرِ يُسْرًا",
        "source": "سورة الإنشراح: ٥-٦",
        "meaning": "مع كل صعوبة يأتي الفرج — بل مرتين في آيتين متتاليتين.",
        "color": "#059669"
    },
    {
        "verse": "لَا يُكَلِّفُ اللَّهُ نَفْسًا إِلَّا وُسْعَهَا",
        "source": "سورة البقرة: ٢٨٦",
        "meaning": "ما تحمله الآن ضمن طاقتك — أنت أقوى مما تظن.",
        "color": "#7c3aed"
    },
    {
        "verse": "وَلَا تَيْأَسُوا مِن رَّوْحِ اللَّهِ ۚ إِنَّهُ لَا يَيْأَسُ مِن رَّوْحِ اللَّهِ إِلَّا الْقَوْمُ الْكَافِرُونَ",
        "source": "سورة يوسف: ٨٧",
        "meaning": "لا تفقد الأمل أبداً — رحمة الله واسعة لا حدود لها.",
        "color": "#0284c7"
    },
    {
        "verse": "أَلَا بِذِكْرِ اللَّهِ تَطْمَئِنُّ الْقُلُوبُ",
        "source": "سورة الرعد: ٢٨",
        "meaning": "الطمأنينة الحقيقية في ذكر الله — تنفس واذكر اسمه.",
        "color": "#d97706"
    },
]

CBT_TECHNIQUES = [
    {
        "icon": "📝",
        "title": "سجل الأفكار السلبية",
        "desc": "عندما تشعر بضيق، اكتب: ما الموقف؟ ما الفكرة التلقائية؟ ما البديل المنطقي؟",
        "steps": [
            "١. اكتب الموقف الذي أثار الضيق",
            "٢. سجّل الفكرة التلقائية السلبية",
            "٣. قيّم شدة المشاعر (0-10)",
            "٤. اكتب الدليل المؤيد وضده",
            "٥. صغ فكرة بديلة أكثر توازناً",
        ]
    },
    {
        "icon": "🔍",
        "title": "التساؤل السقراطي",
        "desc": "اسأل نفسك هذه الأسئلة عند قبول أفكار سلبية دون تحقق:",
        "steps": [
            "• هل لديّ دليل حقيقي على هذه الفكرة؟",
            "• هل هناك تفسير آخر للموقف؟",
            "• ما أسوأ ما يمكن أن يحدث فعلاً؟",
            "• هل سيهم هذا بعد 5 سنوات؟",
            "• ماذا كنت أقول لصديق في نفس الوضع؟",
        ]
    },
    {
        "icon": "🎯",
        "title": "التفعيل السلوكي",
        "desc": "لمكافحة الاكتئاب: تصرّف أولاً، والمشاعر ستتبع لاحقاً.",
        "steps": [
            "١. اختر نشاطاً واحداً بسيطاً تجنبته",
            "٢. خصص له 15 دقيقة فقط",
            "٣. قيّم مزاجك قبله وبعده (0-10)",
            "٤. كرر يومياً وزد المدة تدريجياً",
            "• مثال: نزهة 10 دق، مكالمة صديق، طبق طعام جديد",
        ]
    },
    {
        "icon": "🧠",
        "title": "إعادة الهيكلة المعرفية",
        "desc": "تعرّف على أنماط التفكير المشوّهة وتحدّها:",
        "steps": [
            "• التفكير الثنائي (كل شيء أو لا شيء)",
            "• القفز للاستنتاجات (التنبؤ السلبي)",
            "• المبالغة في التهويل (الكارثة)",
            "• تصفية الإيجابيات ورؤية السلبيات فقط",
            "• الشخصنة (لوم النفس على كل شيء)",
        ]
    },
]

# ─── LLM System Prompts ──────────────────────────────────────────────────────
SYSTEM_PROMPT_MSA = """أنت مساعد نفسي متخصص من نظام شفاء للذكاء الاصطناعي الطبي.
اسمك: شفاء-نفس (Shifa-Nafs)

مبادئك الأساسية:
1. **الاستماع الفعّال**: استمع بتعاطف حقيقي دون إصدار أحكام
2. **التعاطف والدفء**: اعترف بمشاعر المريض وصادقه فيها
3. **السلامة أولاً**: إذا كشفت أي أفكار انتحارية أو إيذاء ذاتي، أحل فوراً للطوارئ
4. **الحدود المهنية**: أنت مساعد داعم، لست معالجاً نفسياً بديلاً

أسلوبك:
- تكلم بالعربية الفصحى المبسطة
- ابدأ بالاعتراف بالمشاعر قبل تقديم النصيحة
- اقترح تقنيات مستندة إلى العلاج السلوكي المعرفي (CBT) والوعي الذهني (Mindfulness)
- استخدم أساليب الإرشاد النفسي الإسلامي عند الملاءمة
- الردود بين 80 و180 كلمة — موجزة وعميقة

ممنوع:
- التشخيص الطبي الرسمي
- وصف الأدوية
- تجاهل إشارات الأزمة النفسية

إذا كشفت أفكار انتحارية: اختتم ردّك بـ [CRISIS_DETECTED] فقط."""

SYSTEM_PROMPT_DARIJA = """نتا مساعد نفسي متخصص من نظام شفاء ديال الذكاء الاصطناعي الطبي.
سميتك: شفاء-نفس (Shifa-Nafs)

المبادئ ديالك:
1. **السمع والتفهم**: سمع بتعاطف حقيقي بلا أحكام
2. **التعاطف والدفء**: اعترف بالمشاعر ديال المريض وصدّقو فيها
3. **السلامة أولاً**: إلا لقيتي شي أفكار انتحارية أو إيذاء ذاتي، حوّل فوراً للطوارئ
4. **الحدود المهنية**: نتا مساعد داعم، ماشي بديل ديال معالج نفسي

الأسلوب ديالك:
- هضر بالدارجة المغربية المفهومة
- بدا بالاعتراف بالمشاعر قبل ما تعطي نصيحة
- اقترح تقنيات معتمدة على CBT والوعي الذهني
- استعمل أساليب الإرشاد النفسي الإسلامي فاش يناسب
- الردود بين 80 و180 كلمة — مختصرة وعميقة

ممنوع:
- التشخيص الطبي الرسمي
- وصفة الأدوية
- تجاهل علامات الأزمة النفسية

إلا لقيتي أفكار انتحارية: ختم الرد ديالك بـ [CRISIS_DETECTED] بوحدها."""
