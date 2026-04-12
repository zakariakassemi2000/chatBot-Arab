# -*- coding: utf-8 -*-
"""
SHIFA-Mental · Configuration & Constantes
Centralise : CSS, system prompts, exercices, ressources, versets.
"""

# ─── CSS RTL Dark Premium ─────────────────────────────────────────────────────
MENTAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&family=Cairo:wght@300;400;600;700&display=swap');

:root {
    --bg-deep:      #0a0e1a;
    --bg-card:      #111827;
    --bg-surface:   #1a2235;
    --accent-teal:  #06b6d4;
    --accent-violet:#8b5cf6;
    --accent-rose:  #f43f5e;
    --accent-amber: #f59e0b;
    --accent-green: #10b981;
    --text-primary: #f0f4ff;
    --text-muted:   #94a3b8;
    --border:       rgba(6,182,212,0.15);
    --glow-teal:    0 0 30px rgba(6,182,212,0.2);
    --glow-violet:  0 0 30px rgba(139,92,246,0.2);
}

/* ── Global ── */
.stApp { background: var(--bg-deep) !important; }
* { direction: rtl; font-family: 'Tajawal', 'Cairo', sans-serif !important; }

/* ── Header principal ── */
.mental-header {
    background: linear-gradient(135deg, #0f172a 0%, #1a1040 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.mental-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%);
    pointer-events: none;
}
.mental-header h1 {
    font-size: 2.2rem !important;
    font-weight: 700;
    background: linear-gradient(135deg, #06b6d4, #8b5cf6);
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
    background: transparent;
    color: var(--text-muted);
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.dialect-btn.active {
    background: linear-gradient(135deg, rgba(6,182,212,0.2), rgba(139,92,246,0.2));
    border-color: var(--accent-teal);
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
}
.mental-card:hover {
    border-color: rgba(6,182,212,0.35);
    box-shadow: var(--glow-teal);
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
    background: rgba(255,255,255,0.1);
    transition: background 0.5s ease;
}
.gauge-bar.active-green  { background: #10b981; box-shadow: 0 0 8px #10b981; }
.gauge-bar.active-yellow { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
.gauge-bar.active-orange { background: #f97316; box-shadow: 0 0 8px #f97316; }
.gauge-bar.active-red    { background: #f43f5e; box-shadow: 0 0 8px #f43f5e; }

/* ── Severity Badge ── */
.severity-badge {
    display: inline-block;
    padding: 0.3rem 1.2rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0.5rem 0;
}
.sev-0 { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.sev-1 { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.sev-2 { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
.sev-3 { background: rgba(244,63,94,0.15);  color: #f43f5e; border: 1px solid rgba(244,63,94,0.3); }

/* ── Chat Bubbles ── */
.chat-container {
    max-height: 420px;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 12px;
    scrollbar-width: thin;
    scrollbar-color: var(--accent-teal) transparent;
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
    background: var(--bg-surface);
    border: 1px solid rgba(139,92,246,0.2);
    align-self: flex-start;
    border-radius: 16px 16px 16px 4px;
    color: var(--text-primary);
}
.bubble-user {
    background: linear-gradient(135deg, rgba(6,182,212,0.15), rgba(139,92,246,0.15));
    border: 1px solid rgba(6,182,212,0.2);
    align-self: flex-end;
    border-radius: 16px 16px 4px 16px;
    color: var(--text-primary);
    direction: rtl;
}
.bubble-crisis {
    background: rgba(244,63,94,0.1);
    border: 1px solid rgba(244,63,94,0.35);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    align-self: stretch;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Crisis Full-Screen ── */
.crisis-screen {
    background: linear-gradient(135deg, rgba(244,63,94,0.08), rgba(220,38,38,0.12));
    border: 2px solid rgba(244,63,94,0.4);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    animation: pulseGlow 2s ease-in-out infinite;
}
@keyframes pulseGlow {
    0%,100% { box-shadow: 0 0 20px rgba(244,63,94,0.1); }
    50%     { box-shadow: 0 0 50px rgba(244,63,94,0.25); }
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
    background: radial-gradient(circle, rgba(6,182,212,0.3), rgba(139,92,246,0.1));
    border: 3px solid rgba(6,182,212,0.4);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; color: var(--text-primary); font-weight: 600;
    animation: breathe 8s ease-in-out infinite;
    box-shadow: 0 0 40px rgba(6,182,212,0.2);
}
@keyframes breathe {
    0%,100% { transform: scale(1); box-shadow: 0 0 20px rgba(6,182,212,0.15); }
    50%      { transform: scale(1.35); box-shadow: 0 0 60px rgba(139,92,246,0.3); }
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
    background: var(--bg-surface);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    display: flex; align-items: center; gap: 1rem;
    transition: all 0.2s ease;
}
.resource-card:hover {
    border-color: rgba(139,92,246,0.5);
    background: rgba(139,92,246,0.05);
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
    background: var(--bg-card) !important;
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
    background: linear-gradient(135deg, rgba(6,182,212,0.2), rgba(139,92,246,0.2)) !important;
    color: var(--text-primary) !important;
}

/* ── Buttons ── */
.stButton>button {
    background: linear-gradient(135deg, rgba(6,182,212,0.15), rgba(139,92,246,0.15)) !important;
    border: 1px solid rgba(6,182,212,0.35) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: 'Tajawal', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton>button:hover {
    border-color: var(--accent-teal) !important;
    box-shadow: var(--glow-teal) !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    direction: rtl !important;
    font-family: 'Tajawal', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent-teal) !important;
    box-shadow: 0 0 0 2px rgba(6,182,212,0.15) !important;
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
    background: rgba(139,92,246,0.06);
    border: 1px solid rgba(139,92,246,0.2);
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
        "color": "#10b981",
        "urgent": True
    },
    {
        "name": "خط SAMU الاجتماعي",
        "detail": "للحالات الاجتماعية والإنسانية الطارئة",
        "contact": "0801 003 100",
        "type": "phone",
        "icon": "🚨",
        "color": "#f43f5e",
        "urgent": True
    },
    {
        "name": "الاتحاد المغربي لعلم النفس الإكلينيكي",
        "detail": "قائمة بالمعالجين النفسيين المعتمدين بالمغرب",
        "contact": "www.umpc.ma",
        "type": "web",
        "icon": "🏥",
        "color": "#8b5cf6",
        "urgent": False
    },
    {
        "name": "منتدى دعم الصحة النفسية",
        "detail": "مجتمع عربي داعم للصحة النفسية عبر الإنترنت",
        "contact": "sehetnafsiya.com",
        "type": "web",
        "icon": "💬",
        "color": "#06b6d4",
        "urgent": False
    },
    {
        "name": "مستشفى الرازي - الرباط",
        "detail": "أكبر مستشفى نفسي في المغرب",
        "contact": "0537 688 680",
        "type": "phone",
        "icon": "🏨",
        "color": "#f59e0b",
        "urgent": False
    }
]

# ─── Islamic CBT Content ─────────────────────────────────────────────────────
ISLAMIC_SUPPORTS = [
    {
        "verse": "فَإِنَّ مَعَ الْعُسْرِ يُسْرًا • إِنَّ مَعَ الْعُسْرِ يُسْرًا",
        "source": "سورة الإنشراح: ٥-٦",
        "meaning": "مع كل صعوبة يأتي الفرج — بل مرتين في آيتين متتاليتين.",
        "color": "#10b981"
    },
    {
        "verse": "لَا يُكَلِّفُ اللَّهُ نَفْسًا إِلَّا وُسْعَهَا",
        "source": "سورة البقرة: ٢٨٦",
        "meaning": "ما تحمله الآن ضمن طاقتك — أنت أقوى مما تظن.",
        "color": "#8b5cf6"
    },
    {
        "verse": "وَلَا تَيْأَسُوا مِن رَّوْحِ اللَّهِ ۚ إِنَّهُ لَا يَيْأَسُ مِن رَّوْحِ اللَّهِ إِلَّا الْقَوْمُ الْكَافِرُونَ",
        "source": "سورة يوسف: ٨٧",
        "meaning": "لا تفقد الأمل أبداً — رحمة الله واسعة لا حدود لها.",
        "color": "#06b6d4"
    },
    {
        "verse": "أَلَا بِذِكْرِ اللَّهِ تَطْمَئِنُّ الْقُلُوبُ",
        "source": "سورة الرعد: ٢٨",
        "meaning": "الطمأنينة الحقيقية في ذكر الله — تنفس واذكر اسمه.",
        "color": "#f59e0b"
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
