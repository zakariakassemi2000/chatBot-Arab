"""
═══════════════════════════════════════════════════════════════════════
  الخطوة 4: الحارس (مرشحات السلامة)
  Step 4: The Guardian (Safety Filters)

  Critical safety layer for a medical assistant. Detects emergencies,
  enforces disclaimers, and prevents the bot from overstepping its
  competence boundaries.
═══════════════════════════════════════════════════════════════════════
"""

import re

# ═══════════════════════════════════════════════════════════════════
#  🔴 RED FLAGS — Immediate emergency keywords
# ═══════════════════════════════════════════════════════════════════
EMERGENCY_KEYWORDS = [
    # Breathing
    "لا أستطيع التنفس", "ضيق تنفس شديد", "اختناق", "توقف التنفس",
    "صعوبة بالتنفس", "مش قادر اتنفس", "ما اقدر اتنفس",
    # Cardiac
    "ألم شديد في الصدر", "نوبة قلبية", "توقف القلب",
    "ألم في صدري مفاجئ", "وجع صدر قوي",
    # Bleeding
    "نزيف شديد", "نزيف لا يتوقف", "نزف حاد",
    "نزيف من الرأس", "نزيف دموي",
    # Consciousness
    "فقدت الوعي", "إغماء", "غيبوبة", "فاقد الوعي",
    "أغمي عليه", "مش واعي",
    # Poisoning
    "تسمم", "ابتلع سم", "ابتلع مادة", "شرب كلور",
    "ابتلع دواء كثير",
    # Stroke
    "شلل مفاجئ", "فقدت النطق", "لا أستطيع الكلام فجأة",
    "وجهي مائل", "تنميل نصف الجسم",
    # Severe trauma
    "حادث سير", "سقوط من ارتفاع", "كسر مفتوح",
    "حروق شديدة", "جرح عميق",
    # Suicidal ideation
    "أفكار انتحارية", "أريد الانتحار", "أريد أموت",
    "بدي اموت", "نفسي اخلص",
]

# ═══════════════════════════════════════════════════════════════════
#  ❤️ CARDIAC EMERGENCY — Multi-keyword threshold detection
# ═══════════════════════════════════════════════════════════════════
CARDIAC_KEYWORDS = [
    "ألم في الصدر", "ضيق التنفس", "تعرق غزير",
    "ألم في الذراع", "غثيان مفاجئ", "خفقان شديد",
    "وجع الصدر", "ألم صدري", "نبضات سريعة",
]

CARDIAC_EMERGENCY_RESPONSE = """
🚨 تحذير طبي عاجل

الأعراض التي وصفتها قد تشير إلى أزمة قلبية حادة.

━━━━━━━━━━━━━━━━━━━━━━━
🚑 اتصل بالإسعاف فوراً
    📞 15 — SAMU Maroc
━━━━━━━━━━━━━━━━━━━━━━━

لا تنتظر. لا تقد السيارة. لا تأخذ دواء.
اطلب المساعدة الآن.

⚠️ هذا النظام لا يستطيع تشخيص حالتك.
   فقط الطبيب في المستشفى يمكنه مساعدتك.
"""

# ═══════════════════════════════════════════════════════════════════
#  🟡 YELLOW FLAGS — Caution keywords (recommend seeing a doctor)
# ═══════════════════════════════════════════════════════════════════
CAUTION_KEYWORDS = [
    "دم في البول", "دم في البراز", "ألم مستمر",
    "فقدان وزن مفاجئ", "كتلة غريبة", "ورم",
    "ارتفاع حرارة شديد", "حرارة أكثر من 39",
    "صداع شديد جداً", "تشنجات", "صرع",
    "ألم في الصدر", "خدر في الذراع",
]

# ═══════════════════════════════════════════════════════════════════
#  🚫 COMPETENCE BOUNDARIES — Things the bot should NEVER do
# ═══════════════════════════════════════════════════════════════════
FORBIDDEN_PATTERNS = [
    r"اكتب\s*لي\s*وصفة",     # "Write me a prescription"
    r"حدد\s*لي\s*الجرعة",    # "Determine the dosage for me"
    r"هل\s*عندي\s*سرطان",    # "Do I have cancer?"
    r"شخص\s*لي",             # "Diagnose me"
    r"أنا\s*مصاب\s*ب",       # "I'm infected with..."
]


# ═══════════════════════════════════════════════════════════════════
#  Emergency response messages
# ═══════════════════════════════════════════════════════════════════

EMERGENCY_RESPONSE = """🚨 **تنبيه: حالة طوارئ محتملة!**

بناءً على ما ذكرته، قد تكون حالتك تستدعي **تدخلاً طبياً عاجلاً**.

### 🏥 ما يجب فعله الآن:

1. **اتصل بالطوارئ فوراً** — في المغرب: **141** أو **15**
2. **توجه لأقرب مستشفى أو مركز صحي**
3. **لا تتناول أي دواء** دون إشراف طبي
4. **لا تقم بمجهود بدني**

### ⚠️ أرقام الطوارئ:
- 🇲🇦 المغرب: **141** (SAMU) / **15** (Protection Civile)
- 🇸🇦 السعودية: **997** (الطوارئ) / **937** (صحة)
- 🇪🇬 مصر: **123** (الإسعاف)
- 🇦🇪 الإمارات: **998** / **999**

> ⚕️ هذا المساعد لا يمكنه تقديم مساعدة في حالات الطوارئ. يُرجى الاتصال بالمختصين فوراً.
"""

CAUTION_RESPONSE = """⚠️ **تنبيه: يُنصح بمراجعة طبيب**

الأعراض التي وصفتها قد تحتاج إلى **تقييم طبي مباشر**.

ما يمكنني تقديمه هو معلومات عامة فقط، لكن حالتك تستحق فحصاً من طبيب مختص.

📋 **نصائح:**
- سجّل أعراضك ومتى بدأت
- لاحظ أي تغيرات في الأعراض
- لا تؤخر زيارة الطبيب

"""

BOUNDARY_RESPONSE = """⚕️ **حدود الخدمة**

عذراً، لا أستطيع **تشخيص الأمراض** أو **وصف الأدوية** أو **تحديد الجرعات**.

هذا المساعد مصمم لتقديم **معلومات صحية عامة** و**توجيهك** للمختص المناسب فقط.

👨‍⚕️ يُرجى مراجعة طبيب مؤهل لأي:
- تشخيص طبي
- وصفة دوائية
- تحديد جرعات
- تفسير تحاليل أو أشعة

"""


# ═══════════════════════════════════════════════════════════════════
#  🔍 POST-LLM CHECK — Patterns to catch in LLM-generated responses
# ═══════════════════════════════════════════════════════════════════
POST_LLM_FORBIDDEN = [
    # Prescription indicators
    r"أصف\s+لك",
    r"الجرعة\s+هي",
    r"خذ\s+\d+\s+ملغ",
    r"تناول\s+\d+",
    r"وصفة\s+طبية",
    # Hard diagnosis
    r"مصاب\s+بـ?",
    r"لديك\s+مرض",
    r"تشخيصك\s+هو",
]

POST_LLM_REPLACEMENT = (
    "\n\n> ⚕️ *تذكير: هذه المعلومات للتوعية الصحية العامة فقط."
    " يُرجى استشارة طبيب مختص قبل اتخاذ أي قرار طبي.*"
)


class SafetyGuard:
    """
    Multi-layer safety system for the Arabic medical chatbot.
    
    Layers:
      1. Emergency detection  → Redirect to emergency services
      2. Caution detection    → Recommend seeing a doctor
      3. Boundary enforcement → Refuse diagnosis / prescription
      4. Disclaimer injection → Always add medical disclaimer
    """

    URGENCE_THRESHOLD = 2  # 2 cardiac keywords = emergency

    def __init__(self):
        # Compile forbidden patterns for speed
        self._forbidden_re = [re.compile(p) for p in FORBIDDEN_PATTERNS]

    def check(self, user_message: str) -> dict:
        """
        Run all safety checks on the user's message.
        
        Returns:
            {
                "level": "safe" | "emergency" | "caution" | "boundary",
                "override_response": str | None,
                "add_disclaimer": bool,
                "flags": list of triggered keywords,
            }
        """
        msg = user_message.strip()
        result = {
            "level": "safe",
            "override_response": None,
            "add_disclaimer": True,
            "flags": [],
        }

        # Layer 0: CARDIAC EMERGENCY (HIGHEST PRIORITY — threshold-based)
        cardiac_emergency, cardiac_flags = self.detect_emergency(msg)
        if cardiac_emergency:
            result["level"] = "emergency"
            result["emergency"] = True
            result["override_response"] = CARDIAC_EMERGENCY_RESPONSE
            result["flags"] = cardiac_flags
            return result

        # Layer 1: Emergency detection (single keyword match)
        for kw in EMERGENCY_KEYWORDS:
            if kw in msg:
                result["level"] = "emergency"
                result["override_response"] = EMERGENCY_RESPONSE
                result["flags"].append(f"🚨 {kw}")
                return result  # Immediately return, don't process further

        # Layer 2: Boundary enforcement
        for pattern in self._forbidden_re:
            if pattern.search(msg):
                result["level"] = "boundary"
                result["override_response"] = BOUNDARY_RESPONSE
                result["flags"].append(f"🚫 boundary: {pattern.pattern}")
                return result

        # Layer 3: Caution detection
        for kw in CAUTION_KEYWORDS:
            if kw in msg:
                result["level"] = "caution"
                result["flags"].append(f"⚠️ {kw}")
                # Don't override response, but add caution prefix later

        return result

    def detect_emergency(self, text: str) -> tuple:
        """
        Détecte une urgence cardiaque par seuil de mots-clés.
        Si >= URGENCE_THRESHOLD mots-clés cardiaques sont détectés,
        c'est une urgence — ZERO appel LLM.
        """
        flags = []
        for kw in CARDIAC_KEYWORDS:
            if kw in text:
                flags.append(f"❤️‍🔥 {kw}")
        return len(flags) >= self.URGENCE_THRESHOLD, flags

    @staticmethod
    def add_disclaimer(response: str) -> str:
        """Add a medical disclaimer to any bot response."""
        disclaimer = "\n\n---\n> ⚕️ *هذه المعلومات للتوعية فقط ولا تُغني عن استشارة طبيب مختص.*"
        if "هذه المعلومات للتوعية" not in response:
            return response + disclaimer
        return response

    @staticmethod
    def format_caution_response(answer: str) -> str:
        """Prepend caution notice to a normal answer."""
        return CAUTION_RESPONSE + "\n" + answer

    @staticmethod
    def post_check(response: str) -> str:
        """
        Scan the LLM-generated response for forbidden patterns
        (prescriptions, hard diagnoses) and append a safety notice if found.

        This is a secondary safety net in case the LLM ignores its system prompt.
        It does NOT remove the content, but flags it visibly.
        """
        import re
        if not response:
            return response

        for pattern in POST_LLM_FORBIDDEN:
            if re.search(pattern, response):
                # Append warning if not already present
                if "تذكير: هذه المعلومات" not in response:
                    response += POST_LLM_REPLACEMENT
                break

        return response
