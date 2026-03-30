# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Medical Triage Classifier
  ────────────────────────────────────────────────────────────────────

  Multi-signal triage system for Arabic medical queries.

  Classifies every user input into one of 3 risk levels:

    🔴 EMERGENCY   — Immediate danger (cardiac, stroke, poisoning…)
    🟡 MODERATE    — Needs medical attention soon (caution symptoms)
    🟢 SAFE        — General health question, no immediate danger

  Scoring architecture:
    ┌──────────────────────────────────────────────┐
    │  Signal 1:  Keyword severity scoring (fast)  │
    │  Signal 2:  Cardiac multi-keyword threshold  │
    │  Signal 3:  Symptom combination heuristics   │
    │  Signal 4:  Time-urgency markers             │
    │  Signal 5:  Vital sign thresholds            │
    └──────────────────────────────────────────────┘

  Each signal contributes a weighted score.  The final triage level
  is determined by the aggregate score crossing calibrated thresholds.

  Output: TriageResult dataclass with risk_level, score, actions,
  matched_flags, and recommended_response.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


def _normalise_arabic(text: str) -> str:
    """Normalise Arabic text for robust matching.

    - Strips tashkeel (diacritics)
    - Normalises hamza/alef variants (إ أ آ → ا)
    - Normalises alef maqsura (ى → ي)
    - Collapses whitespace
    """
    if not text:
        return ""
    # Strip tashkeel
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", text)
    # Normalise alef variants
    text = re.sub(r"[إأآا]", "ا", text)
    # Normalise alef maqsura
    text = text.replace("ى", "ي")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═════════════════════════════════════════════════════════════════
#  Risk Levels
# ═════════════════════════════════════════════════════════════════

class RiskLevel(Enum):
    """Medical triage risk classification."""
    EMERGENCY = "emergency"     # 🔴 Call ambulance NOW
    MODERATE  = "moderate"      # 🟡 See a doctor today
    SAFE      = "safe"          # 🟢 General advice OK

    @property
    def emoji(self) -> str:
        return {"emergency": "🔴", "moderate": "🟡", "safe": "🟢"}[self.value]

    @property
    def label_ar(self) -> str:
        return {
            "emergency": "حالة طوارئ",
            "moderate":  "يحتاج متابعة طبية",
            "safe":      "استفسار عام آمن",
        }[self.value]


# ═════════════════════════════════════════════════════════════════
#  Structured Output
# ═════════════════════════════════════════════════════════════════

@dataclass
class TriageResult:
    """Structured triage output.

    Attributes:
        risk_level:   EMERGENCY | MODERATE | SAFE
        score:        Aggregate triage score (0.0 – 1.0).
        actions:      Ordered list of recommended actions (Arabic).
        flags:        List of matched keywords/patterns with their signal source.
        response_ar:  Pre-built Arabic response for the UI.
        metadata:     Signal-level breakdown for debugging/logging.
    """
    risk_level: RiskLevel
    score: float
    actions: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    response_ar: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "risk_level": self.risk_level.value,
            "risk_emoji": self.risk_level.emoji,
            "risk_label_ar": self.risk_level.label_ar,
            "score": round(self.score, 3),
            "actions": self.actions,
            "flags": self.flags,
            "response_ar": self.response_ar,
        }


# ═════════════════════════════════════════════════════════════════
#  Keyword Severity Dictionary
#  Each keyword has a base severity score (0.0 – 1.0)
# ═════════════════════════════════════════════════════════════════

# --- Signal 1: Severity-scored keywords -------------------------

_KEYWORDS: list[tuple[str, float, str]] = [
    # ── CRITICAL (0.8 – 1.0) ────────────────────────────────────
    # Breathing failure
    ("لا أستطيع التنفس",     1.0, "تنفس"),
    ("توقف التنفس",          1.0, "تنفس"),
    ("اختناق",               0.95, "تنفس"),
    ("ضيق تنفس شديد",       0.90, "تنفس"),
    ("صعوبة بالتنفس",        0.85, "تنفس"),
    ("مش قادر اتنفس",        0.95, "تنفس"),
    ("ما اقدر اتنفس",        0.95, "تنفس"),
    # Cardiac
    ("نوبة قلبية",           1.0, "قلب"),
    ("توقف القلب",           1.0, "قلب"),
    ("ألم شديد في الصدر",    0.90, "قلب"),
    ("ألم في صدري مفاجئ",    0.90, "قلب"),
    ("وجع صدر قوي",          0.85, "قلب"),
    # Consciousness
    ("فقدت الوعي",           0.95, "وعي"),
    ("إغماء",                0.85, "وعي"),
    ("غيبوبة",               1.0, "وعي"),
    ("فاقد الوعي",           0.95, "وعي"),
    # Bleeding
    ("نزيف شديد",            0.90, "نزيف"),
    ("نزيف لا يتوقف",        0.92, "نزيف"),
    ("نزيف من الرأس",        0.95, "نزيف"),
    # Stroke
    ("شلل مفاجئ",            0.95, "سكتة"),
    ("فقدت النطق",           0.90, "سكتة"),
    ("لا أستطيع الكلام فجأة", 0.90, "سكتة"),
    ("وجهي مائل",            0.88, "سكتة"),
    ("تنميل نصف الجسم",      0.85, "سكتة"),
    # Poisoning
    ("تسمم",                 0.90, "تسمم"),
    ("ابتلع سم",             0.95, "تسمم"),
    ("شرب كلور",             0.95, "تسمم"),
    ("ابتلع دواء كثير",      0.90, "تسمم"),
    # Trauma
    ("حادث سير",             0.85, "إصابة"),
    ("سقوط من ارتفاع",       0.85, "إصابة"),
    ("كسر مفتوح",            0.90, "إصابة"),
    ("حروق شديدة",           0.88, "إصابة"),
    ("جرح عميق",             0.80, "إصابة"),
    # Suicidal
    ("أريد الانتحار",        1.0, "نفسي"),
    ("أفكار انتحارية",       0.95, "نفسي"),
    ("أريد أموت",            0.95, "نفسي"),
    ("بدي اموت",             0.95, "نفسي"),
    ("نفسي اخلص",            0.90, "نفسي"),

    # ── MODERATE (0.35 – 0.75) ──────────────────────────────────
    ("دم في البول",          0.55, "بولي"),
    ("دم في البراز",         0.55, "هضمي"),
    ("ألم مستمر",            0.45, "ألم"),
    ("فقدان وزن مفاجئ",      0.55, "عام"),
    ("كتلة غريبة",           0.55, "ورم"),
    ("ورم",                  0.45, "ورم"),
    ("ارتفاع حرارة شديد",    0.55, "حرارة"),
    ("حرارة أكثر من 39",     0.55, "حرارة"),
    ("صداع شديد جداً",       0.50, "عصبي"),
    ("صداع شديد جدا",        0.50, "عصبي"),
    ("تشنجات",               0.60, "عصبي"),
    ("صرع",                  0.55, "عصبي"),
    ("ألم في الصدر",         0.55, "قلب"),
    ("خدر في الذراع",        0.55, "قلب"),
    ("تعرق غزير",            0.45, "قلب"),
    ("ألم في الذراع",        0.45, "قلب"),
    ("غثيان مفاجئ",          0.40, "هضمي"),
    ("خفقان شديد",           0.50, "قلب"),
    ("نبضات سريعة",          0.45, "قلب"),
    ("ضيق التنفس",           0.50, "تنفس"),
    ("دوخة شديدة",           0.50, "عصبي"),
    ("تقيؤ مستمر",           0.55, "هضمي"),

    # ── LOW (0.05 – 0.30) ───────────────────────────────────────
    ("صداع",                 0.15, "عصبي"),
    ("سعال",                 0.10, "تنفس"),
    ("زكام",                 0.05, "تنفس"),
    ("إسهال",                0.15, "هضمي"),
    ("إمساك",                0.10, "هضمي"),
    ("حكة",                  0.10, "جلدي"),
    ("ألم خفيف",             0.10, "ألم"),
    ("تعب",                  0.10, "عام"),
    ("أرق",                  0.10, "نفسي"),
]

# --- Signal 2: Cardiac multi-keyword (from safety.py) -----------
_CARDIAC_KW = [
    "ألم في الصدر", "ضيق التنفس", "تعرق غزير",
    "ألم في الذراع", "غثيان مفاجئ", "خفقان شديد",
    "وجع الصدر", "ألم صدري", "نبضات سريعة",
]
_CARDIAC_THRESHOLD = 2   # ≥2 cardiac keywords → emergency

# --- Signal 3: Symptom combinations (synergy rules) ------------
_DANGEROUS_COMBOS: list[tuple[list[str], float, str]] = [
    # chest pain + arm numbness → very likely cardiac
    (["ألم في الصدر", "خدر في الذراع"],             0.30, "اشتباه نوبة قلبية"),
    (["ألم في الصدر", "تعرق غزير"],                 0.25, "اشتباه نوبة قلبية"),
    (["صداع شديد", "تنميل نصف الجسم"],              0.30, "اشتباه سكتة دماغية"),
    (["صداع شديد", "تشوش الرؤية"],                  0.25, "اشتباه سكتة دماغية"),
    (["ارتفاع حرارة", "تشنجات"],                    0.25, "تشنجات حرارية"),
    (["غثيان", "دم في البراز"],                      0.20, "نزيف داخلي محتمل"),
    (["ألم بطن شديد", "حرارة"],                      0.20, "التهاب حاد"),
]

# --- Signal 4: Time-urgency markers ----------------------------
_URGENCY_MARKERS: list[tuple[str, float]] = [
    ("فجأة",       0.15),
    ("مفاجئ",      0.15),
    ("شديد جداً",  0.10),
    ("شديد",       0.08),
    ("لا يتوقف",   0.12),
    ("مستمر",      0.05),
    ("يزداد",      0.05),
    ("ساءت",       0.05),
]

# --- Signal 5: Vital sign patterns (regex) ---------------------
_VITAL_PATTERNS: list[tuple[str, float, str]] = [
    (r"حرارة?\s*(?:أكثر|فوق|تجاوز)\s*(?:من\s*)?(?:ال)?(\d{2})",  0.0, "حرارة"),
    (r"نبض\s*(?:أكثر|فوق)\s*(?:من\s*)?(\d+)",                     0.0, "نبض"),
    (r"ضغط\s*(\d+)\s*/\s*(\d+)",                                   0.0, "ضغط"),
]


# ═════════════════════════════════════════════════════════════════
#  Triage Classifier
# ═════════════════════════════════════════════════════════════════

class MedicalTriageClassifier:
    """Multi-signal Arabic medical triage classifier.

    Aggregates 5 signal types into a composite score, then maps
    the score to a risk level via calibrated thresholds.

    Thresholds (calibrated for Arabic medical queries):
        score ≥ 0.70   →  EMERGENCY
        score ≥ 0.35   →  MODERATE
        score <  0.35   →  SAFE
    """

    EMERGENCY_THRESHOLD = 0.75
    MODERATE_THRESHOLD  = 0.35

    def classify(self, message: str) -> TriageResult:
        """Classify a user message into a medical risk level.

        Args:
            message: Arabic text input from the user.

        Returns:
            TriageResult with risk_level, score, actions, flags, response_ar.
        """
        msg = message.strip()
        if not msg:
            return TriageResult(
                risk_level=RiskLevel.SAFE,
                score=0.0,
                actions=["أدخل وصفاً واضحاً للأعراض"],
                response_ar="يرجى إدخال وصف واضح للأعراض.",
            )

        # Normalise Arabic for robust matching
        msg_norm = _normalise_arabic(msg)

        scores: dict[str, float] = {}
        flags: list[str] = []

        # ── Signal 1: Keyword severity ──────────────────────────
        kw_score = 0.0
        for keyword, severity, category in _KEYWORDS:
            if _normalise_arabic(keyword) in msg_norm:
                kw_score = max(kw_score, severity)
                flags.append(f"[keyword] {keyword} ({severity:.2f}) [{category}]")
        scores["keyword"] = kw_score

        # ── Signal 2: Cardiac multi-keyword threshold ───────────
        cardiac_hits = [kw for kw in _CARDIAC_KW if _normalise_arabic(kw) in msg_norm]
        cardiac_score = 0.0
        if len(cardiac_hits) >= _CARDIAC_THRESHOLD:
            cardiac_score = 0.95
            flags.append(f"[cardiac] {len(cardiac_hits)} مؤشرات قلبية: {', '.join(cardiac_hits)}")
        scores["cardiac"] = cardiac_score

        # ── Signal 3: Symptom combinations ──────────────────────
        combo_score = 0.0
        for combo_kws, bonus, label in _DANGEROUS_COMBOS:
            if all(_normalise_arabic(kw) in msg_norm for kw in combo_kws):
                combo_score = max(combo_score, bonus)
                flags.append(f"[combo] {label}: {' + '.join(combo_kws)}")
        scores["combo"] = combo_score

        # ── Signal 4: Time-urgency markers ──────────────────────
        urgency_score = 0.0
        for marker, bonus in _URGENCY_MARKERS:
            if _normalise_arabic(marker) in msg_norm:
                urgency_score += bonus
                flags.append(f"[urgency] {marker} (+{bonus:.2f})")
        urgency_score = min(urgency_score, 0.30)   # cap
        scores["urgency"] = urgency_score

        # ── Signal 5: Vital sign analysis ───────────────────────
        vital_score = 0.0
        vital_score = self._check_vitals(msg_norm, flags)
        scores["vitals"] = vital_score

        # ── Aggregate ───────────────────────────────────────────
        # Take the max of individual signals, then add combo + urgency bonus
        base = max(scores["keyword"], scores["cardiac"])
        total = min(base + scores["combo"] + scores["urgency"] + scores["vitals"], 1.0)

        # ── Classify ────────────────────────────────────────────
        if total >= self.EMERGENCY_THRESHOLD:
            risk = RiskLevel.EMERGENCY
        elif total >= self.MODERATE_THRESHOLD:
            risk = RiskLevel.MODERATE
        else:
            risk = RiskLevel.SAFE

        actions = self._build_actions(risk, flags)
        response = self._build_response(risk, total, flags)

        return TriageResult(
            risk_level=risk,
            score=total,
            actions=actions,
            flags=flags,
            response_ar=response,
            metadata=scores,
        )

    # ─────────────────────────────────────────────────────────────
    #  Vital signs (Signal 5)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _check_vitals(msg: str, flags: list[str]) -> float:
        """Extract and evaluate vital sign values from free text.

        Note: *msg* should already be Arabic-normalised.
        """
        score = 0.0

        # Temperature > 39°C  (normalised: اكثر not أكثر)
        temp_match = re.search(r"حرار[ةه]?\s*(?:اكثر|فوق|تجاوز)?\s*(?:من\s*)?(\d{2}(?:\.\d)?)", msg)
        if temp_match:
            try:
                temp = float(temp_match.group(1))
                if temp >= 40:
                    score = max(score, 0.40)
                    flags.append(f"[vitals] حرارة {temp}°C — خطيرة")
                elif temp >= 39:
                    score = max(score, 0.25)
                    flags.append(f"[vitals] حرارة {temp}°C — مرتفعة")
            except ValueError:
                pass

        # Heart rate > 120 bpm (normalised Arabic)
        hr_match = re.search(r"نبض\s*(?:اكثر|فوق)?\s*(?:من\s*)?(\d+)", msg)
        if hr_match:
            try:
                hr = int(hr_match.group(1))
                if hr > 150:
                    score = max(score, 0.25)
                    flags.append(f"[vitals] نبض {hr}/دقيقة — خطير")
                elif hr > 120:
                    score = max(score, 0.15)
                    flags.append(f"[vitals] نبض {hr}/دقيقة — مرتفع")
            except ValueError:
                pass

        # Blood pressure (systolic > 180 or < 80)
        bp_match = re.search(r"ضغط\s*(\d+)\s*/\s*(\d+)", msg)
        if bp_match:
            try:
                sys_bp = int(bp_match.group(1))
                if sys_bp > 180 or sys_bp < 80:
                    score = max(score, 0.25)
                    flags.append(f"[vitals] ضغط {bp_match.group(0)} — خطير")
                elif sys_bp > 160 or sys_bp < 90:
                    score = max(score, 0.15)
                    flags.append(f"[vitals] ضغط {bp_match.group(0)} — غير طبيعي")
            except ValueError:
                pass

        return score

    # ─────────────────────────────────────────────────────────────
    #  Action builder
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _build_actions(risk: RiskLevel, flags: list[str]) -> list[str]:
        """Build an ordered list of recommended actions."""
        if risk == RiskLevel.EMERGENCY:
            actions = [
                "🚑 اتصل بالإسعاف فوراً — 📞 15",
                "🏥 توجه لأقرب مستشفى فوراً",
                "❌ لا تتناول أي دواء بدون إشراف طبي",
                "❌ لا تقم بأي مجهود بدني",
            ]
            # Add cardiac-specific actions
            if any("[cardiac]" in f for f in flags):
                actions.insert(1, "💊 إذا لديك أسبرين — تناول حبة واحدة مع الماء")
            return actions

        if risk == RiskLevel.MODERATE:
            return [
                "👨‍⚕️ راجع طبيباً في أقرب وقت",
                "📝 سجّل أعراضك وموعد بدايتها",
                "⚠️ إذا ساءت الأعراض → اتصل بـ 15",
                "💊 لا تأخذ أدوية جديدة بدون استشارة",
            ]

        return [
            "📖 يمكنك الاطلاع على المعلومات الصحية",
            "💡 إذا استمرت الأعراض أكثر من 48 ساعة → راجع طبيباً",
        ]

    # ─────────────────────────────────────────────────────────────
    #  Response builder
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _build_response(risk: RiskLevel, score: float, flags: list[str]) -> str:
        """Build a pre-formatted Arabic response for the UI."""
        if risk == RiskLevel.EMERGENCY:
            return (
                "🚨 **تحذير طبي عاجل**\n\n"
                "الأعراض التي وصفتها قد تشير إلى **حالة طوارئ طبية**.\n\n"
                "━━━━━━━━━━━━━━━━━━━\n"
                "🚑 اتصل بالإسعاف فوراً\n"
                "    📞 15 — SAMU Maroc\n"
                "━━━━━━━━━━━━━━━━━━━\n\n"
                "لا تنتظر. اطلب المساعدة الآن.\n\n"
                "⚠️ هذا النظام لا يستطيع تشخيص حالتك. "
                "فقط الطبيب يمكنه مساعدتك."
            )

        if risk == RiskLevel.MODERATE:
            return (
                "⚠️ **تنبيه: يُنصح بمراجعة طبيب**\n\n"
                "الأعراض التي وصفتها قد تحتاج إلى **تقييم طبي مباشر**.\n\n"
                "📋 **نصائح:**\n"
                "- سجّل أعراضك ومتى بدأت\n"
                "- لاحظ أي تغيرات في الأعراض\n"
                "- لا تؤخر زيارة الطبيب\n\n"
                "📞 إذا ساءت الأعراض → اتصل بـ **15**"
            )

        return ""   # Safe → no special response, let the RAG/LLM handle it


# ═════════════════════════════════════════════════════════════════
#  Integration with SafetyAgent
# ═════════════════════════════════════════════════════════════════

def triage_to_safety_level(result: TriageResult) -> dict:
    """Convert a TriageResult into the SafetyGuard.check() dict format.

    This allows the orchestrator to use either the triage classifier
    or the safety guard interchangeably.

    Returns:
        dict matching SafetyGuard.check() output:
        {
            "level":             "emergency" | "caution" | "safe",
            "override_response": str | None,
            "add_disclaimer":    bool,
            "flags":             list[str],
            "triage_score":      float,
            "triage_risk":       str,
        }
    """
    level_map = {
        RiskLevel.EMERGENCY: "emergency",
        RiskLevel.MODERATE:  "caution",
        RiskLevel.SAFE:      "safe",
    }

    override = result.response_ar if result.risk_level != RiskLevel.SAFE else None

    return {
        "level": level_map[result.risk_level],
        "override_response": override,
        "add_disclaimer": True,
        "flags": result.flags,
        "triage_score": result.score,
        "triage_risk": result.risk_level.value,
        "actions": result.actions,
    }
