# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  Tests — Medical Triage Classifier (50+ test cases)
  Validates all 5 signals + thresholds + edge cases.
═══════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.triage import MedicalTriageClassifier, RiskLevel, TriageResult


@pytest.fixture(scope="module")
def clf():
    return MedicalTriageClassifier()


# ═══════════════════════════════════════════════════════════════════
#  Signal 1: Keyword Severity — EMERGENCY (score ≥ 0.75)
# ═══════════════════════════════════════════════════════════════════

class TestEmergencyKeywords:
    """Critical keywords that MUST trigger EMERGENCY."""

    @pytest.mark.parametrize("msg", [
        "لا أستطيع التنفس",
        "توقف التنفس عند أخي",
        "أبي يعاني من نوبة قلبية",
        "توقف القلب فجأة",
        "فقدت الوعي في الشارع",
        "غيبوبة منذ ساعة",
        "نزيف شديد لا يتوقف",
        "نزيف من الرأس بعد سقوط",
        "شلل مفاجئ في الجهة اليمنى",
        "تسمم بعد شرب كلور",
        "ابتلع سم",
        "أريد الانتحار",
        "أريد أموت",
        "أفكار انتحارية مستمرة",
    ])
    def test_critical_keywords_emergency(self, clf, msg):
        result = clf.classify(msg)
        assert result.risk_level == RiskLevel.EMERGENCY, \
            f"'{msg}' should be EMERGENCY, got {result.risk_level.value} (score={result.score:.3f})"

    def test_emergency_has_actions(self, clf):
        result = clf.classify("لا أستطيع التنفس")
        assert len(result.actions) >= 2
        assert any("15" in a for a in result.actions)

    def test_emergency_has_response(self, clf):
        result = clf.classify("نوبة قلبية")
        assert result.response_ar != ""
        assert "طوارئ" in result.response_ar or "15" in result.response_ar


# ═══════════════════════════════════════════════════════════════════
#  Signal 1: Keyword Severity — MODERATE (0.35 ≤ score < 0.75)
# ═══════════════════════════════════════════════════════════════════

class TestModerateKeywords:
    """Moderate-severity keywords → MODERATE risk."""

    @pytest.mark.parametrize("msg", [
        "دم في البول منذ يومين",
        "دم في البراز",
        "فقدان وزن مفاجئ بدون سبب",
        "لاحظت كتلة غريبة في رقبتي",
        "ارتفاع حرارة شديد وتعب",
        "ألم في الصدر مع تعرق",
        "تشنجات مستمرة",
        "صداع شديد جداً",
    ])
    def test_moderate_keywords(self, clf, msg):
        result = clf.classify(msg)
        assert result.risk_level in (RiskLevel.MODERATE, RiskLevel.EMERGENCY), \
            f"'{msg}' should be MODERATE+, got {result.risk_level.value} (score={result.score:.3f})"


# ═══════════════════════════════════════════════════════════════════
#  Signal 1: Keyword Severity — SAFE (score < 0.35)
# ═══════════════════════════════════════════════════════════════════

class TestSafeKeywords:
    """Low-severity or no keywords → SAFE risk."""

    @pytest.mark.parametrize("msg", [
        "ما هي فوائد شرب الماء؟",
        "أريد معلومات عن التغذية الصحية",
        "كيف أحسن نومي؟",
        "ما هو النظام الغذائي المتوازن؟",
        "نصائح للوقاية من البرد",
        "كم ساعة نوم يحتاج الإنسان؟",
        "ما هي فوائد المشي؟",
    ])
    def test_safe_queries(self, clf, msg):
        result = clf.classify(msg)
        assert result.risk_level == RiskLevel.SAFE, \
            f"'{msg}' should be SAFE, got {result.risk_level.value} (score={result.score:.3f})"


# ═══════════════════════════════════════════════════════════════════
#  Signal 2: Cardiac Multi-Keyword Threshold
# ═══════════════════════════════════════════════════════════════════

class TestCardiacThreshold:
    """≥2 cardiac keywords → EMERGENCY via cardiac signal."""

    def test_two_cardiac_keywords(self, clf):
        msg = "عندي ألم في الصدر مع ضيق التنفس"
        result = clf.classify(msg)
        assert result.risk_level == RiskLevel.EMERGENCY
        assert any("[cardiac]" in f for f in result.flags)

    def test_three_cardiac_keywords(self, clf):
        msg = "ألم في الصدر وتعرق غزير وغثيان مفاجئ"
        result = clf.classify(msg)
        assert result.risk_level == RiskLevel.EMERGENCY

    def test_one_cardiac_not_emergency(self, clf):
        """Single cardiac keyword alone should NOT trigger cardiac emergency."""
        msg = "خفقان شديد"
        result = clf.classify(msg)
        # Should be moderate, not emergency via cardiac
        assert not any("[cardiac]" in f for f in result.flags)


# ═══════════════════════════════════════════════════════════════════
#  Signal 3: Dangerous Symptom Combinations
# ═══════════════════════════════════════════════════════════════════

class TestSymptomCombos:
    """Dangerous symptom combinations add bonus score."""

    def test_chest_arm_combo(self, clf):
        msg = "ألم في الصدر وخدر في الذراع"
        result = clf.classify(msg)
        assert result.risk_level == RiskLevel.EMERGENCY
        assert any("[combo]" in f for f in result.flags)

    def test_headache_paralysis_combo(self, clf):
        msg = "صداع شديد مع تنميل نصف الجسم"
        result = clf.classify(msg)
        assert result.risk_level == RiskLevel.EMERGENCY


# ═══════════════════════════════════════════════════════════════════
#  Signal 4: Time-Urgency Markers
# ═══════════════════════════════════════════════════════════════════

class TestUrgencyMarkers:
    """Urgency markers (فجأة, مفاجئ, شديد) add bonus score."""

    def test_sudden_onset_boosts_score(self, clf):
        result_normal = clf.classify("صداع")
        result_sudden = clf.classify("صداع فجأة")
        assert result_sudden.score > result_normal.score

    def test_severity_boosts_score(self, clf):
        result_normal = clf.classify("ألم")
        result_severe = clf.classify("ألم شديد جداً يزداد")
        assert result_severe.score > result_normal.score


# ═══════════════════════════════════════════════════════════════════
#  Signal 5: Vital Signs
# ═══════════════════════════════════════════════════════════════════

class TestVitalSigns:
    """Vital sign values extracted from text and scored."""

    def test_high_temperature(self, clf):
        result = clf.classify("حرارة 40 درجة")
        assert result.score > 0
        assert any("[vitals]" in f for f in result.flags)

    def test_dangerous_blood_pressure(self, clf):
        result = clf.classify("ضغط 190/120")
        assert any("[vitals]" in f for f in result.flags)

    def test_high_heart_rate(self, clf):
        result = clf.classify("نبض أكثر من 160")
        assert any("[vitals]" in f for f in result.flags)

    def test_normal_temperature_no_vital_flag(self, clf):
        result = clf.classify("حرارة 37 درجة")
        assert not any("[vitals]" in f for f in result.flags)


# ═══════════════════════════════════════════════════════════════════
#  Edge Cases & Robustness
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases: empty, diacritics, mixed text, very long."""

    def test_empty_input(self, clf):
        result = clf.classify("")
        assert result.risk_level == RiskLevel.SAFE
        assert result.score == 0.0

    def test_whitespace_only(self, clf):
        result = clf.classify("   ")
        assert result.risk_level == RiskLevel.SAFE

    def test_text_with_diacritics(self, clf):
        """Arabic with tashkeel should still match after normalization."""
        result = clf.classify("لاَ أَسْتَطِيعُ التَّنَفُّسَ")
        assert result.risk_level == RiskLevel.EMERGENCY

    def test_alef_variants(self, clf):
        """إ/أ/آ should normalize to ا for matching."""
        result = clf.classify("إغماء مفاجئ")
        assert result.risk_level in (RiskLevel.EMERGENCY, RiskLevel.MODERATE)

    def test_result_to_dict(self, clf):
        result = clf.classify("صداع")
        d = result.to_dict()
        assert "risk_level" in d
        assert "score" in d
        assert "actions" in d
        assert isinstance(d["score"], float)

    def test_score_capped_at_one(self, clf):
        """Even with many signals, score should not exceed 1.0."""
        msg = "ألم شديد في الصدر مع ضيق التنفس وتعرق غزير وخدر في الذراع فجأة مفاجئ لا يتوقف"
        result = clf.classify(msg)
        assert result.score <= 1.0


# ═══════════════════════════════════════════════════════════════════
#  Non-Regression: CRITICAL safety guarantee
# ═══════════════════════════════════════════════════════════════════

class TestNonRegression:
    """These tests MUST NEVER fail — life-safety critical."""

    LIFE_THREATENING = [
        "لا أستطيع التنفس",
        "نوبة قلبية",
        "أريد الانتحار",
        "نزيف شديد من الرأس",
        "تسمم بمادة كيميائية",
        "غيبوبة",
        "ألم شديد في الصدر مع ضيق التنفس",
    ]

    @pytest.mark.parametrize("msg", LIFE_THREATENING)
    def test_life_threatening_never_safe(self, clf, msg):
        """Life-threatening messages MUST NEVER be classified as SAFE."""
        result = clf.classify(msg)
        assert result.risk_level != RiskLevel.SAFE, \
            f"CRITICAL: '{msg}' was classified as SAFE! This is a patient safety violation."
