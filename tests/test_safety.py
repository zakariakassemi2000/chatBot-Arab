# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  Tests — Safety Guard Module (Reinforced)
  77 test cases covering emergency, boundary, caution, post-check,
  edge cases, non-regression, and cardiac threshold detection.
═══════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.safety import SafetyGuard


@pytest.fixture(scope="module")
def guard():
    return SafetyGuard()


# ═══════════════════════════════════════════════════════════════════
#  Layer 1: Emergency Detection (single keyword)
# ═══════════════════════════════════════════════════════════════════

class TestEmergencyDetection:
    """Critical emergency keywords MUST be caught."""

    @pytest.mark.parametrize("msg", [
        "لا أستطيع التنفس",
        "ضيق تنفس شديد",
        "اختناق",
        "ألم شديد في الصدر",
        "نوبة قلبية",
        "توقف القلب",
        "نزيف شديد",
        "نزيف لا يتوقف",
        "نزيف من الرأس",
        "فقدت الوعي",
        "إغماء",
        "غيبوبة",
        "تسمم",
        "ابتلع سم",
        "شلل مفاجئ",
        "فقدت النطق",
        "حادث سير",
        "كسر مفتوح",
        "حروق شديدة",
        "أريد الانتحار",
        "أفكار انتحارية",
        "أريد أموت",
    ])
    def test_emergency_detected(self, guard, msg):
        result = guard.check(msg)
        assert result["level"] == "emergency", \
            f"'{msg}' should trigger emergency, got {result['level']}"
        assert result["override_response"] is not None

    def test_emergency_has_numbers(self, guard):
        result = guard.check("لا أستطيع التنفس")
        assert "15" in result["override_response"]


# ═══════════════════════════════════════════════════════════════════
#  Layer 0: Cardiac Multi-Keyword Emergency
# ═══════════════════════════════════════════════════════════════════

class TestCardiacEmergency:
    """≥2 cardiac keywords → cardiac emergency (highest priority)."""

    def test_two_cardiac_keywords(self, guard):
        msg = "عندي ألم في الصدر وضيق في التنفس"
        result = guard.check(msg)
        assert result["level"] == "emergency"

    def test_three_cardiac_keywords(self, guard):
        msg = "ألم في الصدر مع تعرق غزير وغثيان مفاجئ"
        result = guard.check(msg)
        assert result["level"] == "emergency"

    def test_detect_emergency_method(self, guard):
        is_emergency, flags = guard.detect_emergency("ألم في الصدر وضيق في التنفس")
        assert is_emergency is True
        assert len(flags) >= 2


# ═══════════════════════════════════════════════════════════════════
#  Layer 2: Boundary Enforcement
# ═══════════════════════════════════════════════════════════════════

class TestBoundaryEnforcement:
    """Prescription/diagnosis requests MUST be blocked."""

    @pytest.mark.parametrize("msg", [
        "اكتب لي وصفة دواء",
        "حدد لي الجرعة المناسبة",
        "هل عندي سرطان؟",
        "شخص لي حالتي",
    ])
    def test_boundary_blocked(self, guard, msg):
        result = guard.check(msg)
        assert result["level"] == "boundary", \
            f"'{msg}' should be blocked as boundary, got {result['level']}"
        assert result["override_response"] is not None


# ═══════════════════════════════════════════════════════════════════
#  Layer 3: Caution Detection
# ═══════════════════════════════════════════════════════════════════

class TestCautionDetection:
    """Caution keywords should flag but not override."""

    @pytest.mark.parametrize("msg", [
        "دم في البول",
        "كتلة غريبة في رقبتي",
        "ارتفاع حرارة شديد",
        "تشنجات",
    ])
    def test_caution_flagged(self, guard, msg):
        result = guard.check(msg)
        assert result["level"] == "caution", \
            f"'{msg}' should be caution, got {result['level']}"
        assert result["override_response"] is None  # No override for caution


# ═══════════════════════════════════════════════════════════════════
#  Safe Messages
# ═══════════════════════════════════════════════════════════════════

class TestSafeMessages:
    """Normal health questions should pass through safely."""

    @pytest.mark.parametrize("msg", [
        "ما هي فوائد شرب الماء؟",
        "ما هو النظام الغذائي الصحي؟",
        "كيف أحسن نومي؟",
        "كم ساعة نوم يحتاج الإنسان؟",
        "نصائح للوقاية من البرد",
    ])
    def test_safe_passes(self, guard, msg):
        result = guard.check(msg)
        assert result["level"] == "safe"
        assert result["override_response"] is None
        assert result["add_disclaimer"] is True


# ═══════════════════════════════════════════════════════════════════
#  Post-LLM Check
# ═══════════════════════════════════════════════════════════════════

class TestPostCheck:
    """Post-check catches dangerous LLM outputs."""

    def test_safe_response_unchanged(self, guard):
        response = "يُنصح بشرب الماء والراحة. استشر طبيبك إذا استمرت الأعراض."
        result = guard.post_check(response)
        assert result == response

    def test_prescription_caught(self, guard):
        response = "أصف لك دواء باراسيتامول 500 ملغ"
        result = guard.post_check(response)
        assert "تذكير" in result
        assert len(result) > len(response)

    def test_dosage_caught(self, guard):
        response = "الجرعة هي 500 ملغ كل 8 ساعات"
        result = guard.post_check(response)
        assert "تذكير" in result

    def test_diagnosis_caught(self, guard):
        response = "تشخيصك هو التهاب المعدة"
        result = guard.post_check(response)
        assert "تذكير" in result

    def test_empty_response(self, guard):
        result = guard.post_check("")
        assert result == ""

    def test_none_response(self, guard):
        result = guard.post_check(None)
        assert result is None

    def test_no_duplicate_disclaimer(self, guard):
        """Post-check should not add duplicate disclaimers."""
        response = "معلومات عامة. تذكير: هذه المعلومات للتوعية الصحية."
        result = guard.post_check(response)
        assert result.count("تذكير") == 1


# ═══════════════════════════════════════════════════════════════════
#  Disclaimer Injection
# ═══════════════════════════════════════════════════════════════════

class TestDisclaimer:
    """Disclaimer is added exactly once."""

    def test_disclaimer_added(self, guard):
        result = SafetyGuard.add_disclaimer("نتيجة التحليل: حالة طبيعية")
        assert "هذا التحليل آلي" in result

    def test_no_duplicate_disclaimer(self, guard):
        text = "نتيجة التحليل. هذا التحليل آلي — استشر طبيباً مختصاً."
        result = SafetyGuard.add_disclaimer(text)
        assert result.count("هذا التحليل آلي") == 1


# ═══════════════════════════════════════════════════════════════════
#  Edge Cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_empty_input(self, guard):
        result = guard.check("")
        assert result["level"] == "safe"

    def test_very_long_input_blocked(self, guard):
        """Payload >1200 chars should be blocked."""
        msg = "أ" * 1500
        result = guard.check(msg)
        assert result["level"] == "blocked"

    def test_mixed_arabic_french(self, guard):
        """Mixed language input with emergency keyword."""
        result = guard.check("J'ai un ألم شديد في الصدر sévère")
        assert result["level"] == "emergency"

    def test_format_response_critique(self, guard):
        result = guard.format_response("تحليل طبي", "critique")
        assert "15" in result

    def test_format_response_elevee(self, guard):
        result = guard.format_response("تحليل طبي", "élevée")
        assert "طبيب" in result


# ═══════════════════════════════════════════════════════════════════
#  NON-REGRESSION: CRITICAL SAFETY GUARANTEES
# ═══════════════════════════════════════════════════════════════════

class TestNonRegression:
    """These tests MUST NEVER FAIL — patient safety depends on them."""

    MUST_BE_EMERGENCY = [
        "لا أستطيع التنفس",
        "نوبة قلبية",
        "أريد الانتحار",
        "نزيف شديد",
        "غيبوبة",
        "ابتلع سم",
        "ألم في الصدر وضيق في التنفس",
    ]

    @pytest.mark.parametrize("msg", MUST_BE_EMERGENCY)
    def test_emergency_never_passes_as_safe(self, guard, msg):
        result = guard.check(msg)
        assert result["level"] != "safe", \
            f"CRITICAL SAFETY FAILURE: '{msg}' was classified as safe!"

    MUST_BE_BOUNDARY = [
        "اكتب لي وصفة دواء",
        "حدد لي الجرعة المناسبة",
    ]

    @pytest.mark.parametrize("msg", MUST_BE_BOUNDARY)
    def test_boundary_never_passes(self, guard, msg):
        result = guard.check(msg)
        assert result["level"] in ("boundary", "emergency"), \
            f"BOUNDARY FAILURE: '{msg}' was not blocked!"
