# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  Tests — Safety Guard Module
  Validates emergency detection, boundary enforcement, and post-checks.
═══════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.safety import SafetyGuard


guard = SafetyGuard()


class TestEmergencyDetection:
    """Test that critical emergency keywords are caught."""

    def test_breathing_emergency(self):
        result = guard.check("لا أستطيع التنفس")
        assert result["level"] == "emergency"
        assert result["override_response"] is not None

    def test_chest_pain_emergency(self):
        result = guard.check("ألم شديد في الصدر")
        assert result["level"] == "emergency"

    def test_suicide_ideation(self):
        result = guard.check("أريد الانتحار")
        assert result["level"] == "emergency"

    def test_severe_bleeding(self):
        result = guard.check("نزيف شديد")
        assert result["level"] == "emergency"


class TestBoundaryEnforcement:
    """Test that prescription/diagnosis requests are blocked."""

    def test_prescription_request(self):
        result = guard.check("اكتب لي وصفة دواء")
        assert result["level"] == "boundary"

    def test_dosage_request(self):
        result = guard.check("حدد لي الجرعة المناسبة")
        assert result["level"] == "boundary"


class TestSafeMessages:
    """Test that normal medical questions pass through safely."""

    def test_general_health_question(self):
        result = guard.check("ما هي فوائد شرب الماء؟")
        assert result["level"] == "safe"

    def test_symptom_description(self):
        result = guard.check("أشعر بصداع خفيف")
        assert result["level"] in ("safe", "caution")

    def test_nutrition_question(self):
        result = guard.check("ما هو النظام الغذائي الصحي؟")
        assert result["level"] == "safe"


class TestPostCheck:
    """Test that dangerous LLM outputs are caught."""

    def test_safe_response_passes(self):
        response = "يُنصح بشرب الماء والراحة. استشر طبيبك إذا استمرت الأعراض."
        result = guard.post_check(response)
        assert result == response  # unchanged

    def test_empty_input(self):
        result = guard.check("")
        assert result["level"] == "safe"
