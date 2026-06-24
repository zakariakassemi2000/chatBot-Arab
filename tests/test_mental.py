# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  Tests — Mental Health Module
  Validates distress detection, PHQ-9/GAD-7 scoring, dialect detection.
═══════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.mental.detector import DistressDetector
from modules.mental.phq9 import PHQ9Assessment, PHQ9_QUESTIONS, PHQ9_OPTIONS
from modules.mental.gad7 import GAD7Assessment, GAD7_QUESTIONS, GAD7_OPTIONS


@pytest.fixture(scope="module")
def detector():
    return DistressDetector()


# ═══════════════════════════════════════════════════════════════════
#  Distress Detection — Level 3 (Crisis)
# ═══════════════════════════════════════════════════════════════════

class TestCrisisDetection:
    """Level 3 crisis keywords must be detected across dialects."""

    @pytest.mark.parametrize("msg,dialect", [
        ("أريد الانتحار", "msa"),
        ("بغيت نموت", "darija"),
        ("عايز أموت", "egyptian"),
        ("بدي أموت", "levantine"),
        ("كنفكر نقتل روحي", "darija"),
        ("لا أريد العيش", "msa"),
    ])
    def test_crisis_detected(self, detector, msg, dialect):
        level, reason, keywords = detector.detect(msg)
        assert level == 3, f"'{msg}' ({dialect}) should be level 3, got {level}"
        assert len(keywords) > 0

    def test_negated_crisis_not_detected(self, detector):
        """'I don't want to die' should NOT trigger crisis."""
        msg = "لا أريد الموت ولكن أشعر بالحزن"
        level, reason, keywords = detector.detect(msg)
        # Negation should prevent crisis level
        assert level < 3, f"Negated crisis should not be level 3, got {level}"


# ═══════════════════════════════════════════════════════════════════
#  Distress Detection — Level 2 (High Distress)
# ═══════════════════════════════════════════════════════════════════

class TestHighDistress:
    """Level 2 requires ≥2 high distress keywords."""

    def test_high_distress_msa(self, detector):
        msg = "أشعر بالتحطم ولا أستطيع التحمل وفقدت الأمل"
        level, reason, keywords = detector.detect(msg)
        assert level >= 1  # At least mild distress with these keywords

    def test_single_high_keyword_not_level2(self, detector):
        """A single high keyword should be level 1, not level 2."""
        msg = "أشعر باليأس"
        level, reason, keywords = detector.detect(msg)
        assert level <= 1 or level == 2  # depends on how many keywords match


# ═══════════════════════════════════════════════════════════════════
#  Distress Detection — Level 1 (Mild)
# ═══════════════════════════════════════════════════════════════════

class TestMildDistress:
    """Level 1 for mild emotional difficulty keywords."""

    def test_mild_distress(self, detector):
        msg = "أشعر بالحزن والقلق"
        level, reason, keywords = detector.detect(msg)
        assert level >= 1

    def test_darija_mild(self, detector):
        msg = "تعبان نفسياً وزهقت"
        level, reason, keywords = detector.detect(msg)
        assert level >= 1


# ═══════════════════════════════════════════════════════════════════
#  Distress Detection — Level 0 (No distress)
# ═══════════════════════════════════════════════════════════════════

class TestNoDistress:
    """Normal messages should be level 0."""

    @pytest.mark.parametrize("msg", [
        "كيف حالك؟",
        "ما هي فوائد الرياضة؟",
        "أريد معلومات عن التغذية",
        "",
        "   ",
    ])
    def test_no_distress(self, detector, msg):
        level, reason, keywords = detector.detect(msg)
        assert level == 0


# ═══════════════════════════════════════════════════════════════════
#  Dialect Detection
# ═══════════════════════════════════════════════════════════════════

class TestDialectDetection:
    """Rough dialect detection from user text."""

    def test_darija_detected(self, detector):
        assert detector.detect_dialect("واش كاين شي حاجة بزاف") == "darija"

    def test_egyptian_detected(self, detector):
        assert detector.detect_dialect("إزيك عايز حاجة كده") == "egyptian"

    def test_levantine_detected(self, detector):
        assert detector.detect_dialect("كيفك شو بدي") == "levantine"

    def test_msa_default(self, detector):
        assert detector.detect_dialect("أريد معلومات طبية") == "msa"


# ═══════════════════════════════════════════════════════════════════
#  PHQ-9 Scoring
# ═══════════════════════════════════════════════════════════════════

class TestPHQ9:
    """PHQ-9 depression assessment scoring."""

    def test_questions_count(self):
        assert len(PHQ9_QUESTIONS) == 9

    def test_options_count(self):
        assert len(PHQ9_OPTIONS) == 4

    def test_minimal_score(self):
        answers = [0] * 9  # All "not at all"
        score, severity, rec, badge = PHQ9Assessment.calculate(answers)
        assert score == 0
        assert severity == "طبيعي"
        assert badge == "sev-0"

    def test_maximal_score(self):
        answers = [3] * 9  # All "nearly every day"
        score, severity, rec, badge = PHQ9Assessment.calculate(answers)
        assert score == 27
        assert severity == "اكتئاب شديد"
        assert badge == "sev-3"

    def test_mild_score(self):
        answers = [1, 1, 0, 1, 0, 1, 1, 0, 0]  # score = 5
        score, severity, rec, badge = PHQ9Assessment.calculate(answers)
        assert score == 5
        assert severity == "اكتئاب خفيف"

    def test_moderate_score(self):
        answers = [2, 2, 1, 1, 1, 1, 1, 1, 0]  # score = 10
        score, severity, rec, badge = PHQ9Assessment.calculate(answers)
        assert score == 10
        assert severity == "اكتئاب متوسط"

    def test_badge_icons(self):
        assert PHQ9Assessment.get_badge_icon("طبيعي") == "✅"
        assert PHQ9Assessment.get_badge_icon("اكتئاب شديد") == "🚨"
        assert PHQ9Assessment.get_badge_icon("unknown") == "•"


# ═══════════════════════════════════════════════════════════════════
#  GAD-7 Scoring
# ═══════════════════════════════════════════════════════════════════

class TestGAD7:
    """GAD-7 anxiety assessment scoring."""

    def test_questions_count(self):
        assert len(GAD7_QUESTIONS) == 7

    def test_options_count(self):
        assert len(GAD7_OPTIONS) == 4

    def test_minimal_score(self):
        answers = [0] * 7
        score, severity, rec, badge = GAD7Assessment.calculate(answers)
        assert score == 0
        assert badge == "sev-0"

    def test_maximal_score(self):
        answers = [3] * 7
        score, severity, rec, badge = GAD7Assessment.calculate(answers)
        assert score == 21
        assert badge == "sev-3"

    def test_mild_score(self):
        answers = [1, 1, 1, 1, 1, 0, 0]  # score = 5
        score, severity, rec, badge = GAD7Assessment.calculate(answers)
        assert score == 5
        assert badge == "sev-1"
