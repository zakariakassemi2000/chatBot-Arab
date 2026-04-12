# -*- coding: utf-8 -*-
"""
SHIFA-Mental · GAD-7 Assessment (مقياس اضطراب القلق العام)
7 questions standardisées, score 0-21.
Standard clinique pour mesurer l'anxiété généralisée.
"""

GAD7_QUESTIONS = [
    "الشعور بالعصبية أو القلق أو التوتر الشديد",
    "عدم القدرة على التوقف عن القلق أو السيطرة عليه",
    "القلق المفرط بشأن أمور مختلفة",
    "صعوبة الاسترخاء",
    "الشعور بالقلق لدرجة صعوبة الجلوس ساكناً",
    "الانزعاج أو الغضب بسهولة",
    "الخوف كما لو أن شيئاً سيئاً سيحدث"
]

GAD7_OPTIONS = ["لا على الإطلاق", "عدة أيام", "أكثر من نصف الأيام", "كل يوم تقريباً"]


class GAD7Assessment:
    """Generalized Anxiety Disorder–7 (Arabic)."""

    @staticmethod
    def calculate(answers: list[int]) -> tuple[int, str, str, str]:
        """
        Calculate GAD-7 score.

        Args:
            answers: List of 7 integers (0-3 each)

        Returns:
            (score, severity_label_ar, recommendation_ar, badge_class)
        """
        score = sum(answers)
        if score <= 4:
            return score, "قلق طبيعي", "لا توجد مؤشرات قلق مرضي. استمر في الحفاظ على توازنك.", "sev-0"
        elif score <= 9:
            return score, "قلق خفيف", "جرب تمارين التنفس والاسترخاء. راقب التطور.", "sev-1"
        elif score <= 14:
            return score, "قلق متوسط", "يُنصح بمراجعة مختص نفسي. العلاج السلوكي فعّال جداً.", "sev-2"
        else:
            return score, "قلق شديد", "يُوصى بالتقييم النفسي العاجل والمتابعة المتخصصة.", "sev-3"

    @staticmethod
    def get_badge_icon(severity: str) -> str:
        badge_map = {
            "قلق طبيعي": "✅",
            "قلق خفيف": "🟡",
            "قلق متوسط": "🟠",
            "قلق شديد": "🔴"
        }
        return badge_map.get(severity, "•")
