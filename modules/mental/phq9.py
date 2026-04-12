# -*- coding: utf-8 -*-
"""
SHIFA-Mental · PHQ-9 Assessment (مقياس تقييم الاكتئاب)
9 questions standardisées, score 0-27.
"""

PHQ9_QUESTIONS = [
    "قلة الاهتمام أو المتعة في القيام بالأشياء",
    "الشعور بالإحباط أو الاكتئاب أو اليأس",
    "صعوبة النوم أو البقاء نائماً، أو النوم الزائد",
    "الشعور بالتعب أو انخفاض الطاقة",
    "ضعف الشهية أو الإفراط في تناول الطعام",
    "الشعور بالفشل أو الشعور بأنك تخذل نفسك أو عائلتك",
    "صعوبة التركيز على الأشياء (مثل قراءة الصحيفة أو مشاهدة التلفاز)",
    "التحرك أو التحدث ببطء شديد لدرجة أن الآخرين لاحظوا ذلك",
    "أفكار تتعلق بالأذى الذاتي أو بأنك ستكون ميتاً"
]

PHQ9_OPTIONS = ["لا على الإطلاق", "عدة أيام", "أكثر من نصف الأيام", "كل يوم تقريباً"]


class PHQ9Assessment:
    """Patient Health Questionnaire–9 (Arabic)."""

    @staticmethod
    def calculate(answers: list[int]) -> tuple[int, str, str, str]:
        """
        Calculate PHQ-9 score.

        Args:
            answers: List of 9 integers (0-3 each)

        Returns:
            (score, severity_label_ar, recommendation_ar, badge_class)
        """
        score = sum(answers)
        if score <= 4:
            return score, "طبيعي", "لا تظهر عليك أعراض الاكتئاب. حافظ على صحتك النفسية.", "sev-0"
        elif score <= 9:
            return score, "اكتئاب خفيف", "قد تستفيد من تقنيات الاسترخاء والمتابعة الذاتية.", "sev-1"
        elif score <= 14:
            return score, "اكتئاب متوسط", "يُنصح بزيارة طبيب نفسي أو مرشد نفسي.", "sev-1"
        elif score <= 19:
            return score, "اكتئاب متوسط إلى شديد", "يُوصى بالتقييم النفسي العاجل.", "sev-2"
        else:
            return score, "اكتئاب شديد", "يُلزم التوجه فوراً لمختص نفسي أو مستشفى.", "sev-3"

    @staticmethod
    def get_badge_icon(severity: str) -> str:
        badge_map = {
            "طبيعي": "✅",
            "اكتئاب خفيف": "🟡",
            "اكتئاب متوسط": "🟠",
            "اكتئاب متوسط إلى شديد": "🔴",
            "اكتئاب شديد": "🚨"
        }
        return badge_map.get(severity, "•")
