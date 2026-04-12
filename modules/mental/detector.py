# -*- coding: utf-8 -*-
"""
SHIFA-Mental · Détecteur de Détresse Multi-Dialectal
Charge les mots-clés depuis crisis_keywords_arabic.json (MSA + Darija + Égyptien + Levantin)
Gère la négation pour éviter les faux positifs.
Intègre le SafetyGuard existant.
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("shifa.mental.detector")

# ─── Negation patterns by dialect ──────────────────────────────────────────────
NEGATION_DICT = {
    "msa": ["لا", "ليس", "لست", "لن", "لم", "ما", "غير", "بدون", "ليست", "لسنا", "لسن", "ليسوا"],
    "darija": ["ما", "ماشي", "مش", "مشي"],
    "egyptian": ["مش", "مابقاش", "ماكنش", "ما"],
    "levantine": ["مو", "ما", "مش"]
}

# Regex for sentence boundaries to limit negation scope
_SENTENCE_BOUNDS_RE = re.compile(r'[.!?،؛,;-]')

# Compiled regex for better negation detection
# Matches: <negation_word> ... <keyword> within ~15 chars window
_NEGATION_WINDOW = 20  # max chars between negation and keyword


class DistressDetector:
    """
    Multi-dialectal Arabic distress detector with negation awareness.

    Loads keywords from `datasets/crisis_keywords_arabic.json` and
    applies layered detection:
      Level 3: Crisis (suicidal ideation, self-harm)
      Level 2: High distress (severe emotional pain)
      Level 1: Mild distress (emotional difficulty)
      Level 0: No distress detected
    """

    def __init__(self, keywords_path: Optional[str] = None):
        """
        Args:
            keywords_path: Path to crisis_keywords_arabic.json.
                          If None, auto-detects from project root.
        """
        self._crisis_kws: list[str] = []
        self._high_kws: list[str] = []
        self._mild_kws: list[str] = []
        self._loaded = False

        # Pre-compile negation regex patterns using word boundaries
        self._neg_regexes = {}
        for dialect, neg_list in NEGATION_DICT.items():
            pattern = r'\b(?:و|ف)?(?:' + '|'.join(neg_list) + r')\b'
            self._neg_regexes[dialect] = re.compile(pattern)
        
        # Global fallback regex (combines all dialects)
        all_negs = list({word for lst in NEGATION_DICT.values() for word in lst})
        self._global_neg_regex = re.compile(r'\b(?:و|ف)?(?:' + '|'.join(all_negs) + r')\b')

        if keywords_path is None:
            project_root = Path(__file__).parent.parent.parent
            keywords_path = str(project_root / "datasets" / "crisis_keywords_arabic.json")

        self._load_keywords(keywords_path)

    def _load_keywords(self, path: str) -> None:
        """Load keywords from JSON, merging all dialects per level."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            levels = data.get("levels", {})

            # Level 3: Crisis
            crisis_data = levels.get("3_crisis", {})
            for dialect_key in ("MSA", "Darija", "Egyptian", "Levantine"):
                self._crisis_kws.extend(crisis_data.get(dialect_key, []))

            # Level 2: High distress
            high_data = levels.get("2_high", {})
            for dialect_key in ("MSA", "Darija", "Egyptian", "Levantine"):
                self._high_kws.extend(high_data.get(dialect_key, []))

            # Level 1: Mild
            mild_data = levels.get("1_mild", {})
            for dialect_key in ("MSA", "Darija", "Egyptian", "Levantine"):
                self._mild_kws.extend(mild_data.get(dialect_key, []))

            # Deduplicate
            self._crisis_kws = list(set(self._crisis_kws))
            self._high_kws = list(set(self._high_kws))
            self._mild_kws = list(set(self._mild_kws))

            self._loaded = True
            total = len(self._crisis_kws) + len(self._high_kws) + len(self._mild_kws)
            logger.info(
                "[DistressDetector] Loaded %d keywords from %s "
                "(crisis=%d, high=%d, mild=%d)",
                total, path,
                len(self._crisis_kws), len(self._high_kws), len(self._mild_kws)
            )

        except FileNotFoundError:
            logger.warning("[DistressDetector] Keywords file not found: %s — using fallback", path)
            self._use_fallback_keywords()
        except Exception as e:
            logger.error("[DistressDetector] Error loading keywords: %s", e)
            self._use_fallback_keywords()

    def _use_fallback_keywords(self) -> None:
        """Minimal fallback if JSON file is missing."""
        self._crisis_kws = [
            "انتحار", "أقتل نفسي", "لا أريد العيش", "أريد الموت",
            "ايذاء نفسي", "بغيت نموت", "كنفكر نقتل روحي",
            "مابغيتش نعيش", "عايز أموت", "بدي أموت",
        ]
        self._high_kws = [
            "يائس", "يأس", "محطم", "لا أستطيع التحمل", "كآبة",
            "اكتئاب حاد", "فقدت الأمل", "مكسور خاطري",
            "عيشتي ما فيها حتى شي", "مش قادر أكمل",
        ]
        self._mild_kws = [
            "حزين", "قلق", "متعب", "مرهق", "ضغط", "توتر",
            "خائف", "محبط", "مضطرب", "زهقت", "تعبان نفسياً",
            "مهموم", "قلقان", "عيان نفسياً",
        ]
        self._loaded = True
        logger.info("[DistressDetector] Using fallback keywords (darija-inclusive)")

    def _is_negated(self, text: str, keyword: str, dialect: str) -> bool:
        """
        Check if a keyword occurrence is preceded by a negation word.

        Example:
            "أنا لا أريد الموت" → 'أريد الموت' is negated → True
            "أريد الموت" → not negated → False
            "مش بغيت نموت" → 'بغيت نموت' is negated → True
        """
        idx = text.find(keyword)
        if idx < 0:
            return False

        # Look at the text window before the keyword
        window_start = max(0, idx - _NEGATION_WINDOW)
        before_text = text[window_start:idx]

        # Prevent negation bleed across sentences
        bounds_match = list(_SENTENCE_BOUNDS_RE.finditer(before_text))
        if bounds_match:
            # Cut the window at the last punctuation mark before the keyword
            last_bound = bounds_match[-1].end()
            before_text = before_text[last_bound:]

        regex = self._neg_regexes.get(dialect, self._global_neg_regex)
        match = regex.search(before_text)
        if match:
            logger.info(
                "[DistressDetector] Negation detected: '%s' before '%s'",
                match.group(), keyword
            )
            return True

        return False

    def detect(self, text: str) -> tuple[int, str, list[str]]:
        """
        Analyse text for distress signals across all Arabic dialects.

        Args:
            text: User input text (any Arabic dialect)

        Returns:
            Tuple of:
              - level (int): 0=None, 1=Mild, 2=High, 3=Crisis
              - reason (str): Arabic explanation
              - matched_keywords (list[str]): The keywords that triggered
        """
        if not text or not text.strip():
            return 0, "لا يوجد نص للتحليل", []

        dialect = self.detect_dialect(text)

        matched_crisis: list[str] = []
        matched_high: list[str] = []
        matched_mild: list[str] = []

        # ── Level 3: Crisis keywords ────────────────────────────
        for kw in self._crisis_kws:
            if kw in text and not self._is_negated(text, kw, dialect):
                matched_crisis.append(kw)

        if matched_crisis:
            logger.warning("[DistressDetector] CRISIS detected — keywords: %s", matched_crisis)
            return 3, f"⚠️ كلمات تشير لأزمة نفسية حادة: {', '.join(matched_crisis[:3])}", matched_crisis

        # ── Level 2: High distress ──────────────────────────────
        for kw in self._high_kws:
            if kw in text and not self._is_negated(text, kw, dialect):
                matched_high.append(kw)

        if len(matched_high) >= 2:
            return 2, f"مؤشرات ضيق شديد ({len(matched_high)} علامات)", matched_high

        # ── Level 1: Mild distress ──────────────────────────────
        for kw in self._mild_kws:
            if kw in text and not self._is_negated(text, kw, dialect):
                matched_mild.append(kw)

        if len(matched_mild) >= 2 or (len(matched_mild) >= 1 and len(matched_high) >= 1):
            return 1, f"مؤشرات ضيق خفيف ({len(matched_mild) + len(matched_high)} علامات)", matched_mild + matched_high

        if len(matched_mild) >= 1 or len(matched_high) >= 1:
            return 1, "علامات ضيق مبكرة", matched_mild + matched_high

        return 0, "لا توجد مؤشرات ضيق واضحة", []

    def detect_dialect(self, text: str) -> str:
        """
        Rough dialect detection from user text.
        Returns: 'darija' | 'egyptian' | 'levantine' | 'msa'
        """
        # Removed ambiguous MSA overlap markers like "كنت", "عندك", "ليك"
        darija_markers = ["كاين", "واش", "كيفاش", "فاش", "ديال", "بزاف", "زوين", "بغيت", "دابا", "شنو", "علاش", "درت", "مزيان"]
        egyptian_markers = ["إزيك", "عايز", "كده", "يعني", "ازاي", "ليه", "انت", "بتاع"]
        levantine_markers = ["كيفك", "شو", "هلق", "رح", "بدي", "هلأ", "منيح"]

        darija_count = sum(1 for m in darija_markers if m in text)
        egyptian_count = sum(1 for m in egyptian_markers if m in text)
        levantine_count = sum(1 for m in levantine_markers if m in text)

        counts = {"darija": darija_count, "egyptian": egyptian_count, "levantine": levantine_count}
        best = max(counts, key=counts.get)

        if counts[best] >= 1:
            return best
        return "msa"  # default
