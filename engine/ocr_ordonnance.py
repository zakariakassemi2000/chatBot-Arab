# engine/ocr_ordonnance.py — VERSION INGÉNIEUR IA
# ══════════════════════════════════════════════════════════════
#  SHIFA AI — Pipeline OCR Ordonnances Médicales
#  Stack: docTR (CPU) | Donut Medical (GPU) | Tesseract (fallback)
#  + Vérification medicament.ma + Extraction regex médicale
# ══════════════════════════════════════════════════════════════

import re
import json
import logging
import time
import requests
import urllib.parse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# SECTION 1 — MODÈLE OCR (docTR — meilleur CPU)
# ══════════════════════════════════════════════

class OCREngine:
    """
    Utilise docTR (mindee) — meilleur rapport précision/vitesse sur CPU.
    Fallback sur Tesseract si docTR non disponible.
    Donut disponible mais avec détection d'hallucinations et fallback auto.
    """

    def __init__(self, use_donut: bool = False):
        self.use_donut = use_donut
        self.model = None
        self.engine_name = "uninitialised"
        self.processor = None
        self.device = "cpu"
        self._load_model()

    def _load_model(self):
        if self.use_donut:
            self._load_donut()
        else:
            self._load_doctr()

    def _load_doctr(self):
        """docTR — DBNet + CRNN, pré-entraîné FR/EN."""
        try:
            from doctr.models import ocr_predictor
            self.model = ocr_predictor(
                det_arch='db_resnet50',
                reco_arch='crnn_vgg16_bn',
                pretrained=True,
                assume_straight_pages=True
            )
            self.engine_name = "docTR (DBNet + CRNN)"
            logger.info("[OCR] docTR chargé — poids FR pré-entraînés")
        except ImportError:
            logger.warning("[OCR] docTR non disponible — fallback Tesseract")
            self._load_tesseract_fallback()

    def _load_donut(self):
        """
        Donut — spécialisé ordonnances médicales.
        HuggingFace: chinmays18/medical-prescription-ocr
        """
        try:
            from transformers import (
                DonutProcessor,
                VisionEncoderDecoderModel
            )
            model_id = "chinmays18/medical-prescription-ocr"
            try:
                self.processor = DonutProcessor.from_pretrained(model_id)
            except Exception as e:
                logger.warning(f"Fast tokenizer issue: {e}, fallback to use_fast=False")
                self.processor = DonutProcessor.from_pretrained(model_id, use_fast=False)

            self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.engine_name = "Donut Medical Prescription OCR"
            logger.info(f"[OCR] Donut chargé — device: {self.device}")
        except Exception as e:
            logger.warning(f"[OCR] Donut échec: {e} — fallback docTR")
            self.use_donut = False
            self._load_doctr()

    def _load_tesseract_fallback(self):
        """Fallback Tesseract si rien d'autre."""
        try:
            import pytesseract  # noqa: F401
            self.model = "tesseract"
            self.engine_name = "Tesseract (fallback)"
            logger.info("[OCR] Tesseract fallback activé")
        except ImportError:
            self.model = None
            self.engine_name = "none"
            logger.error("[OCR] Aucun moteur OCR disponible!")

    # ── Hallucination Detection ──────────────────────────────

    @staticmethod
    def _is_hallucinated(text: str) -> bool:
        """
        Détecte si le texte OCR est une hallucination du modèle Donut.
        Le modèle Donut (chinmays18/medical-prescription-ocr) génère
        du texte anglais aléatoire sur les ordonnances françaises/arabes.
        """
        if not text or len(text.strip()) < 10:
            return True

        # Ordonnances are short — hallucinations are always very long
        if len(text) > 2000:
            return True

        # Check for French medical keywords
        fr_medical_words = [
            'mg', 'ml', 'cp', 'comprimé', 'gélule', 'sirop', 'sachet',
            'fois', 'jour', 'matin', 'soir', 'pendant', 'avant', 'après',
            'ordonnance', 'patient', 'docteur', 'dr', 'traitement',
            'prise', 'dose', 'posologie',
        ]
        text_lower = text.lower()
        fr_hits = sum(1 for w in fr_medical_words if w in text_lower)

        # If at least 2 French medical keywords are present, probably real
        if fr_hits >= 2:
            return False

        # Check for long gibberish lines (hallucination signature)
        lines = text.splitlines()
        long_lines = [l for l in lines if len(l.strip()) > 200]
        if long_lines:
            return True

        # Too many words with zero French context = garbage
        words = text.split()
        if len(words) > 50 and fr_hits == 0:
            return True

        return False

    # ── Text Extraction Main Entry Point ─────────────────────

    def extract_text(self, image: Image.Image) -> tuple:
        """Retourne (texte_extrait, confiance_0_1)."""
        if self.use_donut and self.engine_name.startswith("Donut"):
            text, conf = self._extract_donut(image)

            # Hallucination guard — Donut hallucinates on non-English docs
            if self._is_hallucinated(text):
                logger.warning(
                    "[OCR] Donut output détecté comme hallucination — "
                    "fallback automatique vers docTR"
                )
                text, conf = self._fallback_doctr(image)
            return text, conf

        elif self.engine_name.startswith("docTR"):
            return self._extract_doctr(image)
        else:
            return self._extract_tesseract(image)

    def _fallback_doctr(self, image: Image.Image) -> tuple:
        """Charge docTR à la volée et extrait le texte."""
        try:
            from doctr.models import ocr_predictor
            fallback_model = ocr_predictor(
                det_arch='db_resnet50',
                reco_arch='crnn_vgg16_bn',
                pretrained=True,
                assume_straight_pages=True
            )
            arr = np.array(image.convert("RGB"))
            result = fallback_model([arr])
            lines, confs = [], []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        words = [w.value for w in line.words]
                        word_confs = [w.confidence for w in line.words]
                        lines.append(" ".join(words))
                        confs.extend(word_confs)
            text = "\n".join(lines)
            conf = sum(confs) / len(confs) if confs else 0.0
            self.engine_name = "docTR (fallback auto)"
            logger.info(f"[OCR] docTR fallback: {len(lines)} lignes, conf={conf:.2%}")
            return text, conf
        except Exception as fb_err:
            logger.error(f"[OCR] Fallback docTR échoué: {fb_err}")
            return "", 0.0

    # ── Engine-specific extractors ───────────────────────────

    def _extract_doctr(self, image: Image.Image) -> tuple:
        """Extraction via docTR."""
        arr = np.array(image.convert("RGB"))
        result = self.model([arr])

        lines, confs = [], []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [w.value for w in line.words]
                    word_confs = [w.confidence for w in line.words]
                    lines.append(" ".join(words))
                    confs.extend(word_confs)

        text = "\n".join(lines)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return text, avg_conf

    def _extract_donut(self, image: Image.Image) -> tuple:
        """Extraction via Donut Medical."""
        pixel_values = self.processor(
            images=image.convert("RGB"),
            return_tensors="pt"
        ).pixel_values.to(self.device)

        task_prompt = "<s_ocr>"
        decoder_ids = self.processor.tokenizer(
            task_prompt, return_tensors="pt"
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_ids,
                max_length=512,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                repetition_penalty=1.5,
                use_cache=True,
                num_beams=2,
            )

        seq = self.processor.batch_decode(outputs.tolist())[0]
        seq = seq.replace(self.processor.tokenizer.eos_token, "")
        seq = seq.replace(self.processor.tokenizer.pad_token, "")

        # Inject newlines at item boundaries before removing tags
        separators = ["</s_med>", "<s_med>", "</s_item>", "<s_item>",
                       "<sep>", "</li>", "\\n", " \n", "\\n "]
        for tag in separators:
            seq = seq.replace(tag, f"\n{tag}\n")

        seq = re.sub(r"<.*?>", " ", seq).strip()
        seq = re.sub(r" +", " ", seq).strip()
        seq = re.sub(r"\n\s*\n", "\n", seq).strip()

        # Smart line splitting for single-block output
        if seq.count("\n") <= 2 and len(seq) > 40:
            seq = re.sub(
                r'(\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|mq|m9|g|ml|UI|\u03bcg|\u00b5g|%|cp|cpr|comp|gel|suppo|sachet|amp|inj|iu)\b)',
                r'\1\n', seq, flags=re.IGNORECASE
            )
            seq = re.sub(
                r'(\b(?:inj|cp|cpr|g\u00e9lules?|comprim\u00e9s?|suppo|sachet|sirop|pommade|cr\u00e8me|collyre)s?\b)',
                r'\1\n', seq, flags=re.IGNORECASE
            )
            seq = re.sub(r'\s+(\d+[\.)\]]\s)', r'\n\1', seq)
            seq = re.sub(r'\s+([A-Z][a-z\u00e9\u00e8\u00ea\u00eb\u00e0]{3,})', r'\n\1', seq)
            seq = re.sub(r"\n\s*\n", "\n", seq).strip()

        return seq, 0.85

    def _extract_tesseract(self, image: Image.Image) -> tuple:
        """Extraction via Tesseract (fallback)."""
        import pytesseract
        arr = self._preprocess_cv(np.array(image.convert("RGB")))
        processed = Image.fromarray(arr)
        config = "--psm 6 --oem 3 -l fra+ara"
        data = pytesseract.image_to_data(
            processed, config=config,
            output_type=pytesseract.Output.DICT
        )
        words = [w for w, c in zip(data["text"], data["conf"])
                 if w.strip() and int(c) > 30]
        confs = [int(c) / 100 for c in data["conf"]
                 if c != "-1" and int(c) > 0]
        text = " ".join(words)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return text, avg_conf

    def _preprocess_cv(self, arr: np.ndarray) -> np.ndarray:
        """Pipeline CV2 optimisé pour ordonnances."""
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr

        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )

        # Upscaling si petite image
        h, w = binary.shape
        if w < 1000:
            scale = 1000 / w
            binary = cv2.resize(binary, None,
                                fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
        return binary


# ══════════════════════════════════════════════
# SECTION 2 — VÉRIFICATION MEDICAMENT.MA
# ══════════════════════════════════════════════

class MedicamentMAVerifier:
    """
    Vérifie les médicaments sur medicament.ma (scraping respectueux).
    URL pattern : https://medicament.ma/?s=NOMMED
    Rate limit : 1 requête / 2 secondes
    """

    BASE_URL = "https://medicament.ma"
    SEARCH_URL = "https://medicament.ma/listing-des-medicaments/?search={query}"
    DIRECT_URL = "https://medicament.ma/{slug}/"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; SHIFA-AI-Bot/1.0; "
            "+https://github.com/zakariakassemi2000/chatBot-Arab)"
        ),
        "Accept-Language": "fr-MA,fr;q=0.9",
    }
    CACHE: dict = {}  # Cache en mémoire
    LAST_REQUEST: float = 0

    @classmethod
    def _rate_limit(cls):
        """Respect rate limit — 1 req/2s."""
        elapsed = time.time() - cls.LAST_REQUEST
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
        cls.LAST_REQUEST = time.time()

    @classmethod
    def verify(cls, nom_medicament: str) -> dict:
        """
        Cherche un médicament sur medicament.ma.
        Retourne les informations trouvées ou un dict vide.
        """
        key = nom_medicament.lower().strip()
        if key in cls.CACHE:
            return cls.CACHE[key]

        cls._rate_limit()

        try:
            from bs4 import BeautifulSoup
            import urllib.parse

            url = cls.SEARCH_URL.format(
                query=urllib.parse.quote(nom_medicament)
            )
            resp = requests.get(url, headers=cls.HEADERS, timeout=10)
            resp.raise_for_status()

            result = cls._parse_search_results(resp.text, nom_medicament)
            
            if not result.get("found"):
                slug = nom_medicament.lower().replace(" ","-").replace("é","e")
                direct_url = cls.DIRECT_URL.format(slug=urllib.parse.quote(slug))
                try:
                    resp_direct = requests.get(direct_url, headers=cls.HEADERS, timeout=10)
                    if resp_direct.status_code == 200:
                        soup = BeautifulSoup(resp_direct.text, "html.parser")
                        if soup.find("h1"):
                            result = {
                                "found": True,
                                "count": 1,
                                "medicaments": [{"nom": nom_medicament, "url": direct_url, "source": "medicament.ma"}],
                                "query": nom_medicament,
                                "source_url": direct_url
                            }
                except requests.RequestException:
                    pass

            if not result.get("found"):
                fallback_url = f"https://medicament.ma/?s={urllib.parse.quote(nom_medicament)}"
                result["fallback_url"] = fallback_url
                result["message"] = "Recherche manuelle requise"

            cls.CACHE[key] = result
            return result

        except ImportError:
            logger.warning("[MedicamentMA] beautifulsoup4 non installé")
            return {"found": False, "error": "beautifulsoup4 not installed"}
        except requests.RequestException as e:
            logger.warning(f"[MedicamentMA] Erreur réseau: {e}")
            return {"found": False, "error": str(e)}

    @classmethod
    def _parse_search_results(cls, html: str, query: str) -> dict:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        results = []

        query_word_re = re.compile(rf'\b{re.escape(query)}\b', re.IGNORECASE)

        # Pattern 1 : liens directs vers fiches médicaments
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            if ("/medicament/" in href or "/produit/" in href) and text:
                # IMPORTANT: we must actually check if the text matches the query!
                # Otherwise, it scrapes sidebar links.
                if query_word_re.search(text) or query.lower() in text.lower():
                    results.append({
                        "nom": text,
                        "url": href,
                        "source": "medicament.ma"
                    })

        # Pattern 2 : titres d'articles
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            text = h.get_text(strip=True)
            if query_word_re.search(text) and len(text) < 100:
                link_tag = h.find("a")
                url = link_tag["href"] if link_tag else ""
                if link_tag or ("/medicament/" in str(h) or "/produit/" in str(h)):
                    results.append({
                        "nom": text,
                        "url": url,
                        "source": "medicament.ma"
                    })

        # Déduplication par nom
        seen = set()
        unique = []
        for r in results:
            if r["nom"] not in seen:
                seen.add(r["nom"])
                unique.append(r)

        if unique:
            return {
                "found": True,
                "count": len(unique),
                "medicaments": unique[:3],  # Top 3
                "query": query,
                "source_url": cls.SEARCH_URL.format(
                    query=requests.utils.quote(query)
                )
            }
        else:
            return {
                "found": False,
                "query": query,
                "message": "Non trouvé sur medicament.ma"
            }

    @classmethod
    def get_detail(cls, url: str) -> dict:
        """Récupère la fiche détaillée d'un médicament."""
        from bs4 import BeautifulSoup

        if not url.startswith("http"):
            url = cls.BASE_URL + url

        cls._rate_limit()
        try:
            resp = requests.get(url, headers=cls.HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            detail = {"url": url}

            # Extraction DCI / Substance active
            for label in ["DCI", "Substance active", "Principe actif"]:
                tag = soup.find(
                    string=re.compile(label, re.IGNORECASE)
                )
                if tag and tag.parent:
                    val = tag.parent.find_next_sibling()
                    if val:
                        detail["dci"] = val.get_text(strip=True)
                        break

            # Prix
            prix_tag = soup.find(
                string=re.compile(r"Prix|PPV|PH", re.IGNORECASE)
            )
            if prix_tag and prix_tag.parent:
                val = prix_tag.parent.find_next_sibling()
                if val:
                    detail["prix"] = val.get_text(strip=True)

            # Laboratoire
            labo_tag = soup.find(
                string=re.compile(r"Laboratoire|Fabricant", re.IGNORECASE)
            )
            if labo_tag and labo_tag.parent:
                val = labo_tag.parent.find_next_sibling()
                if val:
                    detail["laboratoire"] = val.get_text(strip=True)

            return detail

        except Exception as e:
            return {"url": url, "error": str(e)}


# ══════════════════════════════════════════════
# SECTION 3 — EXTRACTEUR MÉDICAL
# ══════════════════════════════════════════════

@dataclass
class MedicamentExtrait:
    """Représentation structurée d'un médicament extrait."""
    nom_brut: str
    dci: Optional[str] = None
    nom_commercial: Optional[str] = None
    dosage: Optional[str] = None
    posologie: Optional[str] = None
    duree: Optional[str] = None
    formes_disponibles: list = field(default_factory=list)
    dosages_habituels: list = field(default_factory=list)
    confidence_ocr: float = 0.0
    verification_ma: Optional[dict] = None


@dataclass
class ResultatOrdonnance:
    """Résultat complet du scan d'une ordonnance."""
    texte_brut: str
    langue_detectee: str
    engine_utilise: str
    medicaments: list = field(default_factory=list)
    informations_patient: dict = field(default_factory=dict)
    avertissements: list = field(default_factory=list)
    score_global: float = 0.0


class MedicalExtractor:
    """Extraction regex des éléments médicaux depuis le texte OCR."""

    # ── OCR-tolerant regex: accepts common misreads ──
    DOSAGE_RE = re.compile(
        r'\b[0-9OoIl]+(?:[.,][0-9OoIl]+)?\s*'
        r'(?:mg|mq|mcg|m9|g|ml|UI|μg|µg|%|cp|cpr|comp|gel|suppo|sachet|amp|inj|iu)\b',
        re.IGNORECASE
    )
    POSOLOGIE_RE = re.compile(
        r'(?:\d+\s*(?:fois?\s*(?:par\s*)?(?:jour|j)|comprimés?|cp?|gélules?|supps?)|'
        r'matin(?:\s+et\s+soir)?|soir|midi|à\s+jeun|avant\s+les\s+repas|'
        r'après\s+les\s+repas|/\s*j(?:our)?|par\s*jour|'
        r'صباحاً|مساءً|مرة.*يومياً)',
        re.IGNORECASE
    )
    DUREE_RE = re.compile(
        r'(?:pendant\s*|durant\s*|pour\s*)?\d+\s*(?:jours?|semaines?|mois|j|أيام|أسابيع|شهر)',
        re.IGNORECASE
    )
    ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
    LATIN_RE = re.compile(r'[a-zA-Z]')

    # Mots-clés médicaux qui indiquent qu'une ligne est potentiellement un médicament
    MED_KEYWORD_RE = re.compile(
        r'\b(?:comprim[eé]s?|g[eé]lules?|sirop|sachet|cp|suppo|'
        r'pommade|cr[eè]me|collyre|gel|spray|inj|amp|lyoc)\b',
        re.IGNORECASE
    )

    # Lignes à ignorer (headers, footers, info patient)
    # IMPORTANT: utiliser des mots complets uniquement pour éviter les faux positifs
    # ex: 'mr ' matcherait 'puscomr' en substring — utiliser \bMR\b à la place
    SKIP_KEYWORDS_RE = re.compile(
        r'\b(?:docteur|r(?:humatologue|adiologue|éanimateur)|m(?:édecin|édecin-chef)|'
        r'clinique|h(?:opital|ôpital)|urgence|chef|patient|cabinet|'
        r'spécialiste|infirmier|pharmacien|laboratoire|'
        r'casablanca|rabat|marrakech|tanger|oujda|f(?:ès|es)|'
        r'libraire|cin(?:ema|éma)|rex|'
        r'rdv|fax|email|adresse)\b|'
        r'\bdr\.?\s|(?:tél|tel)\s*:|n°\s*\d',
        re.IGNORECASE
    )

    # ── OCR Normalization Map ──
    OCR_CHAR_FIXES = {
        'O': '0', 'o': '0', 'I': '1', 'l': '1',
        'S': '5', 'B': '8', 'q': 'g',
    }

    @staticmethod
    def _normalize_ocr_text(text: str) -> str:
        """
        Normalise le texte OCR: correction de caractères courants
        confondus par les moteurs OCR (O→0, mq→mg, etc.).
        Ne touche qu'aux zones numériques/dosage pour ne pas casser les noms.
        """
        # Fix common OCR dosage errors
        text = re.sub(r'(\d)\s*mq\b', r'\1 mg', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d)\s*m9\b', r'\1 mg', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d)\s*rng\b', r'\1 mg', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d)\s*rnq\b', r'\1 mg', text, flags=re.IGNORECASE)
        # Fix O/o in numbers near dosage units
        text = re.sub(r'\b(\d*)[Oo](\d*)\s*(mg|g|ml|cp|ui)', 
                       lambda m: f"{m.group(1)}0{m.group(2)} {m.group(3)}", text)
        return text

    def detect_language(self, text: str) -> str:
        """Détecte la langue principale du texte."""
        ar = len(self.ARABIC_RE.findall(text))
        lat = len(self.LATIN_RE.findall(text))
        if ar == 0:
            return "fr"
        if lat == 0:
            return "ar"
        return "mixte" if ar / (ar + lat) < 0.6 else "ar"

    def _should_skip_line(self, line: str) -> bool:
        """
        Vérifie si la ligne est un header/footer/info non-médicale.
        Utilise le regex SKIP_KEYWORDS_RE avec des word boundaries pour
        éviter que 'mr' ne matche 'puscomr', etc.
        """
        if len(line.strip()) < 3:
            return True
        # Pure numbers/punctuation with no letters
        if re.match(r'^[\d\W]+$', line.strip()):
            return True
        # Single character
        if len(line.strip()) <= 2:
            return True
        # Check non-medical header keywords (word boundary safe)
        if self.SKIP_KEYWORDS_RE.search(line):
            # Exception: never skip if there's a dosage on the same line
            # (e.g. a doctor note that still mentions a drug dose)
            if self.DOSAGE_RE.search(line):
                return False
            return True
        return False

    def _fuzzy_match_word(self, word: str, name_index: list, db_meds: list, name_to_dci: dict, process, fuzz):
        """
        Essaie un fuzzy match sur un seul mot ou groupe de mots.
        Retourne (dci, nom_commercial, formes, dosages, score) ou (None, None, [], [], 0).
        """
        if not word or len(word) < 3 or not name_index:
            return None, None, [], [], 0

        # Essai 1: token_set_ratio (tolérant aux mots supplémentaires)
        match = process.extractOne(
            word.lower(), name_index,
            scorer=fuzz.token_set_ratio,
            score_cutoff=70  # Abaissé de 85 pour capturer plus de variantes OCR
        )
        if not match:
            # Essai 2: ratio simple (bon pour les mots isolés courts)
            match = process.extractOne(
                word.lower(), name_index,
                scorer=fuzz.ratio,
                score_cutoff=65
            )
        if not match:
            return None, None, [], [], 0

        matched, score, _ = match

        # Validation anti faux-positifs: au moins un mot doit être similaire
        line_words = [w.lower() for w in word.split() if len(w) >= 3]
        matched_words = [w.lower() for w in matched.split() if len(w) >= 3]

        is_valid = False
        if score >= 85:
            is_valid = True
        elif score >= 70:
            for lw in line_words:
                for mw in matched_words:
                    if fuzz.ratio(lw, mw) >= 65:
                        is_valid = True
                        break
                if is_valid:
                    break

        if not is_valid:
            return None, None, [], [], 0

        dci = name_to_dci.get(matched, matched)
        
        # Look up additional details in DB
        formes, dosages = [], []
        if db_meds:
            for item in db_meds:
                if matched.lower() in [dci.lower()] + [n.lower() for n in item.get('noms_commerciaux', [])]:
                    formes = item.get('formes', [])
                    dosages = item.get('dosages_courants', [])
                    break

        # Si le mot matché est exactement le DCI, pas de nom commercial spécifique détecté
        nom_com = None if matched.lower() == dci.lower() else matched.capitalize()

        return dci, nom_com, formes, dosages, score

    def extract_medicaments(
        self, text: str, conf: float
    ) -> list:
        """Extraction ligne par ligne avec regex + fuzzy matching."""
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            logger.warning("[Extractor] rapidfuzz non installé — fuzzy matching désactivé")
            process = None
            fuzz = None

        # Normaliser le texte OCR d'abord
        text = self._normalize_ocr_text(text)

        # Chargement DB locale
        db_path = Path(__file__).parent.parent / "data" / "medicaments_maroc.json"
        all_names = []
        db_medicaments = []

        if db_path.exists():
            with open(db_path, encoding="utf-8") as f:
                db = json.load(f)

            # Support both old format (flat list) and new format (dict with "medicaments")
            if isinstance(db, dict) and "medicaments" in db:
                db_medicaments = db["medicaments"]
                for m in db_medicaments:
                    dci = m.get("dci", "")
                    if dci:
                        all_names.append((dci.lower(), dci))
                    for cn in m.get("noms_commerciaux", []):
                        all_names.append((cn.lower(), dci or cn))
                    for alias in m.get("ocr_aliases", []):
                        nom_base = dci if dci else (m.get("noms_commerciaux", [""])[0] if m.get("noms_commerciaux") else alias)
                        all_names.append((alias.lower(), nom_base))
            elif isinstance(db, list):
                db_medicaments = db
                for m in db:
                    nom = m.get("nom", "")
                    principe = m.get("principe", "")
                    if nom:
                        all_names.append((nom.lower(), principe or nom))
                    if principe:
                        all_names.append((principe.lower(), principe))

        name_index = [n[0] for n in all_names]
        name_to_dci = {n[0]: n[1] for n in all_names}

        meds = []
        seen_dci = set()  # Éviter les doublons

        for line in text.splitlines():
            line = line.strip()
            if self._should_skip_line(line):
                continue

            # Nettoyer la ligne (enlever numéros de liste, tirets et puces)
            line_clean = re.sub(r'^[\d\W_]+', '', line).strip()
            if not line_clean:
                line_clean = line

            # ── Fuzzy Matching (DCI / Nom Commercial) ──
            dci, nom_commercial = None, None
            formes, dosages_habs = [], []
            best_score = 0

            if name_index and process is not None and len(line_clean) >= 3:
                words = line_clean.split()

                # Stratégie 1: ligne entière
                d, nc, f, dh, s = self._fuzzy_match_word(
                    line_clean, name_index, db_medicaments, name_to_dci, process, fuzz
                )
                if s > best_score:
                    dci, nom_commercial, formes, dosages_habs, best_score = d, nc, f, dh, s

                # Stratégie 2: premier mot (souvent le nom du médicament)
                if words and len(words[0]) >= 3:
                    d, nc, f, dh, s = self._fuzzy_match_word(
                        words[0], name_index, db_medicaments, name_to_dci, process, fuzz
                    )
                    if s > best_score:
                        dci, nom_commercial, formes, dosages_habs, best_score = d, nc, f, dh, s

                # Stratégie 3: deux premiers mots
                if len(words) >= 2:
                    two_words = " ".join(words[:2])
                    if len(two_words) >= 4:
                        d, nc, f, dh, s = self._fuzzy_match_word(
                            two_words, name_index, db_medicaments, name_to_dci, process, fuzz
                        )
                        if s > best_score:
                            dci, nom_commercial, formes, dosages_habs, best_score = d, nc, f, dh, s

            # ── Regex extraction ──
            dosage = self.DOSAGE_RE.search(line_clean)
            posologie = self.POSOLOGIE_RE.search(line_clean)
            duree = self.DUREE_RE.search(line_clean)
            med_keyword = self.MED_KEYWORD_RE.search(line_clean)

            # Extraire le vrai nom brut : isoler la partie de texte avant le dosage
            if dosage:
                nom_brut = line_clean[:dosage.start()].strip()
                if not nom_brut or len(nom_brut) < 3:
                    nom_brut = line_clean
            else:
                nom_brut = line_clean

            # Nettoyer nom_brut: retirer la ponctuation résiduelle
            nom_brut = re.sub(r'[^A-Za-zÀ-ÿéèêëîïôùûü0-9\s\-]', ' ', nom_brut).strip()
            if not nom_brut or len(nom_brut) < 2:
                nom_brut = line_clean

            # ── Décision: ligne utile? ──
            # Plus permissif: dosage OU médicament reconnu OU mot-clé médical
            if dosage or dci or med_keyword:
                # Éviter les doublons du même DCI
                dci_key = (dci or nom_brut).lower()
                if dci_key in seen_dci:
                    continue
                seen_dci.add(dci_key)

                meds.append(MedicamentExtrait(
                    nom_brut=nom_brut,
                    dci=dci,
                    nom_commercial=nom_commercial,
                    dosage=dosage.group(0) if dosage else None,
                    posologie=posologie.group(0) if posologie else None,
                    duree=duree.group(0) if duree else None,
                    formes_disponibles=formes,
                    dosages_habituels=dosages_habs,
                    confidence_ocr=conf,
                ))

        return meds


# ══════════════════════════════════════════════
# SECTION 4 — PIPELINE PRINCIPAL
# ══════════════════════════════════════════════

class OrdonnanceOCR:
    """
    Pipeline complet :
    1. OCR via docTR / Donut / Tesseract
    2. Extraction regex médicaments
    3. Vérification sur medicament.ma (optionnel)
    """

    def __init__(self, use_donut: bool = False,
                 verify_online: bool = True):
        self.ocr = OCREngine(use_donut=use_donut)
        self.extractor = MedicalExtractor()
        self.verify_online = verify_online

    def analyser(self, image: Image.Image) -> ResultatOrdonnance:
        """Analyse complète d'une image d'ordonnance."""
        logger.info(f"[OCR] Pipeline: {self.ocr.engine_name}")

        # Étape 1 — OCR
        texte, conf = self.ocr.extract_text(image)
        if not texte.strip():
            return ResultatOrdonnance(
                texte_brut="",
                langue_detectee="inconnu",
                engine_utilise=self.ocr.engine_name,
                avertissements=["❌ Aucun texte détecté"]
            )

        # Étape 2 — Extraction
        langue = self.extractor.detect_language(texte)
        meds = self.extractor.extract_medicaments(texte, conf)

        # Étape 3 — Vérification medicament.ma
        if self.verify_online:
            for med in meds:
                query = med.dci or med.nom_brut.split()[0]
                logger.info(f"[OCR] Vérification medicament.ma: {query}")
                med.verification_ma = MedicamentMAVerifier.verify(query)

        # Avertissements
        warns = []
        if conf < 0.6:
            warns.append("⚠️ Qualité OCR faible — vérifiez manuellement")
        warns.append(
            "ℹ️ Ce module structure uniquement le texte de l'ordonnance. "
            "Il ne vérifie pas les interactions ni la validité médicale."
        )

        return ResultatOrdonnance(
            texte_brut=texte,
            langue_detectee=langue,
            engine_utilise=self.ocr.engine_name,
            medicaments=meds,
            avertissements=warns,
            score_global=conf
        )


# ══════════════════════════════════════════════
# SECTION 5 — SINGLETON GLOBAL
# ══════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_ocr(use_donut=False, verify_online=True) -> OrdonnanceOCR:
    """Retourne l'instance singleton du pipeline OCR."""
    return OrdonnanceOCR(
        use_donut=use_donut,
        verify_online=verify_online
    )
