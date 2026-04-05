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
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# SECTION 1 — MODÈLE OCR (docTR — meilleur CPU)
# ══════════════════════════════════════════════

class OCREngine:
    """
    Utilise docTR (mindee) — meilleur rapport précision/vitesse sur CPU.
    Fallback sur Tesseract si docTR non disponible.
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
            # det_arch: db_resnet50 = meilleure détection
            # reco_arch: crnn_vgg16_bn = meilleure reconnaissance FR
            self.model = ocr_predictor(
                det_arch='db_resnet50',
                reco_arch='crnn_vgg16_bn',
                pretrained=True,
                assume_straight_pages=True  # ordonnances = pages droites
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

    def extract_text(self, image: Image.Image) -> tuple:
        """Retourne (texte_extrait, confiance_0_1)."""
        if self.use_donut and self.engine_name.startswith("Donut"):
            return self._extract_donut(image)
        elif self.engine_name.startswith("docTR"):
            return self._extract_doctr(image)
        else:
            return self._extract_tesseract(image)

    def _extract_doctr(self, image: Image.Image) -> tuple:
        """Extraction via docTR."""
        # docTR accepte numpy array RGB
        arr = np.array(image.convert("RGB"))

        # docTR est un modèle de Deep Learning (CNN) entraîné sur des images natives.
        # La binarisation (adaptiveThreshold) détruit les ombres et textures nécessaires au modèle, 
        # ce qui produit des lettres éparpillées et illisibles.
        # Nous lui passons donc l'image RGB non-altérée.

        result = self.model([arr])

        # Extraction texte + confiances
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
                max_length=1024,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        seq = self.processor.batch_decode(outputs.tolist())[0]
        seq = seq.replace(self.processor.tokenizer.eos_token, "")
        seq = seq.replace(self.processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", " ", seq).strip()
        return seq, 0.85  # Donut ne retourne pas de confiance directe

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
        # Convertir en niveaux de gris
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr

        # Débruitage
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Binarisation adaptative
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
    SEARCH_URL = "https://medicament.ma/?s={query}"
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

            url = cls.SEARCH_URL.format(
                query=requests.utils.quote(nom_medicament)
            )
            resp = requests.get(url, headers=cls.HEADERS, timeout=10)
            resp.raise_for_status()

            result = cls._parse_search_results(resp.text, nom_medicament)
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

        # Pattern 1 : liens directs vers fiches médicaments
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            if (
                "/medicament/" in href or
                "/produit/" in href
            ) and text:
                results.append({
                    "nom": text,
                    "url": href,
                    "source": "medicament.ma"
                })

        # Pattern 2 : titres d'articles
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            text = h.get_text(strip=True)
            if query.lower() in text.lower() and len(text) < 100:
                link_tag = h.find("a")
                url = link_tag["href"] if link_tag else ""
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


class MedicalExtractor:
    """Extraction regex des éléments médicaux depuis le texte OCR."""

    DOSAGE_RE = re.compile(
        r'\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|UI|μg|µg|%|cp|cpr)\b',
        re.IGNORECASE
    )
    POSOLOGIE_RE = re.compile(
        r'(?:\d+\s*(?:fois?\s*(?:par\s*)?(?:jour|j)|comprimés?|cp?|gélules?)|'
        r'matin(?:\s+et\s+soir)?|soir|midi|à\s+jeun|avant\s+les\s+repas|'
        r'après\s+les\s+repas|صباحاً|مساءً|مرة.*يومياً)',
        re.IGNORECASE
    )
    DUREE_RE = re.compile(
        r'(?:pendant\s*)?\d+\s*(?:jours?|semaines?|mois|j|أيام|أسابيع|شهر)',
        re.IGNORECASE
    )
    ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
    LATIN_RE = re.compile(r'[a-zA-Z]')

    def detect_language(self, text: str) -> str:
        """Détecte la langue principale du texte."""
        ar = len(self.ARABIC_RE.findall(text))
        lat = len(self.LATIN_RE.findall(text))
        if ar == 0:
            return "fr"
        if lat == 0:
            return "ar"
        return "mixte" if ar / (ar + lat) < 0.6 else "ar"

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

        # Chargement DB locale
        db_path = Path(__file__).parent.parent / "data" / "medicaments_maroc.json"
        all_names = []

        if db_path.exists():
            with open(db_path, encoding="utf-8") as f:
                db = json.load(f)

            # Support both old format (flat list) and new format (dict with "medicaments")
            if isinstance(db, dict) and "medicaments" in db:
                for m in db["medicaments"]:
                    dci = m.get("dci", "")
                    if dci:
                        all_names.append((dci.lower(), dci))
                    for cn in m.get("noms_commerciaux", []):
                        all_names.append((cn.lower(), dci or cn))
                    for alias in m.get("ocr_aliases", []):
                        nom_base = dci if dci else (m.get("noms_commerciaux", [""])[0] if m.get("noms_commerciaux") else alias)
                        all_names.append((alias.lower(), nom_base))
            elif isinstance(db, list):
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
        for line in text.splitlines():
            line = line.strip()
            if not line or len(line) < 3:
                continue

            # Nettoyer la ligne (enlever numéros de liste, tirets et puces)
            line_clean = re.sub(r'^[\d\W_]+', '', line).strip()
            if not line_clean:
                line_clean = line

            # Fuzzy match
            dci, nom_commercial = None, None
            if name_index and process is not None:
                # We only attempt fuzzy matching if the line has a reasonable length
                if len(line_clean) >= 4:
                    match = process.extractOne(
                        line_clean.lower(), name_index,
                        scorer=fuzz.token_set_ratio,
                        score_cutoff=85
                    )
                    if match:
                        matched, score, _ = match
                        # Ensure we don't accidentally match a single short hallucinated word to a long medicine name
                        line_words = [w.lower() for w in line_clean.split() if len(w) >= 3]
                        matched_words = [w.lower() for w in matched.split() if len(w) >= 3]
                        
                        # Extra validation: At least one word must have high similarity, or it's a very good token match
                        is_valid = False
                        if score >= 90:
                            is_valid = True
                        else:
                            for lw in line_words:
                                for mw in matched_words:
                                    if fuzz.ratio(lw, mw) >= 80:
                                        is_valid = True
                                        break
                                if is_valid:
                                    break
                                    
                        if is_valid:
                            dci = name_to_dci.get(matched, matched)
                            nom_commercial = matched.title() \
                                if matched != dci.lower() else None

            dosage = self.DOSAGE_RE.search(line_clean)
            posologie = self.POSOLOGIE_RE.search(line_clean)
            duree = self.DUREE_RE.search(line_clean)

            # Extraire le vrai nom brut : isoler la partie de texte avant le dosage
            if dosage:
                nom_brut = line_clean[:dosage.start()].strip()
                if not nom_brut or len(nom_brut) < 3:
                    nom_brut = line_clean
            else:
                nom_brut = line_clean

            # Ligne utile si contient dosage OU médicament reconnu
            if dosage or dci:
                meds.append(MedicamentExtrait(
                    nom_brut=nom_brut,
                    dci=dci,
                    nom_commercial=nom_commercial,
                    dosage=dosage.group(0) if dosage else None,
                    posologie=posologie.group(0) if posologie else None,
                    duree=duree.group(0) if duree else None,
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
            avertissements=warns
        )


# ══════════════════════════════════════════════
# SECTION 5 — SINGLETON GLOBAL
# ══════════════════════════════════════════════

_instance = None


def get_ocr(use_donut=False, verify_online=True) -> OrdonnanceOCR:
    """Retourne l'instance singleton du pipeline OCR."""
    global _instance
    if _instance is None:
        _instance = OrdonnanceOCR(
            use_donut=use_donut,
            verify_online=verify_online
        )
    return _instance
