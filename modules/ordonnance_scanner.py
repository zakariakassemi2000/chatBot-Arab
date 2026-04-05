# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Module Scanner d'Ordonnance
  OCR (PaddleOCR / français) + Extraction + Matching Flou
═══════════════════════════════════════════════════════════════════════
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image
from rapidfuzz import fuzz, process

logger = logging.getLogger("shifa.ordonnance")

# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class MedicamentExtrait:
    """Un médicament extrait de l'ordonnance."""
    nom_brut: str                          # Texte brut extrait par OCR
    dosage: Optional[str] = None           # Ex: "500 mg"
    forme: Optional[str] = None            # Ex: "comprimé"
    frequence: Optional[str] = None        # Ex: "3 fois par jour"
    duree: Optional[str] = None            # Ex: "pendant 5 jours"
    # Résultat du matching
    nom_match: Optional[str] = None        # Nom du médicament trouvé dans la base
    principe_actif: Optional[str] = None   # Principe actif associé
    formes_disponibles: List[str] = field(default_factory=list)
    score_match: float = 0.0              # Score de confiance du matching (0-100)
    est_reference: bool = False           # Trouvé dans la base marocaine ?


@dataclass
class ResultatOrdonnance:
    """Résultat complet de l'analyse d'une ordonnance."""
    texte_brut: str                        # Texte OCR complet
    medicaments: List[MedicamentExtrait]    # Liste des médicaments extraits
    score_global: float = 0.0             # Score de confiance global (0-100)
    erreur: Optional[str] = None           # Message d'erreur éventuel


# ─────────────────────────────────────────────────────────────
# BASE DE DONNÉES MÉDICAMENTS
# ─────────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent / "data" / "medicaments_maroc.json"
_MEDICAMENTS_CACHE: Optional[List[Dict]] = None


def load_medicaments_db() -> List[Dict]:
    """Charge la base de médicaments marocains depuis le fichier JSON.
    Supporte les deux formats :
      - Ancien : [{"nom": ..., "principe": ..., "formes": [...]}]
      - Nouveau : {"medicaments": [{"dci": ..., "noms_commerciaux": [...], "formes": [...]}]}
    Normalise toujours vers le format plat [{"nom", "principe", "formes"}].
    """
    global _MEDICAMENTS_CACHE
    if _MEDICAMENTS_CACHE is not None:
        return _MEDICAMENTS_CACHE

    try:
        if _DB_PATH.exists():
            with open(_DB_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)

            # Nouveau format : {"medicaments": [...]}
            if isinstance(raw, dict) and "medicaments" in raw:
                flat = []
                for entry in raw["medicaments"]:
                    dci = entry.get("dci", "")
                    formes = entry.get("formes", [])
                    # Créer une entrée pour chaque nom commercial
                    for nom_com in entry.get("noms_commerciaux", []):
                        flat.append({
                            "nom": nom_com,
                            "principe": dci,
                            "formes": formes,
                        })
                    # Aussi ajouter le DCI lui-même s'il n'est pas déjà nom commercial
                    noms_lower = [n.lower() for n in entry.get("noms_commerciaux", [])]
                    if dci and dci.lower() not in noms_lower:
                        flat.append({
                            "nom": dci,
                            "principe": dci,
                            "formes": formes,
                        })
                _MEDICAMENTS_CACHE = flat
            # Ancien format : liste plate
            elif isinstance(raw, list):
                _MEDICAMENTS_CACHE = raw
            else:
                logger.warning("[Ordonnance] Format JSON non reconnu")
                _MEDICAMENTS_CACHE = []

            logger.info(f"[Ordonnance] Base chargée: {len(_MEDICAMENTS_CACHE)} médicaments")
            return _MEDICAMENTS_CACHE
        else:
            logger.warning(f"[Ordonnance] Fichier base introuvable: {_DB_PATH}")
            return []
    except Exception as e:
        logger.error(f"[Ordonnance] Erreur chargement base: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# PADDLEOCR — MOTEUR OCR
# ─────────────────────────────────────────────────────────────

_paddle_ocr = None  # Singleton — chargé une seule fois


def _get_ocr():
    """
    Retourne l'instance PaddleOCR 3.x (singleton).
    - lang='fr' : dans LATIN_LANGS → routé automatiquement sur PP-OCRv5
    - use_textline_orientation remplace le param use_angle_cls (déprécié v3)
    - PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK : évite le check réseau au démarrage
    - show_log / use_gpu / enable_mkldnn / ocr_version='PP-OCRv4' supprimés
      (invalides dans PaddleOCR 3.x)
    """
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            import os as _os
            _os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

            from paddleocr import PaddleOCR
            logger.info("[OCR] Initialisation PaddleOCR PP-OCRv5 (français/latin)…")

            _paddle_ocr = PaddleOCR(
                lang="fr",                           # PP-OCRv5 Latin + accents FR
                use_doc_orientation_classify=False,  # inutile pour ordonnances
                use_doc_unwarping=False,             # inutile pour ordonnances
                use_textline_orientation=False,      # désactivé pour perfs
                enable_mkldnn=False,                 # FIX: evite le crash PIR/oneDNN
                text_rec_score_thresh=0.4,
            )
            logger.info("[OCR] PaddleOCR PP-OCRv5 (francais) pret")
        except ImportError:
            raise RuntimeError(
                "PaddleOCR n'est pas installé. "
                "Exécutez : pip install paddlepaddle paddleocr"
            )
        except Exception as e:
            logger.error(f"[OCR] Erreur initialisation PaddleOCR: {e}")
            raise RuntimeError(f"Impossible d'initialiser PaddleOCR: {e}")
    return _paddle_ocr


def _pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convertit une PIL Image en tableau numpy BGR attendu par PaddleOCR.
    Gère les images RGBA, palette, niveaux de gris, etc.
    """
    # Normaliser en RGB
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Redimensionner si trop petit (PaddleOCR préfère ≥ 640px de large)
    w, h = image.size
    if w < 640:
        ratio = 640 / w
        image = image.resize((640, int(h * ratio)), Image.LANCZOS)

    # PIL (RGB) → NumPy (BGR pour OpenCV/Paddle)
    arr = np.array(image)
    return arr[:, :, ::-1]  # RGB → BGR


def ocr_extract_text(image: Image.Image) -> str:
    """
    Exécute PaddleOCR en français sur une image PIL.
    Retourne le texte brut extrait, lignes triées de haut en bas.
    """
    try:
        ocr = _get_ocr()
        img_array = _pil_to_numpy(image)

        # PaddleOCR 3.x : predict() est un generateur — il faut l'iterer
        # Chaque element est un dict PaddleX avec cles rec_text/rec_score/dt_poly
        lines: List[Tuple[float, str, float]] = []

        # --- Format PaddleOCR 3.x (predict() generator) ---
        if hasattr(ocr, 'predict'):
            for page in ocr.predict(img_array):
                if page is None:
                    continue
                # PaddleX OCRResult est un dict ou un objet
                get_attr = lambda obj, k: obj.get(k, []) if isinstance(obj, dict) else getattr(obj, k, [])
                
                boxes = get_attr(page, 'dt_polys') or get_attr(page, 'boxes')
                texts = get_attr(page, 'rec_texts') or get_attr(page, 'texts')
                scores = get_attr(page, 'rec_scores')
                
                if not scores:
                    scores = [1.0] * len(texts)
                    
                for bbox, txt, conf in zip(boxes, texts, scores):
                    if not txt or float(conf) < 0.4:
                        continue
                    try:
                        y_top = float(bbox[0][1])
                    except (TypeError, IndexError, KeyError):
                        y_top = 0.0
                    lines.append((y_top, str(txt), float(conf)))

        # --- Fallback format PaddleOCR 2.x (ocr() liste de listes) ---
        else:
            raw_result = ocr.ocr(img_array)
            raw = raw_result[0] if raw_result else []
            for item in (raw or []):
                if item is None:
                    continue
                try:
                    bbox, (text, confidence) = item
                    y_top = (bbox[0][1] + bbox[1][1]) / 2
                    if confidence >= 0.4:
                        lines.append((y_top, text, confidence))
                except Exception:
                    continue

        # Trier par position verticale
        lines.sort(key=lambda x: x[0])

        # Assembler le texte
        text_parts = [line[1] for line in lines]
        text = "\n".join(text_parts)

        # Nettoyage basique
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        # Log statistiques
        if lines:
            avg_conf = sum(l[2] for l in lines) / len(lines)
            logger.info(
                f"[OCR] Texte extrait: {len(text)} caractères, "
                f"{len(lines)} blocs, confiance moy.: {avg_conf:.1%}"
            )

        return text

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"[OCR] Erreur: {e}")
        raise RuntimeError(f"Erreur OCR: {str(e)}")


# ─────────────────────────────────────────────────────────────
# EXTRACTION DES MÉDICAMENTS (REGEX)
# ─────────────────────────────────────────────────────────────

# Patterns regex pour extraction — français médical
_PATTERNS = {
    "dosage": re.compile(
        r'(\d+(?:[.,]\d+)?)\s*'
        r'(mg|g|ml|µg|mcg|ui|%|cp)',
        re.IGNORECASE
    ),
    "forme": re.compile(
        r'\b(comprim[eé]s?|g[eé]lules?|sirop|suspension|sachet|'
        r'poudre|a[eé]rosol|solution|cr[eè]me|pommade|ovule|'
        r'suppositoire|gouttes?|injection|spray|patch|'
        r'g[eé]l|collyre|lyoc|effervescent|buvable)\b',
        re.IGNORECASE
    ),
    "frequence": re.compile(
        r'(\d+)\s*(?:fois|x)\s*(?:par\s*)?(jour|j(?:our)?|semaine|matin|soir|midi)|'
        r'(matin\s*(?:et|,)?\s*(?:midi\s*(?:et|,)?\s*)?soir)|'
        r'(toutes?\s*les?\s*\d+\s*h(?:eures?)?)|'
        r'(\d+\s*(?:cp|g[eé]l|comp)\s*/\s*(?:j|jour))',
        re.IGNORECASE
    ),
    "duree": re.compile(
        r'(?:pendant|durant|pour)\s*(\d+)\s*(jours?|semaines?|mois)|'
        r'(\d+)\s*(jours?|semaines?|mois)\s*(?:de\s*traitement)',
        re.IGNORECASE
    ),
}

def extract_medications(text: str) -> List[MedicamentExtrait]:
    """
    Parse le texte OCR pour extraire les médicaments et leurs attributs.
    Traite le texte ligne par ligne pour plus de robustesse.
    """
    medicaments = []
    db = load_medicaments_db()

    lines = [L.strip() for L in text.split('\n') if L.strip()]
    
    # Numérotation (ex "1)", "A)", "-") et correctif OCR (ex: "≤" confondu avec un tiret/chevron)
    re_bullet = re.compile(r'^([a-zA-Z0-9]{1,2}[\)\.\-]|[-•●\*>≤])\s*')
    
    re_dosage = re.compile(r'\b(\d+(?:[.,]\d+)?\s*(?:mg|g|ml|µg|ui|cp|suppo|sachet|g[eé]l|amp|inj|iu))\b', re.I)
    re_dosage_ocr = re.compile(r'\b(\d*\s*(?:-g|ig|mg))\b', re.I)
    re_med_keywords = re.compile(r'\b(inj|suppos?|sirop|collyre|pommade|cr[eé]me|g[eé]lules?|comprim[eé]s?|cp|sachets?|ampoules?)\b', re.I)

    for line in lines:
        if len(line) < 4 or any(kw in line.lower() for kw in [
            'docteur', 'dr.', 'tél', 'tel', 'hopital', 'maladie',
            'clinique', 'urgence', 'chef', 'rhumatologue', 'patient', 'rdv'
        ]):
            continue

        clean_line = line.replace('≤', ' ')
        clean_line = re_bullet.sub('', clean_line).strip()
        
        dosage_match = re_dosage.search(clean_line) or re_dosage_ocr.search(clean_line)
        keyword_match = re_med_keywords.search(clean_line)

        words = clean_line.split()
        first_word = words[0] if words else ""
        first_two = " ".join(words[:2]) if len(words) > 1 else ""

        match_result = None
        m1 = m2 = None
        if db:
            m2 = fuzzy_match_medication(first_two, db) if len(first_two) > 3 else None
            m1 = fuzzy_match_medication(first_word, db) if len(first_word) > 3 else None
            match_result = m2 if m2 else m1

        if not dosage_match and not keyword_match and not match_result:
            if first_word and first_word[0].isupper() and len(clean_line) < 30:
                pass
            else:
                continue

        if dosage_match:
            idx = dosage_match.start()
            nom_brut = clean_line[:idx].strip()
        elif keyword_match:
            idx = keyword_match.start()
            nom_brut = clean_line[:idx].strip()
        else:
            nom_brut = first_two if match_result == m2 else first_word

        nom_brut = re.sub(r'[^A-Za-zÀ-ÿéèêëîïôùûü0-9\s-]', ' ', nom_brut).strip()
        
        if not nom_brut:
            nom_brut = first_word

        if len(nom_brut) < 3:
            continue

        med = MedicamentExtrait(nom_brut=nom_brut.capitalize())
        
        if dosage_match:
            med.dosage = dosage_match.group(1).strip()

        forme_m = _PATTERNS["forme"].search(clean_line)
        if forme_m: med.forme = forme_m.group(0).strip().lower()

        freq_m = _PATTERNS["frequence"].search(clean_line)
        if freq_m: med.frequence = freq_m.group(0).strip()

        duree_m = _PATTERNS["duree"].search(clean_line)
        if duree_m: med.duree = duree_m.group(0).strip()

        if match_result:
            med.nom_match = match_result["nom"]
            med.principe_actif = match_result["principe"]
            med.formes_disponibles = match_result["formes"]
            med.score_match = match_result["score"]
            med.est_reference = True

        medicaments.append(med)

    return medicaments


# ─────────────────────────────────────────────────────────────
# MATCHING FLOU
# ─────────────────────────────────────────────────────────────

def fuzzy_match_medication(name: str, db: List[Dict], threshold: int = 70) -> Optional[Dict]:
    """
    Recherche floue d'un nom de médicament dans la base marocaine.
    Exige des scores strictes (ratio >= 70 ou token_sort_ratio >= 80)
    pour éviter les faux positifs aberrants (ex: "matin" -> "Augmentin").
    """
    if not db or not name:
        return None

    db_names = [m["nom"] for m in db]

    result = process.extractOne(name.lower(), db_names, scorer=fuzz.ratio, score_cutoff=threshold)
    if not result:
        result = process.extractOne(name.lower(), db_names, scorer=fuzz.token_sort_ratio, score_cutoff=80)

    if result:
        matched_name, score, idx = result
        med_data = db[idx]
        return {
            "nom": med_data["nom"],
            "principe": med_data["principe"],
            "formes": med_data["formes"],
            "score": round(score, 1)
        }
    return None


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def analyze_ordonnance(image: Image.Image) -> ResultatOrdonnance:
    """
    Pipeline complet d'analyse d'une ordonnance :
    Image → OCR PaddleOCR → Extraction → Matching Flou → Résultats
    """
    try:
        # 1. OCR
        texte = ocr_extract_text(image)

        if not texte or len(texte.strip()) < 10:
            return ResultatOrdonnance(
                texte_brut=texte or "",
                medicaments=[],
                score_global=0.0,
                erreur="Le texte extrait est trop court ou vide. "
                       "Vérifiez la qualité de l'image (netteté, éclairage, cadrage)."
            )

        # 2. Extraction des médicaments
        medicaments = extract_medications(texte)

        # 3. Score global
        if medicaments:
            scores = [m.score_match for m in medicaments if m.est_reference]
            score_global = sum(scores) / len(scores) if scores else 30.0
        else:
            score_global = 10.0

        return ResultatOrdonnance(
            texte_brut=texte,
            medicaments=medicaments,
            score_global=round(score_global, 1)
        )

    except RuntimeError as e:
        return ResultatOrdonnance(
            texte_brut="",
            medicaments=[],
            score_global=0.0,
            erreur=str(e)
        )
    except Exception as e:
        logger.error(f"[Ordonnance] Erreur pipeline: {e}", exc_info=True)
        return ResultatOrdonnance(
            texte_brut="",
            medicaments=[],
            score_global=0.0,
            erreur=f"Erreur inattendue: {str(e)}"
        )
