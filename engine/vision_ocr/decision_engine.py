"""
Moteur de décision : orchestre correction + validation pour chaque médicament.
Combine la confiance VLM avec le score de fuzzy matching.
"""
import logging
from engine.vision_ocr.correction import correct_medication_name
from engine.vision_ocr.scraper_medicament_ma import search_medicament_ma
from engine.vision_ocr.validators import check_referentiel, check_cnops

logger = logging.getLogger("shifa.engine")


def analyze_medication(
    raw_name: str,
    raw_dosage: str = None,
    vlm_confidence: float = 0.0,
    posologie: str = None,
    duree: str = None,
) -> dict:
    """
    Orchestre la vérification complète pour un médicament.

    Args:
        raw_name: Nom brut extrait par le VLM
        raw_dosage: Dosage extrait par le VLM
        vlm_confidence: Confiance globale du VLM (0.0 à 1.0)
        posologie: Posologie extraite par le VLM
        duree: Durée de traitement

    Returns:
        dict avec status, confidence, prix, remboursable, etc.
    """
    result = {
        "raw_name": raw_name,
        "dosage": raw_dosage,
        "posologie": posologie,
        "duree": duree,
        "status": "unknown",
        "confidence": 0.0,
        "price": None,
        "remboursable": False,
        "type": "Unknown",
    }

    # 1. Correction fuzzy matching
    corrected_name, match_score = correct_medication_name(raw_name)
    result["corrected_name"] = corrected_name

    # Confiance combinée : moyenne pondérée VLM (40%) + fuzzy (60%)
    # Le fuzzy matching a plus de poids car il vérifie contre la base officielle
    if vlm_confidence > 0:
        combined = (vlm_confidence * 0.4) + (match_score * 0.6)
    else:
        combined = match_score
    result["confidence"] = round(combined, 2)

    # Déterminer le statut
    if combined < 0.5:
        result["status"] = "unknown"
    elif combined < 0.75:
        result["status"] = "suspect"
    else:
        result["status"] = "valid"

    # 2. Vérification Web (medicament.ma)
    web_info = search_medicament_ma(corrected_name)
    if web_info.get("found"):
        result["price"] = web_info.get("price")
        result["type"] = web_info.get("type")
        # Rehausser le statut si trouvé en ligne
        if result["status"] == "unknown" and combined > 0:
            result["status"] = "suspect"

    # 3. Vérification Locale (Prix & Remboursement)
    local_ref = check_referentiel(corrected_name)
    if local_ref.get("found_in_ref"):
        if result["price"] is None:
            result["price"] = local_ref.get("prix_public")
        
        taux = local_ref.get("taux_remboursement", "0%")
        if taux != "0%" and taux != "0" and taux != "NaN" and str(taux).lower() != "nan":
            result["remboursable"] = True

    if not result.get("remboursable"):
        cnops_info = check_cnops(corrected_name)
        result["remboursable"] = cnops_info.get("remboursable", False)

    logger.info(
        f"📋 {raw_name} → {corrected_name} | "
        f"Statut: {result['status']} | Confiance: {result['confidence']:.0%}"
    )

    return result
