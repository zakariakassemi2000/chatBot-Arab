"""
Correction des noms de médicaments via fuzzy matching sur la base CNOPS.
"""
from rapidfuzz import process, fuzz
from engine.vision_ocr.config import LIST_MEDICAMENTS, logger


def correct_medication_name(raw_name: str, threshold: int = 80):
    """
    Corrige le nom du médicament extrait par le VLM en utilisant le fuzzy matching
    sur la base de référence CNOPS.

    Returns:
        tuple: (nom_corrigé, score_de_confiance entre 0 et 1)
    """
    if not isinstance(raw_name, str) or len(raw_name) < 3:
        return raw_name, 0.0

    match = process.extractOne(
        raw_name.upper(),
        [m.upper() for m in LIST_MEDICAMENTS],
        scorer=fuzz.WRatio,
    )

    if match is None:
        return raw_name, 0.0

    best_match, score, _ = match

    # Retrouver le nom avec la casse originale
    try:
        original_match = next(m for m in LIST_MEDICAMENTS if m.upper() == best_match)
    except StopIteration:
        original_match = best_match

    confidence = round(score / 100.0, 2)

    if score >= threshold:
        logger.debug(f"Correction: '{raw_name}' → '{original_match}' ({confidence})")
        return original_match, confidence

    return raw_name, confidence
