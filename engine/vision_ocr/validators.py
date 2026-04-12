"""
Validation des médicaments contre les bases de données locales (CNOPS).
"""
import pandas as pd
from engine.vision_ocr.config import df_medicaments, df_dispositifs, logger


def check_referentiel(med_name: str) -> dict:
    """
    Vérifie si le médicament existe dans le référentiel des prix CNOPS.
    Utilise une recherche case-insensitive sur la colonne NOM.
    """
    if df_medicaments.empty or "NOM" not in df_medicaments.columns:
        return {"found_in_ref": False, "reason": "Dataset manquant"}

    # Recherche case-insensitive
    mask = df_medicaments["NOM"].astype(str).str.upper().str.contains(
        med_name.upper(), regex=False, na=False
    )
    matches = df_medicaments[mask]

    if matches.empty:
        return {"found_in_ref": False}

    row = matches.iloc[0]
    return {
        "found_in_ref": True,
        "prix_public": float(row.get("PPV", 0) or 0),
        "prix_hopital": float(row.get("PH", 0) or 0),
        "taux_remboursement": str(row.get("TAUX_REMBOURSEMENT", "0%")),
    }


def check_cnops(med_name: str) -> dict:
    """
    Vérifie l'éligibilité au remboursement dans la base CNOPS.
    Recherche ciblée au lieu d'un full-scan sur toutes les colonnes.
    """
    if df_dispositifs.empty:
        return {"remboursable": False, "reason": "Dataset manquant"}

    # Chercher dans les colonnes textuelles les plus pertinentes
    text_cols = df_dispositifs.select_dtypes(include=["object"]).columns
    if len(text_cols) == 0:
        return {"remboursable": False, "reason": "Pas de colonnes textuelles"}

    for col in text_cols:
        mask = df_dispositifs[col].astype(str).str.contains(
            med_name, case=False, regex=False, na=False
        )
        if mask.any():
            logger.debug(f"'{med_name}' trouvé dans CNOPS (colonne: {col})")
            return {"remboursable": True}

    return {"remboursable": False}
