"""
Configuration centralisée pour SHIFA AI.
Charge les variables d'environnement et les datasets de référence une seule fois.
"""
import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("shifa")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY absente dans .env — l'extraction VLM ne fonctionnera pas.")

EXCEL_MEDICAMENTS = PROJECT_ROOT / "data" / "ref-des-medicaments-cnops-2014.xlsx"
EXCEL_DISPOSITIFS = PROJECT_ROOT / "data" / "dispositifs-medicaux-admis-au-remboursement-cnops-2014 (1).xls"

MAX_UPLOAD_SIZE_MB = 10

# ── Chargement unique des datasets ───────────────────────────────────
def _load_excel(path: Path, label: str) -> pd.DataFrame:
    """Charge un fichier Excel avec gestion d'erreur propre."""
    if not path.exists():
        logger.warning(f"Fichier {label} introuvable : {path}")
        return pd.DataFrame()
    try:
        df = pd.read_excel(path)
        logger.info(f"✅ {label} chargé ({len(df)} lignes)")
        return df
    except Exception as e:
        logger.error(f"❌ Erreur chargement {label}: {e}")
        return pd.DataFrame()

df_medicaments = _load_excel(EXCEL_MEDICAMENTS, "Référentiel Médicaments")
df_dispositifs = _load_excel(EXCEL_DISPOSITIFS, "Dispositifs CNOPS")

# Liste de tous les noms de médicaments (pour fuzzy matching)
if not df_medicaments.empty and "NOM" in df_medicaments.columns:
    LIST_MEDICAMENTS = df_medicaments["NOM"].dropna().astype(str).tolist()
else:
    logger.warning("Colonne 'NOM' absente — fallback sur liste minimale")
    LIST_MEDICAMENTS = [
        "Doliprane", "Augmentin", "Aspegic", "Spasfon",
        "Smecta", "Amoxicilline", "Clamoxyl", "Voltarene",
    ]
