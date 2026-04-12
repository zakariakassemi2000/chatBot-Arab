"""
Scraper pour medicament.ma — recherche de prix et type de médicaments.
"""
import re
import logging
from functools import lru_cache

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("shifa.scraper")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


@lru_cache(maxsize=256)
def search_medicament_ma(med_name: str) -> dict:
    """
    Recherche le médicament sur medicament.ma et extrait le prix et le type.
    Les résultats sont mis en cache pour éviter les requêtes répétées.

    Returns:
        dict avec {found, name, price, type} ou {found: False, error}
    """
    url = f"https://medicament.ma/?s={requests.utils.quote(med_name)}"

    try:
        response = requests.get(url, headers=_HEADERS, timeout=10)
        if response.status_code != 200:
            logger.warning(f"medicament.ma HTTP {response.status_code} pour '{med_name}'")
            return {"found": False, "error": f"HTTP {response.status_code}"}

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("article")

        if not articles:
            logger.debug(f"Aucun résultat sur medicament.ma pour '{med_name}'")
            return {"found": False}

        first_article = articles[0]

        # Extraire le titre
        title_tag = first_article.find("h2") or first_article.find("h3")
        title = title_tag.text.strip() if title_tag else med_name

        text_content = first_article.text.lower()

        # Type : générique ou princeps
        med_type = "Générique" if "générique" in text_content else "Princeps"

        # Prix en DH
        price_match = re.search(
            r"(\d+[\.,]?\d*)\s*(?:dh|dhs|mad)", text_content, re.IGNORECASE
        )
        price = (
            float(price_match.group(1).replace(",", ".")) if price_match else None
        )

        logger.info(f"✅ medicament.ma: '{med_name}' → {title} | {price} DH | {med_type}")

        return {
            "found": True,
            "name": title,
            "price": price,
            "type": med_type,
        }

    except requests.Timeout:
        logger.warning(f"Timeout medicament.ma pour '{med_name}'")
        return {"found": False, "error": "Timeout"}
    except Exception as e:
        logger.error(f"Erreur scraper pour '{med_name}': {e}")
        return {"found": False, "error": str(e)}
