# Configuration du site
SITE_NAME = "Parapharmacie Pro"
SITE_ICON = "💊"

# Catégories de produits
CATEGORIES = [
    "Visage",
    "Maquillage",
    "Corps",
    "Cheveux",
    "Bébé & Maman",
    "Homme",
    "Hygiène",
    "Solaire",
    "Santé",
    "Para-médical",
    "Bio",
    "PROMOTION"
]

# Configuration base de données
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "pharma_pro"
}

# Prix format
DEVISE = "MAD"

# Seuils d'alerte
STOCK_ALERT_THRESHOLD = 5
EXPIRY_ALERT_DAYS = 30
FREE_SHIPPING_THRESHOLD = 500
SHIPPING_COST = 30