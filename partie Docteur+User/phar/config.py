# Configuration du site
SITE_NAME = "Parapharmacie "
SITE_ICON = "💊"

# Design System - Rouge
COLORS = {
    "primary": "#dc2626",      # Rouge principal
    "primary_dark": "#b91c1c",  # Rouge foncé
    "primary_light": "#ef4444", # Rouge clair
    "primary_bg": "#fef2f2",    # Fond rouge très clair
    "secondary": "#4b5563",     # Gris
    "success": "#10b981",       # Vert
    "warning": "#f59e0b",       # Orange
    "danger": "#dc2626",        # Rouge
    "text_dark": "#1f2937",     # Texte foncé
    "text_light": "#6b7280",    # Texte gris
    "white": "#ffffff",
    "border": "#e5e7eb"
}

# Catégories
CATEGORIES = [
    "Visage", "Maquillage", "Corps", "Cheveux",
    "Bébé & Maman", "Homme", "Hygiène", "Solaire",
    "Santé", "Para-médical", "Bio", "Promotion"
]

# Base de données
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "pharma_pro"
}

# Paramètres
DEVISE = "MAD"
FREE_SHIPPING_THRESHOLD = 500
SHIPPING_COST = 30
STOCK_ALERT = 5
EXPIRY_ALERT_DAYS = 30

# URLs des images par catégorie (placeholders)
CATEGORY_IMAGES = {
    "Visage": "https://images.unsplash.com/photo-1556229010-aa3f7ff66b24?w=300",
    "Maquillage": "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?w=300",
    "Corps": "https://images.unsplash.com/photo-1616394584738-fc6e612e71b9?w=300",
    "Cheveux": "https://images.unsplash.com/photo-1527799820374-dcf8d9d4a388?w=300",
    "Bébé & Maman": "https://images.unsplash.com/photo-1515488042361-ee00e0ddd4e4?w=300",
    "Homme": "https://images.unsplash.com/photo-1617137968427-85924d800a22?w=300",
    "Hygiène": "https://images.unsplash.com/photo-1556228578-567ba127e37c?w=300",
    "Solaire": "https://images.unsplash.com/photo-1505156868549-74bcf7c9900a?w=300",
    "Santé": "https://images.unsplash.com/photo-1505751172876-fa1923c5c528?w=300",
    "Para-médical": "https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?w=300",
    "Bio": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=300",
    "Promotion": "https://images.unsplash.com/photo-1607082348824-0a96f2a4b9da?w=300"
}

# Image par défaut
DEFAULT_PRODUCT_IMAGE = "https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?w=300"