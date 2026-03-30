# app_config.py

class Config:
    # Version de l'application
    VERSION = "2.0.0"
    
    # Configuration de la base de données
    DB_HOST = "localhost"
    DB_USER = "root"
    DB_PASSWORD = ""
    DB_NAME = "ai_shifa_pro"
    
    # Configuration email
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    SMTP_USER = "votre_email@gmail.com"
    SMTP_PASSWORD = "votre_mot_de_passe"
    SMTP_FROM = "AI Shifa Pro <noreply@aishifa.ma>"
    
    # Configuration des dossiers
    UPLOAD_FOLDER = "uploads"
    ORDO_FOLDER = "ordonnances"
    REPORTS_FOLDER = "reports"
    
    # Sécurité
    BCRYPT_ROUNDS = 12
    SECRET_KEY = "votre_clé_secrète_très_longue_et_aléatoire"
    
    # Paramètres de session
    SESSION_TIMEOUT = 3600  # 1 heure
    
    # Paramètres IA
    CONFIDENCE_THRESHOLD = 0.75
    AUTO_ANALYSIS = True
    
    # URLs et endpoints
    BASE_URL = "http://localhost:8501"
    API_URL = "http://localhost:8000"
    
    # Paramètres de logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/app.log"
    
    # Paramètres de l'application
    APP_NAME = "AI Shifa Pro"
    APP_DESCRIPTION = "Plateforme de santé intelligente"
    SUPPORTED_LANGUAGES = ["fr", "ar", "en"]
    DEFAULT_LANGUAGE = "fr"
    
    # Paramètres des fichiers
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'pdf', 'dcm']
    
    # Paramètres des rendez-vous
    MIN_RDV_ADVANCE_DAYS = 1
    MAX_RDV_PER_DAY = 20
    RDV_DURATION_MINUTES = 30
    
    # Paramètres des notifications
    NOTIFICATION_EMAIL = True
    NOTIFICATION_SMS = False
    NOTIFICATION_PUSH = True
    
    # URLs des ressources
    ICON_URL = "https://img.icons8.com/color/96/000000/hospital.png"
    LOGO_URL = "https://img.icons8.com/color/96/000000/stethoscope.png"
      # 👇 AJOUTE CES LIGNES
    BASE_STORAGE = "storage/patients"
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'docx', 'txt'}
    MAX_FILE_SIZE = 10 * 1024 * 1024