# notification_manager.py

import logging
from app_config import Config

logger = logging.getLogger(__name__)

class NotificationManager:
    def __init__(self, db_manager):
        self.db = db_manager
        self.setup_email_config()
    
    def setup_email_config(self):
        """Configure les paramètres email"""
        self.smtp_server = Config.SMTP_SERVER
        self.smtp_port = Config.SMTP_PORT
        self.email_address = Config.SMTP_USER
        self.email_password = Config.SMTP_PASSWORD
        self.email_from = Config.SMTP_FROM
    
    def create_notification(self, user_id, titre, message, type_notif="info"):
        """Crée une notification dans la base de données"""
        try:
            self.db.execute_query(
                "INSERT INTO notifications (user_id, titre, message, type_notification) VALUES (%s, %s, %s, %s)",
                (user_id, titre, message, type_notif)
            )
        except Exception as e:
            logger.error(f"Erreur création notification: {e}")
    
    def get_user_notifications(self, user_id, unread_only=False):
        """Récupère les notifications d'un utilisateur"""
        query = """
        SELECT * FROM notifications 
        WHERE user_id = %s
        """
        if unread_only:
            query += " AND lu = FALSE"
        query += " ORDER BY timestamp DESC LIMIT 10"
        
        return self.db.execute_query(query, (user_id,), fetch_all=True)
    
    def mark_as_read(self, notification_id):
        """Marque une notification comme lue"""
        self.db.execute_query(
            "UPDATE notifications SET lu = TRUE WHERE id = %s",
            (notification_id,)
        )