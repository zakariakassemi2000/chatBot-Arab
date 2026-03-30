# auth_manager.py

import streamlit as st
import bcrypt
import logging
from app_config import Config

logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self, database):
        self.db = database
    
    def hash_password(self, password):
        """Hash le mot de passe avec bcrypt"""
        try:
            salt = bcrypt.gensalt(rounds=Config.BCRYPT_ROUNDS)
            # Retourner le hash comme string
            return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        except Exception as e:
            logger.error(f"Erreur lors du hashage: {e}")
            raise
    
    def verify_password(self, password, hashed):
        """Vérifie le mot de passe"""
        try:
            # Vérifier si le hash est valide
            if not hashed:
                return False
            
            # S'assurer que le hash est en bytes
            if isinstance(hashed, str):
                hashed = hashed.encode('utf-8')
            
            # Vérifier le format bcrypt (commence par $2b$ ou $2a$)
            if not (hashed.startswith(b'$2b$') or hashed.startswith(b'$2a$')):
                logger.warning("Hash non bcrypt détecté")
                return False
            
            return bcrypt.checkpw(password.encode('utf-8'), hashed)
            
        except Exception as e:
            logger.error(f"Erreur vérification mot de passe: {e}")
            return False
    
    def register_user(self, username, email, password, role, **kwargs):
        """Inscription d'un nouvel utilisateur"""
        try:
            # Vérifier si l'utilisateur existe déjà
            existing = self.db.get_user_by_username(username)
            if existing:
                return {"success": False, "message": "Nom d'utilisateur déjà utilisé"}
            
            existing_email = self.db.get_user_by_email(email)
            if existing_email:
                return {"success": False, "message": "Email déjà utilisé"}
            
            # Vérifier la longueur du mot de passe
            if len(password) < 6:
                return {"success": False, "message": "Le mot de passe doit contenir au moins 6 caractères"}
            
            # Hasher le mot de passe
            hashed_password = self.hash_password(password)
            
            # Insérer l'utilisateur
            user_data = {
                'username': username,
                'email': email,
                'password': hashed_password,
                'role': role,
                'full_name': kwargs.get('full_name', ''),
                'phone': kwargs.get('phone', ''),
                'ville': kwargs.get('ville', ''),
                'actif': True,
                'created_at': 'NOW()'
            }
            
            user_id = self.db.insert_user(user_data)
            
            if not user_id:
                return {"success": False, "message": "Erreur lors de la création de l'utilisateur"}
            
            # Créer les paramètres par défaut (si la table existe)
            try:
                self.db.execute_query(
                    "INSERT INTO parametres (user_id) VALUES (%s)",
                    (user_id,)
                )
            except Exception as e:
                logger.warning(f"Table parametres non disponible: {e}")
            
            # Log l'action
            self.log_action(user_id, "INSCRIPTION", f"Nouvel utilisateur {role}")
            
            return {
                "success": True, 
                "message": "Inscription réussie ! Vous pouvez maintenant vous connecter.", 
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'inscription: {e}")
            return {"success": False, "message": f"Erreur lors de l'inscription: {str(e)}"}
    
    def login_user(self, username, password):
        """Connexion utilisateur"""
        try:
            # Validation des entrées
            if not username or not password:
                return {"success": False, "message": "Veuillez remplir tous les champs"}
            
            # Récupérer l'utilisateur
            user = self.db.get_user_by_username(username)
            
            if not user:
                logger.warning(f"Tentative de connexion avec utilisateur inexistant: {username}")
                return {"success": False, "message": "Nom d'utilisateur ou mot de passe incorrect"}
            
            # Vérifier si le compte est actif
            if not user.get('actif', True):
                return {"success": False, "message": "Ce compte est désactivé. Contactez l'administrateur."}
            
            # Vérifier le mot de passe
            password_valid = False
            
            # Cas 1: Mot de passe en texte clair (pour migration)
            if self.is_plaintext_password(user['password']):
                if user['password'] == password:
                    # Migrer vers bcrypt
                    return self.migrate_password(user, password)
                else:
                    password_valid = False
            
            # Cas 2: Mot de passe hashé avec bcrypt
            else:
                password_valid = self.verify_password(password, user['password'])
            
            if password_valid:
                # Préparer les données utilisateur pour la session
                user_data = {
                    'id': user['id'],
                    'username': user['username'],
                    'role': user['role'],
                    'full_name': user.get('full_name', ''),
                    'email': user.get('email', ''),
                    'phone': user.get('phone', ''),
                    'ville': user.get('ville', '')
                }
                
                # Mettre à jour la dernière connexion
                try:
                    self.db.update_last_login(user['id'])
                except Exception as e:
                    logger.warning(f"Erreur mise à jour dernière connexion: {e}")
                
                # Log l'action
                self.log_action(user['id'], "CONNEXION", "Connexion réussie")
                
                logger.info(f"Connexion réussie: {username}")
                return {"success": True, "user": user_data}
            
            # Mot de passe incorrect
            logger.warning(f"Mot de passe incorrect pour: {username}")
            return {"success": False, "message": "Nom d'utilisateur ou mot de passe incorrect"}
            
        except Exception as e:
            logger.error(f"Erreur lors de la connexion: {e}")
            return {"success": False, "message": f"Erreur de connexion: {str(e)}"}
    
    def is_plaintext_password(self, password_hash):
        """Vérifie si le mot de passe est en texte clair (pas un hash bcrypt)"""
        if not password_hash:
            return True
        # Un hash bcrypt commence toujours par $2b$ ou $2a$ et fait 60 caractères
        return not (password_hash.startswith('$2b$') or password_hash.startswith('$2a$'))
    
    def migrate_password(self, user, plain_password):
        """Migre un mot de passe texte clair vers bcrypt"""
        try:
            logger.info(f"Migration du mot de passe pour l'utilisateur {user['username']}")
            
            # Hasher avec bcrypt
            new_hash = self.hash_password(plain_password)
            
            # Mettre à jour dans la base de données
            update_success = self.db.execute_query(
                "UPDATE users SET password = %s WHERE id = %s",
                (new_hash, user['id'])
            )
            
            if update_success:
                logger.info(f"Mot de passe migré avec succès pour {user['username']}")
                
                # Connexion réussie
                user_data = {
                    'id': user['id'],
                    'username': user['username'],
                    'role': user['role'],
                    'full_name': user.get('full_name', ''),
                    'email': user.get('email', ''),
                    'phone': user.get('phone', ''),
                    'ville': user.get('ville', '')
                }
                
                try:
                    self.db.update_last_login(user['id'])
                except:
                    pass
                    
                self.log_action(user['id'], "CONNEXION", "Connexion réussie (migration automatique)")
                
                return {"success": True, "user": user_data}
            else:
                return {"success": False, "message": "Erreur lors de la migration du mot de passe"}
                
        except Exception as e:
            logger.error(f"Erreur lors de la migration: {e}")
            return {"success": False, "message": f"Erreur lors de la migration: {str(e)}"}
    
    def log_action(self, user_id, action, details=""):
        """Enregistre une action dans les logs"""
        try:
            ip_address = "unknown"
            try:
                # Récupérer l'IP si disponible
                ip_address = st.query_params.get('ip', 'unknown')
            except:
                pass
            
            # Vérifier si la table logs existe
            try:
                self.db.execute_query(
                    """INSERT INTO logs (user_id, action, details, ip_address, timestamp) 
                       VALUES (%s, %s, %s, %s, NOW())""",
                    (user_id, action, details, ip_address)
                )
            except Exception as e:
                # La table n'existe peut-être pas, on ignore
                logger.debug(f"Table logs non disponible: {e}")
                
        except Exception as e:
            logger.error(f"Erreur lors du logging: {e}")
    
    def change_password(self, user_id, old_password, new_password):
        """Change le mot de passe d'un utilisateur"""
        try:
            # Récupérer l'utilisateur
            user = self.db.get_user_by_id(user_id)
            
            if not user:
                return {"success": False, "message": "Utilisateur non trouvé"}
            
            # Vérifier l'ancien mot de passe
            if not self.verify_password(old_password, user['password']):
                return {"success": False, "message": "Ancien mot de passe incorrect"}
            
            # Vérifier le nouveau mot de passe
            if len(new_password) < 6:
                return {"success": False, "message": "Le nouveau mot de passe doit contenir au moins 6 caractères"}
            
            # Hasher le nouveau mot de passe
            new_hash = self.hash_password(new_password)
            
            # Mettre à jour
            self.db.execute_query(
                "UPDATE users SET password = %s WHERE id = %s",
                (new_hash, user_id)
            )
            
            self.log_action(user_id, "CHANGE_PASSWORD", "Changement de mot de passe")
            
            return {"success": True, "message": "Mot de passe changé avec succès"}
            
        except Exception as e:
            logger.error(f"Erreur changement mot de passe: {e}")
            return {"success": False, "message": f"Erreur: {str(e)}"}
    
    def reset_password(self, email):
        """Réinitialise le mot de passe (envoi d'email)"""
        try:
            # Vérifier si l'email existe
            user = self.db.get_user_by_email(email)
            
            if not user:
                return {"success": False, "message": "Aucun compte associé à cet email"}
            
            # Générer un nouveau mot de passe temporaire
            import secrets
            import string
            temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
            
            # Hasher le mot de passe temporaire
            temp_hash = self.hash_password(temp_password)
            
            # Mettre à jour
            self.db.execute_query(
                "UPDATE users SET password = %s WHERE id = %s",
                (temp_hash, user['id'])
            )
            
            # TODO: Envoyer l'email avec le mot de passe temporaire
            logger.info(f"Mot de passe réinitialisé pour {email}. Temp: {temp_password}")
            
            self.log_action(user['id'], "RESET_PASSWORD", "Réinitialisation de mot de passe")
            
            return {
                "success": True, 
                "message": "Un email de réinitialisation a été envoyé",
                "temp_password": temp_password  # À retirer en production
            }
            
        except Exception as e:
            logger.error(f"Erreur réinitialisation mot de passe: {e}")
            return {"success": False, "message": f"Erreur: {str(e)}"}