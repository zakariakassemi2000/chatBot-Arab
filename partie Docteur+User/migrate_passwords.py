# migrate_passwords.py

import bcrypt
import logging
from database1 import MySQLDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_all_passwords():
    """Migre tous les mots de passe texte clair vers bcrypt"""
    
    db = MySQLDatabase()
    
    try:
        # Vérifier si la table users existe
        try:
            users = db.execute_query(
                "SELECT id, username, password FROM users",
                fetch_all=True
            )
        except Exception as e:
            logger.error(f"Erreur accès table users: {e}")
            return
        
        if not users:
            logger.info("Aucun utilisateur trouvé")
            return
        
        migrated_count = 0
        error_count = 0
        
        for user in users:
            try:
                password = user['password']
                
                # Vérifier si c'est déjà un hash bcrypt
                if password and (password.startswith('$2b$') or password.startswith('$2a$')):
                    logger.debug(f"Utilisateur {user['username']} a déjà un hash bcrypt")
                    continue
                
                logger.info(f"Migration de l'utilisateur: {user['username']}")
                
                # Hasher le mot de passe texte clair
                salt = bcrypt.gensalt()
                hashed = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
                
                # Mettre à jour dans la base
                result = db.execute_query(
                    "UPDATE users SET password = %s WHERE id = %s",
                    (hashed, user['id'])
                )
                
                if result:
                    migrated_count += 1
                    logger.info(f"✓ Utilisateur {user['username']} migré avec succès")
                else:
                    error_count += 1
                    logger.error(f"✗ Échec migration pour {user['username']}")
                    
            except Exception as e:
                error_count += 1
                logger.error(f"✗ Erreur pour {user['username']}: {e}")
        
        logger.info(f"Migration terminée! {migrated_count} utilisateurs migrés, {error_count} erreurs.")
        
    except Exception as e:
        logger.error(f"Erreur lors de la migration: {e}")

def create_test_user():
    """Crée un utilisateur de test avec mot de passe hashé"""
    
    db = MySQLDatabase()
    
    try:
        # Vérifier si l'utilisateur existe déjà
        existing = db.execute_query(
            "SELECT id FROM users WHERE username = 'test'",
            fetch_one=True
        )
        
        if existing:
            logger.info("L'utilisateur test existe déjà")
            return
        
        # Créer le hash
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw('test123'.encode('utf-8'), salt).decode('utf-8')
        
        # Insérer l'utilisateur
        db.execute_query("""
            INSERT INTO users (username, email, password, role, full_name, phone, ville, actif) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            'test',
            'test@example.com',
            hashed,
            'patient',
            'Utilisateur Test',
            '0612345678',
            'Casablanca',
            True
        ))
        
        logger.info("✅ Utilisateur test créé avec succès (login: test / password: test123)")
        
    except Exception as e:
        logger.error(f"Erreur création utilisateur test: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("MIGRATION DES MOTS DE PASSE")
    print("=" * 50)
    
    choice = input("1. Migrer tous les mots de passe\n2. Créer un utilisateur test\nChoix (1 ou 2): ")
    
    if choice == "1":
        migrate_all_passwords()
    elif choice == "2":
        create_test_user()
    else:
        print("Choix invalide")