# database1.py (Version SQLite)
import sqlite3
import logging
import time
import bcrypt
import os
from datetime import datetime
from app_config import Config

logger = logging.getLogger(__name__)

class MySQLDatabase:  # Keep name for compatibility with app code
    def __init__(self, max_retries=3):
        # We use a local SQLite file instead of a network MySQL connection
        self.db_path = getattr(Config, 'DB_PATH', 'ai_shifa_pro.db')
        self.connection = None
        self.max_retries = max_retries
        self.connect_with_retry()
        if self.connection:
            self.create_tables()

    def dict_factory(self, cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def connect_with_retry(self):
        """Connects to the local SQLite database."""
        try:
            # check_same_thread=False is needed if passing the object between threads (like in Streamlit)
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = self.dict_factory
            logger.info("Connexion à la base de données SQLite (locale) réussie.")
            return True
        except Exception as e:
            logger.error(f"Erreur de connexion SQLite: {e}")
            self.connection = None
            return False

    def ensure_connection(self):
        if not self.connection:
            return self.connect_with_retry()
        try:
            self.connection.cursor()
            return True
        except sqlite3.ProgrammingError:
            return self.connect_with_retry()

    def execute_query(self, query, params=None, fetch_all=False, fetch_one=False):
        import re
        """Exécute une requête SQL avec format SQLite (utilise ? au lieu de %s)"""
        # Conversion du style de paramètre MySQL vers SQLite
        query = query.replace('%s', '?')
        # Conversion DATE_FORMAT
        query = re.sub(r"DATE_FORMAT\(([^,]+),\s*'([^']+)'\)", lambda m: f"strftime('{m.group(2)}', {m.group(1)})", query)
        # Conversion des fonctions spécifiques MySQL
        query = query.replace('CURRENT_TIMESTAMP', "datetime('now', 'localtime')")
        query = query.replace('CURRENT_DATE', "date('now', 'localtime')")
        query = query.replace('CURDATE()', "date('now', 'localtime')")
        query = query.replace('NOW()', "datetime('now', 'localtime')")
        # INSERT IGNORE -> INSERT OR IGNORE
        query = query.replace('INSERT IGNORE', 'INSERT OR IGNORE')

        cursor = None
        try:
            if not self.ensure_connection():
                logger.error("Pas de connexion à la base SQLite")
                return None if fetch_one or fetch_all else 0

            cursor = self.connection.cursor()
            cursor.execute(query, params or ())

            if query.strip().upper().startswith('SELECT'):
                if fetch_one:
                    return cursor.fetchone()
                else:
                    return cursor.fetchall()
            else:
                self.connection.commit()
                return cursor.lastrowid

        except sqlite3.Error as e:
            logger.error(f"Erreur d'exécution de requête: {e}")
            logger.error(f"Query: {query[:200]}...")
            if self.connection:
                try:
                    self.connection.rollback()
                except:
                    pass
            return None if fetch_one or fetch_all else 0
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass

    def create_tables(self):
        """Crée les tables en syntaxe SQLite"""
        if not self.ensure_connection():
            return False

        tables = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                role VARCHAR(20) DEFAULT 'patient',
                full_name VARCHAR(100),
                phone VARCHAR(20),
                ville VARCHAR(50),
                date_naissance DATE,
                groupe_sanguin VARCHAR(5),
                cin VARCHAR(20) UNIQUE,
                allergies TEXT,
                maladies_chroniques TEXT,
                actif BOOLEAN DEFAULT TRUE,
                date_inscription DATETIME DEFAULT (datetime('now', 'localtime')),
                derniere_connexion DATETIME,
                statut_suivi VARCHAR(20) DEFAULT 'termine',
                score_priorite INTEGER DEFAULT 0,
                age INTEGER,
                telephone VARCHAR(20),
                adresse TEXT,
                antecedents TEXT,
                specialite VARCHAR(100)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS parametres (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                notifications_email BOOLEAN DEFAULT TRUE,
                notifications_sms BOOLEAN DEFAULT FALSE,
                rappel_rdv BOOLEAN DEFAULT TRUE,
                theme VARCHAR(20) DEFAULT 'clair',
                langue VARCHAR(10) DEFAULT 'fr',
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                type_message VARCHAR(50) DEFAULT 'symptome',
                urgent BOOLEAN DEFAULT FALSE,
                lu BOOLEAN DEFAULT FALSE,
                timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
                conversation_id INTEGER,
                sender_id INTEGER,
                file_path TEXT,
                file_name VARCHAR(255),
                destinataire_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename VARCHAR(255) NOT NULL,
                type_analyse VARCHAR(100) NOT NULL,
                resultat VARCHAR(255) NOT NULL,
                confiance DECIMAL(5,4),
                recommandations TEXT,
                urgent BOOLEAN DEFAULT FALSE,
                timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rendez_vous (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                medecin_id INTEGER NOT NULL,
                date_rdv DATE NOT NULL,
                heure_rdv TIME NOT NULL,
                motif TEXT,
                notes TEXT,
                statut VARCHAR(20) DEFAULT 'planifie',
                created_at DATETIME DEFAULT (datetime('now', 'localtime')),
                duree INTEGER DEFAULT 30,
                FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (medecin_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                titre VARCHAR(255) NOT NULL,
                message TEXT NOT NULL,
                type_notification VARCHAR(20) DEFAULT 'info',
                lu BOOLEAN DEFAULT FALSE,
                timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action VARCHAR(255) NOT NULL,
                details TEXT,
                ip_address VARCHAR(45),
                user_agent TEXT,
                timestamp DATETIME DEFAULT (datetime('now', 'localtime'))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS photos_patient (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                photo_url VARCHAR(500) NOT NULL,
                date_upload DATETIME DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS notes_medicales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id INTEGER NOT NULL,
                patient_id INTEGER NOT NULL,
                note TEXT NOT NULL,
                type_note VARCHAR(50) DEFAULT 'consultation',
                date_creation DATETIME DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (doctor_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ordonnances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                medecin_id INTEGER NOT NULL,
                date_prescription DATE DEFAULT (date('now', 'localtime')),
                contenu TEXT NOT NULL,
                fichier_pdf VARCHAR(500),
                valide BOOLEAN DEFAULT TRUE,
                timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (medecin_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS specialites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nom VARCHAR(100) UNIQUE NOT NULL,
                description TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS medecin_specialites (
                medecin_id INTEGER NOT NULL,
                specialite_id INTEGER NOT NULL,
                PRIMARY KEY (medecin_id, specialite_id),
                FOREIGN KEY (medecin_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (specialite_id) REFERENCES specialites(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                medecin_id INTEGER NOT NULL,
                date_consultation DATETIME DEFAULT (datetime('now', 'localtime')),
                motif TEXT,
                diagnostic TEXT,
                traitement TEXT,
                notes TEXT,
                statut VARCHAR(20) DEFAULT 'planifie',
                created_at DATETIME DEFAULT (datetime('now', 'localtime')),
                updated_at DATETIME DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (medecin_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nom VARCHAR(255),
                type VARCHAR(20) DEFAULT 'privee',
                date_creation DATETIME DEFAULT (datetime('now', 'localtime'))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversation_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                user_id INTEGER,
                date_ajout DATETIME DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action VARCHAR(100),
                entity_type VARCHAR(50),
                entity_id INTEGER,
                details TEXT,
                ip_address VARCHAR(45),
                user_agent TEXT,
                timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dossiers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                titre VARCHAR(255),
                type_fichier VARCHAR(50),
                chemin_fichier TEXT,
                date_upload DATETIME DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        ]

        success_count = 0
        for i, table_query in enumerate(tables):
            try:
                self.connection.execute(table_query)
                success_count += 1
            except Exception as e:
                logger.error(f"Erreur création table {i+1}: {e}\nQuery: {table_query}")

        self.connection.commit()
        logger.info(f"{success_count}/{len(tables)} tables créées avec succès (SQLite)")

        if success_count > 0:
            self.create_specialites()
            # self.create_default_admin()
            # self.create_test_data()

        return success_count > 0

    def create_specialites(self):
        try:
            specialites = [
                ("Médecine générale", "Médecine généraliste"),
                ("Cardiologie", "Maladies du cœur"),
                ("Pédiatrie", "Médecine pour enfants"),
                ("Gynécologie", "Santé de la femme"),
                ("Dermatologie", "Maladies de la peau"),
                ("Ophtalmologie", "Maladies des yeux"),
                ("ORL", "Oreille, nez, gorge"),
                ("Psychiatrie", "Santé mentale"),
                ("Radiologie", "Imagerie médicale"),
                ("Urgences", "Urgences")
            ]
            for nom, desc in specialites:
                self.execute_query("INSERT OR IGNORE INTO specialites (nom, description) VALUES (?, ?)", (nom, desc))
        except Exception as e:
            pass

    # Rest of the methods follow the same signature...
    def get_user_by_username(self, username):
        return self.execute_query("SELECT * FROM users WHERE username = ?", (username,), fetch_one=True)
    
    def get_user_by_email(self, email):
        return self.execute_query("SELECT * FROM users WHERE email = ?", (email,), fetch_one=True)
    
    def get_user_by_id(self, user_id):
        return self.execute_query("SELECT * FROM users WHERE id = ?", (user_id,), fetch_one=True)
    
    def insert_user(self, user_data):
        query = """
        INSERT INTO users (username, email, password, role, full_name, phone, ville, date_naissance, groupe_sanguin, allergies, maladies_chroniques, cin)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        return self.execute_query(query, (
            user_data['username'], user_data['email'], user_data['password'], user_data['role'],
            user_data.get('full_name', ''), user_data.get('phone', ''), user_data.get('ville', ''),
            user_data.get('date_naissance'), user_data.get('groupe_sanguin'), user_data.get('allergies'),
            user_data.get('maladies_chroniques'), user_data.get('cin')
        ))
    
    def update_user(self, user_id, user_data):
        query = """
        UPDATE users 
        SET full_name = ?, phone = ?, ville = ?, date_naissance = ?, 
            groupe_sanguin = ?, allergies = ?, maladies_chroniques = ?
        WHERE id = ?
        """
        return self.execute_query(query, (
            user_data.get('full_name'), user_data.get('phone'), user_data.get('ville'),
            user_data.get('date_naissance'), user_data.get('groupe_sanguin'),
            user_data.get('allergies'), user_data.get('maladies_chroniques'), user_id
        ))
    
    def update_last_login(self, user_id):
        self.execute_query("UPDATE users SET derniere_connexion = datetime('now', 'localtime') WHERE id = ?", (user_id,))
    
    def update_user_password(self, user_id, new_password):
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), salt)
        return self.execute_query("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
    
    def log_action(self, user_id, action, details="", ip_address=None):
        self.execute_query("INSERT INTO logs (user_id, action, details, ip_address) VALUES (?, ?, ?, ?)", 
                           (user_id, action, details, ip_address))
    
    def get_user_notifications(self, user_id, unread_only=False, limit=10):
        query = "SELECT * FROM notifications WHERE user_id = ?"
        if unread_only: query += " AND lu = 0"
        query += " ORDER BY timestamp DESC LIMIT ?"
        return self.execute_query(query, (user_id, limit), fetch_all=True) or []
    
    def mark_notification_as_read(self, notification_id):
        return self.execute_query("UPDATE notifications SET lu = 1 WHERE id = ?", (notification_id,))
    
    def create_notification(self, user_id, titre, message, type_notification='info'):
        return self.execute_query("INSERT INTO notifications (user_id, titre, message, type_notification) VALUES (?, ?, ?, ?)", 
                                  (user_id, titre, message, type_notification))
    
    def get_doctor_patients(self, doctor_id=None):
        try:
            if doctor_id:
                query = """
                SELECT u.id, u.full_name, u.email, u.phone, u.ville as city,
                    strftime('%d/%m/%Y', u.date_naissance) as birth_date, u.groupe_sanguin as blood_group, u.cin, u.allergies,
                    u.maladies_chroniques as chronic_diseases, u.statut_suivi as status,
                    strftime('%d/%m/%Y', MAX(m.timestamp)) as last_visit,
                    (SELECT 1 FROM messages WHERE user_id = u.id AND urgent = 1 AND date(timestamp) = date('now', 'localtime')) as urgence,
                    (SELECT photo_url FROM photos_patient WHERE patient_id = u.id ORDER BY date_upload DESC LIMIT 1) as photo_url
                FROM users u
                LEFT JOIN messages m ON u.id = m.user_id
                LEFT JOIN rendez_vous r ON u.id = r.patient_id
                WHERE u.role = 'patient' AND u.actif = 1
                    AND (r.medecin_id = ? OR m.user_id IN (SELECT user_id FROM messages WHERE urgent = 1) OR u.statut_suivi = 'en_cours')
                GROUP BY u.id
                ORDER BY urgence DESC, u.statut_suivi ASC, u.full_name
                """
                return self.execute_query(query, (doctor_id,), fetch_all=True) or []
            else:
                query = """
                SELECT u.id, u.full_name, u.email, u.phone, u.ville as city,
                    strftime('%d/%m/%Y', u.date_naissance) as birth_date, u.groupe_sanguin as blood_group, u.cin,
                    u.statut_suivi as status, strftime('%d/%m/%Y', MAX(m.timestamp)) as last_visit,
                    (SELECT 1 FROM messages WHERE user_id = u.id AND urgent = 1 AND date(timestamp) = date('now', 'localtime')) as urgence,
                    (SELECT photo_url FROM photos_patient WHERE patient_id = u.id ORDER BY date_upload DESC LIMIT 1) as photo_url
                FROM users u
                LEFT JOIN messages m ON u.id = m.user_id
                WHERE u.role = 'patient' AND u.actif = 1
                GROUP BY u.id
                ORDER BY urgence DESC, last_visit DESC
                """
                return self.execute_query(query, fetch_all=True) or []
        except Exception as e:
            logger.error(f"Erreur get_doctor_patients: {e}")
            return []
    
    def get_patient_details(self, patient_id):
        # We need to adapt MySQL DATE_FORMAT to SQLite strftime in patient details too
        try:
            query_base = """
            SELECT id, full_name, email, phone, ville as city,
                strftime('%d/%m/%Y', date_naissance) as birth_date, groupe_sanguin as blood_group,
                cin, allergies, maladies_chroniques as chronic_diseases,
                strftime('%d/%m/%Y', date_inscription) as registration_date,
                statut_suivi as status, score_priorite
            FROM users WHERE id = ?
            """
            patient = self.execute_query(query_base, (patient_id,), fetch_one=True)
            if not patient: return None
            
            photo = self.execute_query("SELECT photo_url FROM photos_patient WHERE patient_id = ? ORDER BY date_upload DESC LIMIT 1", (patient_id,), fetch_one=True)
            patient['photo_url'] = photo['photo_url'] if photo else None
            
            last_visit = self.execute_query("SELECT strftime('%d/%m/%Y', timestamp) as last_visit FROM messages WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (patient_id,), fetch_one=True)
            patient['last_visit'] = last_visit['last_visit'] if last_visit else "Jamais"
            
            query_next_rdv = """
            SELECT strftime('%d/%m/%Y', date_rdv) as date, heure_rdv as time, motif as reason, u.full_name as doctor, statut as status
            FROM rendez_vous r JOIN users u ON u.id = r.medecin_id
            WHERE r.patient_id = ? AND r.date_rdv >= date('now', 'localtime')
            ORDER BY r.date_rdv LIMIT 1
            """
            patient['next_appointment'] = self.execute_query(query_next_rdv, (patient_id,), fetch_one=True)
            
            doctor = self.execute_query("SELECT u.full_name FROM users u JOIN rendez_vous r ON r.medecin_id = u.id WHERE r.patient_id = ? ORDER BY r.timestamp DESC LIMIT 1", (patient_id,), fetch_one=True)
            patient['doctor'] = doctor['full_name'] if doctor else "Non assigné"
            
            query_consultations = """
            SELECT strftime('%d/%m/%Y', date_rdv) as date, heure_rdv as time, motif, statut, u.full_name as doctor, notes
            FROM rendez_vous r JOIN users u ON u.id = r.medecin_id
            WHERE r.patient_id = ? ORDER BY r.date_rdv DESC LIMIT 10
            """
            patient['consultations'] = self.execute_query(query_consultations, (patient_id,), fetch_all=True) or []
            
            query_analyses = """
            SELECT id, type_analyse as type, strftime('%d/%m/%Y', timestamp) as date, resultat, confiance, urgent
            FROM analyses WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10
            """
            patient['analyses'] = self.execute_query(query_analyses, (patient_id,), fetch_all=True) or []
            return patient
        except Exception as e:
            logger.error(f"Erreur get_patient_details: {e}")
            return None
    def get_dashboard_stats(self):
        try:
            return {
                'total_users': self.execute_query("SELECT COUNT(*) as count FROM users", fetch_one=True)['count'],
                'total_patients': self.execute_query("SELECT COUNT(*) as count FROM users WHERE role = 'patient'", fetch_one=True)['count'],
                'total_medecins': self.execute_query("SELECT COUNT(*) as count FROM users WHERE role = 'medecin'", fetch_one=True)['count'],
                'total_analyses': self.execute_query("SELECT COUNT(*) as count FROM analyses", fetch_one=True)['count']
            }
        except Exception as e:
            logger.error(f"Erreur get_dashboard_stats: {e}")
            return {'total_users': 0, 'total_patients': 0, 'total_medecins': 0, 'total_analyses': 0}

    def get_doctor_statistics(self, doctor_id):
        try:
            total_patients = self.execute_query(
                "SELECT COUNT(DISTINCT patient_id) as count FROM rendez_vous WHERE medecin_id = ?", 
                (doctor_id,), fetch_one=True
            )['count']
            
            total_consultations = self.execute_query(
                "SELECT COUNT(*) as count FROM consultations WHERE medecin_id = ?",
                (doctor_id,), fetch_one=True
            )['count']
            
            taux_urgence = self.execute_query(
                "SELECT COUNT(*) as count FROM analyses WHERE urgent = 1", fetch_one=True
            )['count'] # Simplification
            
            return {
                'total_patients': total_patients,
                'consultations_month': total_consultations,
                'emergency_cases': taux_urgence,
                'avg_rating': 4.8
            }
        except Exception as e:
            logger.error(f"Erreur get_doctor_statistics: {e}")
            return {'total_patients': 0, 'consultations_month': 0, 'emergency_cases': 0, 'avg_rating': 0.0}
