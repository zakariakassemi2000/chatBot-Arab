#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mysql.connector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = {
    'host': 'localhost',
    'user': 'root',  # Modifiez selon votre config
    'password': '',  # Modifiez selon votre config
    'database': 'ai_shifa_pro'
}

def fix_database():
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        # Vérifier les colonnes existantes
        cursor.execute("DESCRIBE users")
        columns = [col[0] for col in cursor.fetchall()]
        logger.info(f"Colonnes actuelles: {columns}")
        
        # Ajouter date_naissance si manquante
        if 'date_naissance' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN date_naissance DATE")
            logger.info("Colonne 'date_naissance' ajoutée")
        
        # Ajouter statut_suivi si manquante
        if 'statut_suivi' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN statut_suivi ENUM('en_cours', 'traite', 'termine') DEFAULT 'termine'")
            logger.info("Colonne 'statut_suivi' ajoutée")
        
        # Ajouter score_priorite si manquante
        if 'score_priorite' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN score_priorite INT DEFAULT 0")
            logger.info("Colonne 'score_priorite' ajoutée")
        
        # Supprimer la colonne age si elle existe (car elle était générée)
        if 'age' in columns:
            try:
                cursor.execute("ALTER TABLE users DROP COLUMN age")
                logger.info("Colonne 'age' supprimée (sera calculée dynamiquement)")
            except Exception as e:
                logger.warning(f"Erreur lors de l'opération : {e}")
        
        # Créer la table consultations si elle n'existe pas
        cursor.execute("SHOW TABLES LIKE 'consultations'")
        if not cursor.fetchone():
            cursor.execute("""
            CREATE TABLE consultations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id INT NOT NULL,
                medecin_id INT NOT NULL,
                date_consultation DATETIME DEFAULT CURRENT_TIMESTAMP,
                motif TEXT,
                diagnostic TEXT,
                traitement TEXT,
                notes TEXT,
                statut ENUM('planifiee', 'en_cours', 'terminee', 'annulee') DEFAULT 'planifiee',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (medecin_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_patient (patient_id),
                INDEX idx_medecin (medecin_id),
                INDEX idx_date (date_consultation)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            logger.info("Table 'consultations' créée")
        
        conn.commit()
        logger.info("✅ Base de données corrigée avec succès")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")

if __name__ == "__main__":
    fix_database()