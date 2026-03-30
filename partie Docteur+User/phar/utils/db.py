import mysql.connector
from mysql.connector import Error
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_CONFIG

def get_connection():
    """Connexion à la base de données avec gestion d'erreur améliorée"""
    try:
        # Vérifier d'abord si le serveur MySQL est accessible
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            connection_timeout=5
        )
        
        # Sélectionner la base de données
        cursor = conn.cursor()
        cursor.execute(f"USE {DB_CONFIG['database']}")
        cursor.close()
        
        return conn
    except Error as e:
        if "Unknown database" in str(e):
            # La base n'existe pas, on la crée
            return create_database()
        else:
            st.error(f"❌ Erreur de connexion à MySQL: {e}")
            st.info("""
            **Solutions possibles:**
            1. Vérifiez que MySQL est installé et démarré
            2. Vérifiez les identifiants dans config.py
            3. Créez la base de données manuellement
            """)
            return None

def create_database():
    """Crée la base de données si elle n'existe pas"""
    try:
        # Connexion sans base de données
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = conn.cursor()
        
        # Créer la base
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cursor.execute(f"USE {DB_CONFIG['database']}")
        
        # Créer les tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                category VARCHAR(100),
                price DECIMAL(10,2),
                image VARCHAR(255),
                description TEXT,
                promo BOOLEAN DEFAULT FALSE,
                stock INT DEFAULT 10,
                expiration_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                customer_name VARCHAR(255),
                customer_phone VARCHAR(50),
                customer_address TEXT,
                products TEXT,
                total DECIMAL(10,2),
                status VARCHAR(50) DEFAULT 'En attente',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insérer quelques produits de test
        cursor.execute("SELECT COUNT(*) FROM products")
        count = cursor.fetchone()[0]
        
        if count == 0:
            test_products = [
                ("Crème Hydratante", "Visage", 250.00, "Crème pour peau sèche", 0, "visage.jpg", 15, "2025-12-31"),
                ("Shampoing Bio", "Cheveux", 180.00, "Shampoing naturel", 1, "cheveux.jpg", 8, "2025-06-30"),
                ("Gel Douche", "Corps", 120.00, "Gel douche surgras", 0, "corps.jpg", 25, "2025-09-30"),
                ("Rouge à Lèvres", "Maquillage", 150.00, "Rouge à lèvres mat", 1, "maquillage.jpg", 5, "2025-03-31"),
                ("Lait Corporel", "Corps", 200.00, "Lait hydratant", 0, "corps2.jpg", 12, "2025-08-31"),
            ]
            
            for prod in test_products:
                cursor.execute("""
                    INSERT INTO products (name, category, price, description, promo, image, stock, expiration_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, prod)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        st.success("✅ Base de données créée avec succès!")
        return get_connection()
        
    except Error as e:
        st.error(f"❌ Erreur lors de la création de la base: {e}")
        return None

def init_database():
    """Initialise la base de données"""
    return create_database()