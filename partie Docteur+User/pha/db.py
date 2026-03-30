import mysql.connector
from mysql.connector import Error
import streamlit as st
from config import DB_CONFIG

def get_connection():
    """Établit la connexion à la base de données"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        st.error(f"Erreur de connexion à la base de données: {e}")
        return None

def init_database():
    """Initialise la base de données si elle n'existe pas"""
    try:
        # Connexion sans base de données spécifique
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = conn.cursor()
        
        # Créer la base de données si elle n'existe pas
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cursor.execute(f"USE {DB_CONFIG['database']}")
        
        # Créer la table produits
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                category VARCHAR(100),
                price FLOAT,
                image VARCHAR(255),
                description TEXT,
                promo BOOLEAN DEFAULT FALSE,
                stock INT DEFAULT 10,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Créer la table commandes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                customer_name VARCHAR(255),
                customer_phone VARCHAR(50),
                customer_address TEXT,
                products TEXT,
                total FLOAT,
                status VARCHAR(50) DEFAULT 'En attente',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Error as e:
        st.error(f"Erreur d'initialisation: {e}")
        return False