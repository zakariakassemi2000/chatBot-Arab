from utils.db import get_connection
import streamlit as st

def get_all_products(category=None):
    """Récupère tous les produits"""
    conn = get_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    
    if category:
        cursor.execute("SELECT * FROM products WHERE category = %s ORDER BY name", (category,))
    else:
        cursor.execute("SELECT * FROM products ORDER BY name")
    
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return products

def add_product(name, category, price, description, promo, image, stock, expiration_date):
    """Ajoute un produit"""
    conn = get_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO products (name, category, price, description, promo, image, stock, expiration_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (name, category, price, description, promo, image, stock, expiration_date))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Erreur: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def update_product(product_id, name, category, price, description, promo, stock, expiration_date):
    """Met à jour un produit"""
    conn = get_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE products 
            SET name=%s, category=%s, price=%s, description=%s, promo=%s, stock=%s, expiration_date=%s
            WHERE id=%s
        """, (name, category, price, description, promo, stock, expiration_date, product_id))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Erreur: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def delete_product(product_id):
    """Supprime un produit"""
    conn = get_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM products WHERE id=%s", (product_id,))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Erreur: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def create_order(customer_name, customer_phone, customer_address, products, total):
    """Crée une commande"""
    conn = get_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO orders (customer_name, customer_phone, customer_address, products, total)
            VALUES (%s, %s, %s, %s, %s)
        """, (customer_name, customer_phone, customer_address, products, total))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Erreur: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_all_orders():
    """Récupère toutes les commandes"""
    conn = get_connection()
    if not conn:
        return []
    
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM orders ORDER BY created_at DESC")
    orders = cursor.fetchall()
    cursor.close()
    conn.close()
    return orders