import streamlit as st

def init_session_state():
    """Initialise toutes les variables de session"""
    # Panier
    if 'cart' not in st.session_state:
        st.session_state.cart = []
    
    # Filtres catégories
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = "Tous les produits"
    
    if 'price_range' not in st.session_state:
        st.session_state.price_range = (0, 1000)
    
    if 'show_promo_only' not in st.session_state:
        st.session_state.show_promo_only = False
    
    # Admin
    if 'admin_auth' not in st.session_state:
        st.session_state.admin_auth = False
    
    # Utilisateur
    if 'user' not in st.session_state:
        st.session_state.user = {
            'username': 'client',
            'full_name': 'Client',
            'role': 'client'
        }
    
    # Commande
    if 'order_success' not in st.session_state:
        st.session_state.order_success = False
    
    if 'order_number' not in st.session_state:
        st.session_state.order_number = None
    
    if 'order_total' not in st.session_state:
        st.session_state.order_total = 0