import streamlit as st
from utils.models import create_order
from config import DEVISE
import json
import random

st.set_page_config(page_title="Commande", page_icon="📦", layout="wide")
# ✅ INITIALISATION DU PANIER
if 'cart' not in st.session_state:
    st.session_state.cart = []
st.title("📦 Finaliser ma commande")

if not st.session_state.cart:
    st.warning("Votre panier est vide")
    if st.button("🛒 Voir les produits"):
        st.switch_page("pages/1_Categories.py")
else:
    total = sum(float(item['price']) for item in st.session_state.cart)
    frais = 30 if total < 500 else 0
    total_final = total + frais
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        with st.form("order_form"):
            st.subheader("👤 Vos informations")
            
            col_a, col_b = st.columns(2)
            with col_a:
                nom = st.text_input("Nom *")
                telephone = st.text_input("Téléphone *")
            with col_b:
                prenom = st.text_input("Prénom *")
                email = st.text_input("Email")
            
            adresse = st.text_area("Adresse complète *")
            ville = st.text_input("Ville *")
            
            conditions = st.checkbox("J'accepte les CGV *")
            
            if st.form_submit_button("✅ Confirmer la commande", width='stretch'):
                if not all([nom, prenom, telephone, adresse, ville, conditions]):
                    st.error("Veuillez remplir tous les champs obligatoires")
                else:
                    products_list = [{"id": p.get('id'), "name": p['name'], "price": float(p['price'])} 
                                   for p in st.session_state.cart]
                    
                    if create_order(
                        customer_name=f"{prenom} {nom}",
                        customer_phone=telephone,
                        customer_address=f"{adresse}, {ville}",
                        products=json.dumps(products_list),
                        total=total_final
                    ):
                        st.session_state.order_success = True
                        st.session_state.order_number = f"CMD{random.randint(10000, 99999)}"
                        st.session_state.order_total = total_final
                        st.rerun()
    
    with col2:
        with st.container(border=True):
            st.subheader("📋 Récapitulatif")
            for item in st.session_state.cart:
                st.write(f"• {item['name']}: {item['price']} {DEVISE}")
            
            st.divider()
            st.write(f"Sous-total: {total} {DEVISE}")
            st.write(f"Livraison: {'OFFERTE' if frais == 0 else f'{frais} {DEVISE}'}")
            st.divider()
            st.subheader(f"Total: {total_final} {DEVISE}")
    
    if st.session_state.get('order_success', False):
        st.balloons()
        st.success(f"""
        ✅ Commande #{st.session_state.order_number} confirmée !
        Total: {st.session_state.order_total} {DEVISE}
        """)
        if st.button("🗑️ Vider le panier"):
            st.session_state.cart = []
            st.session_state.order_success = False
            st.rerun()