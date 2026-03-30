import streamlit as st
from models import create_order
from config import DEVISE
import json
import random
from datetime import datetime

st.set_page_config(page_title="Finaliser la commande", page_icon="📦", layout="wide")

st.title("📦 Finaliser ma commande")

if not st.session_state.cart:
    st.warning("Votre panier est vide")
    if st.button("🛒 Voir les produits"):
        st.switch_page("pages/2_Categories.py")
else:
    # Calculs
    total = sum(float(item['price']) for item in st.session_state.cart)
    frais_livraison = 30 if total < 500 else 0
    total_final = total + frais_livraison
    
    # Layout en deux colonnes
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Formulaire de commande
        st.markdown("""
            <div style="
                background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin-bottom: 1rem;
            ">
                <h3>👤 Vos informations</h3>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("order_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                nom = st.text_input("Nom complet *")
                telephone = st.text_input("Téléphone *")
                email = st.text_input("Email")
            
            with col_b:
                prenom = st.text_input("Prénom *")
                ville = st.text_input("Ville *")
                code_postal = st.text_input("Code postal")
            
            adresse = st.text_area("Adresse complète *", height=100)
            
            st.markdown("---")
            
            # Mode de livraison
            st.markdown("### 🚚 Mode de livraison")
            livraison = st.radio(
                "Choisissez votre mode de livraison",
                ["Livraison à domicile", "Click & Collect (Gratuit)", "Point relais"],
                horizontal=True
            )
            
            # Mode de paiement
            st.markdown("### 💳 Mode de paiement")
            paiement = st.radio(
                "Choisissez votre mode de paiement",
                ["Paiement à la livraison", "Carte bancaire", "PayPal"],
                horizontal=True
            )
            
            # Informations supplémentaires
            st.markdown("### 📝 Informations complémentaires")
            instructions = st.text_area("Instructions spéciales (code d'accès, étage, etc.)")
            
            # Conditions
            st.markdown("---")
            conditions = st.checkbox("J'accepte les conditions générales de vente *")
            newsletter = st.checkbox("Je souhaite recevoir la newsletter et les offres promotionnelles")
            
            submitted = st.form_submit_button("✅ Confirmer ma commande", use_container_width=True)
            
            if submitted:
                if not nom or not prenom or not telephone or not adresse or not ville or not conditions:
                    st.error("Veuillez remplir tous les champs obligatoires (*)")
                else:
                    # Préparer les données
                    products_list = []
                    for p in st.session_state.cart:
                        products_list.append({
                            "id": p.get('id', 0),
                            "name": p['name'],
                            "price": float(p['price']),
                            "quantity": 1
                        })
                    
                    # Créer la commande
                    success = create_order(
                        customer_name=f"{prenom} {nom}",
                        customer_phone=telephone,
                        customer_address=f"{adresse}, {ville} {code_postal}",
                        products=json.dumps(products_list, ensure_ascii=False),
                        total=total_final
                    )
                    
                    if success:
                        st.session_state.order_success = True
                        st.session_state.order_total = total_final
                        st.session_state.order_number = f"CMD{random.randint(10000, 99999)}"
                        st.rerun()
    
    with col2:
        # Récapitulatif de la commande
        st.markdown("""
            <div style="
                background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin-bottom: 1rem;
            ">
                <h3>📋 Récapitulatif</h3>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container(border=True):
            for item in st.session_state.cart:
                col_x, col_y = st.columns([3, 1])
                with col_x:
                    st.write(f"**{item['name']}**")
                with col_y:
                    st.write(f"{item['price']} {DEVISE}")
            
            st.divider()
            
            # Détails des prix
            col_x, col_y = st.columns([3, 1])
            with col_x:
                st.write("Sous-total")
            with col_y:
                st.write(f"{total} {DEVISE}")
            
            col_x, col_y = st.columns([3, 1])
            with col_x:
                st.write("Frais de livraison")
            with col_y:
                if frais_livraison == 0:
                    st.write("OFFERT")
                else:
                    st.write(f"{frais_livraison} {DEVISE}")
            
            st.divider()
            
            # Total
            col_x, col_y = st.columns([3, 1])
            with col_x:
                st.markdown("### TOTAL")
            with col_y:
                st.markdown(f"### {total_final} {DEVISE}")
            
            # Badge livraison gratuite
            if frais_livraison == 0:
                st.success("🎉 Livraison OFFERTE !")
    
    # Affichage de la confirmation de commande
    if st.session_state.get('order_success', False):
        st.balloons()
        st.markdown(f"""
            <div style="
                background: white;
                padding: 2rem;
                border-radius: 20px;
                text-align: center;
                border: 2px solid #dc2626;
                margin: 2rem 0;
            ">
                <span style="font-size: 4rem;">✅</span>
                <h2 style="color: #dc2626;">Commande confirmée !</h2>
                <p style="font-size: 1.2rem;">Numéro de commande: <strong>{st.session_state.order_number}</strong></p>
                <p>Montant total: <strong>{st.session_state.order_total} {DEVISE}</strong></p>
                <p style="color: #6b7280;">Un email de confirmation vous a été envoyé</p>
                
                <div style="margin-top: 2rem;">
                    <p>📱 Suivez votre commande en temps réel</p>
                </div>
                
                <div style="margin-top: 2rem;">
                    <button onclick="window.print()" style="
                        background: #dc2626;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 5px;
                        cursor: pointer;
                        margin-right: 10px;
                    ">🖨️ Imprimer</button>
                    <button onclick="window.location.href='/'" style="
                        background: #6b7280;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 5px;
                        cursor: pointer;
                    ">🏠 Accueil</button>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Vider le panier après confirmation
        if st.button("🗑️ Vider le panier et continuer"):
            st.session_state.cart = []
            st.session_state.order_success = False
            st.rerun()