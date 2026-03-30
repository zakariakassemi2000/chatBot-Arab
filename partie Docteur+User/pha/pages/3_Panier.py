import streamlit as st
from config import DEVISE
from datetime import datetime
import random

st.set_page_config(page_title="Mon Panier", page_icon="🛒", layout="wide")

st.title("🛒 Mon panier d'achat")

if not st.session_state.cart:
    # Panier vide - Design attrayant
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <div style="
                background: white;
                padding: 3rem;
                border-radius: 20px;
                text-align: center;
                border: 2px dashed #dc2626;
                margin: 2rem 0;
            ">
                <span style="font-size: 5rem;">🛒</span>
                <h2>Votre panier est vide</h2>
                <p style="color: #6b7280;">Découvrez nos produits et faites vos achats</p>
            </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📂 Voir les catégories", use_container_width=True):
                st.switch_page("pages/2_Categories.py")
        with col_b:
            if st.button("🏠 Retour à l'accueil", use_container_width=True):
                st.switch_page("app.py")
else:
    # Affichage du panier
    total = 0
    
    # En-tête du panier
    st.markdown("""
        <div style="
            background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
        ">
            <h3>Récapitulatif de votre commande</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Liste des articles
    for i, item in enumerate(st.session_state.cart):
        price = float(item['price'])
        total += price
        
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
            
            with col1:
                st.image(f"https://via.placeholder.com/100x100?text=Produit", use_column_width=True)
            
            with col2:
                st.markdown(f"### {item['name']}")
                st.markdown(f"<span style='color: #6b7280;'>{item['category']}</span>", unsafe_allow_html=True)
                if item.get('promo', False):
                    st.markdown("""
                        <span style="
                            background: #dc2626;
                            color: white;
                            padding: 2px 8px;
                            border-radius: 20px;
                            font-size: 0.8rem;
                        ">PROMO</span>
                    """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"## {price} {DEVISE}")
                st.markdown("<span style='color: #6b7280;'>Quantité: 1</span>", unsafe_allow_html=True)
            
            with col4:
                if st.button("🗑️", key=f"del_{i}", help="Retirer du panier"):
                    st.session_state.cart.pop(i)
                    st.rerun()
    
    # Résumé de la commande
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Sous-total
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1rem; color: #6b7280;">Sous-total</div>
                <div style="font-size: 2rem; font-weight: bold; color: #dc2626;">{total} {DEVISE}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Frais de livraison
        frais = 30 if total < 500 else 0
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1rem; color: #6b7280;">Frais de livraison</div>
                <div style="font-size: 2rem; font-weight: bold; color: #10b981;">{frais} {DEVISE}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if total >= 500:
            st.success("🎉 Livraison OFFERTE !")
    
    with col3:
        # Total
        total_final = total + frais
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                text-align: center;
            ">
                <div style="font-size: 1rem;">TOTAL</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{total_final} {DEVISE}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Actions
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Continuer mes achats", use_container_width=True):
            st.switch_page("pages/2_Categories.py")
    
    with col2:
        if st.button("🗑️ Vider le panier", use_container_width=True):
            st.session_state.cart = []
            st.rerun()
    
    with col3:
        if st.button("📦 Passer la commande", type="primary", use_container_width=True):
            st.switch_page("pages/4_Commande.py")
    
    with col4:
        # Code promo
        with st.popover("🎫 Code promo"):
            promo_code = st.text_input("Entrez votre code")
            if st.button("Appliquer"):
                if promo_code.upper() == "BIENVENUE20":
                    st.success("✅ Code appliqué ! -20%")
                else:
                    st.error("❌ Code invalide")
    
    st.divider()
    
    # Suggestions de produits
    st.markdown("""
        <div style="
            background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 2rem 0 1rem 0;
        ">
            <h3>🎁 Vous aimerez aussi</h3>
        </div>
    """, unsafe_allow_html=True)
    
    from models import get_all_products
    
    suggestions = get_all_products()
    # Exclure les produits déjà dans le panier
    cart_ids = [p.get('id') for p in st.session_state.cart if p.get('id')]
    suggestions = [p for p in suggestions if p.get('id') not in cart_ids]
    suggestions = random.sample(suggestions, min(4, len(suggestions)))
    
    cols = st.columns(4)
    for i, product in enumerate(suggestions):
        with cols[i]:
            with st.container(border=True):
                st.image(f"https://via.placeholder.com/150x150?text={product['name'][:10]}")
                st.markdown(f"**{product['name']}**")
                st.markdown(f"💰 {product['price']} {DEVISE}")
                
                # Badge promo
                if product['promo']:
                    st.markdown("""
                        <span style="
                            background: #dc2626;
                            color: white;
                            padding: 2px 8px;
                            border-radius: 20px;
                            font-size: 0.7rem;
                        ">PROMO</span>
                    """, unsafe_allow_html=True)
                
                if st.button("➕ Ajouter", key=f"sugg_{product['id']}", use_container_width=True):
                    st.session_state.cart.append(product)
                    st.rerun()
    
    # Barre de progression vers livraison gratuite
    if total < 500:
        reste = 500 - total
        progression = (total / 500) * 100
        st.markdown(f"""
            <div style="margin-top: 2rem;">
                <p>🎁 Plus que <strong>{reste} {DEVISE}</strong> pour la livraison gratuite !</p>
                <div style="
                    background: #e5e7eb;
                    height: 10px;
                    border-radius: 5px;
                    overflow: hidden;
                ">
                    <div style="
                        background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
                        width: {progression}%;
                        height: 100%;
                        border-radius: 5px;
                    "></div>
                </div>
            </div>
        """, unsafe_allow_html=True)