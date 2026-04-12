import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.session import init_session_state
from utils.models import get_all_products
from config import DEVISE, COLORS, CATEGORY_IMAGES, DEFAULT_PRODUCT_IMAGE
import random

st.set_page_config(page_title="Mon Panier", page_icon="🛒", layout="wide")

# Initialisation session
init_session_state()

st.title("🛒 Mon panier d'achat")

if not st.session_state.cart:
    # Panier vide
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown(f"""
            <div style="
                background: white;
                padding: 3rem;
                border-radius: 20px;
                text-align: center;
                border: 2px dashed {COLORS['primary']};
                margin: 2rem 0;
                box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
            ">
                <span style="font-size: 5rem;">🛒</span>
                <h2 style="color: {COLORS['primary']};">Votre panier est vide</h2>
                <p style="color: {COLORS['text_light']};">Découvrez nos produits et faites vos achats</p>
            </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📂 Voir les catégories", width='stretch'):
                st.switch_page("pages/1_Categories.py")
        with col_b:
            if st.button("🏠 Retour à l'accueil", width='stretch'):
                st.switch_page("app.py")
else:
    total = 0
    
    # En-tête
    st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
            padding: 1rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(220, 38, 38, 0.3);
        ">
            <h3>Récapitulatif de votre commande</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Liste articles
    for i, item in enumerate(st.session_state.cart):
        price = float(item['price'])
        total += price
        
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
            
            with col1:
                image_url = CATEGORY_IMAGES.get(item['category'], DEFAULT_PRODUCT_IMAGE)
                st.image(image_url, width='stretch')
            
            with col2:
                st.markdown(f"### {item['name']}")
                st.markdown(f"<span style='color: {COLORS['text_light']};'>{item['category']}</span>", unsafe_allow_html=True)
                if item.get('promo', False):
                    st.markdown(f'<span style="background:{COLORS["primary"]};color:white;padding:2px 8px;border-radius:20px;">PROMO</span>', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"## {price} {DEVISE}")
                st.markdown("Quantité: 1")
            
            with col4:
                if st.button("🗑️", key=f"del_{i}", help="Retirer du panier"):
                    st.session_state.cart.pop(i)
                    st.rerun()
    
    st.divider()
    
    # Calculs
    frais = 30 if total < 500 else 0
    total_final = total + frais
    
    # Résumé
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div style="background:white;padding:1rem;border-radius:15px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.05);">
                <div style="color:{COLORS['text_light']};">Sous-total</div>
                <div style="font-size:2rem;font-weight:bold;color:{COLORS['primary']};">{total} {DEVISE}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background:white;padding:1rem;border-radius:15px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.05);">
                <div style="color:{COLORS['text_light']};">Livraison</div>
                <div style="font-size:2rem;font-weight:bold;color:{COLORS['success']};">{frais} {DEVISE}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if total >= 500:
            st.success("🎉 Livraison OFFERTE !")
    
    with col3:
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
                padding:1rem;
                border-radius:15px;
                text-align:center;
                color:white;
                box-shadow:0 4px 6px rgba(220,38,38,0.3);
            ">
                <div>Total TTC</div>
                <div style="font-size:2.5rem;font-weight:bold;">{total_final} {DEVISE}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Continuer mes achats", width='stretch'):
            st.switch_page("pages/1_Categories.py")
    with col2:
        if st.button("🗑️ Vider le panier", width='stretch'):
            st.session_state.cart = []
            st.rerun()
    with col3:
        if st.button("📦 Passer la commande", type="primary", width='stretch'):
            st.switch_page("pages/3_Commande.py")
    
    st.divider()
    
    # Suggestions
    st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
            padding: 1rem;
            border-radius: 15px;
            color: white;
            margin: 2rem 0;
            box-shadow: 0 4px 6px rgba(220,38,38,0.3);
        ">
            <h3>🎁 Vous aimerez aussi</h3>
        </div>
    """, unsafe_allow_html=True)
    
    suggestions = get_all_products()
    cart_ids = [p.get('id') for p in st.session_state.cart if p.get('id')]
    suggestions = [p for p in suggestions if p.get('id') not in cart_ids]
    suggestions = random.sample(suggestions, min(4, len(suggestions)))
    
    cols = st.columns(4)
    for i, product in enumerate(suggestions):
        with cols[i]:
            with st.container():
                image_url = CATEGORY_IMAGES.get(product['category'], DEFAULT_PRODUCT_IMAGE)
                st.image(image_url, width='stretch')
                st.markdown(f"**{product['name']}**")
                st.markdown(f"💰 {product['price']} {DEVISE}")
                
                if product.get('promo', False):
                    st.markdown(f'<span style="background:{COLORS["primary"]};color:white;padding:2px 8px;border-radius:20px;">PROMO</span>', unsafe_allow_html=True)
                
                if st.button("➕ Ajouter", key=f"sugg_{product['id']}", width='stretch'):
                    st.session_state.cart.append(product)
                    st.rerun()
    
    # Barre progression
    if total < 500:
        reste = 500 - total
        progression = (total / 500) * 100
        st.markdown(f"""
            <div style="margin-top: 2rem;">
                <p>🎁 Plus que <strong>{reste} {DEVISE}</strong> pour la livraison gratuite !</p>
                <div style="
                    background: {COLORS['border']};
                    height: 10px;
                    border-radius: 5px;
                    overflow: hidden;
                ">
                    <div style="
                        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
                        width: {progression}%;
                        height: 100%;
                        border-radius: 5px;
                        transition: width 0.5s ease;
                    "></div>
                </div>
            </div>
        """, unsafe_allow_html=True)