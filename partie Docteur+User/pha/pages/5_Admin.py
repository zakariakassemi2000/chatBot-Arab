import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from models import get_all_products, add_product, update_product, delete_product, get_all_orders
from config import CATEGORIES, DEVISE
import os
from PIL import Image
import json

st.set_page_config(page_title="Administration", page_icon="⚙️", layout="wide")

# Vérification admin
if 'admin_auth' not in st.session_state:
    st.session_state.admin_auth = False

def check_password():
    st.title("🔐 Accès administrateur")
    with st.form("login_form"):
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")
        if submitted and password == "admin123":
            st.session_state.admin_auth = True
            st.rerun()
        elif submitted:
            st.error("Mot de passe incorrect")

if not st.session_state.admin_auth:
    check_password()
else:
    st.title("⚙️ Administration")
    
    # Récupération des données
    products = get_all_products()
    orders = get_all_orders()
    today = datetime.now().date()
    
    # ============================================
    # ALERTES PRODUITS (EXPIRATION)
    # ============================================
    st.markdown("""
        <div style="
            background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        ">
            <h2>🚨 Alertes & Surveillance</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Analyse des expirations
    expired_products = []
    expiring_soon = []
    
    for p in products:
        if p.get('expiration_date'):
            days_to_expiry = (p['expiration_date'] - today).days
            
            if days_to_expiry < 0:
                expired_products.append(p)
            elif days_to_expiry < 30:
                expiring_soon.append(p)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div style="font-size: 2rem;">❌</div>
                <div style="font-size: 2rem; color: #dc2626; font-weight: bold;">""" + str(len(expired_products)) + """</div>
                <div>Produits expirés</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div style="font-size: 2rem;">⚠️</div>
                <div style="font-size: 2rem; color: #f59e0b; font-weight: bold;">""" + str(len(expiring_soon)) + """</div>
                <div>Expirent bientôt</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        stock_faible = len([p for p in products if p['stock'] < 5])
        st.markdown("""
            <div class="metric-card">
                <div style="font-size: 2rem;">📦</div>
                <div style="font-size: 2rem; color: #f59e0b; font-weight: bold;">""" + str(stock_faible) + """</div>
                <div>Stock faible</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Affichage des alertes détaillées
    if expired_products:
        st.error("### ❌ Produits expirés - ACTION REQUISE")
        for p in expired_products:
            days = (p['expiration_date'] - today).days
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([3,2,2,1])
                with col1:
                    st.write(f"**{p['name']}**")
                with col2:
                    st.write(f"Expiré depuis {abs(days)} jours")
                with col3:
                    st.write(f"Stock: {p['stock']} unités")
                with col4:
                    if st.button("🗑️ Retirer", key=f"del_exp_{p['id']}"):
                        delete_product(p['id'])
                        st.rerun()
    
    if expiring_soon:
        st.warning("### ⚠️ Produits expirant dans moins de 30 jours")
        for p in expiring_soon:
            days = (p['expiration_date'] - today).days
            with st.container(border=True):
                col1, col2, col3 = st.columns([3,2,2])
                with col1:
                    st.write(f"**{p['name']}**")
                with col2:
                    st.write(f"Expire dans {days} jours")
                with col3:
                    st.write(f"Stock: {p['stock']} unités")
    
    if stock_faible > 0:
        st.warning("### 📦 Produits en stock faible (<5 unités)")
        for p in products:
            if p['stock'] < 5:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([3,2,2])
                    with col1:
                        st.write(f"**{p['name']}**")
                    with col2:
                        st.write(f"Stock: {p['stock']} unités")
                    with col3:
                        if st.button("➕ Réapprovisionner", key=f"restock_{p['id']}"):
                            # Logique de réapprovisionnement
                            st.info("Fonctionnalité à implémenter")
    
    st.divider()
    
    # ============================================
    # DASHBOARD STATISTIQUES
    # ============================================
    st.markdown("""
        <div style="
            background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        ">
            <h2>📊 Tableau de bord</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Produits total", len(products))
    
    with col2:
        stock_total = sum(p['stock'] for p in products)
        st.metric("📊 Stock total", stock_total)
    
    with col3:
        produits_promo = len([p for p in products if p['promo']])
        st.metric("🏷️ En promotion", produits_promo)
    
    with col4:
        if orders:
            total_ca = sum(order['total'] for order in orders)
            st.metric("💰 CA Total", f"{total_ca} {DEVISE}")
        else:
            st.metric("💰 CA Total", f"0 {DEVISE}")
    
    st.divider()
    
    # Menu admin
    menu = st.tabs(["➕ Ajouter produit", "✏️ Gérer produits", "📦 Commandes", "📈 Statistiques détaillées"])
    
    # Ajouter produit
    with menu[0]:
        st.subheader("➕ Ajouter un nouveau produit")
        
        with st.form("add_product_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Nom du produit *")
                category = st.selectbox("Catégorie *", CATEGORIES)
                price = st.number_input("Prix *", min_value=0.0, step=10.0)
                stock = st.number_input("Stock initial", min_value=0, value=10)
            
            with col2:
                promo = st.checkbox("En promotion")
                expiration_date = st.date_input("Date d'expiration", min_value=today)
                image = st.file_uploader("Image du produit", type=['png', 'jpg', 'jpeg'])
            
            description = st.text_area("Description", height=100)
            
            submitted = st.form_submit_button("✅ Ajouter le produit")
            
            if submitted:
                if not name or not price:
                    st.error("Veuillez remplir tous les champs obligatoires")
                else:
                    image_path = "default.jpg"
                    if image:
                        if not os.path.exists("images"):
                            os.makedirs("images")
                        img = Image.open(image)
                        image_path = f"images/{image.name}"
                        img.save(image_path)
                    
                    if add_product(name, category, price, description, promo, image_path, stock, expiration_date):
                        st.success(f"✅ Produit '{name}' ajouté avec succès!")
                        st.rerun()
    
    # Gérer produits
    with menu[1]:
        st.subheader("✏️ Gérer les produits")
        
        for product in products:
            with st.expander(f"📦 {product['name']} - {product['price']} {DEVISE}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(f"https://via.placeholder.com/150x150?text={product['name'][:10]}")
                
                with col2:
                    with st.form(f"edit_form_{product['id']}"):
                        new_name = st.text_input("Nom", value=product['name'])
                        new_category = st.selectbox("Catégorie", CATEGORIES, 
                                                   index=CATEGORIES.index(product['category']) if product['category'] in CATEGORIES else 0)
                        new_price = st.number_input("Prix", value=float(product['price']))
                        new_stock = st.number_input("Stock", value=int(product['stock']))
                        new_description = st.text_area("Description", value=product['description'] or "")
                        new_promo = st.checkbox("Promotion", value=bool(product['promo']))
                        
                        # Date d'expiration
                        current_exp = product.get('expiration_date')
                        if current_exp:
                            new_expiration = st.date_input("Date d'expiration", value=current_exp)
                        else:
                            new_expiration = st.date_input("Date d'expiration", value=today)
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if st.form_submit_button("💾 Mettre à jour"):
                                if update_product(product['id'], new_name, new_category, new_price, 
                                                new_description, new_promo, new_stock, new_expiration):
                                    st.success("Produit mis à jour")
                                    st.rerun()
                        
                        with col_b:
                            if st.form_submit_button("🗑️ Supprimer"):
                                if delete_product(product['id']):
                                    st.success("Produit supprimé")
                                    st.rerun()
    
    # Commandes
    with menu[2]:
        st.subheader("📦 Gestion des commandes")
        
        if not orders:
            st.info("Aucune commande pour le moment")
        else:
            for order in orders:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**Commande #{order['id']}**")
                        st.write(f"Client: {order['customer_name']}")
                        st.write(f"Téléphone: {order['customer_phone']}")
                    
                    with col2:
                        st.write(f"Date: {order['created_at']}")
                        st.write(f"Total: {order['total']} {DEVISE}")
                    
                    with col3:
                        status = st.selectbox(
                            "Statut",
                            ["En attente", "Confirmée", "Expédiée", "Livrée", "Annulée"],
                            index=0,
                            key=f"status_{order['id']}"
                        )
                    
                    with st.expander("Voir détails"):
                        st.write("**Adresse de livraison:**")
                        st.write(order['customer_address'])
                        st.write("**Produits commandés:**")
                        products_list = json.loads(order['products'])
                        for p in products_list:
                            st.write(f"- {p['name']}: {p['price']} {DEVISE}")
    
    # Statistiques détaillées
    with menu[3]:
        st.subheader("📈 Analyses détaillées")
        
        if products:
            # DataFrame des produits
            df = pd.DataFrame(products)
            
            # Graphique 1: Répartition par catégorie
            fig1 = px.pie(df, names='category', title='Répartition des produits par catégorie',
                         color_discrete_sequence=px.colors.sequential.Reds)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Graphique 2: Stock par catégorie
            stock_by_cat = df.groupby('category')['stock'].sum().reset_index()
            fig2 = px.bar(stock_by_cat, x='category', y='stock', 
                         title='Stock total par catégorie',
                         color_discrete_sequence=['#dc2626'])
            st.plotly_chart(fig2, use_container_width=True)
            
            # Graphique 3: Produits par prix
            fig3 = px.histogram(df, x='price', nbins=20, 
                               title='Distribution des prix',
                               color_discrete_sequence=['#ef4444'])
            st.plotly_chart(fig3, use_container_width=True)
            
            # Tableau des expirations
            st.subheader("📅 Calendrier des expirations")
            
            exp_df = df[df['expiration_date'].notna()].copy()
            if not exp_df.empty:
                exp_df['days_to_expiry'] = (pd.to_datetime(exp_df['expiration_date']).dt.date - today).apply(lambda x: x.days)
                exp_df = exp_df.sort_values('days_to_expiry')
                
                # Colorer selon l'urgence
                def color_days(val):
                    if val < 0:
                        return 'background-color: #fee2e2'
                    elif val < 30:
                        return 'background-color: #fff3cd'
                    return ''
                
                st.dataframe(
                    exp_df[['name', 'category', 'stock', 'expiration_date', 'days_to_expiry']]
                    .style.applymap(color_days, subset=['days_to_expiry'])
                    .format({'days_to_expiry': '{:.0f} jours'})
                )
    
    # Déconnexion
    st.divider()
    if st.button("🚪 Se déconnecter", use_container_width=True):
        st.session_state.admin_auth = False
        st.rerun()