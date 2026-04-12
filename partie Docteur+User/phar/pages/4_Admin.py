import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.session import init_session_state
from utils.models import get_all_products, add_product, update_product, delete_product, get_all_orders
from config import CATEGORIES, DEVISE, COLORS

st.set_page_config(page_title="Administration", page_icon="⚙️", layout="wide")

# Initialisation session
init_session_state()

# Vérification admin
if 'admin_auth' not in st.session_state:
    st.session_state.admin_auth = False

# CSS personnalisé pour l'admin
st.markdown(f"""
    <style>
        /* Style des cartes admin */
        .admin-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 4px solid {COLORS['primary']};
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }}
        
        .admin-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 20px -5px rgba(220, 38, 38, 0.2);
        }}
        
        /* Alertes */
        .alert-box {{
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            font-weight: 500;
        }}
        
        .alert-danger {{
            background: {COLORS['primary_bg']};
            border-left: 4px solid {COLORS['primary']};
            color: {COLORS['primary']};
        }}
        
        .alert-warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }}
        
        .alert-success {{
            background: #d1fae5;
            border-left: 4px solid {COLORS['success']};
            color: #065f46;
        }}
        
        /* Métriques */
        .metric-container {{
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }}
        
        .metric-container:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(220, 38, 38, 0.2);
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: {COLORS['primary']};
        }}
        
        .metric-label {{
            color: {COLORS['text_light']};
            font-size: 1rem;
        }}
        
        /* Boutons admin */
        .admin-button {{
            background: {COLORS['primary']};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .admin-button:hover {{
            background: {COLORS['primary_dark']};
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(220, 38, 38, 0.3);
        }}
        
        /* Tableaux */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        
        .data-table th {{
            background: {COLORS['primary']};
            color: white;
            padding: 1rem;
            text-align: left;
        }}
        
        .data-table td {{
            padding: 1rem;
            border-bottom: 1px solid {COLORS['border']};
        }}
        
        .data-table tr:hover {{
            background: {COLORS['primary_bg']};
        }}
        
        /* Status badges */
        .status-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }}
        
        .status-pending {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status-confirmed {{
            background: #d1fae5;
            color: #065f46;
        }}
        
        .status-shipped {{
            background: #cffafe;
            color: #0e7490;
        }}
        
        .status-delivered {{
            background: #dcfce7;
            color: {COLORS['success']};
        }}
        
        .status-cancelled {{
            background: {COLORS['primary_bg']};
            color: {COLORS['primary']};
        }}
    </style>
""", unsafe_allow_html=True)

def check_password():
    """Page de connexion admin"""
    st.markdown(f"""
        <div style="
            background: white;
            padding: 3rem;
            border-radius: 20px;
            text-align: center;
            max-width: 500px;
            margin: 5rem auto;
            box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
        ">
            <span style="font-size: 4rem;">🔐</span>
            <h2 style="color: {COLORS['primary']};">Accès Administrateur</h2>
            <p style="color: {COLORS['text_light']};">Veuillez vous connecter pour accéder au tableau de bord</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        password = st.text_input("Mot de passe", type="password", placeholder="Entrez le mot de passe admin")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            submitted = st.form_submit_button("🔑 Se connecter", width='stretch')
        
        if submitted:
            if password == "admin123":
                st.session_state.admin_auth = True
                st.rerun()
            else:
                st.error("❌ Mot de passe incorrect")

if not st.session_state.admin_auth:
    check_password()
else:
    # Header admin
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(220, 38, 38, 0.3);
        ">
            <h1 style="font-size: 2.5rem;">⚙️ Administration</h1>
            <p style="opacity: 0.9;">Gérez vos produits, commandes et statistiques</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Bouton déconnexion
    col1, col2, col3, col4, col5 = st.columns([3,1,1,1,2])
    with col5:
        if st.button("🚪 Déconnexion", width='stretch'):
            st.session_state.admin_auth = False
            st.rerun()
    
    st.divider()
    
    # Récupération des données
    products = get_all_products()
    orders = get_all_orders()
    today = datetime.now().date()
    
    # ============================================
    # ALERTES PRODUITS
    # ============================================
    st.markdown("## 🚨 Alertes & Surveillance")
    
    # Analyse des expirations (corrigé)
    expired_products = []
    expiring_soon = []
    low_stock_products = []
    
    for p in products:
        # Vérification expiration
        if p.get('expiration_date'):
            if isinstance(p['expiration_date'], str):
                try:
                    exp_date = datetime.strptime(p['expiration_date'], '%Y-%m-%d').date()
                except:
                    exp_date = today
            else:
                exp_date = p['expiration_date']
            
            days_to_expiry = (exp_date - today).days
            
            if days_to_expiry < 0:
                expired_products.append(p)
            elif days_to_expiry < 30:
                expiring_soon.append(p)
        
        # Stock faible
        if p.get('stock', 0) < 5:
            low_stock_products.append(p)
    
    # Affichage des alertes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="admin-card">
                <div style="font-size: 2rem;">❌</div>
                <div style="font-size: 2rem; color: {COLORS['primary']};">{len(expired_products)}</div>
                <div>Produits expirés</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="admin-card">
                <div style="font-size: 2rem;">⚠️</div>
                <div style="font-size: 2rem; color: #f59e0b;">{len(expiring_soon)}</div>
                <div>Expirent bientôt</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="admin-card">
                <div style="font-size: 2rem;">📦</div>
                <div style="font-size: 2rem; color: #f59e0b;">{len(low_stock_products)}</div>
                <div>Stock faible</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Affichage détaillé des alertes
    if expired_products:
        st.markdown("### ❌ Produits expirés - ACTION REQUISE")
        for p in expired_products:
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([3,2,2,1])
                with col1:
                    st.write(f"**{p['name']}**")
                with col2:
                    exp_date = p.get('expiration_date', 'N/A')
                    st.write(f"Expiré")
                with col3:
                    st.write(f"Stock: {p.get('stock', 0)} unités")
                with col4:
                    if st.button("🗑️ Retirer", key=f"del_exp_{p['id']}"):
                        if delete_product(p['id']):
                            st.success("Produit supprimé")
                            st.rerun()
    
    if expiring_soon:
        st.markdown("### ⚠️ Produits expirant dans moins de 30 jours")
        for p in expiring_soon:
            with st.container(border=True):
                col1, col2, col3 = st.columns([3,2,2])
                with col1:
                    st.write(f"**{p['name']}**")
                with col2:
                    exp_date = p.get('expiration_date', 'N/A')
                    st.write(f"Expire bientôt")
                with col3:
                    st.write(f"Stock: {p.get('stock', 0)} unités")
    
    if low_stock_products:
        st.markdown("### 📦 Produits en stock faible (<5 unités)")
        for p in low_stock_products:
            with st.container(border=True):
                col1, col2, col3 = st.columns([3,2,2])
                with col1:
                    st.write(f"**{p['name']}**")
                with col2:
                    st.write(f"Stock: {p.get('stock', 0)} unités")
                with col3:
                    if st.button("➕ Réapprovisionner", key=f"restock_{p['id']}"):
                        st.info("Fonctionnalité de réapprovisionnement à implémenter")
    
    st.divider()
    
    # ============================================
    # DASHBOARD STATISTIQUES (CORRIGÉ)
    # ============================================
    st.markdown("## 📊 Tableau de bord")
    
    # Métriques principales (corrigé)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{len(products)}</div>
                <div class="metric-label">📦 Produits total</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        stock_total = sum(p.get('stock', 0) for p in products)
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{stock_total}</div>
                <div class="metric-label">📊 Stock total</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # CORRECTION ICI: Convertir en liste au lieu de générateur
        produits_promo = [p for p in products if p.get('promo', False)]
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{len(produits_promo)}</div>
                <div class="metric-label">🏷️ En promotion</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if orders:
            total_ca = sum(float(order.get('total', 0)) for order in orders)
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{total_ca:.0f} {DEVISE}</div>
                    <div class="metric-label">💰 CA Total</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">0 {DEVISE}</div>
                    <div class="metric-label">💰 CA Total</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Menu admin avec onglets
    tab1, tab2, tab3, tab4 = st.tabs(["➕ Ajouter un produit", "✏️ Gérer les produits", "📦 Commandes", "📈 Statistiques détaillées"])
    
    # ============================================
    # TAB 1: AJOUTER PRODUIT
    # ============================================
    with tab1:
        st.markdown("### ➕ Ajouter un nouveau produit")
        
        with st.form("add_product_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Nom du produit *", placeholder="Ex: Crème hydratante")
                category = st.selectbox("Catégorie *", CATEGORIES)
                price = st.number_input("Prix *", min_value=0.0, step=10.0, format="%.2f")
                stock = st.number_input("Stock initial", min_value=0, value=10, step=1)
            
            with col2:
                promo = st.checkbox("En promotion")
                expiration_date = st.date_input("Date d'expiration", min_value=today, value=today + timedelta(days=365))
                image_url = st.text_input("URL de l'image", placeholder="https://...")
            
            description = st.text_area("Description", height=100, placeholder="Description détaillée du produit...")
            
            submitted = st.form_submit_button("✅ Ajouter le produit", width='stretch')
            
            if submitted:
                if not name or price <= 0:
                    st.error("❌ Veuillez remplir tous les champs obligatoires")
                else:
                    if add_product(
                        name, category, price, description, promo, 
                        image_url or "default.jpg", stock, expiration_date
                    ):
                        st.success(f"✅ Produit '{name}' ajouté avec succès!")
                        st.balloons()
                        st.rerun()
    
    # ============================================
    # TAB 2: GÉRER PRODUITS
    # ============================================
    with tab2:
        st.markdown("### ✏️ Gestion des produits")
        
        # Recherche
        search = st.text_input("🔍 Rechercher un produit", placeholder="Nom du produit...")
        
        if search:
            filtered_products = [p for p in products if search.lower() in p['name'].lower()]
        else:
            filtered_products = products
        
        if not filtered_products:
            st.info("Aucun produit trouvé")
        else:
            for product in filtered_products:
                with st.expander(f"📦 {product['name']} - {product['price']} {DEVISE}"):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Image placeholder
                        st.image("https://via.placeholder.com/150x150?text=Produit", width='stretch')
                    
                    with col2:
                        with st.form(f"edit_form_{product['id']}"):
                            new_name = st.text_input("Nom", value=product['name'])
                            new_category = st.selectbox(
                                "Catégorie", 
                                CATEGORIES, 
                                index=CATEGORIES.index(product['category']) if product['category'] in CATEGORIES else 0
                            )
                            new_price = st.number_input("Prix", value=float(product['price']), min_value=0.0)
                            new_stock = st.number_input("Stock", value=int(product.get('stock', 0)), min_value=0)
                            new_description = st.text_area("Description", value=product.get('description', '') or "")
                            new_promo = st.checkbox("Promotion", value=bool(product.get('promo', False)))
                            
                            # Date d'expiration
                            current_exp = product.get('expiration_date')
                            if current_exp:
                                if isinstance(current_exp, str):
                                    try:
                                        current_exp = datetime.strptime(current_exp, '%Y-%m-%d').date()
                                    except:
                                        current_exp = today
                                new_expiration = st.date_input("Date d'expiration", value=current_exp)
                            else:
                                new_expiration = st.date_input("Date d'expiration", value=today)
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                if st.form_submit_button("💾 Mettre à jour", width='stretch'):
                                    if update_product(
                                        product['id'], new_name, new_category, new_price,
                                        new_description, new_promo, new_stock, new_expiration
                                    ):
                                        st.success("✅ Produit mis à jour")
                                        st.rerun()
                            
                            with col_b:
                                if st.form_submit_button("🗑️ Supprimer", width='stretch'):
                                    if delete_product(product['id']):
                                        st.success("✅ Produit supprimé")
                                        st.rerun()
    
    # ============================================
    # TAB 3: COMMANDES
    # ============================================
    with tab3:
        st.markdown("### 📦 Gestion des commandes")
        
        if not orders:
            st.info("📭 Aucune commande pour le moment")
        else:
            # Filtres
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.selectbox(
                    "Filtrer par statut",
                    ["Toutes", "En attente", "Confirmée", "Expédiée", "Livrée", "Annulée"]
                )
            
            # Tableau des commandes
            for order in orders:
                # Appliquer le filtre
                if status_filter != "Toutes" and order.get('status', 'En attente') != status_filter:
                    continue
                
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**Commande #{order['id']}**")
                        st.write(f"👤 {order['customer_name']}")
                        st.write(f"📞 {order['customer_phone']}")
                    
                    with col2:
                        st.write(f"📅 {order['created_at']}")
                        st.write(f"💰 {order['total']} {DEVISE}")
                    
                    with col3:
                        # Statut avec badge
                        status = order.get('status', 'En attente')
                        status_class = {
                            'En attente': 'status-pending',
                            'Confirmée': 'status-confirmed',
                            'Expédiée': 'status-shipped',
                            'Livrée': 'status-delivered',
                            'Annulée': 'status-cancelled'
                        }.get(status, 'status-pending')
                        
                        st.markdown(f'<span class="status-badge {status_class}">{status}</span>', unsafe_allow_html=True)
                        
                        # Changement de statut
                        new_status = st.selectbox(
                            "Changer statut",
                            ["En attente", "Confirmée", "Expédiée", "Livrée", "Annulée"],
                            key=f"status_{order['id']}"
                        )
                    
                    with col4:
                        if st.button("📋 Détails", key=f"details_{order['id']}"):
                            st.session_state[f"show_details_{order['id']}"] = True
                    
                    # Détails de la commande
                    if st.session_state.get(f"show_details_{order['id']}", False):
                        with st.expander("Détails de la commande", expanded=True):
                            st.write("**Adresse de livraison:**")
                            st.write(order['customer_address'])
                            st.write("**Produits commandés:**")
                            try:
                                import json
                                products_list = json.loads(order['products'])
                                for p in products_list:
                                    st.write(f"- {p.get('name', 'Produit')}: {p.get('price', 0)} {DEVISE}")
                            except:
                                st.write(order['products'])
    
    # ============================================
    # TAB 4: STATISTIQUES DÉTAILLÉES
    # ============================================
    with tab4:
        st.markdown("### 📈 Analyses détaillées")
        
        if products:
            # Création DataFrame
            df = pd.DataFrame(products)
            
            # Graphique 1: Répartition par catégorie
            fig1 = px.pie(
                df, 
                names='category', 
                title='Répartition des produits par catégorie',
                color_discrete_sequence=px.colors.sequential.Reds,
                hole=0.3
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, width='stretch')
            
            # Graphique 2: Stock par catégorie
            stock_by_cat = df.groupby('category')['stock'].sum().reset_index()
            fig2 = px.bar(
                stock_by_cat, 
                x='category', 
                y='stock', 
                title='Stock total par catégorie',
                color_discrete_sequence=[COLORS['primary']]
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, width='stretch')
            
            # Graphique 3: Distribution des prix
            fig3 = px.histogram(
                df, 
                x='price', 
                nbins=20, 
                title='Distribution des prix',
                color_discrete_sequence=[COLORS['primary_light']]
            )
            st.plotly_chart(fig3, width='stretch')
            
            # Graphique 4: Évolution des ventes (si commandes)
            if orders:
                orders_df = pd.DataFrame(orders)
                orders_df['created_at'] = pd.to_datetime(orders_df['created_at'])
                orders_df['date'] = orders_df['created_at'].dt.date
                
                daily_sales = orders_df.groupby('date')['total'].sum().reset_index()
                fig4 = px.line(
                    daily_sales, 
                    x='date', 
                    y='total', 
                    title='Évolution du chiffre d\'affaires',
                    markers=True
                )
                fig4.update_traces(line_color=COLORS['primary'])
                st.plotly_chart(fig4, width='stretch')
            
            # Tableau des expirations
            st.subheader("📅 Calendrier des expirations")
            
            exp_data = []
            for p in products:
                if p.get('expiration_date'):
                    exp_date = p['expiration_date']
                    if isinstance(exp_date, str):
                        try:
                            exp_date = datetime.strptime(exp_date, '%Y-%m-%d').date()
                        except:
                            exp_date = today
                    
                    days_to_expiry = (exp_date - today).days
                    exp_data.append({
                        'Produit': p['name'],
                        'Catégorie': p['category'],
                        'Stock': p.get('stock', 0),
                        'Date expiration': exp_date,
                        'Jours restants': days_to_expiry,
                        'Statut': 'Expiré' if days_to_expiry < 0 else 'Critique' if days_to_expiry < 30 else 'Normal'
                    })
            
            if exp_data:
                exp_df = pd.DataFrame(exp_data)
                exp_df = exp_df.sort_values('Jours restants')
                
                # Fonction pour colorer
                def color_status(val):
                    if val == 'Expiré':
                        return 'background-color: #fee2e2'
                    elif val == 'Critique':
                        return 'background-color: #fff3cd'
                    return ''
                
                st.dataframe(
                    exp_df.style.applymap(color_status, subset=['Statut']),
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info("Aucun produit avec date d'expiration")
    
    # Footer
    st.divider()
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-top: 2rem;
        ">
            <p>© 2024 - Panneau d'administration {DEVISE}</p>
        </div>
    """, unsafe_allow_html=True)