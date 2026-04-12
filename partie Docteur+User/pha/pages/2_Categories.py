import streamlit as st
from models import get_all_products
from config import CATEGORIES, DEVISE
from datetime import datetime

st.set_page_config(page_title="Catégories", page_icon="📂", layout="wide")

st.title("📂 Nos produits par catégorie")

# Initialisation des filtres dans session state
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = "Tous les produits"
if 'price_range' not in st.session_state:
    st.session_state.price_range = (0, 1000)
if 'show_promo_only' not in st.session_state:
    st.session_state.show_promo_only = False

# Sidebar avec filtres - Design rouge
with st.sidebar:
    st.markdown("""
        <div class="sidebar-title">
            <span style="font-size: 2rem;">🔍</span><br>
            Filtres
        </div>
    """, unsafe_allow_html=True)
    
    # Filtre par catégorie
    st.markdown("### 📂 Catégorie")
    category_options = ["Tous les produits"] + CATEGORIES
    selected_category = st.selectbox(
        "Choisir une catégorie",
        category_options,
        index=category_options.index(st.session_state.selected_category) if st.session_state.selected_category in category_options else 0,
        key="cat_filter"
    )
    st.session_state.selected_category = selected_category
    
    st.markdown("---")
    
    # Filtre par prix
    st.markdown("### 💰 Prix")
    min_price, max_price = st.slider(
        "Intervalle de prix",
        min_value=0,
        max_value=1000,
        value=st.session_state.price_range,
        step=50,
        key="price_slider"
    )
    st.session_state.price_range = (min_price, max_price)
    
    st.markdown("---")
    
    # Filtre promo
    st.markdown("### 🏷️ Promotions")
    show_promo_only = st.checkbox(
        "Afficher uniquement les promotions",
        value=st.session_state.show_promo_only,
        key="promo_check"
    )
    st.session_state.show_promo_only = show_promo_only
    
    st.markdown("---")
    
    # Bouton réinitialiser
    if st.button("🔄 Réinitialiser les filtres", width='stretch'):
        st.session_state.selected_category = "Tous les produits"
        st.session_state.price_range = (0, 1000)
        st.session_state.show_promo_only = False
        st.rerun()
    
    # Résumé du panier
    st.markdown("---")
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #dc2626;
        ">
            <span style="font-size: 2rem;">🛒</span>
            <div style="font-size: 1.2rem; font-weight: bold; color: #991b1b;">{len(st.session_state.cart)} article(s)</div>
        </div>
    """, unsafe_allow_html=True)

# Récupération des produits
products = get_all_products(selected_category if selected_category != "Tous les produits" else None)

# Application des filtres
filtered_products = []
for p in products:
    # Filtre prix
    if min_price <= p['price'] <= max_price:
        # Filtre promo
        if show_promo_only and not p['promo']:
            continue
        filtered_products.append(p)

# Affichage du nombre de produits
st.markdown(f"""
    <div style="
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #dc2626;
    ">
        <span style="font-size: 1.2rem; font-weight: bold;">📦 {len(filtered_products)} produit(s) trouvé(s)</span>
    </div>
""", unsafe_allow_html=True)

if not filtered_products:
    st.warning("😕 Aucun produit ne correspond à vos critères")
    
    # Suggestions
    st.markdown("### 💡 Suggestions")
    suggestions = products[:3] if products else []
    cols = st.columns(3)
    for i, product in enumerate(suggestions):
        with cols[i]:
            with st.container(border=True):
                st.image(f"https://via.placeholder.com/200x150?text={product['name'][:10]}")
                st.markdown(f"**{product['name']}**")
                st.markdown(f"💰 {product['price']} {DEVISE}")
                if st.button("➕ Ajouter", key=f"sugg_{product['id']}"):
                    st.session_state.cart.append(product)
                    st.rerun()
else:
    # Affichage en grille (3 colonnes)
    cols = st.columns(3)
    for i, product in enumerate(filtered_products):
        with cols[i % 3]:
            # Déterminer le statut du stock
            stock_status = "⚠️ Stock faible" if product['stock'] < 5 else "✅ En stock"
            stock_color = "#dc2626" if product['stock'] < 5 else "#10b981"
            
            # Vérifier l'expiration
            today = datetime.now().date()
            expiry_warning = ""
            if product.get('expiration_date'):
                days_to_expiry = (product['expiration_date'] - today).days
                if days_to_expiry < 0:
                    expiry_warning = "❌ Expiré"
                elif days_to_expiry < 30:
                    expiry_warning = f"⚠️ Expire dans {days_to_expiry} jours"
            
            # Carte produit avec design rouge
            with st.container(border=True):
                # Image
                st.image(f"https://via.placeholder.com/300x200?text={product['name'][:15]}",  width="stretch")
                
                # Badge promo si applicable
                if product['promo']:
                    st.markdown("""
                        <div style="
                            background: #dc2626;
                            color: white;
                            padding: 3px 10px;
                            border-radius: 20px;
                            display: inline-block;
                            font-size: 0.8rem;
                            font-weight: bold;
                            margin-bottom: 5px;
                        ">PROMO -20%</div>
                    """, unsafe_allow_html=True)
                
                # Nom du produit
                st.markdown(f"### {product['name']}")
                
                # Catégorie
                st.markdown(f"<span style='color: #6b7280; font-size: 0.9rem;'>{product['category']}</span>", unsafe_allow_html=True)
                
                # Prix
                st.markdown(f"<span class='price'>{product['price']} {DEVISE}</span>", unsafe_allow_html=True)
                
                # Stock
                st.markdown(f"<span style='color: {stock_color};'>{stock_status} ({product['stock']})</span>", unsafe_allow_html=True)
                
                # Alerte expiration
                if expiry_warning:
                    if "Expiré" in expiry_warning:
                        st.error(expiry_warning)
                    else:
                        st.warning(expiry_warning)
                
                # Description (dans expander)
                if product['description']:
                    with st.expander("Description"):
                        st.write(product['description'])
                
                # Bouton d'achat
                if product['stock'] > 0:
                    if st.button("🛒 Ajouter au panier", key=f"add_{product['id']}", width='stretch'):
                        st.session_state.cart.append(product)
                        st.success("✅ Ajouté au panier!")
                        st.rerun()
                else:
                    st.button("❌ Rupture de stock", disabled=True, width='stretch')

# Pagination simple
if len(filtered_products) > 9:
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns([2,1,1,1,2])
    with col2:
        st.button("◀ Précédent", width='stretch')
    with col3:
        st.button("1", width='stretch')
    with col4:
        st.button("2", width='stretch')
    with col5:
        st.button("Suivant ▶", width='stretch')