import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.session import init_session_state
from utils.models import get_all_products
from config import CATEGORIES, DEVISE, COLORS, CATEGORY_IMAGES, DEFAULT_PRODUCT_IMAGE

st.set_page_config(page_title="Catégories", page_icon="📂", layout="wide")

# Initialisation session
init_session_state()

# CSS
st.markdown(f"""
    <style>
        /* Mêmes styles que app.py */
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            background: white;
            border-radius: 15px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            margin-bottom: 1rem;
        }}
        
        div[data-testid="stVerticalBlockBorderWrapper"] > div:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 20px -5px rgba(220, 38, 38, 0.2);
        }}
        
        .filter-section {{
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }}
        
        .price-value {{
            font-size: 1.2rem;
            font-weight: bold;
            color: {COLORS['primary']};
        }}
    </style>
""", unsafe_allow_html=True)

st.title("📂 Nos produits par catégorie")

# Sidebar filtres améliorée
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span style="font-size: 2rem;">🔍</span>
            <h3>Filtres</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("### 📂 Catégorie")
        selected_category = st.selectbox(
            "Choisir une catégorie",
            ["Tous les produits"] + CATEGORIES,
            key="cat_filter"
        )
        st.session_state.selected_category = selected_category
    
    st.markdown("---")
    
    with st.container():
        st.markdown("### 💰 Prix")
        min_price, max_price = st.slider(
            "Intervalle de prix",
            min_value=0, max_value=1000, value=st.session_state.price_range, step=50
        )
        st.session_state.price_range = (min_price, max_price)
        st.markdown(f'<div class="price-value">{min_price} - {max_price} {DEVISE}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.container():
        st.markdown("### 🏷️ Promotions")
        show_promo_only = st.checkbox("Afficher uniquement les promotions", value=st.session_state.show_promo_only)
        st.session_state.show_promo_only = show_promo_only
    
    st.markdown("---")
    
    if st.button("🔄 Réinitialiser les filtres", width='stretch'):
        st.session_state.selected_category = "Tous les produits"
        st.session_state.price_range = (0, 1000)
        st.session_state.show_promo_only = False
        st.rerun()
    
    st.markdown("---")
    
    # Panier résumé
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {COLORS['primary_bg']} 0%, #fee2e2 100%);
            padding: 1rem;
            border-radius: 15px;
            text-align: center;
            border-left: 4px solid {COLORS['primary']};
            margin-top: 1rem;
        ">
            <span style="font-size: 2rem;">🛒</span>
            <h4 style="color: {COLORS['primary']};">{len(st.session_state.cart)} article(s)</h4>
        </div>
    """, unsafe_allow_html=True)

# Récupération produits
category_filter = None if selected_category == "Tous les produits" else selected_category
products = get_all_products(category_filter)

# Filtrage
filtered_products = []
for p in products:
    if min_price <= p['price'] <= max_price:
        if show_promo_only and not p.get('promo', False):
            continue
        filtered_products.append(p)

# En-tête résultats
st.markdown(f"""
    <div style="
        background: white;
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border-left: 4px solid {COLORS['primary']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    ">
        <h3>📦 {len(filtered_products)} produit(s) trouvé(s)</h3>
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
            with st.container():
                image_url = CATEGORY_IMAGES.get(product['category'], DEFAULT_PRODUCT_IMAGE)
                st.image(image_url, width='stretch')
                st.markdown(f"### {product['name']}")
                st.markdown(f"**{product['price']} {DEVISE}**")
                if st.button("➕ Ajouter", key=f"sugg_{product['id']}", width='stretch'):
                    st.session_state.cart.append(product)
                    st.rerun()
else:
    # Grille produits
    cols = st.columns(3)
    for i, product in enumerate(filtered_products):
        with cols[i % 3]:
            with st.container():
                # Image
                image_url = CATEGORY_IMAGES.get(product['category'], DEFAULT_PRODUCT_IMAGE)
                st.image(image_url, width='stretch')
                
                # Badge promo
                if product.get('promo', False):
                    st.markdown(f'<div style="background:{COLORS["primary"]};color:white;padding:2px 8px;border-radius:20px;display:inline-block;">PROMO -20%</div>', unsafe_allow_html=True)
                
                # Nom
                st.markdown(f"### {product['name']}")
                
                # Catégorie
                st.markdown(f"<span style='color:{COLORS['text_light']};'>{product['category']}</span>", unsafe_allow_html=True)
                
                # Prix
                st.markdown(f"<span style='font-size:1.5rem;font-weight:bold;color:{COLORS['primary']};'>{product['price']} {DEVISE}</span>", unsafe_allow_html=True)
                
                # Stock
                stock_color = COLORS['danger'] if product.get('stock', 0) < 5 else COLORS['success']
                stock_text = "⚠️ Stock faible" if product.get('stock', 0) < 5 else "✅ En stock"
                st.markdown(f"<span style='color:{stock_color};'>{stock_text} ({product.get('stock', 0)})</span>", unsafe_allow_html=True)
                
                # Bouton
                if product.get('stock', 0) > 0:
                    if st.button("🛒 Ajouter au panier", key=f"add_{product['id']}", width='stretch'):
                        st.session_state.cart.append(product)
                        st.balloons()
                        st.rerun()
                else:
                    st.button("❌ Rupture de stock", disabled=True, width='stretch')