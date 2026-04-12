import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.session import init_session_state
from utils.models import get_all_products
from config import SITE_NAME, DEVISE, COLORS, CATEGORY_IMAGES, DEFAULT_PRODUCT_IMAGE

# Configuration
st.set_page_config(
    page_title=SITE_NAME,
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation session
init_session_state()

# CSS personnalisé - Design Rouge
st.markdown(f"""
    <style>
        /* === FOND GLOBAL SOMBRE === */
        .stApp {{
            background: #0d0d1a !important;
            color: #FFFFFF !important;
        }}

        div.block-container {{
            background-color: #0a0a1a !important;
        }}

        /* === CONTAINERS TRANSPARENTS === */
        section[data-testid="stForm"],
        div[data-testid="stVerticalBlock"] > div,
        .element-container, .stForm {{
            background-color: transparent !important;
        }}
        
        /* Header avec dégradé rouge */
        .main-header {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(230, 57, 70, 0.3);
            text-align: center;
            animation: slideIn 0.5s ease-out;
        }}
        
        /* Animation header */
        @keyframes slideIn {{
            from {{
                transform: translateY(-20px);
                opacity: 0;
            }}
            to {{
                transform: translateY(0);
                opacity: 1;
            }}
        }}
        
        /* Cartes produits sombres */
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            background: #1a1a2e !important;
            border-radius: 15px;
            padding: 1rem;
            border: 1px solid #e63946 !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        
        div[data-testid="stVerticalBlockBorderWrapper"] > div:hover {{
            transform: translateY(-8px);
            box-shadow: 0 20px 25px -5px rgba(230, 57, 70, 0.3), 0 10px 10px -5px rgba(0,0,0,0.1);
        }}
        
        /* Badge promo */
        .promo-badge {{
            position: absolute;
            top: 10px;
            right: -30px;
            background: {COLORS['primary']};
            color: white;
            padding: 5px 30px;
            transform: rotate(45deg);
            font-size: 0.8rem;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            z-index: 10;
        }}
        
        /* === BOUTONS === */
        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
            color: white !important;
            border: none !important;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 1rem;
            box-shadow: 0 4px 6px rgba(230, 57, 70, 0.25);
            transition: all 0.3s ease;
            width: 100%;
            position: relative;
            overflow: hidden;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 15px -3px rgba(230, 57, 70, 0.4);
            background: linear-gradient(135deg, {COLORS['primary_dark']} 0%, {COLORS['primary']} 100%);
        }}
        
        .stButton > button:active {{
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(230, 57, 70, 0.3);
        }}
        
        .stButton > button::after {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255,255,255,0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }}
        
        .stButton > button:hover::after {{
            width: 300px;
            height: 300px;
        }}
        
        /* Bouton secondaire */
        .stButton > button.secondary {{
            background: #1a1a2e !important;
            color: {COLORS['primary']} !important;
            border: 2px solid {COLORS['primary']} !important;
            box-shadow: none;
        }}
        
        .stButton > button.secondary:hover {{
            background: #2a2a4e !important;
            color: {COLORS['primary_dark']} !important;
            border-color: {COLORS['primary_dark']} !important;
        }}
        
        /* Prix */
        .price {{
            font-size: 1.8rem;
            font-weight: bold;
            color: {COLORS['primary']} !important;
            margin: 0.5rem 0;
        }}
        
        /* Stock indicator */
        .stock-indicator {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        .stock-low {{
            background: rgba(230, 57, 70, 0.15) !important;
            color: {COLORS['primary']} !important;
        }}
        
        .stock-ok {{
            background: rgba(16, 185, 129, 0.15) !important;
            color: {COLORS['success']} !important;
        }}
        
        /* Images */
        .product-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 10px;
            margin-bottom: 1rem;
            transition: transform 0.3s ease;
        }}
        
        .product-image:hover {{
            transform: scale(1.05);
        }}
        
        /* Catégorie badge */
        .category-badge {{
            background: rgba(230, 57, 70, 0.15) !important;
            color: {COLORS['primary']} !important;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            display: inline-block;
            margin-bottom: 0.5rem;
        }}
        
        /* === SIDEBAR SOMBRE === */
        section[data-testid="stSidebar"] {{
            background: #0d0d1a !important;
            border-right: 1px solid #e63946;
        }}
        
        /* Cart summary */
        .cart-summary {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
            padding: 1rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(230, 57, 70, 0.3);
        }}
        
        /* Navigation pills */
        .nav-pill {{
            background: #1a1a2e !important;
            padding: 0.75rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #e63946 !important;
            transition: all 0.3s ease;
            cursor: pointer;
            margin: 0.25rem 0;
            color: #FFFFFF !important;
        }}
        
        .nav-pill:hover {{
            background: #2a2a4e !important;
            border-color: {COLORS['primary']};
            transform: translateX(5px);
        }}
        
        .nav-pill.active {{
            background: {COLORS['primary']} !important;
            color: white !important;
            border-color: {COLORS['primary']};
        }}
        
        /* Success animation */
        @keyframes checkmark {{
            0% {{ transform: scale(0); }}
            50% {{ transform: scale(1.2); }}
            100% {{ transform: scale(1); }}
        }}
        
        .success-icon {{
            animation: checkmark 0.5s ease-in-out;
        }}

        /* === INPUTS SOMBRES === */
        input, textarea, select,
        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stSelectbox > div > div > select,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {{
            background-color: #1a1a2e !important;
            color: #FFFFFF !important;
            border: 1px solid #e63946 !important;
            border-radius: 8px !important;
            caret-color: #e63946 !important;
        }}

        /* === PLACEHOLDERS === */
        ::placeholder {{ color: #888888 !important; }}
        input::placeholder, textarea::placeholder {{
            color: #888888 !important;
        }}

        /* === LABELS & TEXTE BLANCS === */
        label, .stTextInput label, .stSelectbox label {{
            color: #FFFFFF !important;
            font-weight: 500 !important;
        }}

        label, p, span, div, h1, h2, h3, h4, h5, h6 {{
            color: #FFFFFF !important;
        }}

        /* === TABS SOMBRES === */
        .stTabs [data-baseweb="tab"] {{
            background-color: #1a1a2e !important;
            color: #FFFFFF !important;
            border: 1px solid #e63946 !important;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: #e63946 !important;
            color: #FFFFFF !important;
        }}

        /* === DIVIDERS === */
        hr {{
            border-color: rgba(230, 57, 70, 0.3) !important;
        }}

        /* === SCROLLBAR SOMBRE === */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: #0d0d1a;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #e63946;
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #c1121f;
        }}
    </style>
""", unsafe_allow_html=True)

# Header principal
st.markdown(f"""
    <div class="main-header">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">💊 {SITE_NAME}</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Votre santé, notre priorité ✨</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span style="font-size: 3rem;">💊</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Résumé panier
    st.markdown(f"""
        <div class="cart-summary">
            <span style="font-size: 2rem;">🛒</span>
            <h3 style="margin: 0.5rem 0;">{len(st.session_state.cart)} article(s)</h3>
            <p style="margin: 0; opacity: 0.9;">dans votre panier</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### 📍 Navigation")
    
    pages = [
        ("🏠 Accueil", "app.py", True),
        ("📂 Catégories", "pages/1_Categories.py", False),
        ("🛒 Panier", "pages/2_Panier.py", False),
        ("📦 Commander", "pages/3_Commande.py", False),
        ("⚙️ Admin", "pages/4_Admin.py", False)
    ]
    
    for label, path, active in pages:
        if st.button(label, key=f"nav_{label}", width='stretch'):
            st.switch_page(path)

# Bouton vers autre app
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
with col1:
    
    if st.button("📂مساعدك الطبي الذكي", width='stretch'):
                try:
                    subprocess.Popen(
                        [sys.executable, "-m", "streamlit", "run", "chatBot-Arab-main/pp.py", "--server.port", "8502"]
                    )
                    st.success("Application 2 lancée ! Ouvrez http://localhost:8502")
                except Exception as e:
                    st.error(f"Erreur lancement app2: {e}")

with col2:
    
    if st.button("📂APP", width='stretch'):
                try:
                    subprocess.Popen(
                        [sys.executable, "-m", "streamlit", "run", "PROJET - ROUGE\\main.py", "--server.port", "8501"]
                    )
                    st.success("Application 2 lancée ! Ouvrez http://localhost:8501")
                except Exception as e:
                    st.error(f"Erreur lancement app2: {e}")


# Contenu principal
st.markdown("## 🌟 Bienvenue sur notre plateforme")

# Bannières promotionnelles
col1, col2, col3 = st.columns(3)
with col1:
    with st.container():
        st.image("https://images.unsplash.com/photo-1607082348824-0a96f2a4b9da?w=400", width='stretch')
        st.markdown("""
            <div style="text-align: center;">
                <h3>🔥 -20%</h3>
                <p>sur tout le maquillage</p>
            </div>
        """, unsafe_allow_html=True)
with col2:
    with st.container():
        st.image("https://images.unsplash.com/photo-1542838132-92c53300491e?w=400", width='stretch')
        st.markdown("""
            <div style="text-align: center;">
                <h3>🚚 Livraison gratuite</h3>
                <p>dès 500 MAD</p>
            </div>
        """, unsafe_allow_html=True)
with col3:
    with st.container():
        st.image("https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400", width='stretch')
        st.markdown("""
            <div style="text-align: center;">
                <h3>✨ Nouveautés</h3>
                <p>Découvrez nos produits</p>
            </div>
        """, unsafe_allow_html=True)

st.divider()

# Catégories rapides
st.markdown("## 📂 Catégories populaires")
cols = st.columns(6)
categories = ["Visage", "Cheveux", "Corps", "Maquillage", "Bébé", "Promotion"]
for i, cat in enumerate(categories):
    with cols[i]:
        with st.container():
            st.image(CATEGORY_IMAGES.get(cat, DEFAULT_PRODUCT_IMAGE), width='stretch')
            if st.button(f"🔹 {cat}", key=f"cat_{i}", width='stretch'):
                st.session_state.selected_category = cat
                st.switch_page("pages/1_Categories.py")

st.divider()

# Produits en vedette
st.markdown("## 🔥 Produits en vedette")
products = get_all_products()

if products:
    cols = st.columns(4)
    for i, product in enumerate(products[:8]):
        with cols[i % 4]:
            with st.container():
                # Image
                image_url = CATEGORY_IMAGES.get(product['category'], DEFAULT_PRODUCT_IMAGE)
                st.image(image_url, width='stretch')
                
                # Badge promo
                if product.get('promo', False):
                    st.markdown('<div class="promo-badge">PROMO</div>', unsafe_allow_html=True)
                
                # Catégorie
                st.markdown(f'<span class="category-badge">{product["category"]}</span>', unsafe_allow_html=True)
                
                # Nom
                st.markdown(f"### {product['name']}")
                
                # Prix
                st.markdown(f'<div class="price">{product["price"]} {DEVISE}</div>', unsafe_allow_html=True)
                
                # Stock
                stock_class = "stock-low" if product.get('stock', 0) < 5 else "stock-ok"
                stock_text = "⚠️ Stock faible" if product.get('stock', 0) < 5 else "✅ En stock"
                st.markdown(f'<span class="stock-indicator {stock_class}">{stock_text}</span>', unsafe_allow_html=True)
                
                # Bouton
                if product.get('stock', 0) > 0:
                    if st.button("🛒 Ajouter au panier", key=f"add_{product['id']}", width='stretch'):
                        st.session_state.cart.append(product)
                        st.balloons()
                        st.rerun()
                else:
                    st.button("❌ Rupture de stock", disabled=True, width='stretch')
else:
    st.info("Aucun produit disponible pour le moment")

# Footer
st.divider()
st.markdown(" 🏥 app")
if st.button("📂APP", width='stretch'):
            try:
                subprocess.Popen(
                    [sys.executable, "-m", "streamlit", "run", "PROJET - ROUGE\\main.py", "--server.port", "8501"]
                )
                st.success("Application 2 lancée ! Ouvrez http://localhost:8501")
            except Exception as e:
                st.error(f"Erreur lancement app2: {e}")
st.divider()
st.markdown(" 🏥 إعداد المساعد الصحي الذكي بالعربية")
if st.button("📂مساعدك الطبي الذكي", width='stretch'):
            try:
                subprocess.Popen(
                    [sys.executable, "-m", "streamlit", "run", "chatBot-Arab-main/pp.py", "--server.port", "8502"]
                )
                st.success("Application 2 lancée ! Ouvrez http://localhost:8502")
            except Exception as e:
                st.error(f"Erreur lancement app2: {e}")
st.divider()

st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-top: 2rem;
    ">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 2rem;">
            <div>
                <h4>À propos</h4>
                <p>Notre histoire<br>Nos magasins<br>Recrutement</p>
            </div>
            <div>
                <h4>Service client</h4>
                <p>Contact<br>Livraison<br>Paiement sécurisé</p>
            </div>
            <div>
                <h4>Conseils</h4>
                <p>Blog<br>Guides d'achat<br>Questions fréquentes</p>
            </div>
            <div>
                <h4>Suivez-nous</h4>
                <p>Facebook<br>Instagram<br>LinkedIn</p>
            </div>
        </div>
        <div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
            © 2024 {SITE_NAME} - Tous droits réservés
        </div>
    </div>
""", unsafe_allow_html=True)