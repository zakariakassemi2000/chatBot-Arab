import streamlit as st
from config import SITE_NAME, SITE_ICON
from db import init_database
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title=SITE_NAME,
    page_icon=SITE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de la base de données
if 'db_initialized' not in st.session_state:
    if init_database():
        st.session_state.db_initialized = True
    else:
        st.error("⚠️ Problème de connexion à la base de données")

# Initialisation du panier
if 'cart' not in st.session_state:
    st.session_state.cart = []

# Initialisation de l'utilisateur (simulé - à remplacer par vrai auth)
if 'user' not in st.session_state:
    st.session_state.user = {
        'username': 'client',
        'full_name': 'Client',
        'role': 'client'
    }

# ============================================
# DESIGN ROUGE PROFESSIONNEL
# ============================================
st.markdown("""
    <style>
    /* === FOND GLOBAL SOMBRE === */
    .stApp, .main, section.main {
        background-color: #0d0d1a !important;
        color: #FFFFFF !important;
    }

    div.block-container {
        background-color: #0a0a1a !important;
    }

    /* === CONTAINERS TRANSPARENTS === */
    section[data-testid="stForm"],
    div[data-testid="stVerticalBlock"] > div,
    .element-container, .stForm {
        background-color: transparent !important;
    }

    /* Style général - Thème Rouge Sombre */
    .main-header {
        background: linear-gradient(90deg, #e63946 0%, #ef4444 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(230, 57, 70, 0.3);
    }
    
    /* Style des boutons de navigation */
    div.row-widget.stRadio > div {
        flex-direction: column;
        gap: 0.5rem;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        background-color: #1a1a2e !important;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1px solid #e63946 !important;
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        color: #FFFFFF !important;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
        background-color: #2a2a4e !important;
        border-color: #e63946 !important;
        transform: translateX(5px);
        box-shadow: 0 4px 6px rgba(230, 57, 70, 0.3);
    }
    
    /* Bouton sélectionné - ROUGE */
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: #e63946 !important;
        border-color: #e63946 !important;
        box-shadow: 0 2px 8px rgba(230, 57, 70, 0.4) !important;
    }
    
    /* Texte du bouton sélectionné */
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    /* === BOUTONS === */
    .stButton > button {
        background-color: #e63946 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(230, 57, 70, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #c1121f !important;
        box-shadow: 0 6px 10px rgba(230, 57, 70, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 4px rgba(230, 57, 70, 0.3) !important;
    }
    
    /* Style des boutons secondaires */
    .stButton > button.secondary {
        background-color: #1a1a2e !important;
        color: #e63946 !important;
        border: 2px solid #e63946 !important;
        box-shadow: none !important;
    }
    
    .stButton > button.secondary:hover {
        background-color: #2a2a4e !important;
    }
    
    /* Style des expanders */
    .streamlit-expanderHeader {
        background-color: #1a1a2e !important;
        border-radius: 8px;
        border-left: 4px solid #e63946;
        color: #FFFFFF !important;
    }
    
    /* Style des cartes produits */
    .product-card {
        background: #1a1a2e !important;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e63946 !important;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(230, 57, 70, 0.3);
        border-color: #e63946;
    }
    
    .product-card.promo::before {
        content: "PROMO";
        position: absolute;
        top: 10px;
        right: -30px;
        background: #e63946;
        color: white;
        padding: 5px 30px;
        transform: rotate(45deg);
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Style des alertes expiration */
    .expired-alert {
        background: rgba(230, 57, 70, 0.15) !important;
        border-left: 4px solid #e63946;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .warning-alert {
        background: rgba(255, 193, 7, 0.15) !important;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Style des notifications */
    .stAlert {
        border-left: 4px solid #e63946;
        border-radius: 8px;
        background-color: #1a1a2e !important;
    }
    
    /* === SIDEBAR SOMBRE === */
    section[data-testid="stSidebar"] {
        background-color: #0d0d1a !important;
        border-right: 1px solid #e63946;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #e63946 !important;
    }
    
    /* Titre dans la sidebar */
    .sidebar-title {
        background: linear-gradient(135deg, #e63946 0%, #ef4444 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(230, 57, 70, 0.3);
    }
    
    /* Animation pour les icônes */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .nav-icon {
        animation: pulse 2s infinite;
        display: inline-block;
    }
    
    /* Style des métriques */
    div[data-testid="stMetricValue"] {
        color: #e63946 !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    
    /* Style des métriques avec fond sombre */
    .metric-card {
        background: #1a1a2e !important;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e63946 !important;
        border-bottom: 3px solid #e63946;
        text-align: center;
    }
    
    /* === TABS SOMBRES === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid #e63946 !important;
        color: #FFFFFF !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #e63946 !important;
        color: #FFFFFF !important;
    }
    
    /* === FORMULAIRES SOMBRES === */
    div[data-testid="stForm"] {
        background-color: #1a1a2e !important;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e63946 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* === INPUTS SOMBRES === */
    input, textarea, select,
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox > div > div > select,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea,
    .stTextInput input, .stTextArea textarea,
    .stSelectbox select {
        background-color: #1a1a2e !important;
        color: #FFFFFF !important;
        border: 1px solid #e63946 !important;
        border-radius: 8px !important;
        caret-color: #e63946 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #e63946 !important;
        box-shadow: 0 0 0 2px rgba(230, 57, 70, 0.3) !important;
    }
    
    /* === PLACEHOLDERS === */
    ::placeholder { color: #888888 !important; }
    input::placeholder, textarea::placeholder {
        color: #888888 !important;
    }
    
    /* === LABELS BLANCS === */
    label, .stTextInput label, .stSelectbox label,
    .stTextArea label, .stRadio label, .stCheckbox label {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    /* === TEXTE GLOBAL BLANC === */
    label, p, span, div, h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Style des badges promo */
    .promo-badge {
        background: #e63946;
        color: white;
        padding: 3px 8px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Style prix */
    .price {
        font-size: 1.5rem;
        font-weight: bold;
        color: #e63946 !important;
    }
    
    /* Style stock indicator */
    .stock-low {
        color: #e63946 !important;
        font-weight: bold;
    }
    
    .stock-ok {
        color: #10b981 !important;
        font-weight: bold;
    }
    
    /* Style date expiration */
    .expiry-date {
        font-size: 0.8rem;
        padding: 2px 6px;
        border-radius: 4px;
        background: #1a1a2e !important;
    }
    
    .expiry-date.warning {
        background: rgba(230, 57, 70, 0.15) !important;
        color: #e63946 !important;
    }
    
    /* Style du footer */
    .footer {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }

    /* === DIVIDERS === */
    hr {
        border-color: rgba(230, 57, 70, 0.3) !important;
    }

    /* === SCROLLBAR SOMBRE === */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0d0d1a;
    }
    ::-webkit-scrollbar-thumb {
        background: #e63946;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #c1121f;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar avec design rouge
with st.sidebar:
    st.markdown("""
        <div class="sidebar-title">
            <span style="font-size: 2rem;">💊</span><br>
            """ + SITE_NAME + """
        </div>
    """, unsafe_allow_html=True)
    
    # Profil utilisateur
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 4px solid #dc2626;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        ">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2rem;">👤</span>
                <div>
                    <div style="font-weight: bold; color: #991b1b;">{st.session_state.user['full_name']}</div>
                    <div style="font-size: 0.8rem; color: #dc2626;">{st.session_state.user['role'].upper()}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Panier résumé
    st.markdown(f"""
        <div style="
            background: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border: 1px solid #e5e7eb;
            text-align: center;
        ">
            <span style="font-size: 2rem;">🛒</span>
            <div style="font-size: 1.2rem; font-weight: bold;">{len(st.session_state.cart)} article(s)</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### 📍 Navigation")
    if st.button("🏠 Accueil", width='stretch'):
        st.switch_page("app.py")
    if st.button("📂 Catégories", width='stretch'):
        st.switch_page("pages/2_Categories.py")
    if st.button("🛒 Panier", width='stretch'):
        st.switch_page("pages/3_Panier.py")
    if st.button("📦 Commander", width='stretch'):
        st.switch_page("pages/4_Commande.py")
    if st.button("⚙️ Admin", width='stretch'):
        st.switch_page("pages/5_Admin.py")

# Header principal
st.markdown("""
    <div class="main-header">
        <h1>💊 """ + SITE_NAME + """</h1>
        <p>Votre santé, notre priorité ✨</p>
    </div>
""", unsafe_allow_html=True)

# Page d'accueil
st.title("Bienvenue sur notre Parapharmacie")

# Bannières promotionnelles
col1, col2, col3 = st.columns(3)
with col1:
    with st.container(border=True):
        st.image("https://via.placeholder.com/300x150?text=Promo+-20%25")
        st.markdown("**🔥 Promotion -20% sur tout le maquillage**")
with col2:
    with st.container(border=True):
        st.image("https://via.placeholder.com/300x150?text=Livraison+Gratuite")
        st.markdown("**🚚 Livraison gratuite dès 500 MAD**")
with col3:
    with st.container(border=True):
        st.image("https://via.placeholder.com/300x150?text=Nouveautés")
        st.markdown("**✨ Découvrez nos nouveautés**")

st.divider()

# Navigation rapide
st.subheader("📂 Catégories populaires")
cols = st.columns(6)
categories = ["Visage", "Cheveux", "Corps", "Maquillage", "Bébé", "Promotions"]
for i, cat in enumerate(categories):
    with cols[i]:
        if st.button(f"🔹 {cat}", width='stretch'):
            st.session_state.selected_category = cat
            st.switch_page("pages/2_Categories.py")

st.divider()

# Produits en vedette
st.subheader("🔥 Produits en vedette")
from models import get_all_products
from datetime import datetime, timedelta

products = get_all_products()
cols = st.columns(4)
for i, product in enumerate(products[:8]):
    with cols[i % 4]:
        # Déterminer le statut du stock
        stock_status = "stock-low" if product['stock'] < 5 else "stock-ok"
        
        # Vérifier l'expiration
        expiry_class = ""
        if product.get('expiration_date'):
            days_to_expiry = (product['expiration_date'] - datetime.now().date()).days
            if days_to_expiry < 0:
                expiry_class = "expired"
            elif days_to_expiry < 30:
                expiry_class = "warning"
        
        with st.container(border=True):
            st.image(f"https://via.placeholder.com/200x150?text={product['name'][:10]}", width="stretch")
            st.markdown(f"### {product['name']}")
            if product['promo']:
                st.markdown('<span class="promo-badge">PROMO</span>', unsafe_allow_html=True)
            
            # Prix
            st.markdown(f'<span class="price">{product["price"]} MAD</span>', unsafe_allow_html=True)
            
            # Stock
            if product['stock'] < 5:
                st.markdown(f'<span class="stock-low">⚠️ Plus que {product["stock"]} en stock</span>', unsafe_allow_html=True)
            
            # Date d'expiration
            if product.get('expiration_date'):
                days = (product['expiration_date'] - datetime.now().date()).days
                if days < 0:
                    st.markdown(f'<span class="expiry-date warning">❌ Expiré</span>', unsafe_allow_html=True)
                elif days < 30:
                    st.markdown(f'<span class="expiry-date warning">⚠️ Expire dans {days} jours</span>', unsafe_allow_html=True)
            
            if st.button(f"🛒 Ajouter au panier", key=f"home_add_{product['id']}", width='stretch'):
                st.session_state.cart.append(product)
                st.rerun()

# Footer
st.divider()
st.markdown("""
    <div class="footer">
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
        <div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #475569;">
            © 2024 """ + SITE_NAME + """ - Tous droits réservés
        </div>
    </div>
""", unsafe_allow_html=True)