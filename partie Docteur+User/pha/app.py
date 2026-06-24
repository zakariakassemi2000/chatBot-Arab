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
# ====================================st.markdown("""
    <style>
    /* === FOND GLOBAL === */
    .stApp, .main, section.main {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    div.block-container {
        background-color: transparent !important;
    }

    /* === CONTAINERS TRANSPARENTS === */
    section[data-testid="stForm"],
    div[data-testid="stVerticalBlock"] > div,
    .element-container, .stForm {
        background-color: transparent !important;
    }

    /* Style général - Thème Bleu Médical */
    .main-header {
        background: linear-gradient(90deg, #2563eb 0%, #0ea5e9 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }
    
    /* Style des boutons de navigation */
    div.row-widget.stRadio > div {
        flex-direction: column;
        gap: 0.5rem;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        background-color: #ffffff !important;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        color: #0f172a !important;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
        background-color: #f1f5f9 !important;
        border-color: #2563eb !important;
        transform: translateX(5px);
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.1);
    }
    
    /* Bouton sélectionné - BLEU */
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Texte du bouton sélectionné */
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] p {
        color: #2563eb !important;
        font-weight: 600 !important;
    }
    
    /* === BOUTONS === */
    .stButton > button {
        background-color: #2563eb !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 6px 10px rgba(37, 99, 235, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
    }
    
    /* Style des boutons secondaires */
    .stButton > button.secondary {
        background-color: #ffffff !important;
        color: #2563eb !important;
        border: 2px solid #2563eb !important;
        box-shadow: none !important;
    }
    
    .stButton > button.secondary:hover {
        background-color: #f1f5f9 !important;
    }
    
    /* Style des expanders */
    .streamlit-expanderHeader {
        background-color: #ffffff !important;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
        color: #0f172a !important;
    }
    
    /* Style des cartes produits */
    .product-card {
        background: #ffffff !important;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.15);
        border-color: #2563eb;
    }
    
    .product-card.promo::before {
        content: "PROMO";
        position: absolute;
        top: 10px;
        right: -30px;
        background: #2563eb;
        color: white;
        padding: 5px 30px;
        transform: rotate(45deg);
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Style des alertes expiration */
    .expired-alert {
        background: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .warning-alert {
        background: rgba(245, 158, 11, 0.1) !important;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Style des notifications */
    .stAlert {
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* === SIDEBAR CLAIR === */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #2563eb !important;
    }
    
    /* Titre dans la sidebar */
    .sidebar-title {
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }
    
    /* Style des métriques */
    div[data-testid="stMetricValue"] {
        color: #2563eb !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #0f172a !important;
        font-weight: 500 !important;
    }
    
    /* Style des métriques avec fond clair */
    .metric-card {
        background: #ffffff !important;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0 !important;
        border-bottom: 3px solid #2563eb;
        text-align: center;
    }
    
    /* === TABS CLAIRS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none !important;
        color: #475569 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    
    /* === FORMULAIRES CLAIRS === */
    div[data-testid="stForm"] {
        background-color: #ffffff !important;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* === INPUTS === */
    input, textarea, select,
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox > div > div > select,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea,
    .stTextInput input, .stTextArea textarea,
    .stSelectbox select {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        caret-color: #2563eb !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }
    
    /* === PLACEHOLDERS === */
    ::placeholder { color: #64748b !important; }
    input::placeholder, textarea::placeholder {
        color: #64748b !important;
    }
    
    /* === LABELS === */
    label, .stTextInput label, .stSelectbox label,
    .stTextArea label, .stRadio label, .stCheckbox label {
        color: #0f172a !important;
        font-weight: 500 !important;
    }

    /* === TEXTE GLOBAL === */
    label, p, span, div, h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }
    
    /* Style des badges promo */
    .promo-badge {
        background: #2563eb;
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
        color: #2563eb !important;
    }
    
    /* Style stock indicator */
    .stock-low {
        color: #ef4444 !important;
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
        background: #f1f5f9 !important;
    }
    
    .expiry-date.warning {
        background: rgba(245, 158, 11, 0.1) !important;
        color: #f59e0b !important;
    }
    
    /* Style du footer */
    .footer {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #0f172a;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        border: 1px solid #e2e8f0;
    }

    /* === DIVIDERS === */
    hr {
        border-color: #e2e8f0 !important;
    }

    /* === SCROLLBAR CLAIR === */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f8fafc;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #2563eb;
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