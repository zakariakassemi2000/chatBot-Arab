import streamlit as st
from models import get_all_products

st.set_page_config(page_title="Accueil", page_icon="🏠", layout="wide")

st.title("🏠 Accueil")

# Barre de navigation
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("🏠 Accueil", use_container_width=True):
        st.switch_page("app.py")
with col2:
    if st.button("📂 Catégories", use_container_width=True):
        st.switch_page("pages/2_Categories.py")
with col3:
    if st.button("🛒 Panier", use_container_width=True):
        st.switch_page("pages/3_Panier.py")
with col4:
    if st.button("📦 Commander", use_container_width=True):
        st.switch_page("pages/4_Commande.py")
with col5:
    if st.button("⚙️ Admin", use_container_width=True):
        st.switch_page("pages/5_Admin.py")

st.divider()

# Bannière principale
with st.container(border=True):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("## 🌟 Bienvenue dans votre parapharmacie en ligne")
        st.markdown("Découvrez notre sélection de produits de qualité pour prendre soin de vous et votre famille")
    with col2:
        st.markdown("### 🔥 -20% sur votre première commande")
        if st.button("Je découvre", type="primary"):
            st.switch_page("pages/2_Categories.py")

st.divider()

# Catégories rapides
st.subheader("📂 Nos catégories")
cols = st.columns(4)
categories = ["Visage", "Cheveux", "Corps", "Bébé", "Solaire", "Bio", "Promotions", "Nouveautés"]
for i, cat in enumerate(categories):
    with cols[i % 4]:
        with st.container(border=True):
            st.markdown(f"### {cat}")
            st.image(f"https://via.placeholder.com/150x100?text={cat}")
            if st.button(f"Voir {cat}", key=f"cat_{i}"):
                st.session_state.selected_category = cat
                st.switch_page("pages/2_Categories.py")

st.divider()

# Nouveautés
st.subheader("✨ Nouveautés")
products = get_all_products()
cols = st.columns(4)
for i, product in enumerate(products[:4]):
    with cols[i]:
        with st.container(border=True):
            st.image(f"https://via.placeholder.com/150x150?text={product['name'][:10]}")
            st.markdown(f"**{product['name']}**")
            st.markdown(f"💰 {product['price']} MAD")
            if st.button("➕ Ajouter", key=f"new_{product['id']}"):
                st.session_state.cart.append(product)
                st.success("✅ Ajouté au panier")
                st.rerun()