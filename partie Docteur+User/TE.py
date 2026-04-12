# main.py

import streamlit as st
import os
import logging
import warnings
from app_config import Config
from database1 import MySQLDatabase
from auth_manager import AuthManager
from notification_manager import NotificationManager
from pdf_generator import PDFGenerator
from ui_components import *
from florence_ui import show_florence_analysis_interface
from doctor_patients_ui1 import show_doctor_patients_ui
from ia_priorisation import PatientPriorisationIA
import subprocess
import sys
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================= CONFIGURATION INITIALE =================
st.set_page_config(
    page_title="AI Shifa - Maroc",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= INITIALISATION DES DOSSIERS =================
for folder in [Config.UPLOAD_FOLDER, Config.ORDO_FOLDER, Config.REPORTS_FOLDER, "temp", "logs"]:
    os.makedirs(folder, exist_ok=True)

# ================= GESTIONNAIRE DE BASE DE DONNÉES =================
@st.cache_resource
def init_database():
    """Initialise la connexion à la base de données"""
    return MySQLDatabase()

db = init_database()

# ================= INTERFACE PRINCIPALE =================
def main():
    """Fonction principale de l'application"""
    
    # Initialisation des gestionnaires
    if 'auth' not in st.session_state:
        st.session_state.auth = AuthManager(db)
    if 'notif' not in st.session_state:
        st.session_state.notif = NotificationManager(db)
    if 'pdf_gen' not in st.session_state:
        st.session_state.pdf_gen = PDFGenerator()
    if 'ia_priorisation' not in st.session_state:
        st.session_state.ia_priorisation = PatientPriorisationIA(db)
    
    # Gestion de l'authentification
    if 'user' not in st.session_state:
        show_auth_interface(st.session_state.auth)
    else:
        show_main_interface()

def show_main_interface():
    """Interface principale après connexion"""
    
    # CSS GLOBAL POUR AMÉLIORER LE DESIGN - VERSION ROUGE
    st.markdown("""
        <style>
        /* Style général */
        .main-header {
            background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(220, 38, 38, 0.3);
        }
        
        /* Style des boutons de navigation */
        div.row-widget.stRadio > div {
            flex-direction: column;
            gap: 0.5rem;
        }
        
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            background-color: white;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            transition: all 0.3s ease;
            cursor: pointer;
            margin: 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
            background-color: #fef2f2;
            border-color: #dc2626;
            transform: translateX(5px);
            box-shadow: 0 4px 6px rgba(220, 38, 38, 0.2);
        }
        
        /* Bouton sélectionné - ROUGE */
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
            background-color: #dc2626 !important;
            border-color: #dc2626 !important;
            box-shadow: 0 2px 8px rgba(220, 38, 38, 0.4) !important;
        }
        
        /* Texte du bouton sélectionné */
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] p {
            color: #dc2626 !important;
            font-weight: 600 !important;
        }
        
        /* Style des boutons Streamlit */
        .stButton > button {
            background-color: #dc2626 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px rgba(220, 38, 38, 0.3) !important;
            transition: all 0.3s ease !important;
            width: 100%;
        }
        
        .stButton > button:hover {
            background-color: #b91c1c !important;
            box-shadow: 0 6px 10px rgba(220, 38, 38, 0.4) !important;
            transform: translateY(-2px) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0px) !important;
            box-shadow: 0 2px 4px rgba(220, 38, 38, 0.3) !important;
        }
        
        /* Style des boutons secondaires */
        .stButton > button.secondary {
            background-color: white !important;
            color: #dc2626 !important;
            border: 2px solid #dc2626 !important;
            box-shadow: none !important;
        }
        
        .stButton > button.secondary:hover {
            background-color: #fef2f2 !important;
        }
        
        /* Style des expanders */
        .streamlit-expanderHeader {
            background-color: #f8fafc;
            border-radius: 8px;
            border-left: 4px solid #dc2626;
        }
        
        /* Style des cartes */
        div.stMarkdown div:has(> .patient-card) {
            transition: all 0.3s ease;
        }
        
        /* Style des notifications */
        .stAlert {
            border-left: 4px solid #dc2626;
            border-radius: 8px;
        }
        
        /* Style de la sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e5e7eb;
        }
        
        section[data-testid="stSidebar"] .stButton > button {
            background-color: #dc2626 !important;
        }
        
        /* Titre dans la sidebar */
        .sidebar-title {
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(220, 38, 38, 0.3);
        }
        
        /* Style du bouton de déconnexion */
        .logout-button .stButton > button {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
            box-shadow: 0 4px 6px rgba(30, 41, 59, 0.3) !important;
        }
        
        .logout-button .stButton > button:hover {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
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
            color: #dc2626 !important;
            font-size: 2rem !important;
        }
        
        div[data-testid="stMetricLabel"] {
            color: #4b5563 !important;
            font-weight: 500 !important;
        }
        
        /* Style des tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: 1px solid #e5e7eb;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #dc2626 !important;
            color: white !important;
        }
        
        /* Style des formulaires */
        div[data-testid="stForm"] {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Style des champs de saisie */
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #dc2626;
            box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.2);
        }
        
        /* Style des selectbox */
        div[data-testid="stSelectbox"] > div > div {
            border-radius: 8px;
        }
        
        /* Style des graphiques - éléments rouges */
        .js-plotly-plot .plotly .main-svg {
            background: transparent !important;
        }
        
        .js-plotly-plot .plotly .g-xtitle text,
        .js-plotly-plot .plotly .g-ytitle text {
            fill: #dc2626 !important;
        }
        
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar avec infos utilisateur - DESIGN AMÉLIORÉ EN ROUGE
    with st.sidebar:
        # En-tête avec dégradé rouge
        st.markdown("""
            <div class="sidebar-title">
                <span style="font-size: 2rem;">🏥</span><br>
                AI Shifa Pro
            </div>
        """, unsafe_allow_html=True)
        
        # Profil utilisateur avec style
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
                        <div style="font-weight: bold; color: #991b1b;">{st.session_state.user['full_name'] or st.session_state.user['username']}</div>
                        <div style="font-size: 0.8rem; color: #dc2626;">{st.session_state.user['role'].upper()}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Menu de navigation avec icônes
        menu_options = {
            "patient": [
                ("🏠", "Accueil"),
                ("💬", "Chat Santé"),
                ("📊", "Mon Dossier"),
                ("📎", "Analyses"),
                ("📅", "Rendez-vous"),
                ("⚙️", "Paramètres")
            ],
            "medecin": [
                ("🏠", "Dashboard"),
                ("👥", "Patients"),
                ("📊", "Analyses IA"),
                ("📅", "Consultations"),
                ("📈", "Statistiques"),
                ("⚙️", "Paramètres")
            ],
            "admin": [
                ("🏠", "Dashboard"),
                ("👥", "Utilisateurs"),
                ("📊", "Système"),
                ("📈", "Rapports"),
                ("⚙️", "Configuration")
            ]
        }
        
        # Récupérer la sélection précédente
        if 'nav' not in st.session_state:
            st.session_state.nav = menu_options[st.session_state.user['role']][0][1]
        
        # Créer les options avec émoticônes
        options = [f"{icon} {label}" for icon, label in menu_options[st.session_state.user['role']]]
        
        selected = st.radio(
            "Navigation",
            options,
            key="nav_radio",
            index=options.index(st.session_state.nav) if st.session_state.nav in options else 0,
            label_visibility="collapsed"
        )
        
        st.session_state.nav = selected
        
        st.divider()
        
        # Notifications avec meilleur design
        with st.expander("🔔 Notifications", expanded=False):
            notifications = st.session_state.notif.get_user_notifications(
                st.session_state.user['id'], 
                unread_only=True
            )
            if notifications:
                for notif in notifications:
                    st.markdown(f"""
                        <div style="
                            background-color: white;
                            padding: 0.75rem;
                            border-radius: 8px;
                            margin-bottom: 0.5rem;
                            border-left: 3px solid #dc2626;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                        ">
                            <div style="font-weight: bold; color: #dc2626;">{notif['titre']}</div>
                            <div style="font-size: 0.9rem; margin: 0.25rem 0;">{notif['message'][:50]}...</div>
                            <div style="font-size: 0.7rem; color: #666;">{notif['timestamp']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.button("✓ Marquer comme lu", key=f"read_{notif['id']}"):
                        st.session_state.notif.mark_as_read(notif['id'])
                        st.rerun()
            else:
                st.info("📭 Aucune nouvelle notification")
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
        
        # Bouton de déconnexion stylisé
        st.markdown('<div class="logout-button">', unsafe_allow_html=True)
        if st.button("🚪 Déconnexion", width='stretch'):
            del st.session_state.user
            if 'nav' in st.session_state:
                del st.session_state.nav
            if 'selected_patient' in st.session_state:
                del st.session_state.selected_patient
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Version et copyright
        st.markdown("""
            <div style="
                text-align: center;
                font-size: 0.7rem;
                color: #94a3b8;
                margin-top: 2rem;
                padding: 0.5rem;
                border-top: 1px solid #e5e7eb;
            ">
                Version 2.0.0<br>
                © 2024 AI Shifa Pro
            </div>
        """, unsafe_allow_html=True)
    
    # Interface principale selon le rôle
    if st.session_state.user['role'] == 'patient':
        selected_clean = selected.split(' ', 1)[1] if ' ' in selected else selected
        show_patient_interfaces(selected_clean)
    elif st.session_state.user['role'] == 'medecin':
        selected_clean = selected.split(' ', 1)[1] if ' ' in selected else selected
        show_doctor_interfaces(selected_clean)
    else:
        selected_clean = selected.split(' ', 1)[1] if ' ' in selected else selected
        show_admin_interfaces(selected_clean)

def get_patient_folder(user_id):
    """Crée et retourne le dossier spécifique au patient"""
    folder = os.path.join(Config.BASE_STORAGE, str(user_id))
    os.makedirs(folder, exist_ok=True)
    return folder

# ========== INTERFACES PATIENT ==========

def show_patient_interfaces(selected):
    """Interfaces pour les patients"""
    if selected == "Accueil":
        show_patient_dashboard(db, st.session_state.user['id'])
    elif selected == "Chat Santé":
        show_chat_interface(db, st.session_state.notif, st.session_state.pdf_gen, st.session_state.user['id'])
    elif selected == "Mon Dossier":
        show_patient_records(db, st.session_state.user['id'])
    elif selected == "Analyses":
        show_florence_analysis_interface(db, st.session_state.notif)
    elif selected == "Rendez-vous":
        show_appointments_interface(db, st.session_state.notif, st.session_state.user['id'])
    else:
        show_settings_interface(db, st.session_state.auth, st.session_state.user['id'])

def show_patient_dashboard(db, user_id):
    """Dashboard patient"""
    st.markdown("""
        <div class="main-header">
            <h2>🏠 Tableau de bord patient</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"## 👋 Bonjour, {st.session_state.user['full_name']}!")
    
    # Récupérer les statistiques
    try:
        stats = {
            'messages': db.execute_query(
                "SELECT COUNT(*) as count FROM messages WHERE user_id = %s",
                (user_id,), fetch_one=True
            )['count'],
            
            'analyses': db.execute_query(
                "SELECT COUNT(*) as count FROM analyses WHERE user_id = %s",
                (user_id,), fetch_one=True
            )['count'],
            
            'rdv': db.execute_query(
                """SELECT COUNT(*) as count FROM rendez_vous 
                   WHERE patient_id = %s AND statut = 'planifie'""",
                (user_id,), fetch_one=True
            )['count'],
            
            'prochain_rdv': db.execute_query(
                """SELECT date_rdv FROM rendez_vous 
                   WHERE patient_id = %s AND date_rdv >= CURDATE() 
                   ORDER BY date_rdv LIMIT 1""",
                (user_id,), fetch_one=True
            )
        }
    except:
        stats = {'messages': 0, 'analyses': 0, 'rdv': 0, 'prochain_rdv': None}
    
    # Métriques avec style amélioré
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💬 Messages", stats['messages'])
    with col2:
        st.metric("🔬 Analyses", stats['analyses'])
    with col3:
        st.metric("📅 Rendez-vous", stats['rdv'])
    with col4:
        next_date = stats['prochain_rdv']['date_rdv'].strftime('%d/%m/%Y') if stats['prochain_rdv'] else "Aucun"
        st.metric("📌 Prochain RDV", next_date)
    
    st.divider()
    
    # Derniers messages
    st.subheader("💬 Derniers échanges")
    try:
        derniers_messages = db.execute_query(
            """SELECT message, type_message, timestamp, urgent 
               FROM messages 
               WHERE user_id = %s 
               ORDER BY timestamp DESC 
               LIMIT 5""",
            (user_id,), fetch_all=True
        )
        
        if derniers_messages:
            for msg in derniers_messages:
                with st.container():
                    if msg['urgent']:
                        st.error(f"🚨 {msg['message']}")
                    else:
                        st.info(f"💬 {msg['message']}")
                    st.caption(f"📅 {msg['timestamp']}")
        else:
            st.info("Aucun message pour le moment")
    except Exception as e:
        st.error(f"Erreur lors du chargement des messages: {e}")

def show_chat_interface(db, notif_manager, pdf_gen, user_id):
    """Interface de chat"""
    st.markdown("""
        <div class="main-header">
            <h2>💬 Chat Santé</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialisation de l'historique
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Affichage de l'historique
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Zone de saisie
    if prompt := st.chat_input("Posez votre question..."):
        # Message utilisateur
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Sauvegarde
        try:
            db.execute_query(
                """INSERT INTO messages (user_id, message, type_message) 
                   VALUES (%s, %s, 'symptome')""",
                (user_id, prompt)
            )
        except Exception as e:
            logger.error(f"Erreur sauvegarde message: {e}")
        
        # Réponse simulée
        reponse = generate_response(prompt)
        
        # Affichage réponse
        st.session_state.chat_history.append({"role": "assistant", "content": reponse})
        with st.chat_message("assistant"):
            st.markdown(reponse)

def generate_response(prompt):
    """Génère une réponse simulée"""
    return f"Merci pour votre message. Un médecin vous répondra dans les plus brefs délais."

def show_patient_records(db, user_id):
    """Dossier médical du patient"""
    st.markdown("""
        <div class="main-header">
            <h2>📊 Mon Dossier Médical</h2>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Consultations", "🔬 Analyses", "💬 Messages", "📚 Documents"])
    
    with tab1:
        st.subheader("Historique des consultations")
        try:
            consultations = db.execute_query("""
                SELECT r.date_rdv, r.heure_rdv, r.motif, r.statut, u.full_name as medecin
                FROM rendez_vous r
                LEFT JOIN users u ON u.id = r.medecin_id
                WHERE r.patient_id = %s
                ORDER BY r.date_rdv DESC
            """, (user_id,), fetch_all=True)
            
            if consultations:
                for cons in consultations:
                    with st.container():
                        st.write(f"**{cons['date_rdv']}** à {cons['heure_rdv']} - Dr. {cons['medecin']}")
                        st.caption(f"Motif: {cons['motif']} - Statut: {cons['statut']}")
                        st.divider()
            else:
                st.info("Aucune consultation enregistrée")
        except Exception as e:
            st.error(f"Erreur: {e}")
    
    with tab2:
        st.subheader("Analyses médicales")
        try:
            analyses = db.execute_query("""
                SELECT type_analyse, resultat, confiance, timestamp
                FROM analyses
                WHERE user_id = %s
                ORDER BY timestamp DESC
            """, (user_id,), fetch_all=True)
            
            if analyses:
                for ana in analyses:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{ana['type_analyse']}** - {ana['resultat']}")
                        with col2:
                            if ana['confiance']:
                                st.caption(f"Confiance: {float(ana['confiance'])*100:.0f}%")
                        st.caption(f"📅 {ana['timestamp']}")
                        st.divider()
            else:
                st.info("Aucune analyse enregistrée")
        except Exception as e:
            st.error(f"Erreur: {e}")
    
    with tab3:
        st.subheader("Messages échangés")
        try:
            messages = db.execute_query("""
                SELECT message, type_message, timestamp, urgent
                FROM messages
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT 20
            """, (user_id,), fetch_all=True)
            
            if messages:
                for msg in messages:
                    with st.container():
                        if msg['urgent']:
                            st.error(f"🚨 {msg['message'][:100]}...")
                        else:
                            st.write(f"💬 {msg['message'][:100]}...")
                        st.caption(f"📅 {msg['timestamp']} - {msg['type_message']}")
            else:
                st.info("Aucun message")
        except Exception as e:
            st.error(f"Erreur: {e}")
    
    with tab4:
        st.subheader("📁 Mes documents médicaux")

        # Vérifier d'abord si la table existe, sinon créer
        try:
            db.execute_query("SELECT 1 FROM dossiers LIMIT 1")
        except Exception as e:
            if "doesn't exist" in str(e):
                st.warning("⚙️ Initialisation de la base de données...")
                try:
                    db.execute_query("""
                        CREATE TABLE IF NOT EXISTS dossiers (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            user_id INT NOT NULL,
                            titre VARCHAR(255),
                            type_fichier VARCHAR(50),
                            chemin_fichier TEXT,
                            date_upload TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                        )
                    """)
                    st.success("✅ Base de données initialisée!")
                    st.rerun()
                except Exception as create_error:
                    st.error(f"Erreur création table: {create_error}")
            else:
                st.error(f"Erreur base de données: {e}")

        # Définir le dossier de stockage
        BASE_STORAGE = "storage/patients"
        patient_folder = os.path.join(BASE_STORAGE, str(user_id))
        os.makedirs(patient_folder, exist_ok=True)

        # ---------- UPLOAD ----------
        with st.expander("📤 Ajouter un document", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                titre = st.text_input("Titre du document", placeholder="Ex: Ordonnance mars 2024")
                uploaded_file = st.file_uploader(
                    "Choisir un fichier",
                    type=["pdf", "png", "jpg", "jpeg", "docx", "txt"],
                    help="Formats acceptés: PDF, images, DOCX, TXT"
                )
            
            with col2:
                st.write("")  # Espacement
                st.write("")  # Espacement
                if uploaded_file and st.button("📥 Enregistrer", width='stretch', type="primary"):
                    try:
                        # Limite de taille (10MB)
                        MAX_SIZE = 10 * 1024 * 1024
                        if uploaded_file.size > MAX_SIZE:
                            st.error("Fichier trop volumineux. Maximum 10MB")
                        else:
                            # Nettoyer le nom du fichier
                            safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in ' ._-')
                            file_path = os.path.join(patient_folder, safe_filename)
                            
                            # Éviter les doublons
                            base, ext = os.path.splitext(file_path)
                            counter = 1
                            while os.path.exists(file_path):
                                file_path = f"{base}_{counter}{ext}"
                                counter += 1
                            
                            # Sauvegarder le fichier
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Enregistrer dans la base de données
                            db.execute_query("""
                                INSERT INTO dossiers (user_id, titre, type_fichier, chemin_fichier)
                                VALUES (%s, %s, %s, %s)
                            """, (user_id, titre or "Sans titre", uploaded_file.type, file_path))
                            
                            st.success("✅ Document ajouté avec succès!")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Erreur lors de l'upload: {str(e)}")

        st.divider()

        # ---------- LISTE DES FICHIERS ----------
        st.markdown("### 📚 Mes documents")

        try:
            dossiers = db.execute_query("""
                SELECT id, titre, type_fichier, chemin_fichier, date_upload
                FROM dossiers
                WHERE user_id = %s
                ORDER BY date_upload DESC
            """, (user_id,), fetch_all=True)

            if dossiers:
                for doc in dossiers:
                    with st.container():
                        cols = st.columns([4, 1, 1])
                        
                        with cols[0]:
                            # Icône selon le type
                            ext = os.path.splitext(doc['chemin_fichier'])[1].lower()
                            icon = "📄"
                            if ext in ['.jpg', '.jpeg', '.png']:
                                icon = "🖼️"
                            elif ext == '.pdf':
                                icon = "📕"
                            elif ext == '.docx':
                                icon = "📝"
                            
                            st.write(f"{icon} **{doc['titre']}**")
                            st.caption(f"📅 {doc['date_upload']}")
                        
                        # Télécharger
                        with cols[1]:
                            if os.path.exists(doc['chemin_fichier']):
                                with open(doc['chemin_fichier'], "rb") as file:
                                    st.download_button(
                                        "📥 Télécharger",
                                        file,
                                        file_name=os.path.basename(doc['chemin_fichier']),
                                        key=f"dl_{doc['id']}"
                                    )
                            else:
                                st.warning("Fichier manquant")
                        
                        # Supprimer
                        with cols[2]:
                            if st.button("🗑️ Supprimer", key=f"del_{doc['id']}"):
                                try:
                                    if os.path.exists(doc['chemin_fichier']):
                                        os.remove(doc['chemin_fichier'])
                                    db.execute_query("DELETE FROM dossiers WHERE id = %s", (doc['id'],))
                                    st.success("Document supprimé!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Erreur suppression: {e}")
                        
                        st.divider()
            else:
                st.info("📭 Aucun document pour le moment")
                st.caption("Utilisez le bouton 'Ajouter un document' ci-dessus pour commencer")
                
        except Exception as e:
            st.error(f"Erreur lors du chargement des documents: {str(e)}")

def show_appointments_interface(db, notif_manager, user_id):
    """Interface de rendez-vous"""
    st.markdown("""
        <div class="main-header">
            <h2>📅 Gestion des rendez-vous</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Nouveau rendez-vous
    with st.expander("➕ Prendre un rendez-vous", expanded=True):
        with st.form("new_appointment"):
            col1, col2 = st.columns(2)
            
            with col1:
                date_rdv = st.date_input(
                    "Date",
                    min_value=datetime.now().date(),
                    value=datetime.now().date() + timedelta(days=1)
                )
                
                # Récupérer les médecins
                try:
                    medecins = db.execute_query(
                        "SELECT id, full_name FROM users WHERE role = 'medecin' AND actif = TRUE", 
                        fetch_all=True
                    )
                    medecin_options = {f"Dr. {m['full_name']}": m['id'] for m in medecins}
                    medecin_choice = st.selectbox("Médecin", list(medecin_options.keys()))
                except:
                    st.warning("Aucun médecin disponible")
                    medecin_choice = None
            
            with col2:
                heure_rdv = st.time_input("Heure", value=datetime.now().time().replace(hour=9, minute=0))
                motif = st.text_area("Motif de la consultation")
            
            submit = st.form_submit_button("Confirmer", width='stretch')
            
            if submit and medecin_choice:
                try:
                    db.execute_query("""
                        INSERT INTO rendez_vous (patient_id, medecin_id, date_rdv, heure_rdv, motif)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        user_id,
                        medecin_options[medecin_choice],
                        date_rdv,
                        heure_rdv.strftime('%H:%M:%S'),
                        motif
                    ))
                    st.success(f"✅ Rendez-vous confirmé")
                    
                    # Notification
                    notif_manager.create_notification(
                        user_id,
                        "Rendez-vous confirmé",
                        f"Votre rendez-vous avec {medecin_choice} est confirmé",
                        "info"
                    )
                except Exception as e:
                    st.error(f"Erreur: {e}")
    
    # Rendez-vous à venir
    st.subheader("📌 Rendez-vous à venir")
    try:
        prochains_rdv = db.execute_query("""
            SELECT r.date_rdv, r.heure_rdv, r.motif, r.statut, u.full_name as medecin
            FROM rendez_vous r
            JOIN users u ON u.id = r.medecin_id
            WHERE r.patient_id = %s AND r.date_rdv >= CURDATE()
            ORDER BY r.date_rdv, r.heure_rdv
        """, (user_id,), fetch_all=True)
        
        if prochains_rdv:
            for rdv in prochains_rdv:
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"📅 {rdv['date_rdv']} à {rdv['heure_rdv']}")
                    with col2:
                        st.write(f"👨‍⚕️ Dr. {rdv['medecin']}")
                    with col3:
                        status = rdv['statut']
                        if status == 'planifie':
                            st.info("⏳ En attente")
                        elif status == 'confirme':
                            st.success("✅ Confirmé")
                        else:
                            st.write(status)
                    st.divider()
        else:
            st.info("Aucun rendez-vous planifié")
    except Exception as e:
        st.error(f"Erreur: {e}")

# ========== INTERFACES MÉDECIN ==========

def show_doctor_interfaces(selected):
    """Interfaces pour les médecins"""
    if selected == "Dashboard":
        show_doctor_dashboard_enhanced(db, st.session_state.ia_priorisation, st.session_state.user['id'])
    elif selected == "Patients":
        show_doctor_patients_ui(db, st.session_state.user['id'])
    elif selected == "Analyses IA":
        show_florence_analysis_interface(db, st.session_state.notif)
    elif selected == "Consultations":
        show_doctor_appointments_enhanced(db, st.session_state.user['id'])
    elif selected == "Statistiques":
        show_doctor_statistics_enhanced(db, st.session_state.user['id'])
    else:
        show_settings_interface(db, st.session_state.auth, st.session_state.user['id'])

def show_doctor_dashboard_enhanced(db, ia_priorisation, doctor_id):
    """Dashboard médecin amélioré avec IA"""
    st.markdown("""
        <div class="main-header">
            <h2>🏠 Dashboard Médecin</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Statistiques
    try:
        stats = db.get_dashboard_stats()
        doctor_stats = db.get_doctor_statistics(doctor_id)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Patients suivis", doctor_stats.get('mes_patients', 0))
        with col2:
            st.metric("📅 RDV aujourd'hui", stats['consultations_aujourdhui'])
        with col3:
            st.metric("🚨 Analyses urgentes", stats['analyses_urgentes'])
        with col4:
            st.metric("⚠️ Alertes critiques", stats['alertes_aujourdhui'])
    except:
        st.warning("Impossible de charger les statistiques")
    
    st.divider()
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Courbe activité consultations
        try:
            consultations_data = db.execute_query("""
                SELECT DATE(date_rdv) as jour, COUNT(*) as nb
                FROM rendez_vous
                WHERE medecin_id = %s AND date_rdv >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
                GROUP BY jour
                ORDER BY jour
            """, (doctor_id,), fetch_all=True)
            
            if consultations_data:
                df = pd.DataFrame(consultations_data)
                fig = px.line(df, x='jour', y='nb', title="📊 Consultations - 7 derniers jours")
                fig.update_traces(line_color='#dc2626')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#991b1b')
                )
                st.plotly_chart(fig, width='stretch')
        except:
            st.info("Données de consultation non disponibles")
    
    with col2:
        # Graph analyses par type
        try:
            analyses_data = db.execute_query("""
                SELECT type_analyse, COUNT(*) as nb
                FROM analyses
                WHERE user_id IN (SELECT patient_id FROM rendez_vous WHERE medecin_id = %s)
                GROUP BY type_analyse
                LIMIT 5
            """, (doctor_id,), fetch_all=True)
            
            if analyses_data:
                df = pd.DataFrame(analyses_data)
                fig = px.pie(df, values='nb', names='type_analyse', title="🔬 Types d'analyses")
                fig.update_traces(marker=dict(colors=['#dc2626', '#ef4444', '#f87171', '#fca5a5', '#fecaca']))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#991b1b')
                )
                st.plotly_chart(fig, width='stretch')
        except:
            st.info("Données d'analyses non disponibles")
    
    st.divider()
    
    # Patients priorisés par IA
    st.subheader("🤖 Patients priorisés par IA")
    
    try:
        prioritized_patients = ia_priorisation.get_prioritized_patients_for_doctor(doctor_id, limit=10)
        
        if prioritized_patients:
            for p in prioritized_patients:
                color = "🔴" if "Urgent" in p['priorite'] else "🟠" if "À voir" in p['priorite'] else "🟢"
                
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"{color} **{p['full_name']}**")
                    with col2:
                        st.write(f"Score: {p['score_ia']} - {p['priorite']}")
                    with col3:
                        if st.button("Voir", key=f"view_{p['id']}"):
                            st.session_state.selected_patient = p['id']
                            st.session_state.nav = "Patients"
                            st.rerun()
                    st.divider()
        else:
            st.info("Aucun patient priorisé")
    except Exception as e:
        st.error(f"Erreur chargement patients priorisés: {e}")

def show_doctor_appointments_enhanced(db, doctor_id):
    """Consultations pour médecin"""
    st.markdown("""
        <div class="main-header">
            <h2>📅 Consultations</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("📋 RDV du jour")
        try:
            rdvs = db.execute_query("""
                SELECT u.full_name as patient, r.date_rdv, r.heure_rdv, r.motif, r.statut
                FROM rendez_vous r
                JOIN users u ON u.id = r.patient_id
                WHERE r.medecin_id = %s AND r.date_rdv = CURDATE()
                ORDER BY r.heure_rdv
            """, (doctor_id,), fetch_all=True)
            
            if rdvs:
                for rdv in rdvs:
                    with st.container():
                        st.markdown(f"""
                            <div style="
                                background-color: white;
                                padding: 1rem;
                                border-radius: 8px;
                                margin-bottom: 0.5rem;
                                border-left: 3px solid #dc2626;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                            ">
                                <div style="font-weight: bold;">{rdv['heure_rdv']} - {rdv['patient']}</div>
                                <div style="font-size: 0.9rem; color: #666;">{rdv['motif']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        if rdv['statut'] == 'planifie':
                            if st.button(f"Démarrer", key=f"start_{rdv['patient']}"):
                                st.session_state.current_consultation = rdv
                        st.divider()
            else:
                st.info("Aucun RDV aujourd'hui")
        except Exception as e:
            st.error(f"Erreur: {e}")
    
    with col_right:
        st.subheader("🩺 Consultation active")
        
        if 'current_consultation' in st.session_state:
            consultation = st.session_state.current_consultation
            
            with st.form("consultation_form"):
                st.markdown(f"### Patient: {consultation['patient']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Âge", "45 ans")
                with col2:
                    st.metric("Tension", "120/80")
                with col3:
                    st.metric("Pouls", "75 bpm")
                
                st.markdown("#### Symptômes")
                symptomes = st.text_area("", value=consultation.get('motif', ''))
                
                st.markdown("#### Diagnostic")
                diagnostic = st.text_input("")
                
                st.markdown("#### Note médicale")
                note = st.text_area("", height=100)
                
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    if st.form_submit_button("💾 Enregistrer"):
                        st.success("Consultation enregistrée")
                with col_b2:
                    if st.form_submit_button("📄 Ordonnance"):
                        st.info("Génération ordonnance")
                with col_b3:
                    if st.form_submit_button("❌ Fermer"):
                        del st.session_state.current_consultation
                        st.rerun()
        else:
            st.info("Sélectionnez une consultation à gauche")

def show_doctor_statistics_enhanced(db, doctor_id):
    """Statistiques médecin"""
    st.markdown("""
        <div class="main-header">
            <h2>📈 Statistiques</h2>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        # Statistiques générales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_patients = db.execute_query("""
                SELECT COUNT(DISTINCT patient_id) as count 
                FROM rendez_vous WHERE medecin_id = %s
            """, (doctor_id,), fetch_one=True)['count']
            st.metric("👥 Patients uniques", total_patients)
        
        with col2:
            total_consultations = db.execute_query("""
                SELECT COUNT(*) as count 
                FROM rendez_vous WHERE medecin_id = %s
            """, (doctor_id,), fetch_one=True)['count']
            st.metric("📅 Total consultations", total_consultations)
        
        with col3:
            taux_urgence = db.execute_query("""
                SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM rendez_vous WHERE medecin_id = %s) as taux
                FROM rendez_vous 
                WHERE medecin_id = %s AND motif LIKE '%urgence%'
            """, (doctor_id, doctor_id), fetch_one=True)
            st.metric("🚨 Taux d'urgence", f"{taux_urgence.get('taux', 0):.1f}%" if taux_urgence else "0%")
        
        st.divider()
        
        # Graphique évolution
        st.subheader("📊 Évolution des consultations")
        
        evolution = db.execute_query("""
            SELECT DATE_FORMAT(date_rdv, '%Y-%m') as mois, COUNT(*) as nb
            FROM rendez_vous
            WHERE medecin_id = %s AND date_rdv >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
            GROUP BY mois
            ORDER BY mois
        """, (doctor_id,), fetch_all=True)
        
        if evolution:
            df = pd.DataFrame(evolution)
            fig = px.bar(df, x='mois', y='nb', title="Consultations par mois")
            fig.update_traces(marker_color='#dc2626')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#991b1b')
            )
            st.plotly_chart(fig, width='stretch')
        
        # Répartition par motif
        st.subheader("📋 Motifs de consultation")
        
        motifs = db.execute_query("""
            SELECT motif, COUNT(*) as nb
            FROM rendez_vous
            WHERE medecin_id = %s AND motif IS NOT NULL
            GROUP BY motif
            ORDER BY nb DESC
            LIMIT 10
        """, (doctor_id,), fetch_all=True)
        
        if motifs:
            df_motifs = pd.DataFrame(motifs)
            fig_motifs = px.pie(df_motifs, values='nb', names='motif', title="Motifs de consultation")
            fig_motifs.update_traces(marker=dict(colors=['#dc2626', '#ef4444', '#f87171', '#fca5a5', '#fecaca']))
            fig_motifs.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#991b1b')
            )
            st.plotly_chart(fig_motifs, width='stretch')
            
    except Exception as e:
        st.error(f"Erreur chargement statistiques: {e}")

# ========== INTERFACES ADMIN ==========

def show_admin_interfaces(selected):
    """Interfaces pour les administrateurs"""
    if selected == "Dashboard":
        show_admin_dashboard(db)
    elif selected == "Utilisateurs":
        show_users_management(db)
    elif selected == "Système":
        show_system_status(db)
    elif selected == "Rapports":
        show_system_reports(db)
    else:
        show_system_config(db)

def show_admin_dashboard(db):
    """Dashboard admin"""
    st.markdown("""
        <div class="main-header">
            <h2>🏠 Dashboard Administrateur</h2>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        stats = {
            'total_users': db.execute_query("SELECT COUNT(*) as count FROM users", fetch_one=True)['count'],
            'total_patients': db.execute_query("SELECT COUNT(*) as count FROM users WHERE role = 'patient'", fetch_one=True)['count'],
            'total_medecins': db.execute_query("SELECT COUNT(*) as count FROM users WHERE role = 'medecin'", fetch_one=True)['count'],
            'total_analyses': db.execute_query("SELECT COUNT(*) as count FROM analyses", fetch_one=True)['count']
        }
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Utilisateurs total", stats['total_users'])
        with col2:
            st.metric("👤 Patients", stats['total_patients'])
        with col3:
            st.metric("👨‍⚕️ Médecins", stats['total_medecins'])
        with col4:
            st.metric("🔬 Analyses", stats['total_analyses'])
            
    except Exception as e:
        st.error(f"Erreur: {e}")

def show_users_management(db):
    """Gestion des utilisateurs"""
    st.markdown("""
        <div class="main-header">
            <h2>👥 Gestion des utilisateurs</h2>
        </div>
    """, unsafe_allow_html=True)
    st.info("Interface en cours de développement")

def show_system_status(db):
    """Statut système"""
    st.markdown("""
        <div class="main-header">
            <h2>📊 Statut système</h2>
        </div>
    """, unsafe_allow_html=True)
    st.info("Interface en cours de développement")

def show_system_reports(db):
    """Rapports système"""
    st.markdown("""
        <div class="main-header">
            <h2>📈 Rapports système</h2>
        </div>
    """, unsafe_allow_html=True)
    st.info("Interface en cours de développement")

def show_system_config(db):
    """Configuration système"""
    st.markdown("""
        <div class="main-header">
            <h2>⚙️ Configuration</h2>
        </div>
    """, unsafe_allow_html=True)
    st.info("Interface en cours de développement")

# ========== INTERFACE COMMUNE ==========

def show_settings_interface(db, auth_manager, user_id):
    """Paramètres utilisateur"""
    st.markdown("""
        <div class="main-header">
            <h2>⚙️ Paramètres</h2>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Profil", "Notifications", "Sécurité"])
    
    with tab1:
        st.subheader("Informations personnelles")
        
        user = db.get_user_by_id(user_id)
        
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                full_name = st.text_input("Nom complet", value=user.get('full_name', ''))
                email = st.text_input("Email", value=user.get('email', ''))
                phone = st.text_input("Téléphone", value=user.get('phone', ''))
            
            with col2:
                ville = st.text_input("Ville", value=user.get('ville', ''))
            
            submit = st.form_submit_button("Mettre à jour")
            if submit:
                st.success("Profil mis à jour (simulation)")
    
    with tab2:
        st.subheader("Notifications")
        
        notif_email = st.toggle("Notifications email", value=True)
        notif_sms = st.toggle("Notifications SMS", value=False)
        rappel_rdv = st.toggle("Rappels rendez-vous", value=True)
        
        if st.button("Sauvegarder"):
            st.success("Préférences sauvegardées")
    
    with tab3:
        st.subheader("Sécurité")
        
        with st.form("password_form"):
            current_pw = st.text_input("Mot de passe actuel", type="password")
            new_pw = st.text_input("Nouveau mot de passe", type="password")
            confirm_pw = st.text_input("Confirmer", type="password")
            
            submit_pw = st.form_submit_button("Changer")
            if submit_pw:
                if new_pw == confirm_pw and len(new_pw) >= 6:
                    st.success("Mot de passe modifié")
                else:
                    st.error("Vérifiez les mots de passe")

# ========== FONCTIONS D'AUTH ==========

def show_auth_interface(auth_manager):
    """Interface d'authentification"""
    
    # Style CSS pour les boutons rouges avec ombre
    st.markdown("""
        <style>
            /* Style des entêtes */
            .auth-header {
                background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 10px 25px rgba(220, 38, 38, 0.3);
            }
            
            /* Bouton principal (form_submit_button) */
            div.stFormSubmitButton > button {
                background-color: #dc2626 !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 0.75rem !important;
                box-shadow: 0 4px 14px rgba(220, 38, 38, 0.45) !important;
                font-weight: 600 !important;
                transition: all 0.2s ease !important;
                width: 100%;
            }
            
            div.stFormSubmitButton > button:hover {
                background-color: #b91c1c !important;
                box-shadow: 0 6px 20px rgba(220, 38, 38, 0.6) !important;
                transform: translateY(-2px) !important;
            }
            
            div.stFormSubmitButton > button:active {
                transform: translateY(0px) !important;
                box-shadow: 0 2px 8px rgba(220, 38, 38, 0.4) !important;
            }
            
            /* Style des tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background-color: white;
                padding: 0.5rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            .stTabs [data-baseweb="tab"] {
                border-radius: 8px;
                padding: 0.5rem 1rem;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: #dc2626 !important;
                color: white !important;
            }
            
            /* Style des champs de saisie */
            .stTextInput > div > div > input {
                border-radius: 8px;
                border: 1px solid #e5e7eb;
                padding: 0.75rem;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #dc2626;
                box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1);
            }
            
            /* Style des selectbox */
            div[data-testid="stSelectbox"] > div > div {
                border-radius: 8px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="auth-header">
            <h1>🏥 AI Shifa Pro</h1>
            <p>Plateforme médicale intelligente</p>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Connexion", "📝 Inscription"])
    
    with tab1:
        with st.form("login_form"):
            st.markdown("### Connectez-vous")
            username = st.text_input("Nom d'utilisateur", placeholder="Entrez votre nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password", placeholder="Entrez votre mot de passe")
            submit = st.form_submit_button("Se connecter", width='stretch')
            
            if submit:
                if username and password:
                    result = auth_manager.login_user(username, password)
                    if result['success']:
                        st.session_state.user = result['user']
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.warning("Veuillez remplir tous les champs")
    
    with tab2:
        with st.form("register_form"):
            st.markdown("### Créer un compte")
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Nom d'utilisateur *", placeholder="Choisissez un nom d'utilisateur")
                new_email = st.text_input("Email *", placeholder="votre@email.com")
                new_password = st.text_input("Mot de passe *", type="password", placeholder="Minimum 6 caractères")
            
            with col2:
                confirm_password = st.text_input("Confirmer mot de passe *", type="password", placeholder="Confirmez votre mot de passe")
                full_name = st.text_input("Nom complet", placeholder="Votre nom complet")
                phone = st.text_input("Téléphone", placeholder="06 XX XX XX XX")
            
            ville = st.selectbox("Ville", ["Casablanca", "Rabat", "Marrakech", "Fès", "Tanger", "Agadir", "Autre"])
            role = st.selectbox("Je suis", ["patient", "medecin"])
            
            submit = st.form_submit_button("S'inscrire", width='stretch')
            
            if submit:
                if new_password != confirm_password:
                    st.error("Les mots de passe ne correspondent pas")
                elif len(new_password) < 6:
                    st.error("Le mot de passe doit contenir au moins 6 caractères")
                elif not new_username or not new_email:
                    st.error("Veuillez remplir tous les champs obligatoires")
                else:
                    result = auth_manager.register_user(
                        username=new_username,
                        email=new_email,
                        password=new_password,
                        role=role,
                        full_name=full_name,
                        phone=phone,
                        ville=ville
                    )
                    if result['success']:
                        st.success("Inscription réussie ! Vous pouvez maintenant vous connecter")
                    else:
                        st.error(result['message'])

# ========== IMPORT DES MODULES NÉCESSAIRES ==========
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px

# ================= POINT D'ENTRÉE =================
if __name__ == "__main__":
    main()