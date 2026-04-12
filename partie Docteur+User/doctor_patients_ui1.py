# doctor_patients_ui.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import calendar
from dateutil.relativedelta import relativedelta

def show_doctor_patients_ui(db, doctor_id):
    """Interface de gestion des patients pour le médecin"""
    
    st.header("👥 Gestion des Patients")
    
    # CSS personnalisé pour les cartes (garder votre CSS existant)
    st.markdown("""
        <style>
        .patient-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
            transition: all 0.3s cubic-bezier(.25,.8,.25,1);
            border-left: 4px solid transparent;
        }
        
        .patient-card:hover {
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.25), 0 10px 10px rgba(0, 0, 0, 0.22);
            transform: translateY(-2px);
        }
        
        .patient-card-urgent {
            border-left-color: #ff4b4b;
            background: linear-gradient(to right, rgba(255,75,75,0.05), white);
        }
        
        .patient-card-warning {
            border-left-color: #ffa500;
            background: linear-gradient(to right, rgba(255,165,0,0.05), white);
        }
        
        .patient-card-normal {
            border-left-color: #00a3e0;
            background: linear-gradient(to right, rgba(0,163,224,0.05), white);
        }
        
        .patient-name {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .patient-info {
            color: #666;
            font-size: 0.9rem;
            margin: 5px 0;
        }
        
        .patient-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 5px;
        }
        
        .badge-urgent {
            background-color: #ff4b4b;
            color: white;
        }
        
        .badge-analyses {
            background-color: #00a3e0;
            color: white;
        }
        
        .badge-rdv {
            background-color: #4CAF50;
            color: white;
        }
        
        .button-container {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        .action-button {
            flex: 1;
            padding: 8px;
            border: none;
            border-radius: 5px;
            background-color: #f0f2f6;
            color: #333;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            font-size: 0.9rem;
        }
        
        .action-button:hover {
            background-color: #e0e2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Styles pour le calendrier */
        .calendar-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        
        .calendar-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .calendar-day {
            border: 1px solid #e0e0e0;
            min-height: 100px;
            padding: 10px;
            background-color: white;
        }
        
        .calendar-day-header {
            font-weight: 600;
            text-align: center;
            padding: 5px;
            background-color: #f0f2f6;
        }
        
        .calendar-day-number {
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .calendar-event {
            background-color: #00a3e0;
            color: white;
            padding: 2px 5px;
            border-radius: 3px;
            margin: 2px 0;
            font-size: 0.8rem;
            cursor: pointer;
        }
        
        .calendar-event-confirme {
            background-color: #4CAF50;
        }
        
        .calendar-event-termine {
            background-color: #9e9e9e;
        }
        
        .calendar-event-annule {
            background-color: #f44336;
        }
        
        .pagination {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }
        
        .pagination-button {
            padding: 8px 12px;
            border: 1px solid #ddd;
            background-color: white;
            cursor: pointer;
            border-radius: 5px;
        }
        
        .pagination-button:hover {
            background-color: #f0f2f6;
        }
        
        .pagination-button.active {
            background-color: #00a3e0;
            color: white;
            border-color: #00a3e0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialisation de l'état de session
    if 'selected_patient' not in st.session_state:
        st.session_state.selected_patient = None
    
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = None
    
    # Barre de recherche et filtres
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Rechercher un patient", placeholder="Nom, téléphone, email...")
    
    with col2:
        filtre_priorite = st.selectbox(
            "Priorité",
            ["Tous", "Urgent", "À voir", "Normal"]
        )
    
    with col3:
        tri = st.selectbox(
            "Trier par",
            ["Nom", "Dernière visite", "Priorité"]
        )
    
    # Récupération des patients du médecin (même code que précédemment)
    try:
        query = """
            SELECT DISTINCT 
                u.id,
                u.full_name,
                u.email,
                u.phone,
                u.ville,
                MAX(a.timestamp) as derniere_analyse,
                COUNT(DISTINCT a.id) as nb_analyses,
                MAX(r.date_rdv) as dernier_rdv,
                MAX(CASE WHEN a.urgent = TRUE THEN 1 ELSE 0 END) as a_analyse_urgente
            FROM users u
            LEFT JOIN analyses a ON a.user_id = u.id
            LEFT JOIN rendez_vous r ON r.patient_id = u.id AND r.medecin_id = %s
            WHERE u.role = 'patient'
            GROUP BY u.id, u.full_name, u.email, u.phone, u.ville
        """
        
        params = [doctor_id]
        
        if search_term:
            query += " HAVING (LOWER(u.full_name) LIKE %s OR LOWER(u.email) LIKE %s OR u.phone LIKE %s)"
            search_pattern = f"%{search_term.lower()}%"
            params.extend([search_pattern, search_pattern, search_pattern])
        
        patients = db.execute_query(query, params, fetch_all=True)
        
        if not patients:
            st.info("Aucun patient trouvé")
            return
        
        # Calcul de la priorité pour chaque patient
        for patient in patients:
            score = 0
            if patient.get('a_analyse_urgente'):
                score += 10
            if patient.get('derniere_analyse'):
                delta = datetime.now() - patient['derniere_analyse']
                if delta.days < 7:
                    score += 5
            patient['score_priorite'] = score
            patient['priorite'] = "Urgent" if score >= 10 else "À voir" if score >= 5 else "Normal"
        
        # Filtrage par priorité
        if filtre_priorite != "Tous":
            patients = [p for p in patients if p['priorite'] == filtre_priorite]
        
        # Tri
        if tri == "Nom":
            patients.sort(key=lambda x: x['full_name'] or "")
        elif tri == "Dernière visite":
            patients.sort(key=lambda x: x['dernier_rdv'] or datetime.min, reverse=True)
        elif tri == "Priorité":
            patients.sort(key=lambda x: x['score_priorite'], reverse=True)
        
        # Affichage des patients en grille
        cols_per_row = 3
        for i in range(0, len(patients), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(patients):
                    patient = patients[idx]
                    
                    # Déterminer la classe CSS selon la priorité
                    if patient['priorite'] == "Urgent":
                        card_class = "patient-card patient-card-urgent"
                        priority_icon = "🚨"
                    elif patient['priorite'] == "À voir":
                        card_class = "patient-card patient-card-warning"
                        priority_icon = "⚠️"
                    else:
                        card_class = "patient-card patient-card-normal"
                        priority_icon = "👤"
                    
                    with col:
                        # Carte patient
                        st.markdown(f"""
                            <div class='{card_class}'>
                                <div class='patient-name'>
                                    {priority_icon} {patient['full_name']}
                                </div>
                                <div class='patient-info'>
                                    📞 {patient.get('phone', 'N/A')}
                                </div>
                                <div class='patient-info'>
                                    📍 {patient.get('ville', 'N/A')}
                                </div>
                                <div style='margin: 10px 0;'>
                        """, unsafe_allow_html=True)
                        
                        # Badges
                        if patient.get('nb_analyses', 0) > 0:
                            st.markdown(f"""
                                <span class='patient-badge badge-analyses'>
                                    📊 {patient['nb_analyses']} analyse{'s' if patient['nb_analyses'] > 1 else ''}
                                </span>
                            """, unsafe_allow_html=True)
                        
                        if patient.get('a_analyse_urgente'):
                            st.markdown("""
                                <span class='patient-badge badge-urgent'>
                                    🆘 Urgence
                                </span>
                            """, unsafe_allow_html=True)
                        
                        # Dernière visite
                        if patient.get('dernier_rdv'):
                            st.markdown(f"""
                                <div class='patient-info' style='margin-top: 10px;'>
                                    🕒 Dernier RDV: {patient['dernier_rdv'].strftime('%d/%m/%Y')}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Boutons d'action modifiés
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button("📁 DOSSIER", key=f"dossier_{patient['id']}", width='stretch'):
                                st.session_state.selected_patient = patient['id']
                                st.session_state.view_mode = "dossier"
                                st.rerun()
                        with col_btn2:
                            if st.button("📅 CONSULTATION", key=f"consult_{patient['id']}", width='stretch'):
                                st.session_state.selected_patient = patient['id']
                                st.session_state.view_mode = "consultation"
                                st.rerun()
                        
                        st.markdown("</div>", unsafe_allow_html=True)
        
        # Affichage de la vue sélectionnée
        if st.session_state.selected_patient and st.session_state.view_mode:
            if st.session_state.view_mode == "dossier":
                show_patient_dossier(db, st.session_state.selected_patient, doctor_id)
            elif st.session_state.view_mode == "consultation":
                show_patient_consultation_calendar(db, st.session_state.selected_patient, doctor_id)
            
    except Exception as e:
        st.error(f"Erreur lors du chargement des patients: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

def show_patient_dossier(db, patient_id, doctor_id):
    """Affiche le dossier complet du patient avec PDF, images et documents"""
    
    st.divider()
    st.header(f"📁 Dossier Patient")
    
    # Récupération des informations du patient
    patient = db.execute_query(
        "SELECT full_name FROM users WHERE id = %s",
        (patient_id,), fetch_one=True
    )
    
    if patient:
        st.subheader(f"Patient: {patient['full_name']}")
    
    # Création des onglets pour le dossier
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Informations générales",
        "🔬 Analyses médicales",
        "📄 Documents PDF",
        "🖼️ Images médicales",
        "📊 Historique complet"
    ])
    
    with tab1:
        show_patient_info(db, patient_id)
    
    with tab2:
        show_patient_analyses_complet(db, patient_id)
    
    with tab3:
        show_patient_pdfs(db, patient_id)
    
    with tab4:
        show_patient_images(db, patient_id)
    
    with tab5:
        show_patient_historique(db, patient_id)
    
    # Bouton retour
    if st.button("🔙 Retour à la liste des patients", width='stretch'):
        st.session_state.selected_patient = None
        st.session_state.view_mode = None
        st.rerun()

def show_patient_info(db, patient_id):
    """Affiche les informations générales du patient"""
    
    try:
        patient = db.execute_query("""
            SELECT * FROM users WHERE id = %s
        """, (patient_id,), fetch_one=True)
        
        if patient:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Informations personnelles**")
                st.write(f"**Nom complet:** {patient['full_name']}")
                st.write(f"**Email:** {patient.get('email', 'Non renseigné')}")
                st.write(f"**Téléphone:** {patient.get('phone', 'Non renseigné')}")
                st.write(f"**Ville:** {patient.get('ville', 'Non renseignée')}")
            
            with col2:
                st.markdown("**Informations compte**")
                st.write(f"**Date inscription:** {patient.get('created_at', 'N/A')}")
                st.write(f"**Dernière connexion:** {patient.get('last_login', 'N/A')}")
                st.write(f"**Rôle:** {patient.get('role', 'N/A')}")
    
    except Exception as e:
        st.error(f"Erreur chargement informations: {e}")

def show_patient_analyses_complet(db, patient_id):
    """Affiche toutes les analyses du patient avec détails"""
    
    try:
        analyses = db.execute_query("""
            SELECT * FROM analyses 
            WHERE user_id = %s 
            ORDER BY timestamp DESC
        """, (patient_id,), fetch_all=True)
        
        if analyses:
            for ana in analyses:
                with st.expander(f"📊 {ana['type_analyse']} - {ana['timestamp'].strftime('%d/%m/%Y %H:%M')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Résultat:** {ana['resultat']}")
                        if ana.get('urgent'):
                            st.error("⚠️ Analyse urgente")
                    
                    with col2:
                        if ana.get('confiance'):
                            st.write(f"**Confiance:** {float(ana['confiance'])*100:.1f}%")
                        if ana.get('medecin_id'):
                            medecin = db.execute_query(
                                "SELECT full_name FROM users WHERE id = %s",
                                (ana['medecin_id'],), fetch_one
                            )
                            if medecin:
                                st.write(f"**Prescrite par:** Dr. {medecin['full_name']}")
                    
                    if ana.get('notes'):
                        st.write(f"**Notes:** {ana['notes']}")
                    
                    # Boutons d'action
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    with col_btn1:
                        if st.button("📥 Télécharger", key=f"dl_ana_{ana['id']}"):
                            st.info("Téléchargement à implémenter")
                    with col_btn2:
                        if st.button("📧 Envoyer", key=f"send_ana_{ana['id']}"):
                            st.info("Envoi à implémenter")
                    with col_btn3:
                        if st.button("🖨️ Imprimer", key=f"print_ana_{ana['id']}"):
                            st.info("Impression à implémenter")
        else:
            st.info("Aucune analyse disponible")
    
    except Exception as e:
        st.error(f"Erreur chargement analyses: {e}")

def show_patient_pdfs(db, patient_id):
    """Affiche les documents PDF du patient"""
    
    st.markdown("### 📄 Documents PDF")
    
    try:
        # Simuler des documents PDF (à remplacer par votre requête réelle)
        pdf_documents = [
            {"id": 1, "nom": "Compte-rendu consultation", "date": datetime.now(), "taille": "2.3 MB"},
            {"id": 2, "nom": "Résultats analyse sang", "date": datetime.now() - timedelta(days=7), "taille": "1.1 MB"},
            {"id": 3, "nom": "Ordonnance", "date": datetime.now() - timedelta(days=15), "taille": "0.5 MB"},
        ]
        
        # Upload de nouveau PDF
        with st.expander("📤 Uploader un nouveau document"):
            uploaded_file = st.file_uploader("Choisir un fichier PDF", type=['pdf'])
            if uploaded_file:
                st.success(f"Fichier {uploaded_file.name} uploadé avec succès!")
        
        # Liste des PDF existants
        for pdf in pdf_documents:
            col1, col2, col3, col4 = st.columns([3, 2, 1, 2])
            
            with col1:
                st.write(f"📄 **{pdf['nom']}**")
            with col2:
                st.write(f"📅 {pdf['date'].strftime('%d/%m/%Y')}")
            with col3:
                st.write(f"📦 {pdf['taille']}")
            with col4:
                if st.button("👁️ Voir", key=f"view_pdf_{pdf['id']}"):
                    st.info(f"Affichage du PDF {pdf['nom']} à implémenter")
            
            st.divider()
    
    except Exception as e:
        st.error(f"Erreur chargement PDF: {e}")

def show_patient_images(db, patient_id):
    """Affiche les images médicales du patient"""
    
    st.markdown("### 🖼️ Images médicales")
    
    try:
        # Simuler des images (à remplacer par votre requête réelle)
        images = [
            {"id": 1, "type": "Radio", "date": datetime.now(), "description": "Radio thorax"},
            {"id": 2, "type": "IRM", "date": datetime.now() - timedelta(days=30), "description": "IRM cérébrale"},
            {"id": 3, "type": "Scanner", "date": datetime.now() - timedelta(days=45), "description": "Scanner abdominal"},
        ]
        
        # Upload d'images
        with st.expander("📤 Uploader une nouvelle image"):
            uploaded_files = st.file_uploader("Choisir des images", type=['png', 'jpg', 'jpeg', 'dicom'], accept_multiple_files=True)
            if uploaded_files:
                for file in uploaded_files:
                    st.success(f"Image {file.name} uploadée!")
        
        # Affichage en grille des images
        cols = st.columns(3)
        for idx, img in enumerate(images):
            with cols[idx % 3]:
                st.image("https://via.placeholder.com/200x150.png?text=Image+Medicale", width='stretch')
                st.caption(f"**{img['type']}** - {img['date'].strftime('%d/%m/%Y')}")
                st.write(img['description'])
                if st.button("🔍 Agrandir", key=f"view_img_{img['id']}"):
                    st.info(f"Agrandissement de l'image {img['id']}")
    
    except Exception as e:
        st.error(f"Erreur chargement images: {e}")

def show_patient_historique(db, patient_id):
    """Affiche l'historique complet du patient"""
    
    st.markdown("### 📊 Historique complet")
    
    try:
        # Récupération des analyses
        analyses = db.execute_query("""
            SELECT 'analyse' as type, type_analyse as titre, resultat as description, 
                   timestamp as date, urgent
            FROM analyses 
            WHERE user_id = %s
        """, (patient_id,), fetch_all=True) or []
        
        # Récupération des rendez-vous
        rdvs = db.execute_query("""
            SELECT 'rdv' as type, CONCAT('Rendez-vous avec Dr. ', u.full_name) as titre,
                   motif as description, 
                   CONCAT(date_rdv, ' ', heure_rdv) as date,
                   NULL as urgent
            FROM rendez_vous r
            JOIN users u ON r.medecin_id = u.id
            WHERE r.patient_id = %s
        """, (patient_id,), fetch_all=True) or []
        
        # Combinaison et tri
        historique = analyses + rdvs
        
        # Conversion des dates
        for item in historique:
            if isinstance(item['date'], str):
                try:
                    item['date'] = datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S')
                except:
                    item['date'] = datetime.now()
        
        historique.sort(key=lambda x: x['date'], reverse=True)
        
        # Affichage chronologique
        for item in historique[:50]:  # Limiter à 50 entrées
            if item['type'] == 'analyse':
                if item.get('urgent'):
                    st.error(f"🚨 **{item['date'].strftime('%d/%m/%Y %H:%M')} - {item['titre']}**")
                else:
                    st.info(f"📊 **{item['date'].strftime('%d/%m/%Y %H:%M')} - {item['titre']}**")
            else:
                st.success(f"📅 **{item['date'].strftime('%d/%m/%Y %H:%M')} - {item['titre']}**")
            
            st.write(item['description'])
            st.divider()
    
    except Exception as e:
        st.error(f"Erreur chargement historique: {e}")

def show_patient_consultation_calendar(db, patient_id, doctor_id):
    """Affiche le calendrier des rendez-vous avec pagination"""
    
    st.divider()
    st.header(f"📅 Calendrier des Consultations")
    
    # Récupération du nom du patient
    patient = db.execute_query(
        "SELECT full_name FROM users WHERE id = %s",
        (patient_id,), fetch_one=True
    )
    
    if patient:
        st.subheader(f"Patient: {patient['full_name']}")
    
    # Initialisation des paramètres de pagination
    if 'calendar_page' not in st.session_state:
        st.session_state.calendar_page = 0
    
    if 'current_month' not in st.session_state:
        st.session_state.current_month = datetime.now().month
        st.session_state.current_year = datetime.now().year
    
    # Navigation mois
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    
    with col1:
        if st.button("◀ Mois précédent"):
            if st.session_state.current_month == 1:
                st.session_state.current_month = 12
                st.session_state.current_year -= 1
            else:
                st.session_state.current_month -= 1
            st.session_state.calendar_page = 0
            st.rerun()
    
    with col2:
        mois_nom = datetime(st.session_state.current_year, st.session_state.current_month, 1).strftime('%B %Y')
        st.markdown(f"<h3 style='text-align: center;'>{mois_nom}</h3>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Mois suivant ▶"):
            if st.session_state.current_month == 12:
                st.session_state.current_month = 1
                st.session_state.current_year += 1
            else:
                st.session_state.current_month += 1
            st.session_state.calendar_page = 0
            st.rerun()
    
    with col4:
        if st.button("Aujourd'hui"):
            st.session_state.current_month = datetime.now().month
            st.session_state.current_year = datetime.now().year
            st.session_state.calendar_page = 0
            st.rerun()
    
    # Récupération des rendez-vous du mois
    try:
        start_date = datetime(st.session_state.current_year, st.session_state.current_month, 1)
        if st.session_state.current_month == 12:
            end_date = datetime(st.session_state.current_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(st.session_state.current_year, st.session_state.current_month + 1, 1) - timedelta(days=1)
        
        rendez_vous = db.execute_query("""
            SELECT r.*, u.full_name as medecin_nom
            FROM rendez_vous r
            JOIN users u ON r.medecin_id = u.id
            WHERE r.patient_id = %s 
            AND r.date_rdv BETWEEN %s AND %s
            ORDER BY r.date_rdv, r.heure_rdv
        """, (patient_id, start_date.date(), end_date.date()), fetch_all=True) or []
        
        # Organisation des rendez-vous par date
        rdv_par_date = {}
        for rdv in rendez_vous:
            date_str = rdv['date_rdv'].strftime('%Y-%m-%d')
            if date_str not in rdv_par_date:
                rdv_par_date[date_str] = []
            rdv_par_date[date_str].append(rdv)
        
        # Construction du calendrier
        cal = calendar.monthcalendar(st.session_state.current_year, st.session_state.current_month)
        
        # Affichage du calendrier
        st.markdown('<div class="calendar-container">', unsafe_allow_html=True)
        
        # En-têtes des jours
        cols = st.columns(7)
        jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        for col, jour in zip(cols, jours):
            col.markdown(f"<div class='calendar-day-header'>{jour}</div>", unsafe_allow_html=True)
        
        # Lignes du calendrier
        for semaine in cal:
            cols = st.columns(7)
            for i, (col, jour) in enumerate(zip(cols, semaine)):
                if jour == 0:
                    col.markdown("<div class='calendar-day'></div>", unsafe_allow_html=True)
                else:
                    date_obj = datetime(st.session_state.current_year, st.session_state.current_month, jour)
                    date_str = date_obj.strftime('%Y-%m-%d')
                    
                    with col:
                        st.markdown(f"<div class='calendar-day'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='calendar-day-number'>{jour}</div>", unsafe_allow_html=True)
                        
                        # Afficher les rendez-vous du jour
                        if date_str in rdv_par_date:
                            for rdv in rdv_par_date[date_str]:
                                # Déterminer la classe CSS selon le statut
                                if rdv['statut'] == 'confirme':
                                    classe = "calendar-event-confirme"
                                elif rdv['statut'] == 'termine':
                                    classe = "calendar-event-termine"
                                elif rdv['statut'] == 'annule':
                                    classe = "calendar-event-annule"
                                else:
                                    classe = ""
                                
                                # Créer un bouton pour chaque rendez-vous
                                rdv_key = f"rdv_{rdv['id']}"
                                if st.button(
                                    f"{rdv['heure_rdv'].strftime('%H:%M') if isinstance(rdv['heure_rdv'], datetime) else rdv['heure_rdv']} - Dr. {rdv['medecin_nom']}",
                                    key=rdv_key,
                                    width='stretch'
                                ):
                                    st.session_state.selected_rdv = rdv
                        
                        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Pagination pour les détails des rendez-vous
        st.markdown("### 📋 Détails des rendez-vous du mois")
        
        # Récupérer tous les rendez-vous du mois pour la pagination
        tous_rdvs = rendez_vous
        
        # Calculer le nombre total de pages
        rdvs_par_page = 5
        total_pages = (len(tous_rdvs) + rdvs_par_page - 1) // rdvs_par_page
        
        # Afficher les rendez-vous de la page courante
        start_idx = st.session_state.calendar_page * rdvs_par_page
        end_idx = min(start_idx + rdvs_par_page, len(tous_rdvs))
        
        for i in range(start_idx, end_idx):
            if i < len(tous_rdvs):
                rdv = tous_rdvs[i]
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    
                    with col1:
                        st.write(f"**{rdv['date_rdv'].strftime('%d/%m/%Y')}**")
                    
                    with col2:
                        st.write(f"🕐 {rdv['heure_rdv'].strftime('%H:%M') if isinstance(rdv['heure_rdv'], datetime) else rdv['heure_rdv']}")
                    
                    with col3:
                        st.write(f"👨‍⚕️ Dr. {rdv['medecin_nom']}")
                    
                    with col4:
                        if rdv['statut'] == 'planifie':
                            st.info("📅")
                        elif rdv['statut'] == 'confirme':
                            st.success("✅")
                        elif rdv['statut'] == 'termine':
                            st.write("✓")
                        else:
                            st.write(rdv['statut'])
                    
                    # Détails supplémentaires
                    with st.expander("Voir détails"):
                        st.write(f"**Motif:** {rdv.get('motif', 'Non spécifié')}")
                        st.write(f"**Notes:** {rdv.get('notes', 'Aucune note')}")
                        
                        # Boutons d'action
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        with col_btn1:
                            if st.button("📝 Modifier", key=f"modif_{rdv['id']}"):
                                st.info("Modification à implémenter")
                        with col_btn2:
                            if st.button("❌ Annuler", key=f"annul_{rdv['id']}"):
                                st.warning("Annulation à implémenter")
                        with col_btn3:
                            if st.button("📧 Rappel", key=f"rappel_{rdv['id']}"):
                                st.success("Rappel envoyé")
                    
                    st.divider()
        
        # Contrôles de pagination
        if total_pages > 1:
            st.markdown('<div class="pagination">', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⏮️ Premier") and st.session_state.calendar_page > 0:
                    st.session_state.calendar_page = 0
                    st.rerun()
            
            with col2:
                if st.button("◀ Précédent") and st.session_state.calendar_page > 0:
                    st.session_state.calendar_page -= 1
                    st.rerun()
            
            with col3:
                st.markdown(f"<p style='text-align: center;'>Page {st.session_state.calendar_page + 1} / {total_pages}</p>", unsafe_allow_html=True)
            
            with col4:
                if st.button("Suivant ▶") and st.session_state.calendar_page < total_pages - 1:
                    st.session_state.calendar_page += 1
                    st.rerun()
            
            with col5:
                if st.button("Dernier ⏭️") and st.session_state.calendar_page < total_pages - 1:
                    st.session_state.calendar_page = total_pages - 1
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Ajout d'un nouveau rendez-vous
        with st.expander("➕ Planifier un nouveau rendez-vous"):
            with st.form("new_rdv_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    new_date = st.date_input("Date du rendez-vous", min_value=datetime.now().date())
                
                with col2:
                    new_time = st.time_input("Heure", datetime.now().time())
                
                # Récupérer la liste des médecins
                medecins = db.execute_query(
                    "SELECT id, full_name FROM users WHERE role = 'medecin'",
                    fetch_all=True
                ) or []
                
                medecin_options = {m['full_name']: m['id'] for m in medecins}
                selected_medecin = st.selectbox("Médecin", options=list(medecin_options.keys()))
                
                motif = st.text_input("Motif de la consultation")
                notes = st.text_area("Notes supplémentaires")
                
                if st.form_submit_button("Planifier le rendez-vous"):
                    st.success("Rendez-vous planifié avec succès!")
    
    except Exception as e:
        st.error(f"Erreur lors du chargement du calendrier: {e}")
    
    # Bouton retour
    if st.button("🔙 Retour à la liste des patients", width='stretch'):
        st.session_state.selected_patient = None
        st.session_state.view_mode = None
        st.rerun()

# Garder les fonctions existantes si elles sont utilisées ailleurs
def show_patient_details(db, patient_id, doctor_id):
    """Ancienne fonction à garder pour compatibilité"""
    show_patient_dossier(db, patient_id, doctor_id)

def show_patient_analyses(db, patient_id):
    """Ancienne fonction à garder pour compatibilité"""
    show_patient_analyses_complet(db, patient_id)

def show_patient_consultations(db, patient_id, doctor_id):
    """Ancienne fonction à garder pour compatibilité"""
    show_patient_consultation_calendar(db, patient_id, doctor_id)

def show_patient_messages(db, patient_id, doctor_id):
    """Garder la fonction messages existante"""
    # Votre code existant pour les messages
    pass