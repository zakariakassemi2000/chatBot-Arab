# ui_components.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def show_auth_interface(auth_manager):
    """Interface d'authentification"""
    st.sidebar.header("🔐 Authentification")
    
    tab1, tab2 = st.sidebar.tabs(["Connexion", "Inscription"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("Se connecter", use_container_width=True)
            
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
            new_username = st.text_input("Nom d'utilisateur *")
            new_email = st.text_input("Email *")
            new_password = st.text_input("Mot de passe *", type="password")
            confirm_password = st.text_input("Confirmer mot de passe *", type="password")
            full_name = st.text_input("Nom complet")
            phone = st.text_input("Téléphone")
            ville = st.selectbox("Ville", ["Casablanca", "Rabat", "Marrakech", "Fès", "Tanger", "Agadir", "Autre"])
            role = st.selectbox("Je suis", ["patient", "medecin"])
            
            submit = st.form_submit_button("S'inscrire", use_container_width=True)
            
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

def show_patient_dashboard(db, user_id):
    """Dashboard patient"""
    st.header(f"🏠 Bonjour, {st.session_state.user['full_name'] or st.session_state.user['username']}")
    
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
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Messages", stats['messages'])
    with col2:
        st.metric("Analyses", stats['analyses'])
    with col3:
        st.metric("Rendez-vous", stats['rdv'])
    with col4:
        next_date = stats['prochain_rdv']['date_rdv'] if stats['prochain_rdv'] else "Aucun"
        st.metric("Prochain RDV", next_date)
    
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
    """Interface de chat avec l'assistant"""
    st.header("💬 Chat avec l'assistant santé")
    
    # Initialisation de l'historique
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Affichage de l'historique
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Zone de saisie
    if prompt := st.chat_input("Posez votre question sur vos symptômes..."):
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
            print(f"Erreur sauvegarde message: {e}")
        
        # Détection d'urgence
        mots_urgence = [
            'douleur poitrine', 'étouffe', 'inconscient', 'saigne', 
            'crise', 'urgent', 'infarctus', 'avc', 'paralysie',
            'difficulté respiratoire', 'perte connaissance'
        ]
        urgent = any(mot in prompt.lower() for mot in mots_urgence)
        
        if urgent:
            reponse = generate_urgence_response()
            
            # Sauvegarde avec flag urgent
            db.execute_query(
                """INSERT INTO messages (user_id, message, type_message, urgent) 
                   VALUES (%s, %s, 'reponse_ia', TRUE)""",
                (user_id, reponse)
            )
            
            # Notification
            notif_manager.create_notification(
                user_id,
                "🚨 ALERTE URGENCE",
                "Symptômes critiques détectés - Consultez immédiatement",
                "urgence"
            )
        else:
            reponse = generate_regular_response(prompt)
            
            # Sauvegarde
            db.execute_query(
                """INSERT INTO messages (user_id, message, type_message) 
                   VALUES (%s, %s, 'reponse_ia')""",
                (user_id, reponse)
            )
        
        # Affichage réponse
        st.session_state.chat_history.append({"role": "assistant", "content": reponse})
        with st.chat_message("assistant"):
            st.markdown(reponse)
        
        # Bouton ordonnance
        if st.button("📝 Générer une ordonnance"):
            generate_prescription(pdf_gen, user_id, prompt, reponse)

def generate_urgence_response():
    """Génère une réponse d'urgence"""
    return """
🚨 **ALERTE URGENCE DÉTECTÉE**

Vos symptômes semblent critiques. **ACTION IMMÉDIATE REQUISE**:

1. 📞 **Appelez le SAMU 141** immédiatement
2. 🏥 Rendez-vous aux URGENCES les plus proches
3. 👥 Ne restez pas seul(e)
4. 📋 Préparez vos documents médicaux

**Numéros d'urgence Maroc:**
- SAMU: **141**
- Pompiers: **150**
- Police: **190**
- Urgences hospitalières: **05 22 23 45 67**

⏱️ **NE PAS ATTENDRE - AGISSEZ MAINTENANT**
"""

def generate_regular_response(prompt):
    """Génère une réponse normale"""
    return f"""
**Conseil médical:**

Basé sur vos symptômes, voici mes recommandations:

1. **Repos**: Prenez du repos et hydratez-vous
2. **Surveillance**: Surveillez votre température
3. **Médication**: Paracétamol si fièvre (>38.5°C)
4. **Consultation**: Consultez si les symptômes persistent >48h

⚠️ **Consultez en urgence si:**
- Aggravation soudaine
- Difficultés respiratoires
- Fièvre > 39°C persistante

ℹ️ *Ceci est un conseil préliminaire. Consultez toujours un médecin pour un diagnostic précis.*
"""

def generate_prescription(pdf_gen, user_id, question, reponse):
    """Génère une ordonnance PDF"""
    patient_info = {
        'id': user_id,
        'full_name': st.session_state.user.get('full_name', 'Patient'),
        'date_naissance': 'Non renseignée'
    }
    
    prescription = f"""
Symptômes rapportés:
{question}

Recommandations médicales:
{reponse}

Traitement suggéré (à valider par médecin):
- Repos: 48h
- Hydratation: 1.5L d'eau par jour
- Paracétamol: 1g si fièvre, max 3g/jour
- Consultation si persistance >48h
"""
    
    filename = pdf_gen.generate_ordonnance(patient_info, prescription)
    
    st.success(f"✅ Ordonnance générée")
    
    with open(filename, "rb") as f:
        st.download_button(
            "📥 Télécharger l'ordonnance",
            f,
            file_name=os.path.basename(filename),
            mime="application/pdf"
        )

def show_patient_records(db, user_id):
    """Dossier médical du patient"""
    st.header("📊 Mon Dossier Médical")
    
    tab1, tab2, tab3 = st.tabs(["📝 Consultations", "🔬 Analyses", "💬 Messages"])
    
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
                df = pd.DataFrame(consultations)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Aucune consultation enregistrée")
        except Exception as e:
            st.error(f"Erreur: {e}")
    
    with tab2:
        st.subheader("Analyses médicales")
        try:
            analyses = db.execute_query("""
                SELECT type_analyse, resultat, confiance, timestamp, urgent
                FROM analyses
                WHERE user_id = %s
                ORDER BY timestamp DESC
            """, (user_id,), fetch_all=True)
            
            if analyses:
                for ana in analyses:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"**{ana['type_analyse']}** - {ana['resultat']}")
                        with col2:
                            st.caption(f"Confiance: {float(ana['confiance']):.1%}" if ana['confiance'] else "N/A")
                        with col3:
                            if ana['urgent']:
                                st.error("🚨 URGENT")
                            else:
                                st.success("✅ Normal")
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

def show_appointments_interface(db, notif_manager, user_id):
    """Interface de rendez-vous"""
    st.header("📅 Gestion des rendez-vous")
    
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
                        "SELECT id, full_name FROM users WHERE role = 'medecin' AND actif = TRUE"
                    )
                    medecin_options = {f"{m['full_name']}": m['id'] for m in medecins}
                    medecin_choice = st.selectbox("Médecin", list(medecin_options.keys()))
                except:
                    st.warning("Aucun médecin disponible")
                    medecin_choice = None
            
            with col2:
                heure_rdv = st.time_input("Heure", value=datetime.now().time().replace(hour=9, minute=0))
                motif = st.text_area("Motif de la consultation")
            
            submit = st.form_submit_button("Confirmer", use_container_width=True)
            
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
        else:
            st.info("Aucun rendez-vous planifié")
    except Exception as e:
        st.error(f"Erreur: {e}")

def show_settings_interface(db, auth_manager, user_id):
    """Paramètres utilisateur"""
    st.header("⚙️ Paramètres")
    
    tab1, tab2, tab3 = st.tabs(["Profil", "Notifications", "Sécurité"])
    
    with tab1:
        st.subheader("Informations personnelles")
        
        with st.form("profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                full_name = st.text_input("Nom complet", value=st.session_state.user.get('full_name', ''))
                email = st.text_input("Email", value=st.session_state.user.get('email', ''))
                phone = st.text_input("Téléphone")
            
            with col2:
                date_naissance = st.date_input("Date naissance", value=None)
                ville = st.selectbox("Ville", ["Casablanca", "Rabat", "Marrakech", "Fès", "Tanger", "Agadir", "Autre"])
            
            submit = st.form_submit_button("Mettre à jour")
            if submit:
                try:
                    db.execute_query("""
                        UPDATE users 
                        SET full_name = %s, email = %s, phone = %s, ville = %s
                        WHERE id = %s
                    """, (full_name, email, phone, ville, user_id))
                    st.session_state.user['full_name'] = full_name
                    st.session_state.user['email'] = email
                    st.success("Profil mis à jour")
                except Exception as e:
                    st.error(f"Erreur: {e}")
    
    with tab2:
        st.subheader("Préférences de notifications")
        
        notif_email = st.toggle("Notifications par email", value=True)
        notif_sms = st.toggle("Notifications SMS", value=False)
        rappel_rdv = st.toggle("Rappels de rendez-vous", value=True)
        alert_urgente = st.toggle("Alertes urgentes", value=True)
        
        if st.button("Sauvegarder préférences"):
            st.success("Préférences sauvegardées")
    
    with tab3:
        st.subheader("Sécurité")
        
        with st.form("password_form"):
            current_pw = st.text_input("Mot de passe actuel", type="password")
            new_pw = st.text_input("Nouveau mot de passe", type="password")
            confirm_pw = st.text_input("Confirmer le nouveau mot de passe", type="password")
            
            submit_pw = st.form_submit_button("Changer le mot de passe")
            if submit_pw:
                if new_pw == confirm_pw and len(new_pw) >= 6:
                    # Vérifier l'ancien mot de passe
                    user = db.get_user_by_id(user_id)
                    if user and auth_manager.verify_password(current_pw, user['password']):
                        new_hash = auth_manager.hash_password(new_pw)
                        db.execute_query(
                            "UPDATE users SET password = %s WHERE id = %s",
                            (new_hash, user_id)
                        )
                        st.success("Mot de passe modifié avec succès")
                    else:
                        st.error("Mot de passe actuel incorrect")
                else:
                    st.error("Vérifiez les mots de passe (6 caractères minimum)")