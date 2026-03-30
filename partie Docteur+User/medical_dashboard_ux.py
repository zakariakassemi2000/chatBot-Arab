# medical_dashboard_ux.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

def show_medical_ux_demo():
    """Démonstration de l'UX complète du produit médical"""
    
    st.set_page_config(layout="wide", page_title="Florence Medical UX")
    
    # CSS personnalisé pour l'UX médical
    st.markdown("""
    <style>
    /* Style général médical */
    .medical-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .patient-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 5px solid;
        transition: transform 0.2s;
    }
    
    .patient-card:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .priority-urgent { border-left-color: #e74c3c; background: #fff5f5; }
    .priority-high { border-left-color: #f39c12; background: #fff9e6; }
    .priority-normal { border-left-color: #27ae60; background: #f0fff4; }
    
    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    
    .badge-urgent { background: #e74c3c; color: white; }
    .badge-warning { background: #f39c12; color: white; }
    .badge-success { background: #27ae60; color: white; }
    .badge-info { background: #3498db; color: white; }
    
    .consultation-area {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        min-height: 500px;
    }
    
    .timeline-item {
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #3498db;
        background: #f8f9fa;
    }
    
    .stat-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stat-value {
        font-size: 2em;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .stat-label {
        color: #7f8c8d;
        font-size: 0.9em;
    }
    
    .action-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        width: 100%;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="medical-header">
        <h1>🏥 Florence - Plateforme de Gestion Médicale</h1>
        <p>Dr. Alami Mohammed • Cardiologie • Casablanca</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== DASHBOARD MÉDECIN ==========
    st.header("📊 Dashboard Médecin")
    
    # Ligne 1: Statistiques rapides
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">1,247</div>
            <div class="stat-label">Patients suivis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">12</div>
            <div class="stat-label">RDV aujourd'hui</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">3</div>
            <div class="stat-label">Analyses urgentes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">5</div>
            <div class="stat-label">Alertes critiques</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Ligne 2: Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Courbe activité consultations
        dates = [(datetime.now() - timedelta(days=i)).strftime('%d/%m') for i in range(7, 0, -1)]
        consultations = [random.randint(5, 15) for _ in range(7)]
        
        fig = px.line(x=dates, y=consultations, title="Activité des consultations - 7 derniers jours")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Graph analyses par type
        types = ['Radiologie', 'Biologie', 'IA Images', 'ECG']
        counts = [45, 32, 28, 19]
        
        fig = px.pie(values=counts, names=types, title="Répartition des analyses")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========== LISTE PATIENTS PRIORISÉE IA ==========
    st.header("👥 Liste des Patients - Priorisation IA")
    
    # Simulation de scoring IA
    patients = generate_patients_with_score()
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("🔍 Rechercher", placeholder="Nom, téléphone...")
    with col2:
        priority_filter = st.selectbox("Filtrer priorité", ["Tous", "Urgent", "À voir vite", "Stable"])
    with col3:
        tri = st.selectbox("Trier par", ["Score IA", "Nom", "Dernier contact"])
    
    # Affichage des patients
    for patient in patients[:10]:  # Top 10 pour la démo
        priority_class = {
            "Urgent": "priority-urgent",
            "À voir vite": "priority-high",
            "Stable": "priority-normal"
        }[patient['priority']]
        
        badge_class = {
            "Urgent": "badge-urgent",
            "À voir vite": "badge-warning",
            "Stable": "badge-success"
        }[patient['priority']]
        
        st.markdown(f"""
        <div class="patient-card {priority_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 2">
                    <strong>{patient['name']}</strong> • {patient['age']} ans • {patient['city']}
                </div>
                <div style="flex: 1">
                    <span class="badge {badge_class}">{patient['priority']} (Score: {patient['score']})</span>
                </div>
                <div style="flex: 1">
                    {patient['last_contact']}
                </div>
                <div>
                    <button class="action-button" style="width: auto; padding: 5px 15px;">Voir</button>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== DOSSIER PATIENT COMPLET ==========
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 👤 Fiche Patient")
        
        # Header patient
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <img src="https://img.icons8.com/color/96/user.png" width="100">
            <h3>Ahmed Benchekroun</h3>
            <p>📧 ahmed.b@email.com • 📞 06 12 34 56 78</p>
            <p>🏙 Casablanca • 45 ans</p>
            <div style="display: flex; gap: 10px; justify-content: center;">
                <span class="badge badge-warning">🟠 En cours</span>
                <span class="badge badge-info">🩸 A+</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Boutons d'action
        st.markdown("#### ⚡ Actions")
        if st.button("✏️ Ajouter note médicale", use_container_width=True):
            pass
        if st.button("📅 Planifier consultation", use_container_width=True):
            pass
        if st.button("📄 Générer PDF", use_container_width=True):
            pass
    
    with col2:
        # Onglets du dossier patient
        tab1, tab2, tab3, tab4 = st.tabs([
            "🩺 Résumé médical",
            "📜 Historique",
            "🧾 Documents",
            "📝 Notes"
        ])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Consultations", "12")
                st.metric("Analyses", "8")
                st.metric("Dernier message", "02/03/2024")
            with col_b:
                st.metric("Dernière visite", "28/02/2024")
                st.metric("Prochain RDV", "15/03/2024")
                st.warning("⚠️ Allergie: Pénicilline")
        
        with tab2:
            # Timeline chronologique
            st.markdown("#### 📅 Timeline médicale")
            
            events = [
                ("🧪 Analyse IA", "Pneumonie détectée", "02/03/2024", True),
                ("💬 Message", "Douleur thoracique", "01/03/2024", False),
                ("📅 Consultation", "Suivi cardiologue", "28/02/2024", False),
                ("📝 Note", "Tension élevée", "27/02/2024", False),
                ("🧾 Ordonnance", "Traitement antibiotique", "25/02/2024", False)
            ]
            
            for event in events:
                st.markdown(f"""
                <div class="timeline-item">
                    <strong>{event[0]}</strong> • {event[2]}
                    <br>
                    {event[1]}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== CONSULTATION ACTIVE ==========
    st.header("📅 Consultation en cours")
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("### 📋 RDV du jour")
        
        rdvs = [
            ("09:00", "Fatima Zahra", "Suivi cardiologie"),
            ("10:30", "Karim Mansouri", "Douleur thoracique"),
            ("11:45", "Nadia Tazi", "Résultats analyses"),
            ("14:00", "Youssef Alaoui", "Consultation routine"),
        ]
        
        for heure, nom, motif in rdvs:
            st.markdown(f"""
            <div class="patient-card" style="border-left-color: #3498db;">
                <strong>{heure}</strong> - {nom}
                <br>
                <small>{motif}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("### 🩺 Consultation active: Fatima Zahra")
        
        with st.container():
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.info("45 ans")
            with col_info2:
                st.warning("⚠️ Hypertension")
            with col_info3:
                st.success("✅ Dernière analyse: normale")
        
        # Formulaire de consultation
        with st.form("consultation_form"):
            st.markdown("#### Symptômes rapportés")
            symptomes = st.text_area("", value="Douleur thoracique légère, essoufflement")
            
            st.markdown("#### Diagnostic")
            diagnostic = st.text_input("", value="Angine de poitrine suspectée")
            
            st.markdown("#### Note médicale")
            note = st.text_area("", height=100)
            
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                if st.form_submit_button("📄 Ordonnance"):
                    st.success("Ordonnance générée")
            with col_b2:
                if st.form_submit_button("🧪 Demander analyse"):
                    st.success("Demande d'analyse envoyée")
            with col_b3:
                if st.form_submit_button("📅 Reprogrammer"):
                    st.info("Sélectionner nouvelle date")
            with col_b4:
                if st.form_submit_button("💬 Message"):
                    st.success("Message envoyé au patient")

def generate_patients_with_score():
    """Génère des patients avec scores IA simulés"""
    names = [
        "Ahmed Benchekroun", "Fatima Zahra", "Karim Mansouri", 
        "Nadia Tazi", "Youssef Alaoui", "Samira Bennis",
        "Mohammed El Fassi", "Khadija Amrani", "Rachid Berrada"
    ]
    
    cities = ["Casablanca", "Rabat", "Marrakech", "Fès", "Tanger"]
    ages = [35, 42, 28, 55, 48, 31, 62, 39, 51]
    
    patients = []
    for i, name in enumerate(names):
        score = random.randint(20, 95)
        if score >= 80:
            priority = "Urgent"
        elif score >= 50:
            priority = "À voir vite"
        else:
            priority = "Stable"
        
        patients.append({
            'name': name,
            'age': ages[i],
            'city': random.choice(cities),
            'score': score,
            'priority': priority,
            'last_contact': (datetime.now() - timedelta(days=random.randint(1, 60))).strftime('%d/%m/%Y')
        })
    
    # Trier par score décroissant
    patients.sort(key=lambda x: x['score'], reverse=True)
    return patients

if __name__ == "__main__":
    show_medical_ux_demo()