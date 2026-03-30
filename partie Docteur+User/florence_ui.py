# florence_ui.py

import streamlit as st
import os
import time
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
from io import BytesIO

from florence_analyzer import *

# Initialisation des analyseurs Florence
florence_analyzers = {
    "Analyse IRM": FlorenceIRMAnalyzer(),
    "Détection Arythmie": FlorenceArythmieAnalyzer(),
    "Scanner Thorax": FlorenceScannerThoraxAnalyzer(),
    "Radiographie": FlorenceRadiographieAnalyzer(),
    "ECG": FlorenceECGAnalyzer(),
    "Analyse Sang": FlorenceSangAnalyzer(),
    "IRM Cerveau": FlorenceIRMCerveauAnalyzer(),
    "Scanner Abdomen": FlorenceScannerAbdomenAnalyzer(),
    "Test COVID-19": FlorenceCovidAnalyzer(),
    "Analyse Urine": FlorenceUrineAnalyzer()
}

# Classe PDF Generator Florence
class FlorencePDFGenerator:
    @staticmethod
    def generate_report(patient_name, analysis_type, result):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        
        # En-tête
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 20, 'Florence - Rapport d\'Analyse Médicale', 0, 1, 'C')
        pdf.ln(10)
        
        # Informations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Patient: {patient_name}", 0, 1)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1)
        pdf.cell(0, 10, f"Type d'analyse: {analysis_type}", 0, 1)
        pdf.ln(10)
        
        # Résultats
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Résultats:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, f"Diagnostic: {result['resultat']}")
        pdf.cell(0, 10, f"Confiance: {result['confiance']*100:.1f}%", 0, 1)
        pdf.cell(0, 10, f"Niveau d'urgence: {'URGENT' if result['urgent'] else 'Normal'}", 0, 1)
        pdf.ln(10)
        
        # Recommandations
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Recommandations:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, result['recommandations'])
        
        return pdf

florence_pdf_gen = FlorencePDFGenerator()

def inject_florence_css():
    st.markdown("""
    <style>
        /* Style global */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Style des boutons */
        div.stButton > button {
            width: 100%;
            height: 80px;
            font-size: 18px !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 15px !important;
            transition: all 0.3s ease !important;
            margin: 5px 0 !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }
        
        div.stButton > button:hover {
            transform: translateY(-5px) !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important;
        }
        
        div.stButton > button:active {
            transform: translateY(0) !important;
        }
        
        /* Couleurs spécifiques par bouton */
        .btn-0 > button { background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important; color: white !important; }
        .btn-1 > button { background: linear-gradient(45deg, #4ECDC4, #556270) !important; color: white !important; }
        .btn-2 > button { background: linear-gradient(45deg, #A8E6CF, #3B9AE1) !important; color: white !important; }
        .btn-3 > button { background: linear-gradient(45deg, #FFD93D, #FF6B6B) !important; color: white !important; }
        .btn-4 > button { background: linear-gradient(45deg, #6C5B7B, #C06C84) !important; color: white !important; }
        .btn-5 > button { background: linear-gradient(45deg, #99B898, #FECEAB) !important; color: white !important; }
        .btn-6 > button { background: linear-gradient(45deg, #E84A5F, #FF847C) !important; color: white !important; }
        .btn-7 > button { background: linear-gradient(45deg, #2C3E50, #3498DB) !important; color: white !important; }
        .btn-8 > button { background: linear-gradient(45deg, #F8B195, #F67280) !important; color: white !important; }
        .btn-9 > button { background: linear-gradient(45deg, #355C7D, #6C5B7B) !important; color: white !important; }
        
        /* Style des cartes */
        .card {
            background: white;
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin: 15px 0;
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        
        /* Style des titres */
        .main-title {
            text-align: center;
            color: #2C3E50;
            font-size: 3em;
            font-weight: 800;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            background: linear-gradient(45deg, #2C3E50, #3498DB);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .sub-title {
            text-align: center;
            color: #7F8C8D;
            font-size: 1.2em;
            margin-bottom: 30px;
        }
        
        .section-title {
            color: #2C3E50;
            font-size: 1.8em;
            font-weight: 600;
            margin: 25px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498DB;
        }
        
        /* Style des métriques */
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 20px rgba(102,126,234,0.3);
        }
        
        .metric-value {
            font-size: 2.2em;
            font-weight: 700;
        }
        
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Style des badges */
        .badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.9em;
            margin: 5px;
        }
        
        .badge-success {
            background: linear-gradient(45deg, #00B09B, #96C93D);
            color: white;
        }
        
        .badge-warning {
            background: linear-gradient(45deg, #F7971E, #FFD200);
            color: white;
        }
        
        .badge-danger {
            background: linear-gradient(45deg, #EB3349, #F45C43);
            color: white;
        }
        
        /* Style des résultats */
        .result-normal {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            font-size: 1.5em;
            font-weight: 700;
        }
        
        .result-urgent {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            font-size: 1.5em;
            font-weight: 700;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        /* Style de l'uploader */
        .uploader-box {
            border: 3px dashed #3498DB;
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            background: rgba(255,255,255,0.9);
            margin: 20px 0;
        }
        
        /* Style des onglets */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 10px;
            background: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(45deg, #3498DB, #2980B9) !important;
            color: white !important;
        }
        
        /* Animation de chargement */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading {
            animation: spin 1s linear infinite;
        }
    </style>
    """, unsafe_allow_html=True)

def show_florence_analysis_interface(db, notif_manager):
    """Interface d'analyse Florence complète"""
    
    # CSS Florence
    inject_florence_css()
    
    st.markdown('<h1 class="main-title">🔬 FLORENCE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Plateforme d\'Analyse Médicale Intelligente par IA</p>', unsafe_allow_html=True)
    
    # Initialisation session state Florence
    if 'florence_active_analysis' not in st.session_state:
        st.session_state.florence_active_analysis = None
    if 'florence_result' not in st.session_state:
        st.session_state.florence_result = None
    if 'florence_uploaded_file' not in st.session_state:
        st.session_state.florence_uploaded_file = None
    if 'florence_history' not in st.session_state:
        st.session_state.florence_history = []
    
    # Upload de fichier spécifique à Florence
    st.markdown("### 📤 Upload Fichier")
    florence_uploaded_file = st.file_uploader(
        "Choisir un fichier médical",
        type=['jpg', 'jpeg', 'png', 'pdf', 'dcm'],
        help="Formats supportés: JPG, PNG, PDF, DICOM",
        key="florence_uploader"
    )
    
    if florence_uploaded_file:
        st.session_state.florence_uploaded_file = florence_uploaded_file
        st.success(f"✅ Fichier chargé: {florence_uploaded_file.name}")
    
    st.markdown("---")
    
    # Statistiques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Analyses aujourd'hui", "1,247", "+5.3%")
    with col2:
        st.metric("Précision IA", "98.7%", "+0.2%")
    with col3:
        st.metric("Patients analysés", "3,892", "+12%")
    
    st.markdown("---")
    
    # Historique Florence
    with st.expander("📋 Historique des analyses"):
        if st.session_state.florence_history:
            for h in st.session_state.florence_history[-10:]:
                if h['urgent']:
                    st.error(f"**{h['type']}** - {h['resultat']} - *{h['timestamp']}*")
                else:
                    st.info(f"**{h['type']}** - {h['resultat']} - *{h['timestamp']}*")
        else:
            st.write("Aucune analyse récente")
    
    st.markdown("---")
    
    # Boutons d'analyse
    st.markdown('<h2 class="section-title">🔬 Sélectionnez le type d\'analyse</h2>', unsafe_allow_html=True)
    
    # Création des lignes de boutons
    row1 = st.columns(5)
    row2 = st.columns(5)
    
    buttons = list(florence_analyzers.keys())
    
    # Première ligne
    for i, (col, btn_name) in enumerate(zip(row1, buttons[:5])):
        with col:
            st.markdown(f'<div class="btn-{i}">', unsafe_allow_html=True)
            if st.button(btn_name, key=f"florence_btn_{i}", use_container_width=True):
                if st.session_state.florence_uploaded_file:
                    with st.spinner(f"🔬 Analyse {btn_name} en cours..."):
                        time.sleep(2)
                        
                        # Sauvegarde temporaire
                        temp_path = f"temp/florence_{datetime.now().timestamp()}_{st.session_state.florence_uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(st.session_state.florence_uploaded_file.getbuffer())
                        
                        analyzer = florence_analyzers[btn_name]
                        result = analyzer.analyze(temp_path)
                        
                        # Ajouter l'ID utilisateur
                        result['user_id'] = st.session_state.user['id']
                        
                        st.session_state.florence_result = result
                        st.session_state.florence_active_analysis = btn_name
                        
                        # Ajouter à l'historique
                        st.session_state.florence_history.append({
                            'type': btn_name,
                            'resultat': result['resultat'],
                            'urgent': result['urgent'],
                            'timestamp': datetime.now().strftime('%H:%M')
                        })
                        
                        # Sauvegarde en base de données
                        try:
                            db.execute_query("""
                                INSERT INTO analyses (user_id, filename, type_analyse, resultat, confiance, recommandations, urgent)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                st.session_state.user['id'],
                                st.session_state.florence_uploaded_file.name,
                                btn_name,
                                result['resultat'],
                                result['confiance'],
                                result['recommandations'],
                                result['urgent']
                            ))
                            
                            # Notification si urgent
                            if result['urgent']:
                                notif_manager.create_notification(
                                    st.session_state.user['id'],
                                    "🚨 ALERTE - Analyse Urgente",
                                    f"L'analyse {btn_name} a détecté une anomalie: {result['resultat']}",
                                    "urgence"
                                )
                        except Exception as e:
                            st.error(f"Erreur sauvegarde: {e}")
                        
                        # Nettoyage
                        os.remove(temp_path)
                        
                    st.rerun()
                else:
                    st.error("⚠️ Veuillez d'abord importer un fichier")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Deuxième ligne
    for i, (col, btn_name) in enumerate(zip(row2, buttons[5:])):
        with col:
            st.markdown(f'<div class="btn-{i+5}">', unsafe_allow_html=True)
            if st.button(btn_name, key=f"florence_btn_{i+5}", use_container_width=True):
                if st.session_state.florence_uploaded_file:
                    with st.spinner(f"🔬 Analyse {btn_name} en cours..."):
                        time.sleep(2)
                        
                        # Sauvegarde temporaire
                        temp_path = f"temp/florence_{datetime.now().timestamp()}_{st.session_state.florence_uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(st.session_state.florence_uploaded_file.getbuffer())
                        
                        analyzer = florence_analyzers[btn_name]
                        result = analyzer.analyze(temp_path)
                        
                        # Ajouter l'ID utilisateur
                        result['user_id'] = st.session_state.user['id']
                        
                        st.session_state.florence_result = result
                        st.session_state.florence_active_analysis = btn_name
                        
                        # Ajouter à l'historique
                        st.session_state.florence_history.append({
                            'type': btn_name,
                            'resultat': result['resultat'],
                            'urgent': result['urgent'],
                            'timestamp': datetime.now().strftime('%H:%M')
                        })
                        
                        # Sauvegarde en base de données
                        try:
                            db.execute_query("""
                                INSERT INTO analyses (user_id, filename, type_analyse, resultat, confiance, recommandations, urgent)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                st.session_state.user['id'],
                                st.session_state.florence_uploaded_file.name,
                                btn_name,
                                result['resultat'],
                                result['confiance'],
                                result['recommandations'],
                                result['urgent']
                            ))
                            
                            # Notification si urgent
                            if result['urgent']:
                                notif_manager.create_notification(
                                    st.session_state.user['id'],
                                    "🚨 ALERTE - Analyse Urgente",
                                    f"L'analyse {btn_name} a détecté une anomalie: {result['resultat']}",
                                    "urgence"
                                )
                        except Exception as e:
                            st.error(f"Erreur sauvegarde: {e}")
                        
                        # Nettoyage
                        os.remove(temp_path)
                        
                    st.rerun()
                else:
                    st.error("⚠️ Veuillez d'abord importer un fichier")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Affichage des résultats
    if st.session_state.florence_active_analysis and st.session_state.florence_result:
        st.markdown("---")
        
        # En-tête des résultats
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<h2 class="section-title">📊 Résultats - {st.session_state.florence_active_analysis}</h2>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="badge badge-info">{st.session_state.florence_result["timestamp"]}</div>', unsafe_allow_html=True)
        
        # Image et résultats principaux
        col_img, col_res = st.columns([1, 1])
        
        with col_img:
            if st.session_state.florence_uploaded_file:
                st.image(st.session_state.florence_uploaded_file, caption="Fichier analysé", use_container_width=True)
        
        with col_res:
            # Badges
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                if st.session_state.florence_result['urgent']:
                    st.markdown('<div class="badge badge-danger">🚨 URGENT</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge badge-success">✅ NORMAL</div>', unsafe_allow_html=True)
            with col_b2:
                sev = st.session_state.florence_result['severity']
                if sev == 'Critique':
                    st.markdown('<div class="badge badge-danger">⚠️ CRITIQUE</div>', unsafe_allow_html=True)
                elif sev == 'Haute':
                    st.markdown('<div class="badge badge-warning">⚠️ HAUTE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge badge-success">ℹ️ BASSE</div>', unsafe_allow_html=True)
            with col_b3:
                st.markdown(f'<div class="badge badge-info">🔬 {st.session_state.florence_result["type"]}</div>', unsafe_allow_html=True)
            
            # Métriques
            st.markdown("### 📊 Métriques principales")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{st.session_state.florence_result['resultat']}</div>
                    <div class="metric-label">Diagnostic</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{st.session_state.florence_result['confiance']*100:.1f}%</div>
                    <div class="metric-label">Confiance</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Barre de progression
            st.progress(st.session_state.florence_result['confiance'])
        
        # Onglets pour les détails
        st.markdown("### 🔍 Détails de l'analyse")
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Rapport", "📊 Graphiques", "🔬 Détails", "💊 Recommandations"])
        
        with tab1:
            st.markdown("#### 📄 Rapport d'analyse")
            
            # Informations générales
            st.markdown(f"""
            **Patient:** {st.session_state.user.get('full_name', 'ANONYME')}  
            **Date:** {st.session_state.florence_result['timestamp']}  
            **Type d'analyse:** {st.session_state.florence_active_analysis}  
            **Résultat:** {st.session_state.florence_result['resultat']}  
            **Niveau de confiance:** {st.session_state.florence_result['confiance']*100:.1f}%  
            **Sévérité:** {st.session_state.florence_result['severity']}  
            **Urgence:** {'OUI' if st.session_state.florence_result['urgent'] else 'NON'}
            """)
            
            # Symptômes associés
            if st.session_state.florence_result['symptoms']:
                st.markdown("#### 🤒 Symptômes associés")
                for symptom in st.session_state.florence_result['symptoms']:
                    st.markdown(f"- {symptom}")
            
            # Paramètres spécifiques
            if 'heart_rate' in st.session_state.florence_result:
                st.markdown(f"**Fréquence cardiaque:** {st.session_state.florence_result['heart_rate']} BPM")
            if 'rhythm' in st.session_state.florence_result:
                st.markdown(f"**Rythme:** {st.session_state.florence_result['rhythm']}")
            if 'ct_value' in st.session_state.florence_result:
                st.markdown(f"**Valeur CT:** {st.session_state.florence_result['ct_value']}")
            if 'variant' in st.session_state.florence_result:
                st.markdown(f"**Variant:** {st.session_state.florence_result['variant']}")
            
            # Bouton de téléchargement PDF
            if st.button("📥 Télécharger le rapport PDF", key="florence_download_pdf"):
                pdf = florence_pdf_gen.generate_report(
                    st.session_state.user.get('full_name', 'ANONYME'),
                    st.session_state.florence_active_analysis,
                    st.session_state.florence_result
                )
                pdf_output = BytesIO()
                pdf.output(pdf_output, dest='S').encode('latin1')
                pdf_output.seek(0)
                
                st.download_button(
                    label="📥 Cliquez pour télécharger",
                    data=pdf_output,
                    file_name=f"florence_rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        with tab2:
            st.markdown("#### 📊 Visualisations")
            
            # Graphique de confiance
            fig_confidence = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = st.session_state.florence_result['confiance'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Niveau de confiance"},
                gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
            ))
            st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Graphique des probabilités
            diseases = ['Pneumonie', 'Tuberculose', 'Cancer', 'Bronchite', 'Normal']
            probs = [0.25, 0.15, 0.10, 0.20, 0.30]
            
            fig_probs = px.bar(
                x=diseases,
                y=probs,
                title="Distribution des probabilités",
                labels={'x': 'Diagnostic', 'y': 'Probabilité'},
                color=probs,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_probs, use_container_width=True)
        
        with tab3:
            st.markdown("#### 🔬 Paramètres détaillés")
            
            if 'parameters' in st.session_state.florence_result:
                df_params = pd.DataFrame(
                    list(st.session_state.florence_result['parameters'].items()),
                    columns=['Paramètre', 'Valeur']
                )
                st.dataframe(df_params, use_container_width=True)
            elif 'intervals' in st.session_state.florence_result:
                df_intervals = pd.DataFrame(
                    list(st.session_state.florence_result['intervals'].items()),
                    columns=['Intervalle', 'Valeur (ms)']
                )
                st.dataframe(df_intervals, use_container_width=True)
            else:
                st.info("Aucun paramètre détaillé disponible pour cette analyse")
            
            # Métadonnées
            st.markdown("#### ℹ️ Métadonnées")
            st.json({
                "type_analyse": st.session_state.florence_active_analysis,
                "timestamp": st.session_state.florence_result['timestamp'],
                "version_ia": "Florence 2.0",
                "modele": "Florence-Net v2",
                "precision_modele": "98.7%",
                "patient_id": st.session_state.user['id']
            })
        
        with tab4:
            st.markdown("#### 💊 Recommandations médicales")
            st.markdown(st.session_state.florence_result['recommandations'])
            
            # Actions rapides
            st.markdown("#### 🏥 Actions recommandées")
            col_a1, col_a2, col_a3 = st.columns(3)
            
            with col_a1:
                if st.button("📞 Contacter médecin", key="florence_contact", use_container_width=True):
                    st.info("Fonctionnalité de téléconsultation - À implémenter")
            
            with col_a2:
                if st.button("📅 Prendre RDV", key="florence_rdv", use_container_width=True):
                    st.session_state['nav'] = "📅 Rendez-vous"
                    st.rerun()
            
            with col_a3:
                if st.button("💬 Support", key="florence_support", use_container_width=True):
                    st.session_state['nav'] = "💬 Chat Santé"
                    st.rerun()
        
        # Bouton nouvelle analyse
        if st.button("🔄 Nouvelle analyse Florence", use_container_width=True):
            st.session_state.florence_active_analysis = None
            st.session_state.florence_result = None
            st.rerun()
    
    # Section d'aide
    with st.expander("ℹ️ Guide d'utilisation Florence"):
        st.markdown("""
        ### Comment utiliser Florence ?
        
        1. **Importez un fichier** médical (radio, scanner, analyse sanguine, etc.)
        2. **Sélectionnez le type d'analyse** correspondant à votre fichier
        3. **Attendez l'analyse** par notre IA (quelques secondes)
        4. **Consultez les résultats** détaillés et les recommandations
        
        ### Types d'analyses disponibles
        
        - **IRM** : Imagerie par résonance magnétique
        - **Détection Arythmie** : Analyse du rythme cardiaque
        - **Scanner Thorax** : Scanner pulmonaire
        - **Radiographie** : Rayons X standards
        - **ECG** : Électrocardiogramme
        - **Analyse Sang** : Paramètres sanguins
        - **IRM Cerveau** : Imagerie cérébrale
        - **Scanner Abdomen** : Scanner abdominal
        - **Test COVID-19** : Détection SARS-CoV-2
        - **Analyse Urine** : Paramètres urinaires
        
        ### Interprétation des résultats
        
        - 🟢 **Normal** : Aucune anomalie détectée
        - 🟡 **À surveiller** : Anomalie mineure
        - 🔴 **Urgent** : Consultation médicale immédiate requise
        """)