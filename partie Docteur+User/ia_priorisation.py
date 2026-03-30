# ia_priorisation.py
"""
Module d'IA pour la priorisation automatique des patients
Basé sur l'analyse des données médicales en temps réel
"""

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class PatientPriorisationIA:
    """
    Moteur d'IA pour calculer le score de priorité des patients
    """
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        self.weights = {
            'urgence_message': 50,
            'analyse_critique': 40,
            'rdv_aujourdhui': 30,
            'symptomes_graves': 35,
            'absence_suivi': 10,
            'antecedents_graves': 20,
            'age_superieur_65': 15,
            'pathologie_chronique': 25
        }
        
        # Mots-clés d'urgence dans les messages
        self.urgence_keywords = [
            'douleur poitrine', 'infarctus', 'avc', 'paralysie',
            'difficulté respiratoire', 'étouffe', 'inconscient',
            'crise', 'convulsion', 'saignement', 'traumatisme',
            'brûlure grave', 'intoxication', 'overdose'
        ]
        
        # Symptômes graves
        self.symptomes_graves = [
            'douleur thoracique', 'essoufflement', 'fièvre élevée',
            'vomissements sanglants', 'perte connaissance',
            'confusion', 'vision trouble'
        ]
    
    def calculate_patient_score(self, patient_data: Dict) -> Dict:
        """
        Calcule le score de priorité pour un patient
        
        Args:
            patient_data: Dictionnaire contenant toutes les données du patient
            
        Returns:
            Dict avec score total et détails
        """
        score = 0
        details = {}
        
        # 1. Vérifier les messages urgents aujourd'hui
        if self.check_urgence_message(patient_data):
            score += self.weights['urgence_message']
            details['urgence_message'] = self.weights['urgence_message']
        
        # 2. Vérifier les analyses critiques récentes
        analyse_score = self.check_analyses_critiques(patient_data)
        if analyse_score > 0:
            score += analyse_score
            details['analyse_critique'] = analyse_score
        
        # 3. Vérifier les rendez-vous aujourd'hui
        if self.check_rdv_aujourdhui(patient_data):
            score += self.weights['rdv_aujourdhui']
            details['rdv_aujourdhui'] = self.weights['rdv_aujourdhui']
        
        # 4. Analyser les symptômes graves dans les messages
        symptomes_score = self.check_symptomes_graves(patient_data)
        if symptomes_score > 0:
            score += symptomes_score
            details['symptomes_graves'] = symptomes_score
        
        # 5. Vérifier l'absence de suivi prolongée
        absence_score = self.check_absence_suivi(patient_data)
        if absence_score > 0:
            score += absence_score
            details['absence_suivi'] = absence_score
        
        # 6. Vérifier les antécédents graves
        antecedents_score = self.check_antecedents_graves(patient_data)
        if antecedents_score > 0:
            score += antecedents_score
            details['antecedents_graves'] = antecedents_score
        
        # 7. Facteur âge
        if self.check_age_superieur_65(patient_data):
            score += self.weights['age_superieur_65']
            details['age_superieur_65'] = self.weights['age_superieur_65']
        
        # 8. Pathologies chroniques
        if self.check_pathologie_chronique(patient_data):
            score += self.weights['pathologie_chronique']
            details['pathologie_chronique'] = self.weights['pathologie_chronique']
        
        # Déterminer le niveau de priorité
        priorite = self.get_priority_level(score)
        
        return {
            'patient_id': patient_data.get('id'),
            'score_total': score,
            'priorite': priorite,
            'details': details,
            'timestamp': datetime.now()
        }
    
    def check_urgence_message(self, patient_data: Dict) -> bool:
        """Vérifie s'il y a des messages urgents aujourd'hui"""
        messages = patient_data.get('messages_aujourdhui', [])
        for msg in messages:
            if msg.get('urgent'):
                return True
        return False
    
    def check_analyses_critiques(self, patient_data: Dict) -> int:
        """Vérifie les analyses critiques et retourne le score"""
        analyses = patient_data.get('analyses_recentes', [])
        max_score = 0
        
        for ana in analyses:
            if ana.get('urgent'):
                # Plus l'analyse est récente, plus le score est élevé
                jours_ecoules = ana.get('jours_ecoules', 0)
                if jours_ecoules <= 1:
                    max_score = max(max_score, self.weights['analyse_critique'])
                elif jours_ecoules <= 3:
                    max_score = max(max_score, self.weights['analyse_critique'] * 0.7)
                elif jours_ecoules <= 7:
                    max_score = max(max_score, self.weights['analyse_critique'] * 0.4)
        
        return int(max_score)
    
    def check_rdv_aujourdhui(self, patient_data: Dict) -> bool:
        """Vérifie si le patient a un rendez-vous aujourd'hui"""
        rdvs = patient_data.get('rdv_aujourdhui', [])
        return len(rdvs) > 0
    
    def check_symptomes_graves(self, patient_data: Dict) -> int:
        """Analyse les symptômes dans les messages récents"""
        messages = patient_data.get('messages_recents', [])
        score = 0
        
        for msg in messages:
            contenu = msg.get('contenu', '').lower()
            for symptome in self.symptomes_graves:
                if symptome in contenu:
                    score += 15  # Score progressif
                    break
            
            # Bonus pour multiples symptômes
            if score > 30:
                score = min(score, 35)  # Plafond à 35
        
        return min(score, self.weights['symptomes_graves'])
    
    def check_absence_suivi(self, patient_data: Dict) -> int:
        """Vérifie la durée depuis le dernier contact"""
        dernier_contact = patient_data.get('dernier_contact')
        if not dernier_contact:
            return self.weights['absence_suivi']
        
        jours_ecoules = (datetime.now() - dernier_contact).days
        
        if jours_ecoules > 60:
            return self.weights['absence_suivi']
        elif jours_ecoules > 30:
            return int(self.weights['absence_suivi'] * 0.7)
        elif jours_ecoules > 15:
            return int(self.weights['absence_suivi'] * 0.3)
        
        return 0
    
    def check_antecedents_graves(self, patient_data: Dict) -> int:
        """Vérifie les antécédents médicaux graves"""
        antecedents = patient_data.get('antecedents', '').lower()
        pathologies_graves = [
            'infarctus', 'avc', 'cancer', 'diabète', 'insuffisance rénale',
            'insuffisance cardiaque', 'hépatite', 'sida', 'transplantation'
        ]
        
        for pathologie in pathologies_graves:
            if pathologie in antecedents:
                return self.weights['antecedents_graves']
        
        return 0
    
    def check_age_superieur_65(self, patient_data: Dict) -> bool:
        """Vérifie si le patient a plus de 65 ans"""
        age = patient_data.get('age', 0)
        return age >= 65
    
    def check_pathologie_chronique(self, patient_data: Dict) -> bool:
        """Vérifie la présence de pathologies chroniques"""
        maladies = patient_data.get('maladies_chroniques', '').lower()
        pathologies_chroniques = [
            'diabète', 'hypertension', 'asthme', 'bpco', 'insuffisance',
            'arthrite', 'parkinson', 'alzheimer', 'épilepsie'
        ]
        
        for pathologie in pathologies_chroniques:
            if pathologie in maladies:
                return True
        
        return False
    
    def get_priority_level(self, score: int) -> str:
        """Détermine le niveau de priorité basé sur le score"""
        if score >= 80:
            return "🔴 Urgent"
        elif score >= 50:
            return "🟠 À voir vite"
        else:
            return "🟢 Stable"
    
    def batch_calculate_scores(self, patients_list: List[Dict]) -> List[Dict]:
        """
        Calcule les scores pour une liste de patients
        
        Args:
            patients_list: Liste des dictionnaires de données patients
            
        Returns:
            Liste des résultats avec scores
        """
        results = []
        for patient in patients_list:
            try:
                score_result = self.calculate_patient_score(patient)
                results.append(score_result)
            except Exception as e:
                logger.error(f"Erreur calcul score patient {patient.get('id')}: {e}")
                results.append({
                    'patient_id': patient.get('id'),
                    'score_total': 0,
                    'priorite': "🟢 Stable",
                    'details': {},
                    'error': str(e)
                })
        
        # Trier par score décroissant
        results.sort(key=lambda x: x['score_total'], reverse=True)
        return results
    
    def get_prioritized_patients_for_doctor(self, doctor_id: int, limit: int = 50) -> List[Dict]:
        """
        Récupère les patients priorisés pour un médecin spécifique
        
        Args:
            doctor_id: ID du médecin
            limit: Nombre maximum de patients à retourner
            
        Returns:
            Liste des patients avec leur priorité
        """
        if not self.db:
            logger.error("Pas de connexion base de données")
            return []
        
        try:
            # Récupérer les patients du médecin
            query = """
            SELECT 
                u.id,
                u.full_name,
                u.age,
                u.ville,
                u.statut_suivi,
                u.allergies,
                u.maladies_chroniques,
                MAX(m.timestamp) as dernier_message,
                COUNT(CASE WHEN m.urgent = TRUE AND DATE(m.timestamp) = CURDATE() THEN 1 END) as messages_urgents,
                COUNT(CASE WHEN a.urgent = TRUE AND a.timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 END) as analyses_urgentes,
                COUNT(CASE WHEN c.date_rdv = CURDATE() THEN 1 END) as rdv_aujourdhui
            FROM users u
            LEFT JOIN messages m ON u.id = m.user_id
            LEFT JOIN analyses a ON u.id = a.user_id
            LEFT JOIN consultations c ON u.id = c.patient_id
            WHERE u.role = 'patient' 
                AND u.actif = TRUE
                AND (c.medecin_id = %s OR EXISTS (
                    SELECT 1 FROM messages WHERE user_id = u.id AND urgent = TRUE
                ))
            GROUP BY u.id
            """
            
            patients_data = self.db.execute_query(query, (doctor_id,), fetch_all=True)
            
            # Transformer les données pour l'IA
            patients_processed = []
            for p in patients_data:
                patient_dict = {
                    'id': p['id'],
                    'full_name': p['full_name'],
                    'age': p['age'],
                    'ville': p['ville'],
                    'maladies_chroniques': p['maladies_chroniques'],
                    'messages_aujourdhui': [{'urgent': p['messages_urgents'] > 0}],
                    'analyses_recentes': [{'urgent': p['analyses_urgentes'] > 0, 'jours_ecoules': 1}],
                    'rdv_aujourdhui': [{}] if p['rdv_aujourdhui'] > 0 else [],
                    'dernier_contact': p['dernier_message']
                }
                patients_processed.append(patient_dict)
            
            # Calculer les scores
            scored_patients = self.batch_calculate_scores(patients_processed)
            
            # Fusionner avec les données originales
            result = []
            for score_result in scored_patients:
                patient_original = next(
                    (p for p in patients_data if p['id'] == score_result['patient_id']), 
                    {}
                )
                result.append({
                    **patient_original,
                    'score_ia': score_result['score_total'],
                    'priorite': score_result['priorite'],
                    'details_ia': score_result['details']
                })
            
            return result[:limit]
            
        except Exception as e:
            logger.error(f"Erreur get_prioritized_patients: {e}")
            return []

# ========== EXEMPLE D'UTILISATION ==========

def test_ia_priorisation():
    """Test du module d'IA avec des données simulées"""
    
    ia = PatientPriorisationIA()
    
    # Données de test
    test_patients = [
        {
            'id': 1,
            'full_name': 'Ahmed Benchekroun',
            'age': 70,
            'messages_aujourdhui': [{'urgent': True, 'contenu': 'douleur poitrine intense'}],
            'analyses_recentes': [{'urgent': True, 'jours_ecoules': 1}],
            'rdv_aujourdhui': [{'date': datetime.now()}],
            'dernier_contact': datetime.now() - timedelta(days=1),
            'antecedents': 'infarctus en 2020',
            'maladies_chroniques': 'diabète, hypertension'
        },
        {
            'id': 2,
            'full_name': 'Fatima Zahra',
            'age': 45,
            'messages_aujourdhui': [],
            'analyses_recentes': [],
            'rdv_aujourdhui': [],
            'dernier_contact': datetime.now() - timedelta(days=45),
            'antecedents': '',
            'maladies_chroniques': ''
        },
        {
            'id': 3,
            'full_name': 'Karim Mansouri',
            'age': 35,
            'messages_aujourdhui': [{'urgent': False, 'contenu': 'fièvre et toux'}],
            'analyses_recentes': [{'urgent': False, 'jours_ecoules': 5}],
            'rdv_aujourdhui': [],
            'dernier_contact': datetime.now() - timedelta(days=20),
            'antecedents': 'asthme',
            'maladies_chroniques': 'asthme'
        }
    ]
    
    # Calculer les scores
    results = ia.batch_calculate_scores(test_patients)
    
    print("=== RÉSULTATS IA PRIORISATION ===")
    for r in results:
        print(f"\nPatient {r['patient_id']}:")
        print(f"  Score: {r['score_total']}")
        print(f"  Priorité: {r['priorite']}")
        print(f"  Détails: {r['details']}")

if __name__ == "__main__":
    test_ia_priorisation()