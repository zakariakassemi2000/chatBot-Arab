# florence_analyzer.py

import random
from PIL import Image
from datetime import datetime

class FlorenceBaseAnalyzer:
    def __init__(self):
        self.diseases = {
            'pneumonie': {
                'name': 'Pneumonie',
                'symptoms': ['Toux', 'Fièvre', 'Difficulté respiratoire', 'Douleur thoracique'],
                'severity': 'Haute',
                'urgence': True,
                'color': '#ff4444',
                'probability': 0.25
            },
            'tuberculose': {
                'name': 'Tuberculose',
                'symptoms': ['Toux chronique', 'Fatigue', 'Perte de poids', 'Sueurs nocturnes'],
                'severity': 'Haute',
                'urgence': True,
                'color': '#ff6b6b',
                'probability': 0.15
            },
            'cancer': {
                'name': 'Cancer du poumon',
                'symptoms': ['Toux sanglante', 'Douleur thoracique', 'Essoufflement', 'Fatigue'],
                'severity': 'Critique',
                'urgence': True,
                'color': '#ff1744',
                'probability': 0.10
            },
            'bronchite': {
                'name': 'Bronchite',
                'symptoms': ['Toux grasse', 'Expectorations', 'Fatigue', 'Fièvre légère'],
                'severity': 'Moyenne',
                'urgence': False,
                'color': '#ffa726',
                'probability': 0.20
            },
            'normal': {
                'name': 'Normal',
                'symptoms': [],
                'severity': 'Basse',
                'urgence': False,
                'color': '#00C851',
                'probability': 0.30
            }
        }
    
    def analyze(self, image_path):
        try:
            # Simulation d'analyse
            img = Image.open(image_path)
            
            # Sélection aléatoire pondérée
            diseases = list(self.diseases.keys())
            probabilities = [self.diseases[d]['probability'] for d in diseases]
            
            # Ajustement des probabilités selon le type d'analyse
            probabilities = self.adjust_probabilities(probabilities)
            
            resultat = random.choices(diseases, probabilities)[0]
            confiance = round(random.uniform(0.75, 0.99), 3)
            
            disease_info = self.diseases[resultat]
            
            return {
                'resultat': disease_info['name'],
                'code': resultat,
                'confiance': confiance,
                'recommandations': self.generate_recommendations(resultat, confiance),
                'urgent': disease_info['urgence'],
                'severity': disease_info['severity'],
                'symptoms': disease_info['symptoms'],
                'type': self.get_type_name(),
                'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            }
        except Exception as e:
            return {
                'resultat': 'Erreur',
                'code': 'error',
                'confiance': 0,
                'recommandations': f"Erreur d'analyse: {str(e)}",
                'urgent': False,
                'severity': 'Inconnue',
                'symptoms': [],
                'type': self.get_type_name(),
                'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            }
    
    def adjust_probabilities(self, probabilities):
        """À surcharger par les classes filles"""
        return probabilities
    
    def get_type_name(self):
        return "Analyse Générale"
    
    def generate_recommendations(self, disease, confiance):
        recommendations = {
            'pneumonie': f"""
### 🏥 RECOMMANDATIONS POUR PNEUMONIE (Confiance: {confiance:.1%})

**URGENCE MÉDICALE - CONSULTATION IMMÉDIATE REQUISE**

1. **Consultation médicale** : Rendez-vous en urgence chez un médecin ou aux urgences
2. **Traitement** : Antibiothérapie probable (à adapter après consultation)
3. **Repos** : Repos absolu pendant 48-72h
4. **Hydratation** : Boire 1.5-2L d'eau par jour
5. **Surveillance** :
   - Température toutes les 4h
   - Fréquence respiratoire
   - Couleur des expectorations

⚠️ **SIGNES D'ALARME** : Difficultés respiratoires, douleur thoracique intense, fièvre >39°C
""",
            'tuberculose': f"""
### ⚠️ RECOMMANDATIONS POUR SUSPICION DE TUBERCULOSE (Confiance: {confiance:.1%})

**MESURES D'ISOLEMENT ET CONSULTATION URGENTE**

1. **Isolement** : Limiter les contacts familiaux, porter un masque
2. **Diagnostic** : Test de confirmation (bacilloscopie, PCR GeneXpert)
3. **Traitement** : Traitement antibacillaire à débuter rapidement
4. **Dépistage** : Dépistage des contacts familiaux
5. **Suivi** : Suivi rapproché par un pneumologue

📞 **CONTACTS UTILES** :
- Centre de lutte antituberculeux : 05 22 23 45 67
- Pneumologue de garde : 05 22 23 45 68
""",
            'cancer': f"""
### 🚨 ALERTE CRITIQUE - SUSPICION DE CANCER (Confiance: {confiance:.1%})

**ACTION IMMÉDIATE REQUISE - URGENCE ONCOLOGIQUE**

1. **Consultation** : Rendez-vous URGENT en oncologie (délai < 48h)
2. **Examens complémentaires** :
   - Biopsie guidée
   - TEP-scan
   - Bilan d'extension complet
3. **Support** :
   - Consultation psychologique
   - Information à la famille
   - Mise en place rapide du traitement

📋 **DOCUMENTS À APPORTER** :
- Pièce d'identité
- Carte de santé
- Antécédents médicaux
- Traitements en cours

📞 **URGENCES ONCOLOGIQUES 24h/24** : 05 22 23 45 69
""",
            'bronchite': f"""
### 📋 RECOMMANDATIONS POUR BRONCHITE (Confiance: {confiance:.1%})

**TRAITEMENT AMBULATOIRE ET SURVEILLANCE**

1. **Repos** : 2-3 jours de repos à domicile
2. **Hydratation** : Boire abondamment (tisanes, eau)
3. **Traitement symptomatique** :
   - Paracétamol si fièvre (>38.5°C)
   - Sirop pour la toux si nécessaire
   - Humidifier l'air ambiant
4. **Surveillance** : 48-72h

⚠️ **CONSULTEZ EN URGENCE SI** :
- Fièvre > 39°C persistante > 48h
- Difficultés respiratoires
- Expectorations sanglantes
- Douleur thoracique
""",
            'normal': f"""
### ✅ RÉSULTAT NORMAL (Confiance: {confiance:.1%})

**AUCUNE ANOMALIE DÉTECTÉE**

L'analyse n'a révélé aucune anomalie significative.

**RECOMMANDATIONS GÉNÉRALES** :
- Maintenir une hygiène de vie saine
- Alimentation équilibrée
- Activité physique régulière
- Sommeil suffisant

**PROCHAIN CONTRÔLE** : Dans 1 an ou selon recommandations de votre médecin traitant

📅 **SI SYMPTÔMES PERSISTANTS** : Consultez votre médecin traitant
"""
        }
        return recommendations.get(disease, "Consultation médicale recommandée")

class FlorenceIRMAnalyzer(FlorenceBaseAnalyzer):
    def get_type_name(self):
        return "IRM"
    
    def adjust_probabilities(self, probabilities):
        return [p * 1.1 if i != 4 else p * 0.7 for i, p in enumerate(probabilities)]

class FlorenceArythmieAnalyzer(FlorenceBaseAnalyzer):
    def get_type_name(self):
        return "Détection Arythmie"
    
    def analyze(self, image_path):
        result = super().analyze(image_path)
        result['heart_rate'] = random.randint(60, 180)
        result['rhythm'] = random.choice(['Sinusal', 'Tachycardie', 'Bradycardie', 'Fibrillation'])
        return result

class FlorenceScannerThoraxAnalyzer(FlorenceBaseAnalyzer):
    def get_type_name(self):
        return "Scanner Thorax"
    
    def adjust_probabilities(self, probabilities):
        return [p * 1.2 if i == 0 else p * 0.9 for i, p in enumerate(probabilities)]

class FlorenceRadiographieAnalyzer(FlorenceBaseAnalyzer):
    def get_type_name(self):
        return "Radiographie"

class FlorenceECGAnalyzer(FlorenceBaseAnalyzer):
    def get_type_name(self):
        return "ECG"
    
    def analyze(self, image_path):
        result = super().analyze(image_path)
        result['heart_rate'] = random.randint(55, 120)
        result['rhythm'] = random.choice(['Sinusal', 'Arythmie sinusale', 'Tachycardie sinusale'])
        result['intervals'] = {
            'PR': random.randint(120, 200),
            'QRS': random.randint(70, 110),
            'QT': random.randint(350, 450)
        }
        return result

class FlorenceSangAnalyzer(FlorenceBaseAnalyzer):
    def get_type_name(self):
        return "Analyse Sang"
    
    def analyze(self, image_path):
        result = super().analyze(image_path)
        result['parameters'] = {
            'Globules rouges': round(random.uniform(4.0, 6.0), 2),
            'Globules blancs': round(random.uniform(4.0, 11.0), 2),
            'Plaquettes': random.randint(150, 450),
            'Hémoglobine': round(random.uniform(12, 18), 1),
            'Hématocrite': round(random.uniform(36, 48), 1),
            'VGM': random.randint(80, 100)
        }
        return result

class FlorenceIRMCerveauAnalyzer(FlorenceBaseAnalyzer):
    def get_type_name(self):
        return "IRM Cerveau"
    
    def adjust_probabilities(self, probabilities):
        return [p * 1.3 if i == 2 else p * 0.8 for i, p in enumerate(probabilities)]

class FlorenceScannerAbdomenAnalyzer(FlorenceBaseAnalyzer):
    def get_type_name(self):
        return "Scanner Abdomen"

class FlorenceCovidAnalyzer(FlorenceBaseAnalyzer):
    def get_type_name(self):
        return "Test COVID-19"
    
    def adjust_probabilities(self, probabilities):
        probs = list(probabilities)
        probs[4] *= 1.5  # normal
        probs[0] *= 0.5  # pneumonie
        return probs
    
    def analyze(self, image_path):
        result = super().analyze(image_path)
        result['ct_value'] = random.randint(15, 40)
        result['variant'] = random.choice(['Omicron', 'Delta', 'Non détecté'])
        return result

class FlorenceUrineAnalyzer(FlorenceBaseAnalyzer):
    def get_type_name(self):
        return "Analyse Urine"
    
    def analyze(self, image_path):
        result = super().analyze(image_path)
        result['parameters'] = {
            'pH': round(random.uniform(4.5, 8.0), 1),
            'Densité': round(random.uniform(1.005, 1.030), 3),
            'Protéines': random.choice(['Négatif', 'Traces', '+', '++']),
            'Glucose': random.choice(['Négatif', 'Positif']),
            'Leucocytes': random.choice(['Négatif', 'Positif']),
            'Nitrites': random.choice(['Négatif', 'Positif'])
        }
        return result