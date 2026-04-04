import re
import json
from pathlib import Path
from rapidfuzz import process, fuzz

def extract_medications_new(text):
    # Load DB
    db_path = Path('data/medicaments_maroc.json')
    if db_path.exists():
        with open(db_path, "r", encoding="utf-8") as f:
            db_medicaments = json.load(f)
    else:
        db_medicaments = []
        
    db_names = [m["nom"].lower() for m in db_medicaments]

    lines = [L.strip() for L in text.split('\n') if L.strip()]
    results = []
    
    # Regex for bullets/numbers at start
    re_bullet = re.compile(r'^([a-zA-Z0-9]{1,2}[\)\.\-]|[-•●\*>])\s*')
    
    # Regex dosages
    re_dosage = re.compile(r'\b(?:\d+(?:[.,]\d+)?\s*(?:mg|g|ml|µg|ui|cp|suppo|sachet|gél|amp|inj))\b', re.I)
    
    # Custom tweaks for OCR mistakes
    re_dosage_ocr = re.compile(r'\b(?:\d+\s*(?:-g|ig))\b', re.I)

    for line in lines:
        # Ignore lines like "Docteur...", "Tél:", "Fes le"
        if len(line) < 4 or any(kw in line.lower() for kw in ['docteur', 'dr.', 'tél', 'tel', 'hopital', 'maladie', 'clinique', 'urgence', 'chef']):
            continue
            
        clean_line = re_bullet.sub('', line)
        
        # Est-ce une ligne de médicament ?
        # Condition 1: Contient un dosage
        dosage_match = re_dosage.search(clean_line) or re_dosage_ocr.search(clean_line)
        
        # Condition 2: Le premier mot correspond à un médoc en base
        first_word = clean_line.split()[0].lower() if clean_line.split() else ""
        db_match = process.extractOne(first_word, db_names, scorer=fuzz.ratio, score_cutoff=75)
        
        # On peut aussi vérifier les 2 premiers mots
        first_two_words = " ".join(clean_line.split()[:2]).lower() if len(clean_line.split()) > 1 else ""
        db_match_2 = process.extractOne(first_two_words, db_names, scorer=fuzz.ratio, score_cutoff=80)
        
        best_db_match = db_match_2 if db_match_2 else db_match
        
        # Si ni dosage ni match DB, on passe
        if not dosage_match and not best_db_match:
            continue
            
        # Extraction du nom du médoc
        # Si on a un match DB, on prend le match
        nom_match = best_db_match[0].capitalize() if best_db_match else None
        score_match = best_db_match[1] if best_db_match else 0.0
        
        # Le nom brut = les premiers mots avant le dosage, ou jusqu'à 2 mots
        if dosage_match:
            idx = dosage_match.start()
            nom_brut = clean_line[:idx].strip()
            # Nettoyer nom_brut
            nom_brut = re.sub(r'[^A-Za-zÀ-ÿéèêëîïôùûü0-9\s-]', '', nom_brut).strip()
            if not nom_brut: # Le dosage est au tout début ? bizarre
                nom_brut = clean_line.split()[0]
        else:
            # S'il y a un match DB mais pas de dosage (ex: Voltarene Fort)
            parts = clean_line.split()
            nom_brut = parts[0]
            if len(parts) > 1 and len(parts[1]) > 2:
                nom_brut += " " + parts[1]
                
        # Dosage
        dosage = dosage_match.group(0) if dosage_match else None
        
        results.append({
            "line": line,
            "nom_brut": nom_brut,
            "nom_match": nom_match,
            "score": score_match,
            "dosage": dosage
        })
        
    return results

text = """RHUMATOLOGUE
aladies des os, articulations et de
la colonne vértébrale
Médecin chef de l'hôpital Ibn Baytar
05.35.94.21.75
Tél.: 05.35.94.21.75
En Cas d'Urgent : 06.68.75.69.44
One Jeffruly El Kamlo
Fès, le 31103/2026:
) pirocom inj + cottrex ig
(em essociation)
Minj /joun pedat O6jours
puis
a 3 piscom 20mg
despiequres = Peduade)
3≤PP 20-9(5128)
1gellj Ce muat- è jem
Vuteevrel f2t
"""

import pprint
pprint.pprint(extract_medications_new(text))
