from modules.ordonnance_scanner import MedicamentExtrait, _PATTERNS, load_medicaments_db
import re
from typing import List, Dict, Optional
from rapidfuzz import process, fuzz

def fuzzy_match_medication(name: str, db: List[Dict], threshold: int = 75) -> Optional[Dict]:
    if not db or not name: return None
    db_names = [m["nom"] for m in db]

    result = process.extractOne(name, db_names, scorer=fuzz.ratio, score_cutoff=threshold)
    if not result:
        result = process.extractOne(name, db_names, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)

    if result:
        matched_name, score, idx = result
        med_data = db[idx]
        return {
            "nom": med_data["nom"],
            "principe": med_data["principe"],
            "formes": med_data["formes"],
            "score": score
        }
    return None

def extract_medications(text: str) -> List[MedicamentExtrait]:
    medicaments = []
    db = load_medicaments_db()

    lines = [L.strip() for L in text.split('\n') if L.strip()]
    re_bullet = re.compile(r'^([a-zA-Z0-9]{1,2}[\)\.\-]|[-•●\*>])\s*')
    re_dosage = re.compile(r'\b(\d+(?:[.,]\d+)?\s*(?:mg|g|ml|µg|ui|cp|suppo|sachet|g[eé]l|amp|inj|iu))\b', re.I)
    re_dosage_ocr = re.compile(r'\b(\d+\s*(?:-g|ig))\b', re.I)

    for line in lines:
        if len(line) < 4 or any(kw in line.lower() for kw in ['docteur', 'dr.', 'tél', 'tel', 'hopital', 'maladie', 'clinique', 'urgence', 'chef', 'rhumatologue', 'patient', 'rdv', 'rendez-vous']):
            continue

        clean_line = re_bullet.sub('', line).strip()
        dosage_match = re_dosage.search(clean_line) or re_dosage_ocr.search(clean_line)

        words = clean_line.split()
        first_word = words[0].lower() if words else ""
        first_two = " ".join(words[:2]).lower() if len(words) > 1 else ""

        match_result = None
        m1 = m2 = None
        if db:
            m2 = fuzzy_match_medication(first_two, db) if len(first_two) > 3 else None
            m1 = fuzzy_match_medication(first_word, db) if len(first_word) > 3 else None
            match_result = m2 if m2 else m1

        if not dosage_match and not match_result:
            has_other_attr = _PATTERNS["forme"].search(clean_line) or _PATTERNS["frequence"].search(clean_line)
            if not has_other_attr:
                continue

        if dosage_match:
            nom_brut = clean_line[:dosage_match.start()].strip()
            nom_brut = re.sub(r'[^A-Za-zÀ-ÿéèêëîïôùûü0-9\s-]', '', nom_brut).strip()
            if not nom_brut and words:
                nom_brut = words[0]
        else:
            nom_brut = first_two if match_result == m2 else first_word
            nom_brut = nom_brut.capitalize()

        if not nom_brut or len(nom_brut) < 3:
            continue

        med = MedicamentExtrait(nom_brut=nom_brut)
        if dosage_match:
            med.dosage = dosage_match.group(1).strip()

        forme_m = _PATTERNS["forme"].search(clean_line)
        if forme_m: med.forme = forme_m.group(0).strip().lower()

        freq_m = _PATTERNS["frequence"].search(clean_line)
        if freq_m: med.frequence = freq_m.group(0).strip()

        duree_m = _PATTERNS["duree"].search(clean_line)
        if duree_m: med.duree = duree_m.group(0).strip()

        if match_result:
            med.nom_match = match_result["nom"]
            med.principe_actif = match_result["principe"]
            med.formes_disponibles = match_result["formes"]
            med.score_match = match_result["score"]
            med.est_reference = True

        medicaments.append(med)

    return medicaments

text = """RHUMATOLOGUE
aladies des os, articulations et de
la colonne vértébrale
l
Médecin chef de l'hôpital Ibn Baytar
05.35.94.21.75
Tél.: 05.35.94.21.75
06.68.75.69.44
En Cas d'Urgent : 06.68.75.69.44
One Jeffruly El Kamlo
Fès, le 31103/2026:
) pirocom inj + cottrex ig
141063289
(em essociation)
Minj /joun pedat O6jours
puis
a 3 piscom 20mg
141063289
despiequres = Peduade)
3≤PP 20-9(5128)
68 756944
1gellj Ce muat- è jem
Vuteevrel f2t
hles
55
1-1- 1"""

import pprint
res = extract_medications(text)
for r in res:
    print(r.nom_brut, "| Match:", r.nom_match, r.score_match, "| Dos:", r.dosage, "| Frme:", r.forme, "| Freq:", r.frequence)
