from modules.ordonnance_scanner import extract_medications
import rapidfuzz

text = """Docteur ALAMI Abdelaziz
RHUMATOLOGUE
Maladies des os, Articulations et de
la colonne vértébrale
Médecin Chef de l'Hôpital  Ibn Bajja
Tél : 05.35.61.21.78
En Cas d'Urgence : 06.68.75.69.49

Fes le 31/03/2016
Mme Jalliouly

A) piroxicam 20mg
(en reservation)
1inj / jour pendant 03 jours

2) Piroxicam 20mg
NB: 1 suppo / jour

3) IPP 20 mg
1gel / j (le matin à jeun)

4) Voltarene Fort
a
"""

extracted = extract_medications(text)
print(f"Trouvé {len(extracted)} medicaments:")
for m in extracted:
    print(f" - Brut: {repr(m.nom_brut)}, Match: {m.nom_match} ({m.score_match}), Dosage: {m.dosage}, Forme: {m.forme}, Freq: {m.frequence}")
