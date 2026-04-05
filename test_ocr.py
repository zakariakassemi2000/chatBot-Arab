import sys
from PIL import Image
from engine.ocr_ordonnance import get_ocr

# Disable verify online for faster testing
ocr = get_ocr(use_donut=False, verify_online=False)

img_path = r"C:\Users\zakar\Downloads\WhatsApp Image 2024-07-06 at 15.01.20.jpeg"
try:
    img = Image.open(img_path)
    res = ocr.analyser(img)
    print("----- TEXTE BRUT -----")
    print(res.texte_brut)
    print("----- EXTRACTS -----")
    for m in res.medicaments:
        print(f"Brut: {m.nom_brut}")
        print(f"  DCI: {m.dci} | NomCom: {m.nom_commercial} | Dosage: {m.dosage}")
except Exception as e:
    import traceback
    traceback.print_exc()
