import sys
from PIL import Image
from engine.ocr_ordonnance import OCREngine
import numpy as np

img = Image.open('WhatsApp Image 2026-04-04 at 13.21.38.jpeg')
ocr = OCREngine()
# test standard _extract
print("--- NORMAL (with preprocessing) ---")
txt, conf = ocr.extract_text(img)
print(txt)

print("\n--- PURE RGB (no preprocessing) ---")
arr = np.array(img.convert("RGB"))
res = ocr.model([arr])
lines = []
for page in res.pages:
    for block in page.blocks:
        for line in block.lines:
            words = [w.value for w in line.words]
            lines.append(" ".join(words))
print("\n".join(lines))
