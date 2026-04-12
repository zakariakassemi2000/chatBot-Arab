"""
Extraction d'ordonnances via VLM (Vision Language Model).
Remplace : preprocessing.py + ocr.py + nlp_extraction.py en un seul appel API.
Utilise OpenRouter avec le modèle google/gemini-3-flash-preview.
"""
import base64
import json
import logging
from typing import Optional

import requests
from pydantic import BaseModel, Field

from engine.vision_ocr.config import OPENROUTER_API_KEY

logger = logging.getLogger("shifa.vlm")

# ── Modèles Pydantic pour validation stricte du JSON ────────────────

class Medecin(BaseModel):
    nom: Optional[str] = None
    specialite: Optional[str] = None

class Patient(BaseModel):
    nom: Optional[str] = None

class Medicament(BaseModel):
    nom: str
    dosage: Optional[str] = None
    posologie: Optional[str] = None
    duree: Optional[str] = None

class OrdonnanceResult(BaseModel):
    medecin: Optional[Medecin] = None
    patient: Optional[Patient] = None
    medicaments: list[Medicament] = Field(default_factory=list)
    confiance_globale: float = Field(default=0.0, ge=0.0, le=1.0)


# ── Prompt système ──────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant pharmacien expert au Maroc. 
Analyse cette ordonnance médicale (manuscrite ou imprimée, en français et/ou arabe).

INSTRUCTIONS :
- Identifie TOUS les médicaments prescrits, même si l'écriture est difficile à lire.
- Pour chaque médicament, extrais le nom commercial, le dosage, la posologie et la durée.
- Si un champ est illisible ou absent, mets null.
- Évalue ta confiance globale de 0.0 (rien n'est lisible) à 1.0 (tout est parfaitement clair).
- Si tu n'es pas sûr d'un nom de médicament, donne ta meilleure approximation.

Retourne UNIQUEMENT ce JSON valide selon ce format précis SANS markdown :
{
  "medecin": {"nom": "string", "specialite": "string"},
  "patient": {"nom": "string"},
  "medicaments": [
    {"nom": "string", "dosage": "string", "posologie": "string", "duree": "string"}
  ],
  "confiance_globale": 0.0
}"""


# ── Fonction principale ────────────────────────────────────────────

def extract_from_image(image_bytes: bytes) -> OrdonnanceResult:
    """
    Envoie l'image d'ordonnance au VLM via OpenRouter et retourne un résultat structuré.
    """
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY non configurée")
        return OrdonnanceResult()
        
    # Encoder l'image en base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{base64_image}"
    
    logger.info("📤 Envoi de l'image au modèle Gemini 3 Flash Preview (OpenRouter)...")
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "SHIFA-AI"
            },
            json={
                "model": "google/gemini-3-flash-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url
                                }
                            }
                        ]
                    }
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 1024
            },
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Erreur OpenRouter HTTP {response.status_code}: {response.text}")
            return OrdonnanceResult()
            
        json_resp = response.json()
        raw_text = json_resp['choices'][0]['message']['content'].strip()
        
        logger.info(f"📥 Réponse VLM reçue ({len(raw_text)} chars)")
        logger.debug(f"Réponse brute: {raw_text}")
        
        # Nettoyer si le modèle ajoute des backticks markdown (au cas où)
        if raw_text.startswith("```"):
            raw_text = "\n".join([line for line in raw_text.split("\n") if not line.strip().startswith("```")])
            
        # Parser le JSON
        data = json.loads(raw_text)
        result = OrdonnanceResult(**data)
        
        logger.info(f"✅ {len(result.medicaments)} médicament(s) extraits "
                     f"(confiance: {result.confiance_globale:.0%})")
        
        return result
        
    except requests.Timeout:
        logger.error("❌ Timeout lors de la communication avec OpenRouter")
        return OrdonnanceResult()
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON invalide du VLM: {e}\nRéponse: {raw_text[:500]}")
        return OrdonnanceResult()
    except Exception as e:
        logger.error(f"❌ Erreur OpenRouter API: {e}")
        return OrdonnanceResult()
