# ============================================================
# SHIFA AI · Vision Router
# Description : Routeur central pour l'inférence des modèles de vision (Memory Optimized)
# Auteur : SHIFA AI Team
# ============================================================

import logging
import gc
import torch
from typing import Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

class VisionRouter:
    """
    Routeur de modèles de vision optimisé pour la mémoire (1GB RAM Streamlit Cloud).
    Gère le chargement et le déchargement dynamique des modèles.
    """
    
    def __init__(self):
        self._models = {}

    def _get_model(self, vision_type: str):
        """
        Charge le modèle demandé et libère les autres pour économiser la RAM.
        """
        # 1. Libération de la mémoire des autres modèles
        for key in list(self._models.keys()):
            if key != vision_type:
                logger.info(f"[VisionRouter] Déchargement du modèle: {key} pour libérer la RAM")
                del self._models[key]
        
        # Nettoyage profond
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Chargement du modèle cible s'il n'est pas déjà présent
        if vision_type not in self._models:
            logger.info(f"[VisionRouter] Chargement du modèle: {vision_type}")
            if vision_type == "dermato":
                from engine.dermato import DermatoModel
                self._models[vision_type] = DermatoModel()
            elif vision_type == "xray":
                from engine.xray import XRayModel
                self._models[vision_type] = XRayModel()
            elif vision_type == "brain_mri":
                from engine.brain_mri import BrainMRIModel
                self._models[vision_type] = BrainMRIModel()
            elif vision_type == "cancer":
                from engine.cancer import CancerDetectorTF
                self._models[vision_type] = CancerDetectorTF()
            elif vision_type == "breast":
                from engine.breast import BreastDensityDetector
                self._models[vision_type] = BreastDensityDetector()
            else:
                raise ValueError(f"[VisionRouter] Type de vision inconnu: {vision_type}")
        
        return self._models[vision_type]

    def analyze(self, image: Image.Image, vision_type: str) -> Dict[str, Any]:
        """
        Analyse l'image selon le modèle demandé avec gestion stricte de la mémoire.
        """
        try:
            model = self._get_model(vision_type)
            
            # VALIDATION AVANT INFÉRENCE
            validation = model.is_medical_image(image)
            if not validation["valid"]:
                return {
                    "valid": False,
                    "class": None,
                    "confidence": 0.0,
                    "severity": None,
                    "urgency": None,
                    "gradcam": None,
                    "recommendation_ar": "تعذّر تحليل الصورة — يرجى رفع صورة طبية واضحة",
                    "rejection_reason": validation["reason"],
                    "vision_type": vision_type
                }
            
            # Inférence
            result = model.predict(image)
            result["valid"] = True
            
            # VALIDATION POST-INFÉRENCE
            if result["confidence"] < model.MIN_CONFIDENCE:
                result["valid"] = False
                result["recommendation_ar"] = "⚠️ الصورة غير واضحة أو لا تتطابق مع النموذج المحدد"
            
            return result
            
        except Exception as e:
            logger.error(f"[VisionRouter] Échec critique de l'analyse ({vision_type}): {e}")
            return {
                "valid": False,
                "class": "Erreur d'analyse",
                "confidence": 0.0,
                "all_probs": {},
                "severity": "indéfini",
                "urgency": "consult_doctor",
                "recommendation_ar": "حدث خطأ أثناء تحليل الصورة. يرجى إعادة المحاولة.",
                "gradcam": None,
                "vision_type": vision_type
            }
