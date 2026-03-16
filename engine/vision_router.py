# ============================================================
# SHIFA AI · Vision Router
# Description : Routeur central pour l'inférence des modèles de vision (Lazy Loading)
# Auteur : SHIFA AI Team
# ============================================================

import logging
from typing import Dict, Any
from PIL import Image

import streamlit as st

logger = logging.getLogger(__name__)

class VisionRouter:
    """
    Routeur de modèles de vision. Charge dynamiquement les modèles demandés (Lazy Loading)
    et unifie l'interface d'inférence pour toute l'application SHIFA AI.
    """
    
    def __init__(self):
        # Dict statique pour lazy loading classique hors Streamlit cache si besoin
        self._models = {}

    @staticmethod
    @st.cache_resource
    def _get_model(vision_type: str):
        """
        Charge et met en cache (Streamlit) le modèle spécifique de manière lazy.
        
        Args:
            vision_type (str): Type de vision ('dermato', 'xray', 'brain_mri')
            
        Returns:
            Instance héritant de VisionBase.
        """
        if vision_type == "dermato":
            from engine.dermato import DermatoModel
            return DermatoModel()
        elif vision_type == "xray":
            from engine.xray import XRayModel
            return XRayModel()
        elif vision_type == "brain_mri":
            from engine.brain_mri import BrainMRIModel
            return BrainMRIModel()
        elif vision_type == "cancer":
            from engine.cancer import CancerDetector
            return CancerDetector()
        elif vision_type == "breast":
            from engine.breast import BreastDensityDetector
            return BreastDensityDetector()
        else:
            raise ValueError(f"[VisionRouter] Type de vision inconnu: {vision_type}")

    def analyze(self, image: Image.Image, vision_type: str) -> Dict[str, Any]:
        """
        Analyse l'image selon le modèle demandé et retourne un schéma standardisé.
        Gère les validations strictes pour la sécurité médicale.
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
                    "rejection_reason": "Image non médicale détectée",
                    "vision_type": vision_type
                }
            
            # Inférence seulement si image valide
            result = model.predict(image)
            result["valid"] = True
            
            # VALIDATION POST-INFÉRENCE
            if result["confidence"] < model.MIN_CONFIDENCE:
                result["valid"] = False
                result["recommendation_ar"] = "⚠️ الصورة غير واضحة أو لا تتطابق مع النموذج المحدد"
            
            if result.get("gradcam") is None and result["valid"]:
                logger.warning(f"[VisionRouter] Grad-CAM n'a pas pu être généré pour {vision_type}.")
            
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
