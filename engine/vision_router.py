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
        else:
            raise ValueError(f"[VisionRouter] Type de vision inconnu: {vision_type}")

    def analyze(self, image: Image.Image, vision_type: str) -> Dict[str, Any]:
        """
        Analyse l'image selon le modèle demandé et retourne un schéma standardisé.
        Gère les erreurs et assure qu'un résultat décent est renvoyé même en cas de crash partiel.
        
        Args:
            image (Image.Image): L'image à analyser.
            vision_type (str): Le domaine ('dermato', 'xray', 'brain_mri')
            
        Returns:
            Dict[str, Any]: Schéma unifié SHIFA AI
        """
        try:
            model = self._get_model(vision_type)
            result = model.predict(image)
            
            if result.get("gradcam") is None:
                logger.warning(f"[VisionRouter] Grad-CAM n'a pas pu être généré pour {vision_type}.")
            
            return result
            
        except Exception as e:
            logger.error(f"[VisionRouter] Échec critique de l'analyse ({vision_type}): {e}")
            return {
                "class": "Erreur d'analyse",
                "confidence": 0.0,
                "all_probs": {},
                "severity": "indéfini",
                "urgency": "consult_doctor",
                "recommendation_ar": "حدث خطأ أثناء تحليل الصورة. يرجى إعادة المحاولة أو استشارة طبيب.",
                "gradcam": None,
                "vision_type": vision_type
            }
