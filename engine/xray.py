# ============================================================
# SHIFA AI · X-Ray Model
# Description : Modèle d'analyse de radiographie pulmonaire (DenseNet-121)
# Auteur : SHIFA AI Team
# ============================================================

import logging
import torch
import torch.nn as nn
from torchvision.models import densenet121
from typing import Dict

from engine.vision_base import VisionBase

logger = logging.getLogger(__name__)

class XRayModel(VisionBase):
    """
    Modèle d'analyse de radiographies pulmonaires (Chest X-Ray) basé sur DenseNet-121.
    Reconnaît 3 états: Normal, Pneumonie bactérienne, Pneumonie virale/COVID-19.
    """

    def __init__(self):
        super().__init__(target_size=(224, 224))

    def _load_model(self) -> torch.nn.Module:
        """Charge l'architecture DenseNet-121 avec une tête à 3 classes."""
        try:
            model = densenet121(weights='DEFAULT')
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 3)
            )
            return model
        except Exception as e:
            logger.error(f"[XRayModel] Erreur lors du chargement du modèle: {e}")
            raise

    def _get_target_layer(self) -> torch.nn.Module:
        """Retourne le bloc denseblock4 pour le Grad-CAM."""
        return self.model.features.denseblock4

    def _get_classes(self) -> Dict[int, Dict[str, str]]:
        """Définit les 3 classes pulmonaires مع priorités et actions recommandées."""
        return {
            0: {
                "name": "Normal",
                "severity": "faible",
                "urgency": "home_care",
                "recommendation_ar": "الرئتان سليمات ولا توجد علامات لعدوى. يوصى بالراحة إذا استمرت أعراض أخرى."
            },
            1: {
                "name": "Pneumonie bactérienne",
                "severity": "élevée",
                "urgency": "consult_doctor",
                "recommendation_ar": "علامات لالتهاب رئوي بكتيري. يجب مراجعة الطبيب للحصول على وصفة مضادات حيوية مناسبة. خذ قسطاً كافياً من الراحة."
            },
            2: {
                "name": "Pneumonie virale / COVID-19",
                "severity": "critique",
                "urgency": "emergency",
                "recommendation_ar": "اشتباه قوي بالتهاب رئوي فيروسي أو كوفيد-19. يجب عزل النفس واستشارة الطوارئ فوراً إذا كنت تعاني من ضيق في التنفس."
            }
        }

    def get_vision_type(self) -> str:
        return "xray"
