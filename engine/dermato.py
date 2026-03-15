# ============================================================
# SHIFA AI · Dermato Model
# Description : Modèle de classification dermatologique (7 classes HAM10000)
# Auteur : SHIFA AI Team
# ============================================================

import logging
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from typing import Dict

from engine.vision_base import VisionBase

logger = logging.getLogger(__name__)

class DermatoModel(VisionBase):
    """
    Modèle d'analyse dermatologique basé sur EfficientNet-B3.
    Reconnaît 7 types de lésions cutanées (dataset HAM10000).
    """

    def __init__(self):
        # EfficientNet-B3 optimal resolution is 300x300
        super().__init__(target_size=(300, 300))

    def _load_model(self) -> torch.nn.Module:
        """Charge l'architecture EfficientNet-B3 avec une tête de classification à 7 classes."""
        try:
            model = efficientnet_b3(weights=None)
            # Remplacement de la tête de classification
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(num_ftrs, 7)
            )
            return model
        except Exception as e:
            logger.error(f"[DermatoModel] Erreur lors du chargement du modèle: {e}")
            raise

    def _get_target_layer(self) -> torch.nn.Module:
        """Retourne le dernier bloc de convolution pour le Grad-CAM."""
        return self.model.features[-1]

    def _get_classes(self) -> Dict[int, Dict[str, str]]:
        """Définit les 7 classes avec leur niveau de sévérité et urgence."""
        return {
            0: {
                "name": "Mélanocytaire bénin (nv)",
                "severity": "faible",
                "urgency": "home_care",
                "recommendation_ar": "شامة حميدة أو آفة غير ضارة. يوصى بالمراقبة الروتينية، لا تستدعي القلق."
            },
            1: {
                "name": "Mélanome (mel)",
                "severity": "critique",
                "urgency": "emergency",
                "recommendation_ar": "اشتباه قوي في سرطان الجلد الميلانيني. يجب استشارة طبيب أمراض جلدية فوراً لتأكيد التشخيص واستئصال الآفة."
            },
            2: {
                "name": "Kératose bénigne (bkl)",
                "severity": "faible",
                "urgency": "home_care",
                "recommendation_ar": "آفة جلدية حميدة. لا خطر منها، يمكن إزالتها لأسباب تجميلية فقط."
            },
            3: {
                "name": "Carcinome basocellulaire (bcc)",
                "severity": "élevée",
                "urgency": "consult_doctor",
                "recommendation_ar": "اشتباه في سرطان الخلايا القاعدية. يجب مراجعة الطبيب لتحديد موعد إزالة الآفة جراحياً. قلل التعرض لأشعة الشمس."
            },
            4: {
                "name": "Kératose actinique (akiec)",
                "severity": "modérée",
                "urgency": "consult_doctor",
                "recommendation_ar": "آفة محتملة التسرطن بسب التعرض للشمس. يُنصح بزيارة الطبيب للعلاج الموضعي أو الكي لتجنب تحولها لسرطان."
            },
            5: {
                "name": "Lésion vasculaire (vasc)",
                "severity": "modérée",
                "urgency": "consult_doctor",
                "recommendation_ar": "آفة وعائية. بشكل عام لا تمثل خطراً مباشراً ولكن يفضل تقييم طبي للتﺄكد."
            },
            6: {
                "name": "Dermatofibrome (df)",
                "severity": "faible",
                "urgency": "home_care",
                "recommendation_ar": "ورم ليفي جلدي حميد وشائع، عادة يتكون بعد لدغات الحشرات، لا داعي للقلق."
            }
        }

    def get_vision_type(self) -> str:
        return "dermato"
