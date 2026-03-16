# ============================================================
# SHIFA AI · Cancer Detector (Unified)
# Description : Modèle MobileNetV2 pour la détection du cancer du sein (TF)
# ============================================================

import os
import numpy as np
import logging
from PIL import Image
from engine.vision_base import VisionBase
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Désactiver les logs TF verbeux
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CancerDetector(VisionBase):
    """
    Détecteur de cancer du sein basé sur MobileNetV2 (TensorFlow/Keras).
    Classes : Benign, Malignant, Normal.
    """

    def __init__(self):
        # On n'appelle pas super().__init__ car VisionBase est très orienté Torch
        # On réimplémente le nécessaire manuellement pour TF
        self.target_size = (224, 224)
        self.model_path = "models/breast_cancer_model_v2.keras"
        self.model = self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"Modèle cancer non trouvé : {self.model_path}")
            return None
        try:
            import tensorflow as tf
            model = tf.keras.saving.load_model(self.model_path)
            logger.info(f"CancerDetector (TF) chargé depuis {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Erreur chargement CancerDetector: {e}")
            return None

    def _get_target_layer(self):
        # Grad-CAM non implémenté pour TF dans ce pattern unifié pour l'instant
        return None

    def get_vision_type(self) -> str:
        return "cancer"

    def _get_classes(self) -> Dict[int, Dict[str, str]]:
        return {
            0: {
                "name": "Benign",
                "severity": "modérée",
                "urgency": "suivi recommandé",
                "recommendation_ar": "المؤشرات تبدو حميدة. يُنصح بالمتابعة الدورية مع طبيب مختص."
            },
            1: {
                "name": "Malignant",
                "severity": "critique",
                "urgency": "immédiate",
                "recommendation_ar": "رُصدت مؤشرات خبيثة تستوجب مراجعة أخصائي الأورام فوراً لإجراء مزيد من الفحوصات."
            },
            2: {
                "name": "Normal",
                "severity": "normale",
                "urgency": "aucune",
                "recommendation_ar": "لم تُرصد علامات مثيرة للقلق. يُنصح بالفحص الدوري المعتاد."
            }
        }

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Override predict car TF utilise un pipeline différent de Torch."""
        if self.model is None:
            raise RuntimeError("Modèle Cancer non chargé.")

        try:
            import tensorflow as tf
            # Préparation
            img = image.convert("RGB").resize(self.target_size)
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.array(img, dtype=np.float32)
            )
            arr = np.expand_dims(arr, axis=0)

            # Inférence
            probs = self.model.predict(arr, verbose=0)[0]
            predicted_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_idx])

            classes_meta = self._get_classes()
            predicted_meta = classes_meta.get(predicted_idx)

            all_probs = {
                classes_meta[i]["name"]: float(probs[i])
                for i in range(len(probs))
            }

            return {
                "class": predicted_meta["name"],
                "confidence": confidence,
                "all_probs": all_probs,
                "severity": predicted_meta["severity"],
                "urgency": predicted_meta["urgency"],
                "recommendation_ar": predicted_meta["recommendation_ar"],
                "gradcam": None,
                "vision_type": self.get_vision_type()
            }
        except Exception as e:
            logger.error(f"Erreur prédiction Cancer: {e}")
            raise
