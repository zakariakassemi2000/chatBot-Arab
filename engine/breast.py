# ============================================================
# SHIFA AI · Breast Density Detector (Unified)
# Description : Modèle InceptionV3 MONAI pour la classification BI-RADS
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import logging
from PIL import Image
from engine.vision_base import VisionBase
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class BreastDensityDetector(VisionBase):
    """
    Classifieur de densité mammaire basé sur MONAI InceptionV3.
    Classes BI-RADS : A, B, C, D.
    """

    def __init__(self):
        # On ne passe pas d'arguments à super() car on gère la taille et le modèle spécifiquement
        super().__init__(target_size=(299, 299))
        
    def _load_model(self) -> torch.nn.Module:
        """Charge le modèle InceptionV3 avec les poids MONAI."""
        model_dir = os.path.join("models", "monai_breast_density")
        model_path = os.path.join(model_dir, "model.pt")
        
        # Téléchargement automatique si manquant (logique conservée)
        if not os.path.exists(model_path):
            self._download_monai_model(model_dir, model_path)

        try:
            model = models.inception_v3(pretrained=False, init_weights=False)
            model.fc = nn.Linear(model.fc.in_features, 4)
            model.AuxLogits = None

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extraction du state_dict
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
            else:
                state_dict = checkpoint

            # Nettoyage des clés (prefix module. ou _model.)
            cleaned_state_dict = {
                k.replace("module.", "").replace("_model.", ""): v 
                for k, v in state_dict.items()
            }

            model.load_state_dict(cleaned_state_dict, strict=False)
            logger.info("BreastDensityDetector (MONAI) chargé avec succès.")
            return model
        except Exception as e:
            logger.error(f"Erreur chargement BreastDensityDetector: {e}")
            # Retourne un modèle vide pour éviter un crash au init, mais à gérer
            return models.inception_v3(num_classes=4)

    def _download_monai_model(self, model_dir, model_path):
        """Logique de téléchargement Hugging Face."""
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs(model_dir, exist_ok=True)
            logger.info("Téléchargement du modèle de densité mammaire depuis HF...")
            path = hf_hub_download(
                repo_id="MONAI/breast_density_classification",
                filename="models/model.pt",
                local_dir=os.path.join(model_dir, "_hf_cache")
            )
            import shutil
            shutil.copy2(path, model_path)
        except Exception as e:
            logger.error(f"Échec du téléchargement du modèle : {e}")

    def _get_target_layer(self) -> torch.nn.Module:
        # MixerLayer / Inception Layer final
        return self.model.Mixed_7c

    def get_vision_type(self) -> str:
        return "breast"

    def _get_classes(self) -> Dict[int, Dict[str, str]]:
        return {
            0: {
                "name": "Density A",
                "severity": "faible",
                "urgency": "normale",
                "recommendation_ar": "🟢 فئة A: الثدي دهني بالكامل تقريباً. الكثافة منخفضة مما يُسهّل قراءة صور الماموغرام."
            },
            1: {
                "name": "Density B",
                "severity": "modérée",
                "urgency": "suivi",
                "recommendation_ar": "🟡 فئة B: توجد مناطق متفرقة من الكثافة الليفية الغدية. معظم الأنسجة دهنية."
            },
            2: {
                "name": "Density C",
                "severity": "élevée",
                "urgency": "précaution",
                "recommendation_ar": "🟠 فئة C: الثدي كثيف بشكل غير متجانس، مما قد يُخفي آفات صغيرة. يُنصح بفحوصات إضافية."
            },
            3: {
                "name": "Density D",
                "severity": "très élevée",
                "urgency": "examen complémentaire",
                "recommendation_ar": "🔴 فئة D: الثدي شديد الكثافة مما يُقلل من حساسية الماموغرام. يُوصى بفحوصات تكميلية كالرنين المغناطيسي."
            }
        }
        
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Override preprocess pour normalisation spécifique 0-1."""
        img = image.convert("RGB").resize(self.target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Override pour sigmoid + normalisation des probabilités."""
        tensor_image = self.preprocess(image)
        
        with torch.no_grad():
            output = self.model(tensor_image)
            # Sigmoid car c'est ainsi que le modèle MONAI a été entraîné
            probs = torch.sigmoid(output).cpu().numpy()[0]
            
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
            
        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        
        classes_meta = self._get_classes()
        predicted_meta = classes_meta[predicted_idx]
        
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
            "gradcam": self.compute_gradcam(tensor_image.detach().clone(), predicted_idx),
            "vision_type": self.get_vision_type()
        }
