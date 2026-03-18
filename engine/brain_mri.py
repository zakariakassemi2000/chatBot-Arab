# ============================================================
# SHIFA AI · Brain MRI Model
# Description : Modèle de détection de tumeurs cérébrales (ResNet-50 + MONAI)
# Auteur : SHIFA AI Team
# ============================================================

import logging
import torch
import torch.nn as nn
from torchvision.models import resnet50
from typing import Dict, Tuple
from PIL import Image
import streamlit as st

try:
    from monai.transforms import Compose, ScaleIntensity, Resize, ToTensor
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    from torchvision import transforms

from engine.vision_base import VisionBase

logger = logging.getLogger(__name__)

class BrainMRIModel(VisionBase):
    """
    Modèle d'analyse IRM cérébrales basé sur ResNet-50.
    Intègre les transformations MONAI pour les 4 classes de tumeurs (BraTS-inspired).
    """

    def __init__(self):
        super().__init__(target_size=(224, 224))

    def _build_transform(self):
        """
        Surcharge la méthode transform de VisionBase pour utiliser MONAI si possible
        ou un équivalent torchvision strict.
        """
        if MONAI_AVAILABLE:
            return Compose([
                ScaleIntensity(),
                Resize(self.target_size),
                ToTensor()
            ])
        else:
            from torchvision import transforms
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor()
            ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Prétraite l'image. Comme l'IRM est souvent en niveaux de gris (1 canal),
        on la convertit et réplique en RGB, puis on applique les transformations.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if MONAI_AVAILABLE:
            import numpy as np
            image_np = np.array(image).astype(np.float32) / 255.0
            image_np = np.transpose(image_np, (2, 0, 1)) # (C, H, W)
            tensor = self.transform(image_np).unsqueeze(0)
        else:
            tensor = self.transform(image).unsqueeze(0)
            
        return tensor.to(self.device).float()

    def _load_model(self) -> torch.nn.Module:
        """Charge l'architecture ResNet-50 avec une tête à 4 classes."""
        try:
            model = resnet50(weights='DEFAULT')
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 4)
            return model
        except Exception as e:
            logger.error(f"[BrainMRIModel] Erreur lors du chargement du modèle: {e}")
            raise

    def _get_target_layer(self) -> torch.nn.Module:
        """Retourne le dernier bloc ResNet."""
        return self.model.layer4[-1]

    def _get_classes(self) -> Dict[int, Dict[str, str]]:
        """Définit les 4 classes de tumeurs (ou absence) مع sévérité et avis médical."""
        return {
            0: {
                "name": "Aucune tumeur",
                "severity": "faible",
                "urgency": "home_care",
                "recommendation_ar": "لا توجد علامات واضحة لورم دماغي في هذه الصورة. يوصى بمتابعة الطبيب لأي أعراض أخرى."
            },
            1: {
                "name": "Gliome",
                "severity": "critique",
                "urgency": "emergency",
                "recommendation_ar": "اكتشاف يشير إلى ورم دبقي (Glioma). يتطلب تدخلاً طبياً عاجلاً وجراحة أو علاجاً عصبياً وتنسيقاً فورياً."
            },
            2: {
                "name": "Méningiome",
                "severity": "élevée",
                "urgency": "consult_doctor",
                "recommendation_ar": "اكتشاف يشير إلى ورم سحائي (Meningioma) وهو غالباً حميد لكنه يمارس ضغطاً. يجب استشارة طبيب الجراحة العصبية."
            },
            3: {
                "name": "Tumeur pituitaire",
                "severity": "élevée",
                "urgency": "consult_doctor",
                "recommendation_ar": "اشتباه في ورم في الغدة النخامية. ينصح بمراجعة طاقم الجراحة والغدد الصماء لتقييم التأثير الهرموني والبصري."
            }
        }

    def get_vision_type(self) -> str:
        return "brain_mri"

    def load_model(self):
        """Force le chargement immédiat au démarrage (Warm-up)."""
        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            self.model(dummy_input)

@st.cache_resource(show_spinner="جاري تحميل نموذج الدماغ...")
def load_brain_model():
    model = BrainMRIModel()
    model.load_model()  # Force le chargement immédiat au démarrage
    return model
