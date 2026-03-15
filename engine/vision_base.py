# ============================================================
# SHIFA AI · Vision Base
# Description : Classe abstraite fondamentale pour tous les modèles de vision
# Auteur : SHIFA AI Team
# ============================================================

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

class VisionBase(ABC):
    """
    Classe abstraite fondamentale pour les modèles de vision médicale SHIFA AI.
    Définit le pipeline standard (prétraitement, inférence, Grad-CAM, formatage).
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialise le modèle de vision de base.
        
        Args:
            target_size (Tuple[int, int]): Taille (H, W) attendue par le modèle.
        """
        self.target_size = target_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model().to(self.device)
        self.model.eval()
        self.transform = self._build_transform()
        
        # Variables pour Grad-CAM
        self.gradients = None
        self.activations = None
        self._register_hooks()

    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        """Doit retourner l'instance du modèle pré-entraîné."""
        pass

    @abstractmethod
    def _get_target_layer(self) -> torch.nn.Module:
        """Doit retourner la couche cible pour le calcul du Grad-CAM."""
        pass
        
    @abstractmethod
    def _get_classes(self) -> Dict[int, Dict[str, str]]:
        """
        Doit retourner le dictionnaire des classes avec leurs métadonnées.
        Format attendu:
        {
            0: {"name": "Nom de classe", "severity": "...", "urgency": "...", "recommendation_ar": "..."}
        }
        """
        pass
    
    @abstractmethod
    def get_vision_type(self) -> str:
        """Doit retourner le type de vision (ex: 'dermato', 'xray', 'brain_mri')."""
        pass

    def _build_transform(self) -> transforms.Compose:
        """
        Construit le pipeline de transformation standardisé ImageNet.
        
        Returns:
            transforms.Compose: Pipeline de transformation.
        """
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Prétraite l'image (conversion RGB automatique + transformations).
        
        Args:
            image (Image.Image): Image originale.
            
        Returns:
            torch.Tensor: Tensor prêt pour l'inférence [1, C, H, W].
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def _register_hooks(self):
        """Enregistre les hooks forward et backward sur la couche cible pour Grad-CAM."""
        target_layer = self._get_target_layer()
        if target_layer is None:
            logger.warning(f"[{self.__class__.__name__}] target_layer non définie pour Grad-CAM.")
            return

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def compute_gradcam(self, tensor_image: torch.Tensor, class_idx: int) -> Optional[np.ndarray]:
        """
        Calcule la carte d'activation Grad-CAM pour l'image et la classe données.
        
        Args:
            tensor_image (torch.Tensor): Image prétraitée [1, C, H, W].
            class_idx (int): Index de la classe à expliquer.
            
        Returns:
            Optional[np.ndarray]: Heatmap numpy normalisée [H, W], ou None en cas d'erreur.
        """
        if self._get_target_layer() is None:
            return None

        # S'assurer que les gradients/activations sont réinitialisés
        self.gradients = None
        self.activations = None

        try:
            # Mode d'entraînement temporaire pour autoriser les gradients
            self.model.eval()
            tensor_image.requires_grad_(True)
            
            output = self.model(tensor_image)
            
            self.model.zero_grad()
            # Création du tenseur cible pour la classe spécifique
            target = torch.zeros_like(output)
            target[0][class_idx] = 1
            
            # Retropropagation pour obtenir les gradients
            output.backward(gradient=target, retain_graph=True)
            
            if self.gradients is None or self.activations is None:
                return None
                
            gradients = self.gradients.detach().cpu()
            activations = self.activations.detach().cpu()
            
            # Calcul des poids (moyenne globale spatiale des gradients)
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            
            # Pondération des activations
            cam = torch.sum(weights * activations, dim=1).squeeze()
            cam = F.relu(cam) # ReLU
            
            cam_np = cam.numpy()
            
            # Normalisation entre 0 et 1
            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max - cam_min > 1e-8:
                cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_np = np.zeros_like(cam_np)
            
            # Redimensionnement aux dimensions originales (target_size)
            cam_resized = cv2.resize(cam_np, (self.target_size[1], self.target_size[0]))
            return cam_resized
            
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Erreur Grad-CAM: {e}")
            return None
        finally:
            tensor_image.requires_grad_(False)
            self.model.zero_grad()

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Effectue une prédiction sur l'image et retourne le schéma unifié SHIFA AI.
        
        Args:
            image (Image.Image): L'image d'entrée.
            
        Returns:
            Dict[str, Any]: Dictionnaire au format standardisé.
        """
        tensor_image = self.preprocess(image)
        
        try:
            # Phase 1: Inférence sans gradients
            with torch.no_grad():
                outputs = self.model(tensor_image)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
                
            predicted_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_idx])
            
            classes_meta = self._get_classes()
            predicted_meta = classes_meta.get(predicted_idx, {
                "name": "Inconnu",
                "severity": "indéfini",
                "urgency": "indéfinie",
                "recommendation_ar": "يرجى استشارة طبيب مختص."
            })
            
            all_probs = {classes_meta.get(i, {}).get("name", f"Class_{i}"): float(prob) 
                         for i, prob in enumerate(probabilities)}
            
            # Phase 2: Grad-CAM calculé en dehors de torch.no_grad()
            gradcam = self.compute_gradcam(tensor_image.detach().clone(), predicted_idx)
            
            # Nettoyage mémoire GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "class": predicted_meta["name"],
                "confidence": confidence,
                "all_probs": all_probs,
                "severity": predicted_meta.get("severity", "indéfini"),
                "urgency": predicted_meta.get("urgency", "indéfinie"),
                "recommendation_ar": predicted_meta.get("recommendation_ar", ""),
                "gradcam": gradcam,
                "vision_type": self.get_vision_type()
            }
            
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Erreur de prédiction: {e}")
            raise
