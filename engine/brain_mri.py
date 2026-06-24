# ============================================================
# SHIFA AI · Brain Tumor Detector (Keras — Multi-Output)
# Description : Détection et classification des tumeurs cérébrales
#               via modèle Keras multi-sortie (tumor_presence + tumor_type)
# Performance : +99% détection · +96% identification du type
# Input       : IRM 260×260 RGB
# Outputs     : tumor_presence (sigmoid) + tumor_type (4 classes softmax)
# Auteur      : SHIFA AI Team
# ============================================================

import os
import numpy as np
import logging
from PIL import Image
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Désactiver les logs TF verbeux
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BrainTumorKerasDetector:
    """
    Détecteur de tumeurs cérébrales basé sur un modèle Keras multi-output.

    Architecture du modèle :
        - Entrée  : (260, 260, 3) — IRM cérébrale en RGB
        - Sortie 1: `tumor_presence` — (1,) sigmoid → probabilité de tumeur
        - Sortie 2: `tumor_type`     — (4,) softmax → classification du type

    Classes de tumeurs (sortie tumor_type) :
        0 = Gliome (glioma_tumor)
        1 = Méningiome (meningioma_tumor)
        2 = Pas de tumeur (no_tumor)
        3 = Tumeur pituitaire (pituitary_tumor)

    Standalone TF (pas de dépendance PyTorch / VisionBase / MONAI).
    """

    MIN_CONFIDENCE = 0.45
    DETECTION_THRESHOLD = 0.50  # Seuil de détection: tumor_presence > 0.5

    def __init__(self):
        self.target_size = (260, 260)
        self.model_path = os.path.join("models", "brain_tumor_model.keras")
        self.device = "cpu"  # TF gère seul (GPU si disponible)
        self.model = self._load_model()

    def _load_model(self):
        """Charge le modèle Keras avec compile=False (custom loss)."""
        if not os.path.exists(self.model_path):
            logger.warning(f"[BrainTumorKerasDetector] Modèle non trouvé : {self.model_path}")
            return None
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(self.model_path, compile=False)
            logger.info(
                f"[BrainTumorKerasDetector] Modèle chargé avec succès "
                f"(input={model.input_shape}, outputs={[o.shape for o in model.outputs]})"
            )
            return model
        except Exception as e:
            logger.error(f"[BrainTumorKerasDetector] Erreur chargement : {e}")
            return None

    def get_vision_type(self) -> str:
        return "brain_mri"

    def _get_classes(self) -> Dict[int, Dict[str, str]]:
        """
        Définit les 4 classes de tumeurs cérébrales.
        L'ordre correspond à la sortie `tumor_type` du modèle.
        """
        return {
            0: {
                "name": "Gliome (Glioma)",
                "severity": "critique",
                "urgency": "emergency",
                "recommendation_ar": (
                    "🔴 اكتشاف يشير إلى ورم دبقي (Glioma). "
                    "يتطلب تدخلاً طبياً عاجلاً وتقييماً جراحياً وعصبياً فورياً. "
                    "يُنصح بمراجعة طبيب الجراحة العصبية في أقرب وقت."
                ),
                "recommendation_fr": (
                    "🔴 Détection évocatrice d'un gliome. "
                    "Intervention médicale urgente requise — consultation neurochirurgicale immédiate."
                )
            },
            1: {
                "name": "Méningiome (Meningioma)",
                "severity": "élevée",
                "urgency": "consult_doctor",
                "recommendation_ar": (
                    "🟠 اكتشاف يشير إلى ورم سحائي (Meningioma)، وهو غالباً حميد "
                    "لكنه يمارس ضغطاً على الدماغ. يجب استشارة طبيب الجراحة العصبية."
                ),
                "recommendation_fr": (
                    "🟠 Détection évocatrice d'un méningiome (généralement bénin). "
                    "Consultation en neurochirurgie recommandée pour évaluation."
                )
            },
            2: {
                "name": "Aucune tumeur (No Tumor)",
                "severity": "faible",
                "urgency": "home_care",
                "recommendation_ar": (
                    "🟢 لا توجد علامات واضحة لورم دماغي في هذه الصورة. "
                    "يوصى بمتابعة الطبيب المختص لأي أعراض أخرى."
                ),
                "recommendation_fr": (
                    "🟢 Aucun signe de tumeur cérébrale détecté. "
                    "Suivi médical recommandé pour tout autre symptôme."
                )
            },
            3: {
                "name": "Tumeur pituitaire (Pituitary)",
                "severity": "élevée",
                "urgency": "consult_doctor",
                "recommendation_ar": (
                    "🟠 اشتباه في ورم في الغدة النخامية (Pituitary Tumor). "
                    "ينصح بمراجعة طبيب الجراحة العصبية والغدد الصماء "
                    "لتقييم التأثير الهرموني والبصري."
                ),
                "recommendation_fr": (
                    "🟠 Suspicion de tumeur hypophysaire. "
                    "Consultation neurochirurgicale et endocrinologique recommandée."
                )
            }
        }

    def is_medical_image(self, image: Image.Image) -> dict:
        """
        Valide que l'image est une IRM cérébrale plausible.
        Critères : résolution, aspect ratio, dominance gris, contraste.
        """
        try:
            w, h = image.size
            if w < 100 or h < 100:
                return {"valid": False, "reason": "Résolution insuffisante (< 100×100)"}

            ratio = w / h
            if ratio < 0.5 or ratio > 2.0:
                return {"valid": False, "reason": "Aspect ratio invalide pour une IRM"}

            img_rgb = np.array(image.convert('RGB')).astype(np.float32)
            r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

            # Les IRM sont quasi-monochromes (canaux R≈G≈B)
            if np.std(r - g) >= 20 or np.std(g - b) >= 20:
                return {"valid": False, "reason": "Déséquilibre de couleur détecté — pas une IRM"}

            # Contraste minimum
            import cv2
            gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            if np.std(gray) <= 40:
                return {"valid": False, "reason": "Contraste trop faible pour une IRM"}

            return {"valid": True, "reason": "OK"}
        except Exception as e:
            return {"valid": False, "reason": f"Erreur validation : {str(e)}"}

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Effectue la détection et classification des tumeurs cérébrales.

        Pipeline :
            1. Validation de l'image (is_medical_image)
            2. Prétraitement (resize 260×260, normalisation [0,1])
            3. Inférence multi-output
            4. Interprétation : détection (seuil 0.5) + classification
            5. Format de sortie unifié SHIFA AI

        Returns:
            Dict conforme au schéma VisionRouter :
            {class, confidence, all_probs, severity, urgency,
             recommendation_ar, gradcam, vision_type, tumor_detected,
             detection_confidence}
        """
        if self.model is None:
            raise RuntimeError(
                "[BrainTumorKerasDetector] Modèle non chargé. "
                "Vérifiez que le fichier brain_tumor_model.keras est dans models/."
            )

        try:
            import tensorflow as tf

            # ── Prétraitement ──────────────────────────────────────
            img = image.convert("RGB").resize(self.target_size)
            arr = np.array(img, dtype=np.float32) / 255.0  # Normalisation [0, 1]
            arr = np.expand_dims(arr, axis=0)  # (1, 260, 260, 3)

            # ── Inférence ──────────────────────────────────────────
            predictions = self.model.predict(arr, verbose=0)
            # predictions = dict {'tumor_presence': (1,1), 'tumor_type': (1,4)}
            # ou list [tumor_presence, tumor_type] selon la version TF

            # Extraction robuste (dict ou list)
            if isinstance(predictions, dict):
                raw_presence = predictions['tumor_presence']
                raw_type = predictions['tumor_type']
            else:
                raw_presence = predictions[0]
                raw_type = predictions[1]

            # Sortie 1 : Détection binaire
            detection_prob = float(raw_presence[0][0])
            tumor_detected = detection_prob >= self.DETECTION_THRESHOLD

            # Sortie 2 : Classification du type (4 classes)
            type_probs = raw_type[0].astype(np.float32)

            # Appliquer softmax si nécessaire (le modèle peut sortir des logits)
            if np.any(type_probs < 0) or np.sum(type_probs) > 1.5:
                type_probs = tf.nn.softmax(type_probs).numpy()
            else:
                # Normaliser pour s'assurer que la somme = 1
                prob_sum = type_probs.sum()
                if prob_sum > 0:
                    type_probs = type_probs / prob_sum

            predicted_idx = int(np.argmax(type_probs))
            type_confidence = float(type_probs[predicted_idx])

            # ── Logique de décision combinée ──────────────────────
            # Les deux têtes (presence + type) ne sont pas toujours calibrées
            # ensemble. On utilise une stratégie combinée :
            #   1. Si tumor_presence >= seuil → tumeur détectée, utiliser type
            #   2. Si tumor_presence < seuil MAIS type classifier est très
            #      confiant (>= 70%) sur une classe TUMEUR (pas no_tumor=idx 2)
            #      → faire confiance au type classifier (évite les faux négatifs)
            #   3. Sinon → pas de tumeur
            classes_meta = self._get_classes()

            # Vérifier si le type classifier est confiant sur une classe tumeur
            tumor_class_indices = [0, 1, 3]  # gliome, méningiome, pituitaire
            type_says_tumor = (
                predicted_idx in tumor_class_indices
                and type_confidence >= 0.70
            )

            if tumor_detected or type_says_tumor:
                # Tumeur détectée (par presence OU par type classifier)
                final_idx = predicted_idx
                final_confidence = type_confidence
                tumor_detected = True  # Mettre à jour le flag
            else:
                # Pas de tumeur détectée
                final_idx = 2  # no_tumor
                final_confidence = 1.0 - detection_prob

            predicted_meta = classes_meta.get(final_idx, classes_meta[2])

            # Probabilités pour toutes les classes
            all_probs = {
                classes_meta[i]["name"]: float(type_probs[i])
                for i in range(len(type_probs))
            }

            return {
                "class": predicted_meta["name"],
                "confidence": final_confidence,
                "all_probs": all_probs,
                "severity": predicted_meta["severity"],
                "urgency": predicted_meta["urgency"],
                "recommendation_ar": predicted_meta["recommendation_ar"],
                "gradcam": None,  # Pas de Grad-CAM TF pour l'instant
                "vision_type": self.get_vision_type(),
                # Métadonnées supplémentaires spécifiques au modèle multi-output
                "tumor_detected": tumor_detected,
                "detection_confidence": detection_prob,
            }

        except Exception as e:
            logger.error(f"[BrainTumorKerasDetector] Erreur prédiction : {e}")
            raise
