# -*- coding: utf-8 -*-
"""
Chest X-Ray Classifier — Vision Transformer (ViT) (Hugging Face)
Model: codewithdark/vit-chest-xray
Input:  Chest X-ray images -> [224, 224, 3]
Output: 5 classes (Cardiomegaly, Edema, Consolidation, Pneumonia, No Finding)
Accuracy: 98.46%
Dataset: CheXpert (Stanford)
"""

import os
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# Class definitions
CLASS_NAMES = ["Cardiomegaly", "Edema", "Consolidation", "Pneumonia", "No Finding"]
CLASS_LABELS_AR = {
    "Cardiomegaly": "تضخم القلب (Cardiomegaly)",
    "Edema": "وذمة رئوية (Edema)",
    "Consolidation": "تصلب رئوي (Consolidation)",
    "Pneumonia": "التهاب رئوي (Pneumonia)",
    "No Finding": "لا توجد مشاكل (سليم)",
}
CLASS_ICONS = {
    "Cardiomegaly": "🫀",
    "Edema": "💧",
    "Consolidation": "🫁",
    "Pneumonia": "🔴",
    "No Finding": "🟢",
}

HF_MODEL_ID = "codewithdark/vit-chest-xray"


class ChestXrayAnalyzer:
    """
    Chest X-ray classifier using Vision Transformer (ViT) from Hugging Face.
    Classifies chest X-ray images into 5 categories.
    Uses the CheXpert-trained model with 98.46% validation accuracy.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.load_error = None
        self._load_model()

    def _load_model(self):
        """Load the ViT model from Hugging Face."""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch

            hf_token = os.environ.get("HF_TOKEN", None)

            logger.info("Loading chest X-ray model: %s", HF_MODEL_ID)
            self.processor = AutoImageProcessor.from_pretrained(
                HF_MODEL_ID, token=hf_token
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                HF_MODEL_ID, token=hf_token
            )
            self.model.eval()
            self.load_error = None
            logger.info("ChestXrayAnalyzer initialized: %s", HF_MODEL_ID)

        except Exception as e:
            self.load_error = str(e)
            self.model = None
            self.processor = None
            logger.error("ChestXrayAnalyzer failed to load", exc_info=True)

    def predict_image(self, pil_image):
        """
        Predict chest condition from a PIL Image (chest X-ray).
        Returns a dict with class probabilities.
        """
        if self.model is None or self.processor is None:
            return None

        try:
            import torch

            img = pil_image.convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            class_idx = int(np.argmax(probs))
            class_name = CLASS_NAMES[class_idx]

            result = {
                "class_name": class_name,
                "class_label_ar": CLASS_LABELS_AR[class_name],
                "icon": CLASS_ICONS[class_name],
                "confidence": float(probs[class_idx]),
            }

            for i, name in enumerate(CLASS_NAMES):
                result[f"prob_{name.lower().replace(' ', '_')}"] = float(probs[i])

            return result

        except Exception as e:
            logger.error("ChestXrayAnalyzer prediction failed", exc_info=True)
            return None

    def interpret_result(self, prediction: dict):
        """
        Interpret the chest X-ray classification result.
        Returns: (label_ar, explanation_ar, risk_level, style)
        """
        if prediction is None:
            return "غير متاح", "تعذر التحليل", "unknown", "error"

        cls = prediction["class_name"]
        conf = prediction["confidence"] * 100
        icon = prediction["icon"]

        if cls == "No Finding":
            return (
                f"{icon} سليم — لا توجد مشاكل ({conf:.1f}%)",
                "لم يتم اكتشاف أي مؤشرات مرضية في صورة الأشعة السينية. يُنصح بالفحص الدوري.",
                "normal",
                "success"
            )
        elif cls == "Pneumonia":
            return (
                f"{icon} التهاب رئوي — Pneumonia ({conf:.1f}%)",
                "رُصدت مؤشرات لالتهاب رئوي. يتطلب مراجعة طبيب أمراض صدرية وعلاج فوري.",
                "high",
                "danger"
            )
        elif cls == "Cardiomegaly":
            return (
                f"{icon} تضخم القلب — Cardiomegaly ({conf:.1f}%)",
                "رُصدت مؤشرات لتضخم في عضلة القلب. يتطلب مراجعة أخصائي قلب فوراً.",
                "high",
                "danger"
            )
        elif cls == "Edema":
            return (
                f"{icon} وذمة رئوية — Edema ({conf:.1f}%)",
                "رُصدت مؤشرات لوذمة رئوية (تجمع سوائل). يتطلب تقييماً طبياً عاجلاً.",
                "high",
                "danger"
            )
        else:  # Consolidation
            return (
                f"{icon} تصلب رئوي — Consolidation ({conf:.1f}%)",
                "رُصدت مؤشرات لتصلب في أنسجة الرئة. قد يدل على عدوى أو التهاب. يتطلب مراجعة طبيب.",
                "moderate",
                "warning"
            )
