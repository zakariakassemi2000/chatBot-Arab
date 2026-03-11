# -*- coding: utf-8 -*-
"""
Brain Tumor MRI Detector — Swin Transformer (Hugging Face)
Model: Devarshi/Brain_Tumor_Classification
Input:  MRI brain images -> [224, 224, 3]
Output: 4 classes (glioma, meningioma, no_tumor, pituitary_tumor)
"""

import os
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# Class definitions
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
CLASS_LABELS_AR = {
    "glioma_tumor": "ورم دبقي (Glioma)",
    "meningioma_tumor": "ورم سحائي (Meningioma)",
    "no_tumor": "لا يوجد ورم",
    "pituitary_tumor": "ورم الغدة النخامية (Pituitary)",
}
CLASS_ICONS = {
    "glioma_tumor": "🔴",
    "meningioma_tumor": "🟠",
    "no_tumor": "🟢",
    "pituitary_tumor": "🟡",
}

HF_MODEL_ID = "Devarshi/Brain_Tumor_Classification"


class BrainTumorDetector:
    """
    Brain tumor classifier using Swin Transformer from Hugging Face.
    Classifies MRI brain images into: glioma, meningioma, no_tumor, pituitary_tumor.
    Downloads the model from Hugging Face on first use.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.load_error = None
        self._load_model()

    def _load_model(self):
        """Load the Swin Transformer model from Hugging Face."""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch

            hf_token = os.environ.get("HF_TOKEN", None)

            logger.info("Loading brain tumor model: %s", HF_MODEL_ID)
            self.processor = AutoImageProcessor.from_pretrained(
                HF_MODEL_ID, token=hf_token
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                HF_MODEL_ID, token=hf_token
            )
            self.model.eval()
            self.load_error = None
            logger.info("BrainTumorDetector initialized: %s", HF_MODEL_ID)

        except Exception as e:
            self.load_error = str(e)
            self.model = None
            self.processor = None
            logger.error("BrainTumorDetector failed to load", exc_info=True)

    def predict_image(self, pil_image):
        """
        Predict brain tumor type from a PIL Image (MRI scan).
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

            return {
                "class_name": class_name,
                "class_label_ar": CLASS_LABELS_AR[class_name],
                "icon": CLASS_ICONS[class_name],
                "confidence": float(probs[class_idx]),
                "prob_glioma": float(probs[0]),
                "prob_meningioma": float(probs[1]),
                "prob_no_tumor": float(probs[2]),
                "prob_pituitary": float(probs[3]),
            }

        except Exception as e:
            logger.error("BrainTumorDetector prediction failed", exc_info=True)
            return None

    def interpret_result(self, prediction: dict):
        """
        Interpret the brain tumor classification result.
        Returns: (label_ar, explanation_ar, risk_level, style)
        """
        if prediction is None:
            return "غير متاح", "تعذر التحليل", "unknown", "error"

        cls = prediction["class_name"]
        conf = prediction["confidence"] * 100
        icon = prediction["icon"]

        if cls == "no_tumor":
            return (
                f"{icon} لا يوجد ورم ({conf:.1f}%)",
                "لم يتم اكتشاف أي مؤشرات لأورام في صورة الرنين المغناطيسي. يُنصح بالمتابعة الدورية.",
                "normal",
                "success"
            )
        elif cls == "glioma_tumor":
            return (
                f"{icon} ورم دبقي — Glioma ({conf:.1f}%)",
                "رُصدت مؤشرات لورم دبقي (Glioma). هذا النوع يتطلب تقييماً عصبياً عاجلاً ومراجعة أخصائي أورام.",
                "high",
                "danger"
            )
        elif cls == "meningioma_tumor":
            return (
                f"{icon} ورم سحائي — Meningioma ({conf:.1f}%)",
                "رُصدت مؤشرات لورم سحائي (Meningioma). عادةً حميد لكن يحتاج متابعة جراحة أعصاب.",
                "moderate",
                "warning"
            )
        else:  # pituitary_tumor
            return (
                f"{icon} ورم الغدة النخامية — Pituitary ({conf:.1f}%)",
                "رُصدت مؤشرات لورم في الغدة النخامية. يتطلب تقييماً هرمونياً ومراجعة أخصائي غدد صماء.",
                "moderate",
                "warning"
            )
