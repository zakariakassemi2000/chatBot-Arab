# -*- coding: utf-8 -*-
"""
Dermatology Detector — Skin Disease Classification (Hugging Face)
Model: Jayanth2002/dinov2-base-finetuned-SkinDisease (DinoV2)
Input:  Skin images -> [518, 518, 3]
Output: 31 skin disease classes (ISIC 2018 + Atlas Dermatology)
Accuracy: 95.57%

For educational purposes only — NOT for clinical diagnosis.
"""

import os
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Model config ──
HF_MODEL_ID = "Jayanth2002/dinov2-base-finetuned-SkinDisease"

# ── 31 disease classes with Arabic labels and risk levels ──
CLASS_INFO = {
    "Basal Cell Carcinoma":       {"ar": "سرطان الخلايا القاعدية (BCC)", "risk": "high"},
    "Darier_s Disease":           {"ar": "مرض دارييه", "risk": "moderate"},
    "Epidermolysis Bullosa Pruriginosa": {"ar": "انحلال البشرة الفقاعي الحكّي", "risk": "moderate"},
    "Hailey-Hailey Disease":      {"ar": "مرض هيلي-هيلي", "risk": "moderate"},
    "Herpes Simplex":             {"ar": "الهربس البسيط", "risk": "moderate"},
    "Impetigo":                   {"ar": "القوباء (Impetigo)", "risk": "moderate"},
    "Larva Migrans":              {"ar": "هجرة اليرقات الجلدية", "risk": "moderate"},
    "Leprosy Borderline":         {"ar": "جذام حدّي", "risk": "high"},
    "Leprosy Lepromatous":        {"ar": "جذام ورمي", "risk": "high"},
    "Leprosy Tuberculoid":        {"ar": "جذام درني", "risk": "high"},
    "Lichen Planus":              {"ar": "الحزاز المسطح", "risk": "moderate"},
    "Lupus Erythematosus Chronicus Discoides": {"ar": "الذئبة الحمامية القرصية", "risk": "high"},
    "Melanoma":                   {"ar": "الميلانوما (Melanoma)", "risk": "high"},
    "Molluscum Contagiosum":      {"ar": "المليساء المعدية", "risk": "low"},
    "Mycosis Fungoides":          {"ar": "الفطار الفطراني", "risk": "high"},
    "Neurofibromatosis":          {"ar": "الورم الليفي العصبي", "risk": "moderate"},
    "Papilomatosis Confluentes And Reticulate": {"ar": "الورم الحليمي المتكدس والشبكي", "risk": "low"},
    "Pediculosis Capitis":        {"ar": "قمل الرأس", "risk": "low"},
    "Pityriasis Rosea":           {"ar": "النخالية الوردية", "risk": "low"},
    "Porokeratosis Actinic":      {"ar": "التقرن المسامي الشعاعي", "risk": "moderate"},
    "Psoriasis":                  {"ar": "الصدفية (Psoriasis)", "risk": "moderate"},
    "Tinea Corporis":             {"ar": "سعفة الجسم (Ringworm)", "risk": "low"},
    "Tinea Nigra":                {"ar": "السعفة السوداء", "risk": "low"},
    "Tungiasis":                  {"ar": "داء التونغا", "risk": "moderate"},
    "actinic keratosis":          {"ar": "التقرن السفعي", "risk": "moderate"},
    "dermatofibroma":             {"ar": "الورم الليفي الجلدي", "risk": "low"},
    "nevus":                      {"ar": "الشامات الميلانينية (Nevus)", "risk": "low"},
    "pigmented benign keratosis": {"ar": "التقرن الحميد المصطبغ", "risk": "low"},
    "seborrheic keratosis":       {"ar": "التقرن الدهني", "risk": "low"},
    "squamous cell carcinoma":    {"ar": "سرطان الخلايا الحرشفية (SCC)", "risk": "high"},
    "vascular lesion":            {"ar": "آفات وعائية", "risk": "low"},
}


class DermDetector:
    """
    Skin disease classifier using DinoV2 from Hugging Face.
    Classifies skin images into 31 dermatological conditions.
    Trained on ISIC 2018 + Atlas Dermatology — accuracy: 95.57%.
    For educational purposes only — not a diagnostic tool.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.load_error = None
        self.id2label = None
        self._load_model()

    def _load_model(self):
        """Load the DinoV2 skin disease model from Hugging Face."""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch

            hf_token = os.environ.get("HF_TOKEN", None)
            model_id = HF_MODEL_ID

            logger.info("Loading skin disease model: %s", model_id)
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, token=hf_token
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                model_id, token=hf_token
            )
            logger.info("DermDetector initialized: %s", model_id)

            self.model.eval()
            self.id2label = self.model.config.id2label
            self.load_error = None

        except Exception as e:
            self.load_error = str(e)
            self.model = None
            self.processor = None
            self.id2label = None
            logger.error("DermDetector failed to load", exc_info=True)

    def predict_image(self, pil_image):
        """
        Predict skin disease from a PIL Image.
        Returns a dict with top prediction and all probabilities.
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
            class_name = self.id2label.get(str(class_idx), self.id2label.get(class_idx, f"class_{class_idx}"))
            info = CLASS_INFO.get(class_name, {"ar": class_name, "risk": "unknown"})

            # Build top-5 predictions for display
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5 = []
            for idx in top5_idx:
                label = self.id2label.get(str(idx), self.id2label.get(idx, f"class_{idx}"))
                linfo = CLASS_INFO.get(label, {"ar": label, "risk": "unknown"})
                top5.append({
                    "class_name": label,
                    "class_label_ar": linfo["ar"],
                    "risk": linfo["risk"],
                    "probability": float(probs[idx]),
                })

            result = {
                "class_name": class_name,
                "class_label_ar": info["ar"],
                "risk_level": info["risk"],
                "confidence": float(probs[class_idx]),
                "top5": top5,
            }

            # Add individual probabilities for all classes
            for idx_str, label in self.id2label.items():
                i = int(idx_str)
                safe_key = label.lower().replace(" ", "_").replace("-", "_").replace("'", "")
                result[f"prob_{safe_key}"] = float(probs[i]) if i < len(probs) else 0.0

            return result

        except Exception as e:
            logger.error("DermDetector prediction failed", exc_info=True)
            return None

    def interpret_result(self, prediction: dict):
        """
        Interpret the skin disease classification result.
        Returns: (label_ar, explanation_ar, risk_level, style)
        """
        if prediction is None:
            return "غير متاح", "تعذر التحليل", "unknown", "error"

        cls = prediction["class_name"]
        conf = prediction["confidence"] * 100
        risk = prediction["risk_level"]
        label_ar = prediction["class_label_ar"]

        if risk == "high":
            return (
                f"{label_ar} ({conf:.1f}%)",
                "يُنصح بشدة بمراجعة طبيب جلدية فوراً. لا يعتمد على هذا التحليل للتشخيص.",
                "high",
                "danger",
            )
        elif risk == "moderate":
            return (
                f"{label_ar} ({conf:.1f}%)",
                "يُنصح بمراجعة طبيب جلدية للمتابعة والتقييم.",
                "moderate",
                "warning",
            )
        else:  # low or unknown
            return (
                f"{label_ar} ({conf:.1f}%)",
                "عادةً حميد. يُنصح بالفحص الدوري عند طبيب الجلد.",
                "normal",
                "success",
            )
