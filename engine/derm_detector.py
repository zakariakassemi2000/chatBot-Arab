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
import io
import requests
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Model config ──
# Free Inference API endpoint for the Skin Disease model
API_URL = "https://api-inference.huggingface.co/models/Jayanth2002/dinov2-base-finetuned-SkinDisease"

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
    Skin disease classifier using the FREE Hugging Face Inference API.
    Zero local VRAM usage.
    """

    def __init__(self):
        self.hf_token = os.environ.get("HF_TOKEN")
        # Ensure the model attribute exists to satisfy the UI check `if derm_detector.model is None`
        self.model = True if self.hf_token else None
        
        if not self.hf_token:
            logger.error("HF_TOKEN manquante. DermDetector ne fonctionnera pas.")

    def predict_image(self, pil_image):
        """
        Predict skin disease from a PIL Image by calling HF Inference API.
        """
        if not self.hf_token:
            logger.error("Predict failed: HF_TOKEN is missing")
            return None

        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()

            headers = {"Authorization": f"Bearer {self.hf_token}"}
            
            # Call HF Inference API
            response = requests.post(API_URL, headers=headers, data=img_bytes)
            
            if response.status_code != 200:
                logger.error(f"HF API Error: {response.text}")
                return None
                
            predictions = response.json()
            
            if not predictions or not isinstance(predictions, list):
                logger.error("Invalid response format from HF API")
                return None

            # Top prediction is the first item
            top_pred = predictions[0]
            class_name = top_pred.get("label", "unknown")
            confidence = float(top_pred.get("score", 0.0))
            
            info = CLASS_INFO.get(class_name, {"ar": class_name, "risk": "unknown"})

            # Build top-5 predictions for display
            top5 = []
            for pred in predictions[:5]:
                label = pred.get("label", "unknown")
                prob = float(pred.get("score", 0.0))
                linfo = CLASS_INFO.get(label, {"ar": label, "risk": "unknown"})
                top5.append({
                    "class_name": label,
                    "class_label_ar": linfo["ar"],
                    "risk": linfo["risk"],
                    "probability": prob,
                })

            result = {
                "class_name": class_name,
                "class_label_ar": info["ar"],
                "risk_level": info["risk"],
                "confidence": confidence,
                "top5": top5,
            }

            return result

        except Exception as e:
            logger.error(f"DermDetector prediction API failed: {e}", exc_info=True)
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
