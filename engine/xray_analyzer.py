# -*- coding: utf-8 -*-
"""
Chest X-Ray Classifier — TorchXRayVision (Ultra-light)
Model: DenseNet121-res224-all (trained on 5 datasets)
Input:  Chest X-ray images
Output: 18 pathologies probabilities
Size: ~50MB parameters
"""

import os
import numpy as np
from PIL import Image
from utils.logger import get_logger

logger = get_logger(__name__)

# Arabized generic classes
CLASS_LABELS_AR = {
    "Cardiomegaly": "تضخم القلب (Cardiomegaly)",
    "Edema": "وذمة رئوية (Edema)",
    "Consolidation": "تصلب رئوي (Consolidation)",
    "Pneumonia": "التهاب رئوي (Pneumonia)",
    "Effusion": "انصباب جنبي (Effusion)",
    "Pneumothorax": "استرواح الصدر (Pneumothorax)",
    "Atelectasis": "انخماص الرئة (Atelectasis)",
    "Mass": "كتلة (Mass)",
    "Nodule": "عقدة (Nodule)",
    "Infiltration": "ارتشاح (Infiltration)",
    "Lung Opacity": "عتامة الرئة (Opacity)",
    "Fracture": "كسر (Fracture)",
}

# General threshold for detection
THRESHOLD = 0.5


class ChestXrayAnalyzer:
    """
    Ultra-light Chest X-ray classifier using TorchXRayVision.
    Predicts 18 pathologies.
    """

    def __init__(self):
        self.model = None
        self.transform = None
        self.pathologies = []
        self.load_error = None
        self._load_model()

    def _load_model(self):
        """Load TorchXRayVision DenseNet."""
        try:
            import torchxrayvision as xrv
            import torchvision

            logger.info("Loading chest X-ray model: torchxrayvision DenseNet")
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.model.eval()
            self.pathologies = self.model.pathologies
            
            # XRV transform resizes to 224x224 and normalizes to [-1024, 1024]
            self.transform = torchvision.transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224)
            ])
            
            self.load_error = None
            logger.info("ChestXrayAnalyzer initialized (XRV DenseNet)")

        except Exception as e:
            self.load_error = str(e)
            self.model = None
            self.transform = None
            logger.error(f"ChestXrayAnalyzer failed to load: {e}", exc_info=True)

    def predict_image(self, pil_image):
        """
        Predict chest condition using XRV.
        Returns probabilities for all pathologies.
        """
        if self.model is None:
            return None

        try:
            import torch

            # Convert PIL to standard format for XRV (1-channel grayscale)
            img = pil_image.convert('L')
            img_array = np.array(img)
            
            # Normalize to [-1024, 1024] which is XRV standard for 8-bit images
            # formula: (x / 255.0) * 2048.0 - 1024.0
            img_array = (img_array / 255.0) * 2048.0 - 1024.0
            
            # Add color channel
            img_array = img_array[None, ...] # (1, H, W)
            
            # Resize
            img_array = self.transform(img_array)
            
            # Convert to PyTorch Tensor and add batch dim
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(img_tensor)[0]
                
            probs = outputs.cpu().numpy()
            
            # Store all probabilities
            all_probs = {}
            for i, p in enumerate(self.pathologies):
                all_probs[p] = float(probs[i])
                
            # Find the top pathology (ignore some non-specific ones if you want, but here we take max)
            max_idx = np.argmax(probs)
            max_class = self.pathologies[max_idx]
            max_prob = float(probs[max_idx])
            
            # Also determine if "No Finding"
            # If all probabilities are below THRESHOLD, we consider it No Finding
            is_healthy = np.all(probs < THRESHOLD)

            if is_healthy:
                top_class = "No Finding"
                confidence = 1.0 - max_prob # Confidence of being healthy
            else:
                top_class = max_class
                confidence = max_prob

            result = {
                "class_name": top_class,
                "confidence": confidence,
                "all_probs": all_probs,
                "is_healthy": is_healthy
            }
            
            # Add specific prob fields used in tests/pages
            result["prob_cardiomegaly"] = all_probs.get("Cardiomegaly", 0.0)
            result["prob_edema"] = all_probs.get("Edema", 0.0)
            result["prob_consolidation"] = all_probs.get("Consolidation", 0.0)
            result["prob_pneumonia"] = all_probs.get("Pneumonia", 0.0)
            
            return result

        except Exception as e:
            logger.error("ChestXrayAnalyzer prediction failed", exc_info=True)
            return None

    def interpret_result(self, prediction: dict):
        """
        Interpret the X-ray result. Returns: (label_ar, explanation_ar, risk_level, style)
        """
        if prediction is None:
            return "غير متاح", "تعذر التحليل", "unknown", "error"

        if prediction.get("is_healthy", False) or prediction["class_name"] == "No Finding":
            return (
                "🟢 سليم — لا توجد مشاكل",
                "لم يتم اكتشاف أي مؤشرات واضحة للأمراض في صورة الأشعة السينية. يُنصح بالمتابعة في حال استمرار الأعراض.",
                "normal",
                "success"
            )

        cls = prediction["class_name"]
        conf = prediction["confidence"] * 100
        
        ar_name = CLASS_LABELS_AR.get(cls, cls)
        
        if cls in ["Cardiomegaly", "Pneumonia", "Edema", "Pneumothorax", "Mass"]:
            return (
                f"🔴 {ar_name} ({conf:.1f}%)",
                f"تم رصد مؤشرات لـ {ar_name}. هذا يتطلب تدخلاً ومراجعة طبية عاجلة.",
                "high",
                "danger"
            )
        else:
            return (
                f"🟠 {ar_name} ({conf:.1f}%)",
                f"تم رصد مؤشرات لـ {ar_name}. يُرجى مراجعة طبيب مختص للتشخيص النهائي.",
                "moderate",
                "warning"
            )
