# -*- coding: utf-8 -*-
"""
Breast Density Detector — MONAI InceptionV3 (BI-RADS A/B/C/D)
Model: MONAI/breast_density_classification (Hugging Face)
Input:  Mammogram images -> [299, 299, 3], float32, [0, 1]
Output: 4 BI-RADS density classes (A, B, C, D)
"""

import os
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# BI-RADS Density Classes
CLASS_NAMES = ["A", "B", "C", "D"]
CLASS_LABELS_AR = {
    "A": "دهني (كثافة منخفضة)",
    "B": "ليفي غدي متناثر",
    "C": "كثيف بشكل غير متجانس",
    "D": "كثيف للغاية",
}
CLASS_LABELS_EN = {
    "A": "Fatty (Almost entirely fatty)",
    "B": "Scattered fibroglandular densities",
    "C": "Heterogeneously dense",
    "D": "Extremely dense",
}
CLASS_ICONS = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}
INPUT_SHAPE = (299, 299)

# HF Model ID
HF_MODEL_REPO = "MONAI/breast_density_classification"
MODEL_FILENAME = "models/model.pt"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "monai_breast_density")
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.pt")


class BreastDensityDetector:
    """
    MONAI-based breast density classifier using InceptionV3.
    Classifies mammograms into BI-RADS density categories (A, B, C, D).
    Downloads the model from Hugging Face on first use.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.input_shape = INPUT_SHAPE
        self.load_error = None
        self._load_model()

    def _download_model(self):
        """Download model from Hugging Face if not already cached."""
        if os.path.exists(LOCAL_MODEL_PATH):
            return True

        try:
            from huggingface_hub import hf_hub_download

            hf_token = os.environ.get("HF_TOKEN", None)
            os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

            logger.info("Downloading breast density model from HuggingFace: %s", HF_MODEL_REPO)
            downloaded_path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=MODEL_FILENAME,
                token=hf_token,
                local_dir=os.path.join(LOCAL_MODEL_DIR, "_hf_cache"),
            )

            import shutil
            shutil.copy2(downloaded_path, LOCAL_MODEL_PATH)
            logger.info("Breast density model downloaded: %s", LOCAL_MODEL_PATH)
            return True

        except Exception as e:
            self.load_error = f"Model download failed: {str(e)}"
            logger.error("Breast density model download failed", exc_info=True)
            return False

    def _load_model(self):
        """Load the InceptionV3 model with MONAI weights."""
        try:
            # Step 1: Download if needed
            if not os.path.exists(LOCAL_MODEL_PATH):
                if not self._download_model():
                    return

            # Step 2: Load model
            import torch
            import torchvision.models as models

            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            model = models.inception_v3(pretrained=False, init_weights=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 4)
            model.AuxLogits = None

            checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "").replace("_model.", "")
                cleaned_state_dict[new_key] = v

            model.load_state_dict(cleaned_state_dict, strict=False)
            model.to(self.device)
            model.eval()

            self.model = model
            self.load_error = None
            logger.info("BreastDensityDetector initialized: InceptionV3 (MONAI) on %s", self.device)

        except Exception as e:
            self.load_error = str(e)
            self.model = None
            logger.error("BreastDensityDetector failed to load", exc_info=True)

    def predict_image(self, pil_image):
        """
        Predict breast density from a PIL Image (mammogram).
        Returns a dict: {density_class, density_label_ar, density_label_en, confidence, all_probs}
        """
        if self.model is None:
            return None

        try:
            import torch
            from torchvision import transforms

            img = pil_image.convert("RGB").resize(self.input_shape)
            img_array = np.array(img, dtype=np.float32) / 255.0

            tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.sigmoid(output).cpu().numpy()[0]

            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum

            class_idx = int(np.argmax(probs))
            density_class = CLASS_NAMES[class_idx]

            return {
                "density_class": density_class,
                "density_label_ar": CLASS_LABELS_AR[density_class],
                "density_label_en": CLASS_LABELS_EN[density_class],
                "confidence": float(probs[class_idx]),
                "icon": CLASS_ICONS[density_class],
                "prob_A": float(probs[0]),
                "prob_B": float(probs[1]),
                "prob_C": float(probs[2]),
                "prob_D": float(probs[3]),
            }

        except Exception as e:
            logger.error("BreastDensityDetector prediction failed", exc_info=True)
            return None

    def interpret_density(self, prediction: dict):
        """
        Interpret the density classification result.
        Returns: (label_ar, explanation_ar, risk_level, style)
        """
        if prediction is None:
            return "غير متاح", "تعذر التحليل", "unknown", "error"

        cls = prediction["density_class"]
        conf = prediction["confidence"] * 100
        icon = prediction["icon"]

        if cls == "A":
            return (
                f"{icon} فئة A — دهني ({conf:.1f}%)",
                "الثدي دهني بالكامل تقريباً. الكثافة منخفضة مما يُسهّل قراءة صور الماموغرام.",
                "low",
                "success"
            )
        elif cls == "B":
            return (
                f"{icon} فئة B — ليفي غدي متناثر ({conf:.1f}%)",
                "توجد مناطق متفرقة من الكثافة الليفية الغدية. معظم الأنسجة دهنية.",
                "moderate_low",
                "info"
            )
        elif cls == "C":
            return (
                f"{icon} فئة C — كثيف بشكل غير متجانس ({conf:.1f}%)",
                "الثدي كثيف بشكل غير متجانس، مما قد يُخفي آفات صغيرة. يُنصح بفحوصات إضافية.",
                "moderate_high",
                "warning"
            )
        else:  # D
            return (
                f"{icon} فئة D — كثيف للغاية ({conf:.1f}%)",
                "الثدي شديد الكثافة مما يُقلل من حساسية الماموغرام. يُوصى بفحوصات تكميلية كالرنين المغناطيسي.",
                "high",
                "danger"
            )
