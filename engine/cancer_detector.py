# -*- coding: utf-8 -*-
"""
Cancer Detector Engine v2
Compatible with: breast_cancer_model_v2.keras (MobileNetV2, TF 2.15+)
Classes: benign (0) | malignant (1) | normal (2)
"""

import os
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_PATH = "models/breast_cancer_model_v2.keras"
INPUT_SHAPE = (224, 224)
CLASS_NAMES = ["benign", "malignant", "normal"]


class BreastCancerDetector:

    def __init__(self):
        self.model = None
        self.input_shape = INPUT_SHAPE
        self.load_error = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.load_error = (
                f"Model file not found: {MODEL_PATH}. "
                "Please run: python train_cancer_model.py"
            )
            logger.warning("Breast cancer model not found: %s", MODEL_PATH)
            return
        try:
            import tensorflow as tf
            self.model = tf.keras.saving.load_model(MODEL_PATH)
            self.load_error = None
            logger.info("BreastCancerDetector initialized: %s", MODEL_PATH)
        except Exception as e:
            self.load_error = str(e)
            self.model = None
            logger.error("BreastCancerDetector failed to load", exc_info=True)

    def predict_image(self, pil_image):
        """
        Predict from a PIL Image.
        Returns a dict: {class_name, confidence, all_probs}
        """
        if self.model is None:
            return None
        try:
            import tensorflow as tf
            img = pil_image.convert("RGB").resize(self.input_shape)
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.array(img, dtype=np.float32)
            )
            arr = np.expand_dims(arr, axis=0)
            probs = self.model.predict(arr, verbose=0)[0]
            class_idx = int(np.argmax(probs))
            return {
                "class_name": CLASS_NAMES[class_idx],
                "confidence": float(probs[class_idx]),
                "benign_prob":    float(probs[0]),
                "malignant_prob": float(probs[1]),
                "normal_prob":    float(probs[2]),
            }
        except Exception as e:
            logger.error("BreastCancerDetector prediction failed", exc_info=True)
            return None

    def interpret_risk(self, prediction: dict):
        """
        Returns: (label_ar, explanation_ar, style)
        """
        if prediction is None:
            return "غير متاح", "تعذر التحليل", "error"

        cls = prediction["class_name"]
        conf = prediction["confidence"] * 100

        if cls == "malignant":
            return (
                f"مؤشر خبيث ({conf:.1f}%)",
                "رُصدت مؤشرات تستوجب مراجعة أخصائي الأورام فوراً.",
                "danger"
            )
        elif cls == "benign":
            return (
                f"حميد ({conf:.1f}%)",
                "المؤشرات تبدو حميدة. يُنصح بالمتابعة الدورية.",
                "success"
            )
        else:  # normal
            return (
                f"طبيعي ({conf:.1f}%)",
                "لم تُرصد علامات مثيرة للقلق. يُنصح بالفحص الدوري.",
                "success"
            )
