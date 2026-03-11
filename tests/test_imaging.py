# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  Tests — Medical Imaging Models
  Validates model loading, prediction format, and interpretation.
═══════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

RUN_HEAVY = os.environ.get("RUN_HEAVY_TESTS", "").strip() in ("1", "true", "TRUE", "yes", "YES")


def create_dummy_image(size=(224, 224)):
    """Create a dummy image for testing (random noise)."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ═══════════ Breast Density (MONAI) ═══════════

class TestBreastDensityDetector:
    @pytest.fixture(scope="class")
    def detector(self):
        if not RUN_HEAVY:
            pytest.skip("Heavy imaging tests disabled. Set RUN_HEAVY_TESTS=1 to enable.")
        from engine.breast_density_detector import BreastDensityDetector
        d = BreastDensityDetector()
        if d.model is None:
            pytest.skip("MONAI model not available")
        return d

    def test_model_loaded(self, detector):
        assert detector.model is not None
        assert detector.load_error is None

    def test_predict_returns_dict(self, detector):
        img = create_dummy_image(size=(299, 299))
        result = detector.predict_image(img)
        assert result is not None
        assert "prob_A" in result
        assert "prob_D" in result

    def test_probabilities_sum_to_one(self, detector):
        img = create_dummy_image(size=(299, 299))
        result = detector.predict_image(img)
        total = sum(result[f"prob_{c}"] for c in "ABCD")
        assert abs(total - 1.0) < 0.01

    def test_interpret_density(self, detector):
        img = create_dummy_image(size=(299, 299))
        prediction = detector.predict_image(img)
        label, explanation, risk, style = detector.interpret_density(prediction)
        assert isinstance(label, str) and len(label) > 0
        assert style in ("success", "info", "warning", "danger", "error")


# ═══════════ Brain Tumor (Swin) ═══════════

class TestBrainTumorDetector:
    @pytest.fixture(scope="class")
    def detector(self):
        if not RUN_HEAVY:
            pytest.skip("Heavy imaging tests disabled. Set RUN_HEAVY_TESTS=1 to enable.")
        from engine.brain_tumor_detector import BrainTumorDetector
        d = BrainTumorDetector()
        if d.model is None:
            pytest.skip("Brain tumor model not available")
        return d

    def test_model_loaded(self, detector):
        assert detector.model is not None

    def test_predict_format(self, detector):
        img = create_dummy_image()
        result = detector.predict_image(img)
        assert result is not None
        assert "class_name" in result
        assert "confidence" in result
        assert result["class_name"] in [
            "glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"
        ]

    def test_probabilities_valid(self, detector):
        img = create_dummy_image()
        result = detector.predict_image(img)
        for key in ["prob_glioma", "prob_meningioma", "prob_no_tumor", "prob_pituitary"]:
            assert 0.0 <= result[key] <= 1.0

    def test_interpret_result(self, detector):
        img = create_dummy_image()
        prediction = detector.predict_image(img)
        label, explanation, risk, style = detector.interpret_result(prediction)
        assert risk in ("normal", "moderate", "high", "unknown")


# ═══════════ Chest X-Ray (ViT) ═══════════

class TestChestXrayAnalyzer:
    @pytest.fixture(scope="class")
    def detector(self):
        if not RUN_HEAVY:
            pytest.skip("Heavy imaging tests disabled. Set RUN_HEAVY_TESTS=1 to enable.")
        from engine.xray_analyzer import ChestXrayAnalyzer
        d = ChestXrayAnalyzer()
        if d.model is None:
            pytest.skip("X-ray model not available")
        return d

    def test_model_loaded(self, detector):
        assert detector.model is not None

    def test_predict_format(self, detector):
        img = create_dummy_image()
        result = detector.predict_image(img)
        assert result is not None
        assert "class_name" in result
        assert result["class_name"] in [
            "Cardiomegaly", "Edema", "Consolidation", "Pneumonia", "No Finding"
        ]

    def test_interpret_result(self, detector):
        img = create_dummy_image()
        prediction = detector.predict_image(img)
        label, explanation, risk, style = detector.interpret_result(prediction)
        assert risk in ("normal", "moderate", "high", "unknown")
        assert isinstance(explanation, str)
