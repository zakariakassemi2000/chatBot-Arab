# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  Tests — Medical Image Validator (Image Gatekeeper)
  Validates basic quality checks and color analysis.
  CLIP-based tests are optional (requires model download).
═══════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from PIL import Image
from utils.image_validator import check_basic_quality, check_color_profile, validate_medical_image


def make_image(w, h, color=(128, 128, 128)):
    """Create a solid-color test image."""
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


def make_grayscale_image(w=256, h=256):
    """Simulate a grayscale medical image."""
    gray = np.random.randint(40, 200, (h, w), dtype=np.uint8)
    arr = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(arr)


def make_colorful_image(w=256, h=256):
    """Simulate a colorful non-medical photo."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    # Make it very colorful
    arr[:, :, 0] = np.random.randint(200, 255, (h, w), dtype=np.uint8)  # heavy red
    arr[:, :, 1] = np.random.randint(0, 50, (h, w), dtype=np.uint8)    # low green
    arr[:, :, 2] = np.random.randint(100, 200, (h, w), dtype=np.uint8)  # medium blue
    return Image.fromarray(arr)


class TestBasicQuality:
    """Test image quality checks."""

    def test_valid_image(self):
        img = make_image(256, 256)
        result = check_basic_quality(img)
        assert result["valid"] is True

    def test_too_small(self):
        img = make_image(32, 32)
        result = check_basic_quality(img)
        assert result["valid"] is False
        assert "منخفضة" in result["reason"]

    def test_too_large(self):
        img = make_image(10000, 10000)
        result = check_basic_quality(img)
        assert result["valid"] is False

    def test_weird_aspect_ratio(self):
        img = make_image(1000, 100)  # 10:1 ratio
        result = check_basic_quality(img)
        assert result["valid"] is False

    def test_normal_aspect_ratio(self):
        img = make_image(300, 400)  # 3:4 ratio
        result = check_basic_quality(img)
        assert result["valid"] is True


class TestColorProfile:
    """Test grayscale / color analysis."""

    def test_grayscale_detected(self):
        img = make_grayscale_image()
        result = check_color_profile(img)
        assert result["is_grayscale"] == True
        assert result["medical_likely"] == True

    def test_colorful_detected(self):
        img = make_colorful_image()
        result = check_color_profile(img)
        # Colorful images should be flagged
        assert result["is_grayscale"] == False


class TestFullValidation:
    """Test the full validation pipeline with basic checks only."""

    def test_tiny_image_rejected(self):
        img = make_image(30, 30)
        result = validate_medical_image(img)
        assert result["valid"] is False
        assert "quality_ok" in result
        assert result["quality_ok"] is False

    def test_normal_grayscale_passes_quality(self):
        img = make_grayscale_image()
        result = validate_medical_image(img)
        assert result["quality_ok"] is True
