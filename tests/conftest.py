# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Shared Test Fixtures (conftest.py)
  Provides reusable fixtures for all test modules.
═══════════════════════════════════════════════════════════════════════
"""
import sys
import os
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

# ── Ensure project root is on sys.path ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Environment ─────────────────────────────────────────────────────
RUN_HEAVY = os.environ.get("RUN_HEAVY_TESTS", "").strip().lower() in ("1", "true", "yes")


# ═══════════════════════════════════════════════════════════════════
#  Image Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def dummy_rgb_image():
    """224×224 random RGB image (standard CV input size)."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def dummy_mri_image():
    """260×260 grayscale-like image simulating a brain MRI."""
    gray = np.random.randint(20, 200, (260, 260), dtype=np.uint8)
    arr = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(arr)


@pytest.fixture
def dummy_xray_image():
    """512×512 grayscale-like image simulating a chest X-ray."""
    gray = np.random.randint(10, 240, (512, 512), dtype=np.uint8)
    arr = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(arr)


@pytest.fixture
def tiny_image():
    """30×30 image that should fail quality checks."""
    arr = np.full((30, 30, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr)


# ═══════════════════════════════════════════════════════════════════
#  Safety & Triage Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def safety_guard():
    """Shared SafetyGuard instance."""
    from engine.safety import SafetyGuard
    return SafetyGuard()


@pytest.fixture(scope="module")
def triage_classifier():
    """Shared MedicalTriageClassifier instance."""
    from engine.triage import MedicalTriageClassifier
    return MedicalTriageClassifier()


# ═══════════════════════════════════════════════════════════════════
#  Mental Health Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def distress_detector():
    """Shared DistressDetector instance."""
    from modules.mental.detector import DistressDetector
    return DistressDetector()


# ═══════════════════════════════════════════════════════════════════
#  Skip Markers
# ═══════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "heavy: requires model loading (slow)")
    config.addinivalue_line("markers", "integration: end-to-end integration test")


def pytest_collection_modifyitems(config, items):
    """Auto-skip heavy tests unless RUN_HEAVY_TESTS=1."""
    if not RUN_HEAVY:
        skip_heavy = pytest.mark.skip(reason="Heavy tests disabled. Set RUN_HEAVY_TESTS=1")
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)
