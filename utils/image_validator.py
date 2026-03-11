# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Medical Image Validator (Image Gatekeeper)
  
  Multi-layer validation pipeline:
    1. Basic checks (resolution, format, aspect ratio)
    2. Grayscale / color analysis (medical images are often grayscale)
    3. CLIP zero-shot classification (medical vs non-medical)
    4. Image type detection (X-Ray / MRI / Mammogram / Unknown)
    
  Rejects selfies, memes, screenshots, landscapes, etc.
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
from PIL import Image
from utils.logger import get_logger

logger = get_logger(__name__)

# ── CLIP lazy-load ──
_clip_model = None
_clip_processor = None
CLIP_AVAILABLE = False

def _load_clip():
    """Lazy-load CLIP model (only when first needed)."""
    global _clip_model, _clip_processor, CLIP_AVAILABLE
    if _clip_model is not None:
        return True
    try:
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP model for image validation")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        CLIP_AVAILABLE = True
        logger.info("CLIP model ready for medical image validation")
        return True
    except Exception as e:
        logger.warning("CLIP unavailable: %s", e)
        CLIP_AVAILABLE = False
        return False


# ═══════════════════════════════════════════════════════════════
#  CLIP Labels for Zero-Shot Classification
# ═══════════════════════════════════════════════════════════════

MEDICAL_LABELS = [
    "a medical chest x-ray radiograph",
    "a brain MRI scan image",
    "a mammography breast x-ray image",
    "a medical ultrasound image",
    "a CT scan medical image",
]

NON_MEDICAL_LABELS = [
    "a photo of a person or selfie",
    "a photo of food or a meal",
    "a screenshot of a computer or phone",
    "a photo of a landscape or nature",
    "a meme or cartoon drawing",
    "a photo of an animal or pet",
    "a photo of a building or city",
    "a document or text",
]

IMAGE_TYPE_LABELS = {
    "xray": ["a chest x-ray radiograph", "a lung x-ray image", "a thorax radiograph"],
    "mri": ["a brain MRI scan", "a magnetic resonance imaging scan of the brain", "a head MRI"],
    "mammogram": ["a mammography image", "a breast x-ray", "a mammogram scan"],
}


# ═══════════════════════════════════════════════════════════════
#  Layer 1: Basic Image Checks
# ═══════════════════════════════════════════════════════════════

def check_basic_quality(pil_image: Image.Image) -> dict:
    """
    Basic image quality checks.
    
    Returns:
        {"valid": bool, "reason": str, "details": dict}
    """
    img = pil_image.convert("RGB")
    w, h = img.size
    
    details = {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / max(h, 1), 2),
        "total_pixels": w * h,
    }
    
    # Minimum resolution
    if w < 64 or h < 64:
        return {"valid": False, "reason": "دقة الصورة منخفضة جداً (أقل من 64×64 بيكسل).", "details": details}
    
    # Maximum resolution (avoid memory issues)
    if w > 8000 or h > 8000:
        return {"valid": False, "reason": "الصورة كبيرة جداً. الحد الأقصى 8000×8000 بيكسل.", "details": details}
    
    # Weird aspect ratios (panoramas, banners)
    aspect = w / max(h, 1)
    if aspect > 4.0 or aspect < 0.25:
        return {"valid": False, "reason": "نسبة أبعاد الصورة غير عادية. يُرجى تحميل صورة طبية معيارية.", "details": details}
    
    return {"valid": True, "reason": "", "details": details}


# ═══════════════════════════════════════════════════════════════
#  Layer 2: Color / Grayscale Analysis
# ═══════════════════════════════════════════════════════════════

def check_color_profile(pil_image: Image.Image) -> dict:
    """
    Analyze color profile — medical images are typically grayscale or near-grayscale.
    
    Returns:
        {"is_grayscale": bool, "color_variance": float, "medical_likely": bool}
    """
    img = pil_image.convert("RGB")
    # Resize for fast analysis
    img_small = img.resize((128, 128))
    arr = np.array(img_small, dtype=np.float32)
    
    # Check if effectively grayscale (R ≈ G ≈ B)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    channel_diff = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))
    is_grayscale = channel_diff < 15.0  # threshold
    
    # Color variance (medical images tend to have low variance)
    color_variance = float(arr.var())
    
    # High color variance + not grayscale = probably not medical
    too_colorful = color_variance > 5000 and not is_grayscale
    
    return {
        "is_grayscale": is_grayscale,
        "channel_diff": round(float(channel_diff), 2),
        "color_variance": round(color_variance, 2),
        "medical_likely": not too_colorful,
    }


# ═══════════════════════════════════════════════════════════════
#  Layer 3: CLIP Medical vs Non-Medical Classification
# ═══════════════════════════════════════════════════════════════

def classify_medical_clip(pil_image: Image.Image) -> dict:
    """
    Use CLIP zero-shot to determine if image is medical.
    
    Returns:
        {"is_medical": bool, "medical_score": float, "best_label": str, "all_scores": dict}
    """
    if not _load_clip():
        # If CLIP unavailable, assume medical (don't block user)
        return {"is_medical": True, "medical_score": 1.0, "best_label": "unknown", "all_scores": {}}
    
    import torch
    
    img = pil_image.convert("RGB")
    all_labels = MEDICAL_LABELS + NON_MEDICAL_LABELS
    
    inputs = _clip_processor(
        text=all_labels,
        images=img,
        return_tensors="pt",
        padding=True,
    )
    
    with torch.no_grad():
        outputs = _clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    
    # Sum medical vs non-medical probabilities
    n_medical = len(MEDICAL_LABELS)
    medical_score = float(probs[:n_medical].sum())
    non_medical_score = float(probs[n_medical:].sum())
    
    # Find best label
    best_idx = probs.argmax().item()
    best_label = all_labels[best_idx]
    best_prob = float(probs[best_idx])
    
    # Build score dict
    all_scores = {label: round(float(p), 4) for label, p in zip(all_labels, probs)}
    
    return {
        "is_medical": medical_score > 0.45,  # threshold
        "medical_score": round(medical_score, 4),
        "non_medical_score": round(non_medical_score, 4),
        "best_label": best_label,
        "best_prob": round(best_prob, 4),
        "all_scores": all_scores,
    }


# ═══════════════════════════════════════════════════════════════
#  Layer 4: Image Type Detection (X-Ray / MRI / Mammogram)
# ═══════════════════════════════════════════════════════════════

def detect_image_type(pil_image: Image.Image) -> dict:
    """
    Detect what type of medical image this is.
    
    Returns:
        {"type": "xray"|"mri"|"mammogram"|"unknown", "confidence": float, "scores": dict}
    """
    if not _load_clip():
        return {"type": "unknown", "confidence": 0.0, "scores": {}}
    
    import torch
    
    img = pil_image.convert("RGB")
    all_labels = []
    label_to_type = {}
    for img_type, labels in IMAGE_TYPE_LABELS.items():
        for label in labels:
            all_labels.append(label)
            label_to_type[label] = img_type
    
    inputs = _clip_processor(
        text=all_labels,
        images=img,
        return_tensors="pt",
        padding=True,
    )
    
    with torch.no_grad():
        outputs = _clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    
    # Aggregate scores by type
    type_scores = {}
    for label, prob in zip(all_labels, probs):
        t = label_to_type[label]
        type_scores[t] = type_scores.get(t, 0) + float(prob)
    
    # Normalize
    total = sum(type_scores.values())
    if total > 0:
        type_scores = {k: round(v / total, 4) for k, v in type_scores.items()}
    
    best_type = max(type_scores, key=type_scores.get)
    best_score = type_scores[best_type]
    
    return {
        "type": best_type if best_score > 0.4 else "unknown",
        "confidence": best_score,
        "scores": type_scores,
    }


# ═══════════════════════════════════════════════════════════════
#  MAIN VALIDATION PIPELINE
# ═══════════════════════════════════════════════════════════════

def validate_medical_image(pil_image: Image.Image, expected_type: str = None) -> dict:
    """
    Complete validation pipeline for medical images.
    
    Args:
        pil_image: PIL Image to validate
        expected_type: "xray", "mri", "mammogram", or None (any medical)
    
    Returns:
        {
            "valid": bool,
            "reason": str (Arabic error message if invalid),
            "is_medical": bool,
            "medical_score": float,
            "image_type": str,
            "type_match": bool,
            "quality_ok": bool,
            "details": dict,
        }
    """
    result = {
        "valid": False,
        "reason": "",
        "is_medical": False,
        "medical_score": 0.0,
        "image_type": "unknown",
        "type_match": True,
        "quality_ok": False,
        "details": {},
    }
    
    # ── Layer 1: Basic quality ──
    quality = check_basic_quality(pil_image)
    result["quality_ok"] = quality["valid"]
    result["details"]["quality"] = quality["details"]
    
    if not quality["valid"]:
        result["reason"] = f"جودة الصورة: {quality['reason']}"
        return result

    # Dermatology/skin images: skip color/CLIP (skin photos are naturally colorful)
    if expected_type == "derm":
        result["valid"] = True
        result["is_medical"] = True
        result["type_match"] = True
        return result
    
    # ── Layer 2: Color profile ──
    color = check_color_profile(pil_image)
    result["details"]["color"] = color
    
    if not color["medical_likely"]:
        result["reason"] = "هذه الصورة تحتوي على ألوان كثيرة — الصور الطبية عادةً رمادية اللون. هل أنت متأكد أنها صورة طبية؟"
        # Don't return yet — let CLIP confirm
    
    # ── Layer 3: CLIP medical classification ──
    clip_result = classify_medical_clip(pil_image)
    result["is_medical"] = clip_result["is_medical"]
    result["medical_score"] = clip_result["medical_score"]
    result["details"]["clip"] = {
        "medical_score": clip_result["medical_score"],
        "best_label": clip_result["best_label"],
    }
    
    if not clip_result["is_medical"]:
        result["reason"] = (
            "**هذه ليست صورة طبية**\n\n"
            f"النموذج يعتقد أنها: *{clip_result['best_label']}*\n\n"
            "يُرجى تحميل صورة طبية حقيقية (أشعة سينية، رنين مغناطيسي، أو ماموغرام)."
        )
        return result
    
    # ── Layer 4: Image type detection ──
    type_result = detect_image_type(pil_image)
    result["image_type"] = type_result["type"]
    result["details"]["type_detection"] = type_result
    
    # Check type match if expected
    if expected_type and type_result["type"] != "unknown":
        if type_result["type"] != expected_type:
            type_names = {
                "xray": "أشعة سينية (X-Ray)",
                "mri": "رنين مغناطيسي (MRI)",
                "mammogram": "ماموغرام (Mammogram)",
            }
            expected_name = type_names.get(expected_type, expected_type)
            detected_name = type_names.get(type_result["type"], type_result["type"])
            result["type_match"] = False
            result["reason"] = (
                f"**نوع الصورة غير مطابق**\n\n"
                f"هذه الصفحة تتطلب صورة **{expected_name}**\n"
                f"لكن الصورة تبدو أنها **{detected_name}**\n\n"
                f"هل تريد المتابعة رغم ذلك؟"
            )
            # Still mark as valid but with warning
            result["valid"] = True
            return result
    
    # ── All checks passed ──
    result["valid"] = True
    result["reason"] = ""
    return result
