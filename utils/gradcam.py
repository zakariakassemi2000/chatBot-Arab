# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Grad-CAM Explainability for Medical Imaging
  Generates heatmaps showing WHERE the AI model is looking.
  Supports: Brain Tumor (Swin), Chest X-Ray (ViT), Breast Density.
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
from PIL import Image
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


def generate_gradcam_heatmap(model, processor, pil_image, target_class_idx=None):
    """
    Generate a Grad-CAM heatmap overlay on the input image.
    
    Args:
        model: HuggingFace image classification model
        processor: HuggingFace image processor
        pil_image: PIL Image to explain
        target_class_idx: Which class to explain (None = predicted class)
    
    Returns:
        PIL Image with heatmap overlay, or None if Grad-CAM is not available
    """
    if not GRADCAM_AVAILABLE:
        return None

    try:
        # Preprocess image
        img = pil_image.convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        input_tensor = inputs["pixel_values"]

        # Find the target layer (last attention/conv layer)
        target_layer = _find_target_layer(model)
        if target_layer is None:
            return None

        # Setup Grad-CAM
        targets = None
        if target_class_idx is not None:
            targets = [ClassifierOutputTarget(target_class_idx)]

        # Create a wrapper model that takes pixel_values directly
        class ModelWrapper(torch.nn.Module):
            def __init__(self, hf_model):
                super().__init__()
                self.hf_model = hf_model

            def forward(self, x):
                outputs = self.hf_model(pixel_values=x)
                return outputs.logits

        wrapped = ModelWrapper(model)
        wrapped.eval()

        cam = GradCAM(model=wrapped, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # First image in batch

        # Resize original image to match
        img_resized = img.resize((grayscale_cam.shape[1], grayscale_cam.shape[0]))
        img_np = np.array(img_resized).astype(np.float32) / 255.0

        # Overlay heatmap
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        return Image.fromarray(visualization)

    except Exception as e:
        logger.warning("Grad-CAM generation failed", exc_info=True)
        return None


def _find_target_layer(model):
    """
    Automatically find the best layer for Grad-CAM in different architectures.
    Supports: Swin, ViT, InceptionV3, ResNet, etc.
    """
    try:
        hf_model = model.hf_model if hasattr(model, 'hf_model') else model

        # Swin Transformer (Brain Tumor)
        if hasattr(hf_model, 'swin'):
            layers = hf_model.swin.encoder.layers
            return layers[-1].blocks[-1].layernorm_after

        # ViT (Chest X-Ray)
        if hasattr(hf_model, 'vit'):
            return hf_model.vit.encoder.layer[-1].layernorm_after

        # Generic: try last named module with "norm" or "layer"
        last_layer = None
        for name, module in hf_model.named_modules():
            if 'norm' in name.lower() or 'layer' in name.lower():
                last_layer = module
        return last_layer

    except Exception:
        return None


def is_available():
    """Check if Grad-CAM dependencies are installed."""
    return GRADCAM_AVAILABLE
